"""
Qwen3-VL-Reranker-8B を QLoRA (bitsandbytes 4bit NF4 + LoRA) でファインチューニングする。

Qwen3-VL-Reranker は single-tower の pointwise reranker で、
関連度スコアは最終位置の logit("yes") - logit("no") で表される。
本番検索 (scripts/search.py) は sentence-transformers の CrossEncoder に
Transformer + LogitScore を組み合わせてこのスコアを計算している。

この学習コードは同じ CrossEncoder をそのまま学習対象にする。
つまり学習時の入力構成・チャットテンプレート・スコア計算は、本番推論と完全に同一である。
損失は、そのスコアを logit とみなした二値交差エントロピー (pointwise reranker loss) を使う。

    score = logit("yes") - logit("no")
    loss  = BCEWithLogits(score, label)

LoRA は transformers の PEFT 統合 (add_adapter) でモデル内部へ直接注入する。
モデルを PeftModel でラップしないため、sentence-transformers 側の forward 経路は変わらない。
ViT (vision tower) と merger は既定で凍結し、LLM 側の線形層だけを学習する。

使い方:

    python scripts/train_reranker_qlora.py \\
        --dataset-dir datasets/dashcam_reranker_ft_v1 \\
        --output-dir models/qwen3-vl-reranker-8b-dashcam-v1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from reranker_common import (  # noqa: E402
    DASHCAM_RERANKER_PROMPT,
    RERANKER_MODEL_NAME,
    PairRecord,
    build_predict_pairs,
    load_pairs,
    load_reranker,
    relative_to_project,
    write_json,
)

PROJECT_ROOT = SCRIPTS_DIR.parent

DEFAULT_DATASET_DIR = "datasets/dashcam_reranker_ft_v1"
DEFAULT_OUTPUT_DIR = "models/qwen3-vl-reranker-8b-dashcam-v1"
ADAPTER_NAME = "dashcam"

# Qwen 公式の Qwen3-VL-Reranker 向け LoRA 設定に合わせた既定値。
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# LoRA を LLM 側だけに入れるため、視覚側モジュール名を除外する。
VISION_MODULE_KEYWORDS = ("visual", "vision", "merger", "deepstack")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-Reranker-8B の QLoRA ファインチューニング。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=RERANKER_MODEL_NAME)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--quantization",
        choices=["4bit", "8bit", "none"],
        default="4bit",
        help="4bit が QLoRA。VRAM に余裕がある場合のみ 8bit / none を使う。",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="画像トークン数の上限。既定はモデルの preprocessor_config.json の値。OOM 時に下げる。",
    )
    parser.add_argument("--attn-implementation", default=None, help="例: flash_attention_2")

    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=DEFAULT_TARGET_MODULES,
        help='LoRA を入れる線形層名。"all-linear" を指定すると LLM 側の全線形層を対象にする。',
    )
    parser.add_argument(
        "--train-vision-tower",
        action="store_true",
        help="視覚エンコーダにも LoRA を入れる。既定は凍結。",
    )

    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=2, help="1ステップあたりの前向き件数。")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="正例の重み。既定は train の負例数/正例数から自動計算する。",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        default=True,
    )
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument(
        "--eval-every-steps",
        type=int,
        default=0,
        help="0 ならエポック終了時のみ val 評価する。",
    )
    parser.add_argument("--max-train-pairs", type=int, default=0, help="0 なら全件。動作確認用。")
    parser.add_argument("--max-val-pairs", type=int, default=0, help="0 なら全件。")
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument(
        "--use-caption",
        action="store_true",
        help="候補ドキュメントに事前生成キャプションも含める (caption モード相当)。",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_target_modules(model, requested: list[str], *, train_vision_tower: bool) -> list[str]:
    """
    LoRA を入れる線形層を、実際のモジュール名で列挙する。

    モジュール名を完全修飾で渡すことで、視覚側を確実に除外できる。
    """

    import torch.nn as nn

    suffixes = set(requested)
    use_all_linear = "all-linear" in suffixes

    names: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) and type(module).__name__ not in (
            "Linear4bit",
            "Linear8bitLt",
            "Params4bit",
        ):
            continue
        if name.endswith("lm_head") or "lm_head" in name:
            # 出力ヘッドは yes/no のスコア計算そのものなので凍結する。
            continue
        if not train_vision_tower and any(keyword in name for keyword in VISION_MODULE_KEYWORDS):
            continue
        if use_all_linear or name.split(".")[-1] in suffixes:
            names.append(name)

    if not names:
        raise RuntimeError(
            f"LoRA 対象の線形層が見つかりませんでした: {requested}. "
            "--target-modules を見直してください。"
        )
    return names


def attach_lora(model, args: argparse.Namespace) -> Any:
    """CrossEncoder が内包する transformers モデルへ LoRA を注入する。"""

    from peft import LoraConfig

    transformers_model = model.transformers_model
    target_modules = resolve_target_modules(
        transformers_model,
        args.target_modules,
        train_vision_tower=args.train_vision_tower,
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model.add_adapter(lora_config, adapter_name=ADAPTER_NAME)
    model.set_adapter(ADAPTER_NAME)

    trainable = [
        parameter for parameter in transformers_model.parameters() if parameter.requires_grad
    ]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    total_count = sum(parameter.numel() for parameter in transformers_model.parameters())
    print(
        f"LoRA 対象モジュール: {len(target_modules)} 層 / "
        f"学習パラメータ: {trainable_count:,} ({trainable_count / total_count * 100:.4f}%)"
    )
    return lora_config


def prepare_for_training(model, args: argparse.Namespace) -> None:
    """量子化モデルを学習可能な状態にする。"""

    transformers_model = model.transformers_model
    transformers_model.config.use_cache = False
    if hasattr(transformers_model.config, "text_config"):
        transformers_model.config.text_config.use_cache = False

    if args.gradient_checkpointing:
        transformers_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(transformers_model, "enable_input_require_grads"):
            # gradient checkpointing 下で、凍結した埋め込み層から勾配を流すために必要。
            transformers_model.enable_input_require_grads()

    for name, parameter in transformers_model.named_parameters():
        if "lora_" not in name:
            parameter.requires_grad_(False)


def batched(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def forward_scores(model, pairs: list[PairRecord], *, use_caption: bool):
    """本番と同じ前処理・スコア計算で、勾配の流れる score テンソルを得る。"""

    predict_pairs = build_predict_pairs(pairs, PROJECT_ROOT, use_caption=use_caption)
    features = model.preprocess(predict_pairs, prompt=DASHCAM_RERANKER_PROMPT)
    features = features.to(model.device)
    outputs = model(features)
    return outputs["scores"].view(-1)


def evaluate_val(
    model,
    pairs: list[PairRecord],
    *,
    batch_size: int,
    use_caption: bool,
    pos_weight,
) -> dict[str, float]:
    """val の損失と、順位評価に使う MRR / nDCG@5 を計算する。"""

    import torch
    from torch.nn.functional import binary_cross_entropy_with_logits

    model.eval()
    all_scores: list[float] = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in batched(pairs, batch_size):
            scores = forward_scores(model, batch, use_caption=use_caption)
            labels = torch.tensor(
                [float(pair.label) for pair in batch], device=scores.device, dtype=scores.dtype
            )
            loss = binary_cross_entropy_with_logits(
                scores.float(),
                labels.float(),
                pos_weight=pos_weight.to(scores.device) if pos_weight is not None else None,
            )
            total_loss += float(loss) * len(batch)
            total_count += len(batch)
            all_scores.extend(scores.float().tolist())

    model.train()

    metrics = ranking_metrics(pairs, all_scores)
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics


def ranking_metrics(pairs: list[PairRecord], scores: list[float]) -> dict[str, float]:
    """クエリ単位で候補を並べ替えたときの MRR と nDCG@5 を返す。"""

    by_query: dict[str, list[tuple[float, int]]] = {}
    for pair, score in zip(pairs, scores):
        by_query.setdefault(pair.query_id, []).append((score, pair.label))

    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []

    for candidates in by_query.values():
        if not any(label == 1 for _, label in candidates):
            continue
        ordered = sorted(candidates, key=lambda item: -item[0])

        rank = next(
            (index for index, (_, label) in enumerate(ordered, start=1) if label == 1), None
        )
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        dcg = sum(
            label / math.log2(index + 1) for index, (_, label) in enumerate(ordered[:5], start=1)
        )
        ideal_labels = sorted((label for _, label in candidates), reverse=True)[:5]
        idcg = sum(label / math.log2(index + 1) for index, label in enumerate(ideal_labels, start=1))
        ndcgs.append(dcg / idcg if idcg else 0.0)

    return {
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "ndcg@5": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "queries": float(len(reciprocal_ranks)),
    }


def group_shuffled_batches(
    pairs: list[PairRecord], batch_size: int, rng: random.Random
) -> list[list[PairRecord]]:
    """
    同一クエリの候補が同じバッチへ入りやすいようにまとめる。

    pointwise loss でも、正例と hard negative が同じバッチに入るほうが勾配が安定する。
    """

    by_query: dict[str, list[PairRecord]] = {}
    for pair in pairs:
        by_query.setdefault(pair.query_id, []).append(pair)

    query_ids = list(by_query)
    rng.shuffle(query_ids)

    ordered: list[PairRecord] = []
    for query_id in query_ids:
        group = by_query[query_id]
        rng.shuffle(group)
        ordered.extend(group)

    return batched(ordered, batch_size)


def main() -> None:
    args = parse_args()

    import torch
    from torch.nn.functional import binary_cross_entropy_with_logits

    set_seed(args.seed)

    dataset_dir = (PROJECT_ROOT / args.dataset_dir).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_pairs = load_pairs(dataset_dir / "pairs.train.jsonl")
    val_pairs = load_pairs(dataset_dir / "pairs.val.jsonl")
    if not train_pairs:
        raise RuntimeError(f"{dataset_dir}/pairs.train.jsonl が空です。")

    if args.max_train_pairs > 0:
        train_pairs = train_pairs[: args.max_train_pairs]
    if args.max_val_pairs > 0:
        val_pairs = val_pairs[: args.max_val_pairs]

    positives = sum(1 for pair in train_pairs if pair.label == 1)
    negatives = len(train_pairs) - positives
    print(f"train: {len(train_pairs)} ペア (正例 {positives} / 負例 {negatives})")
    print(f"val:   {len(val_pairs)} ペア")

    pos_weight_value = args.pos_weight
    if pos_weight_value is None:
        pos_weight_value = negatives / positives if positives else 1.0
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32)
    print(f"pos_weight: {pos_weight_value:.3f}")

    print(f"モデルをロードします: {args.model} ({args.quantization})")
    model = load_reranker(
        model_name=args.model,
        device=args.device,
        quantization=args.quantization,
        max_pixels=args.max_pixels,
        attn_implementation=args.attn_implementation,
    )

    prepare_for_training(model, args)
    lora_config = attach_lora(model, args)
    model.train()

    trainable_parameters = [
        parameter for parameter in model.transformers_model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = max(1, len(train_pairs) // args.batch_size)
    total_micro_steps = int(steps_per_epoch * args.epochs)
    total_optimizer_steps = max(1, total_micro_steps // args.gradient_accumulation_steps)
    warmup_steps = max(1, int(total_optimizer_steps * args.warmup_ratio))

    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    history: list[dict[str, Any]] = []
    run_config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "dataset_dir": relative_to_project(dataset_dir, PROJECT_ROOT),
        "quantization": args.quantization,
        "max_pixels": args.max_pixels,
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.target_modules,
            "train_vision_tower": args.train_vision_tower,
            "num_target_layers": len(lora_config.target_modules),
        },
        "optimization": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "pos_weight": pos_weight_value,
            "gradient_checkpointing": args.gradient_checkpointing,
            "seed": args.seed,
            "total_optimizer_steps": total_optimizer_steps,
        },
        "instruction": DASHCAM_RERANKER_PROMPT,
        "use_caption": args.use_caption,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
    }
    write_json(output_dir / "train_config.json", run_config)

    def save_adapter(tag: str, metrics: dict[str, float]) -> Path:
        adapter_dir = output_dir if tag == "best" else output_dir / tag
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.transformers_model.save_pretrained(
            str(adapter_dir), selected_adapters=[ADAPTER_NAME], save_embedding_layers=False
        )
        # PEFT は adapter_name のサブディレクトリへ保存するため、直下へ揃える。
        nested = adapter_dir / ADAPTER_NAME
        if nested.is_dir():
            for path in nested.iterdir():
                path.replace(adapter_dir / path.name)
            nested.rmdir()
        write_json(
            adapter_dir / "adapter_metrics.json",
            {"tag": tag, "metrics": metrics, "config": run_config},
        )
        return adapter_dir

    best_metric = -float("inf")
    micro_step = 0
    optimizer_step = 0
    accumulated_loss = 0.0
    started_at = time.monotonic()
    rng = random.Random(args.seed)

    print(
        f"学習開始: 最適化ステップ {total_optimizer_steps} "
        f"(実効バッチ {args.batch_size * args.gradient_accumulation_steps})"
    )

    if val_pairs:
        baseline = evaluate_val(
            model,
            val_pairs,
            batch_size=args.eval_batch_size,
            use_caption=args.use_caption,
            pos_weight=pos_weight,
        )
        # 学習開始前の val はアダプタ初期値 (LoRA B=0) なのでベースモデルと同じ。
        print(f"[step 0] val(base): {json.dumps(baseline, ensure_ascii=False)}")
        history.append({"optimizer_step": 0, "val": baseline})
        best_metric = baseline["ndcg@5"]

    total_epochs = int(math.ceil(args.epochs))
    stop = False

    for epoch in range(total_epochs):
        if stop:
            break
        for batch in group_shuffled_batches(train_pairs, args.batch_size, rng):
            if micro_step >= total_micro_steps:
                stop = True
                break

            scores = forward_scores(model, batch, use_caption=args.use_caption)
            labels = torch.tensor(
                [float(pair.label) for pair in batch], device=scores.device, dtype=torch.float32
            )
            loss = binary_cross_entropy_with_logits(
                scores.float(), labels, pos_weight=pos_weight.to(scores.device)
            )
            (loss / args.gradient_accumulation_steps).backward()

            accumulated_loss += float(loss.detach())
            micro_step += 1

            if micro_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                if optimizer_step % args.log_every_steps == 0:
                    mean_loss = accumulated_loss / (
                        args.log_every_steps * args.gradient_accumulation_steps
                    )
                    elapsed = time.monotonic() - started_at
                    remaining = (
                        elapsed / optimizer_step * (total_optimizer_steps - optimizer_step)
                        if optimizer_step
                        else 0.0
                    )
                    peak = (
                        torch.cuda.max_memory_allocated() / 1024**3
                        if torch.cuda.is_available()
                        else 0.0
                    )
                    print(
                        f"[epoch {epoch + 1} step {optimizer_step}/{total_optimizer_steps}] "
                        f"loss={mean_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                        f"peak_vram={peak:.1f}GiB 残り約 {remaining / 60:.0f} 分"
                    )
                    history.append(
                        {"optimizer_step": optimizer_step, "train_loss": mean_loss}
                    )
                    accumulated_loss = 0.0

                if (
                    val_pairs
                    and args.eval_every_steps > 0
                    and optimizer_step % args.eval_every_steps == 0
                ):
                    metrics = evaluate_val(
                        model,
                        val_pairs,
                        batch_size=args.eval_batch_size,
                        use_caption=args.use_caption,
                        pos_weight=pos_weight,
                    )
                    print(f"[step {optimizer_step}] val: {json.dumps(metrics, ensure_ascii=False)}")
                    history.append({"optimizer_step": optimizer_step, "val": metrics})
                    if metrics["ndcg@5"] > best_metric:
                        best_metric = metrics["ndcg@5"]
                        save_adapter("best", metrics)
                        print(f"  -> best を更新しました (nDCG@5={best_metric:.4f})")

        if val_pairs and not stop:
            metrics = evaluate_val(
                model,
                val_pairs,
                batch_size=args.eval_batch_size,
                use_caption=args.use_caption,
                pos_weight=pos_weight,
            )
            print(f"[epoch {epoch + 1} 終了] val: {json.dumps(metrics, ensure_ascii=False)}")
            history.append({"epoch": epoch + 1, "optimizer_step": optimizer_step, "val": metrics})
            if metrics["ndcg@5"] > best_metric:
                best_metric = metrics["ndcg@5"]
                save_adapter("best", metrics)
                print(f"  -> best を更新しました (nDCG@5={best_metric:.4f})")

    final_metrics = (
        evaluate_val(
            model,
            val_pairs,
            batch_size=args.eval_batch_size,
            use_caption=args.use_caption,
            pos_weight=pos_weight,
        )
        if val_pairs
        else {}
    )
    save_adapter("last", final_metrics)
    if not (output_dir / "adapter_config.json").exists():
        # val 評価で一度も best を更新しなかった場合でも、直下にアダプタを残す。
        save_adapter("best", final_metrics)

    write_json(
        output_dir / "train_history.json",
        {
            "history": history,
            "best_val_ndcg@5": best_metric,
            "final_val": final_metrics,
            "elapsed_seconds": round(time.monotonic() - started_at, 1),
        },
    )
    print(f"学習完了。アダプタ: {output_dir}")
    print(f"最終 val: {json.dumps(final_metrics, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
