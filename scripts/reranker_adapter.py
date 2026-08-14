"""Qwen3-VL-Reranker の LoRA アダプタを安全に解決・ロードする。"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_ADAPTER_NAME = "dashcam"


def resolve_adapter_path(adapter_path: Path) -> Path:
    """run管理形式の ``best`` と旧形式の直下アダプタの両方を解決する。"""

    adapter_path = adapter_path.expanduser().resolve()
    promoted = adapter_path / "best"
    if (promoted / "adapter_config.json").is_file():
        return promoted.resolve()
    if (adapter_path / "adapter_config.json").is_file():
        return adapter_path
    return adapter_path


def attach_adapter(
    model,
    adapter_path: Path,
    *,
    adapter_name: str = DEFAULT_ADAPTER_NAME,
) -> Path:
    """
    量子化済みCrossEncoderへLoRA層を注入し、重みを読み込む。

    transformersのload_adapterによるモデル全体分のメモリ先取りを避けるため、
    add_adapterで層だけを作成してからPEFTの重みを流し込む。
    """

    from peft import LoraConfig, load_peft_weights, set_peft_model_state_dict

    adapter_path = resolve_adapter_path(adapter_path)
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise RuntimeError(f"{config_path} がありません。アダプタのパスを確認してください。")

    with config_path.open("r", encoding="utf-8") as f:
        raw_config = json.load(f)

    lora_config = LoraConfig(
        **{
            key: value
            for key, value in raw_config.items()
            if key in LoraConfig.__dataclass_fields__ and key != "task_type"
        },
        task_type=raw_config.get("task_type"),
    )
    model.add_adapter(lora_config, adapter_name=adapter_name)
    model.set_adapter(adapter_name)

    state_dict = load_peft_weights(str(adapter_path))
    result = set_peft_model_state_dict(
        model.transformers_model,
        state_dict,
        adapter_name=adapter_name,
    )

    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(f"アダプタの重みに未知のキーがあります (先頭5件): {unexpected[:5]}")

    loaded = sum(
        1
        for name, parameter in model.transformers_model.named_parameters()
        if "lora_B" in name
        and adapter_name in name
        and parameter.detach().abs().sum().item() > 0
    )
    if loaded == 0:
        raise RuntimeError(
            f"{adapter_path} のLoRA重みが読み込まれていません "
            "(lora_Bがすべて0のため、ベースモデルと同じ出力になります)。"
        )
    return adapter_path
