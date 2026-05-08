import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path


DATA_DIR = Path("data")
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_SEED = 42


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"日付は YYYY-MM-DD 形式で指定してください: {value}") from exc


def iter_dates(start_date: date, end_date: date) -> list[date]:
    if start_date > end_date:
        raise ValueError("開始日は終了日以前にしてください。")

    day_count = (end_date - start_date).days + 1
    return [start_date + timedelta(days=offset) for offset in range(day_count)]


def build_balanced_date_labels(
    *,
    image_count: int,
    start_date: date,
    end_date: date,
    seed: int,
) -> list[str]:
    all_dates = iter_dates(start_date, end_date)
    base_count, remainder = divmod(image_count, len(all_dates))

    labels = []
    for index, image_date in enumerate(all_dates):
        count = base_count + (1 if index < remainder else 0)
        labels.extend([image_date.isoformat()] * count)

    rng = random.Random(seed)
    rng.shuffle(labels)
    return labels


def main():
    parser = argparse.ArgumentParser(
        description="image_paths.json に対応する実験用の日付メタデータを生成します。"
    )
    parser.add_argument(
        "--paths-json",
        type=Path,
        default=DATA_DIR / "image_paths.json",
        help="画像パス一覧JSON。デフォルト: data/image_paths.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "image_dates.json",
        help="出力先JSON。デフォルト: data/image_dates.json",
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=parse_date(DEFAULT_START_DATE),
        help=f"割り当てる期間の開始日。デフォルト: {DEFAULT_START_DATE}",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=parse_date(DEFAULT_END_DATE),
        help=f"割り当てる期間の終了日。デフォルト: {DEFAULT_END_DATE}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"再現性のための乱数シード。デフォルト: {DEFAULT_SEED}",
    )
    args = parser.parse_args()

    with open(args.paths_json, "r", encoding="utf-8") as f:
        image_paths = json.load(f)

    if not isinstance(image_paths, list) or not all(
        isinstance(path, str) for path in image_paths
    ):
        raise RuntimeError(f"{args.paths_json} は画像パス文字列のリストである必要があります。")

    date_labels = build_balanced_date_labels(
        image_count=len(image_paths),
        start_date=args.start_date,
        end_date=args.end_date,
        seed=args.seed,
    )

    dates_by_path = dict(zip(image_paths, date_labels, strict=True))
    day_count = (args.end_date - args.start_date).days + 1
    base_count, remainder = divmod(len(image_paths), day_count)

    payload = {
        "schema_version": 1,
        "source_paths_json": str(args.paths_json),
        "start_date": args.start_date.isoformat(),
        "end_date": args.end_date.isoformat(),
        "day_count": day_count,
        "image_count": len(image_paths),
        "seed": args.seed,
        "assignment": "balanced_date_labels_shuffled_across_image_paths",
        "per_day_min_count": base_count,
        "per_day_max_count": base_count + (1 if remainder else 0),
        "dates_by_path": dates_by_path,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Loaded: {args.paths_json}")
    print(f"Images: {len(image_paths)}")
    print(
        f"Date range: {args.start_date.isoformat()} - {args.end_date.isoformat()} "
        f"({day_count} days)"
    )
    print(f"Per-day count: {base_count} or {base_count + (1 if remainder else 0)}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
