#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_CONFIG_ROOT = Path("configs/training")
DEFAULT_OUT_ROOT = Path("runs/benchmarks")


def _repo_src_on_path() -> None:
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if repo_src.exists():
        sys.path.insert(0, str(repo_src))


def _config_records(config_root: Path) -> list[dict[str, str]]:
    records = []
    for path in sorted(config_root.rglob("*.yaml")):
        if path.name.endswith("_bak.yaml"):
            continue
        rel = path.relative_to(config_root)
        if len(rel.parts) != 3:
            continue
        dataset, task, filename = rel.parts
        backbone = Path(filename).stem
        records.append(
            {
                "dataset": dataset,
                "task": task,
                "backbone": backbone,
                "config": str(path),
            }
        )
    return records


def _split_filter(value: Optional[str]) -> Optional[set[str]]:
    if value is None:
        return None
    tokens = {token.strip() for token in value.split(",") if token.strip()}
    return tokens or None


def _matches(record: dict[str, str], args: argparse.Namespace) -> bool:
    datasets = _split_filter(args.datasets)
    backbones = _split_filter(args.backbones)
    tasks = _split_filter(args.tasks)
    if datasets is not None and record["dataset"] not in datasets:
        return False
    if backbones is not None and record["backbone"] not in backbones:
        return False
    if tasks is not None and record["task"] not in tasks:
        return False
    return True


def _read_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _row_from_metrics(
    record: dict[str, str],
    run_dir: Path,
    *,
    status: str,
    elapsed_sec: float,
    error: Optional[str] = None,
) -> dict:
    metrics = _read_metrics(run_dir / "metrics.json")
    return {
        "dataset": record["dataset"],
        "task": record["task"],
        "backbone": record["backbone"],
        "config": record["config"],
        "run_dir": str(run_dir),
        "status": status,
        "elapsed_sec": round(float(elapsed_sec), 3),
        "best_metric": metrics.get("best_metric"),
        "best_score": metrics.get("best_score"),
        "best_cwc": metrics.get("best_cwc"),
        "best_cwc_step": metrics.get("best_cwc_step"),
        "history_steps": metrics.get("history_steps"),
        "has_labels": metrics.get("has_labels"),
        "error": error or "",
    }


def _write_summary(rows: list[dict], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "summary.json"
    csv_path = out_root / "summary.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    fieldnames = [
        "dataset",
        "task",
        "backbone",
        "config",
        "run_dir",
        "status",
        "elapsed_sec",
        "best_metric",
        "best_score",
        "best_cwc",
        "best_cwc_step",
        "history_steps",
        "has_labels",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _extra_train_args(args: argparse.Namespace) -> list[str]:
    extra: list[str] = []
    if args.device:
        extra += ["--device", args.device]
    if args.iter_num is not None:
        extra += ["--iter-num", str(args.iter_num)]
    if args.eval_interval is not None:
        extra += ["--eval-interval", str(args.eval_interval)]
    if args.no_progress:
        extra += ["--no-progress"]
    return extra


def _run_one(record: dict[str, str], args: argparse.Namespace) -> dict:
    from fps_uda.cli import main as fps_main

    run_dir = args.out_root / record["dataset"] / record["task"] / record["backbone"]
    if args.resume and (run_dir / "metrics.json").exists():
        return _row_from_metrics(record, run_dir, status="skipped", elapsed_sec=0.0)
    if args.dry_run:
        return _row_from_metrics(record, run_dir, status="dry_run", elapsed_sec=0.0)

    cmd = [
        "train",
        "--config",
        record["config"],
        "--out",
        str(run_dir),
        *_extra_train_args(args),
    ]
    print()
    print(f"==> {record['dataset']} / {record['task']} / {record['backbone']}")
    print("fps-uda " + " ".join(cmd))
    start = time.time()
    try:
        fps_main(cmd)
    except Exception as exc:
        elapsed = time.time() - start
        if not args.keep_going:
            raise
        return _row_from_metrics(
            record,
            run_dir,
            status="failed",
            elapsed_sec=elapsed,
            error=repr(exc),
        )
    return _row_from_metrics(
        record,
        run_dir,
        status="completed",
        elapsed_sec=time.time() - start,
    )


def run(args: argparse.Namespace) -> list[dict]:
    records = [
        record for record in _config_records(args.config_root) if _matches(record, args)
    ]
    if not records:
        raise SystemExit("No training configs matched the requested filters.")
    if args.limit is not None:
        records = records[: int(args.limit)]

    print(f"Matched {len(records)} training configs.")
    rows: list[dict] = []
    for index, record in enumerate(records, start=1):
        print(
            f"[{index}/{len(records)}] "
            f"{record['dataset']} / {record['task']} / {record['backbone']}"
        )
        row = _run_one(record, args)
        rows.append(row)
        _write_summary(rows, args.out_root)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FPS-UDA benchmark training configs and summarize metrics."
    )
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--datasets", type=str, default=None, help="Comma list filter.")
    parser.add_argument("--backbones", type=str, default=None, help="Comma list filter.")
    parser.add_argument("--tasks", type=str, default=None, help="Comma list filter.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--iter-num", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    _repo_src_on_path()
    parser = build_parser()
    args = parser.parse_args(argv)
    rows = run(args)
    _write_summary(rows, args.out_root)
    print()
    print(f"Wrote {args.out_root / 'summary.json'}")
    print(f"Wrote {args.out_root / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
