"""Write a split-isolated benchmark export for one Day 7 run."""
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "data/benchmark_day7.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    questions = [q for q in source["questions"] if q.get("split") == args.split]
    if not questions:
        raise ValueError(f"Không có câu {args.split} trong benchmark")
    if any(q.get("split") != args.split for q in questions):
        raise ValueError("Export bị lẫn split")
    counts = {
        "answerable": sum(bool(q.get("answerable")) for q in questions),
        "unanswerable": sum(not q.get("answerable") for q in questions),
    }
    output = args.output or ROOT / f"data/{args.split}_benchmark_day7.json"
    result = {
        "version": f"day7-{args.split}-isolated-v1",
        "source": str(args.input.relative_to(ROOT)),
        "split": args.split,
        "note": "Split-isolated export. Gold is for offline scoring only and must never be passed to the agent.",
        "counts": counts,
        "questions": questions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
