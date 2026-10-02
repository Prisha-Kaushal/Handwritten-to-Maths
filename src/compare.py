"""
Build the baseline-vs-fine-tuned results table.
  python compare.py results/baseline_metrics.json results/finetuned_metrics.json
"""

import json
import sys


def load(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    if len(sys.argv) != 3:
        print("usage: python compare.py baseline_metrics.json finetuned_metrics.json")
        return

    base, fine = load(sys.argv[1]), load(sys.argv[2])

    if (base["n"], base["seed"], base["pad"]) != (fine["n"], fine["seed"], fine["pad"]):
        print("WARNING: the two runs used different sample counts, seeds or "
              "padding modes. The comparison is not fair.\n")

    lines = [
        "| Metric | Baseline | Fine-tuned | Change |",
        "|---|---|---|---|",
        f"| Test samples | {base['n']} | {fine['n']} | |",
        f"| Exact Match Rate (%) | {base['exact_match']:.2f} | "
        f"{fine['exact_match']:.2f} | {fine['exact_match'] - base['exact_match']:+.2f} |",
        f"| Character Error Rate (%) | {base['cer_corpus']:.2f} | "
        f"{fine['cer_corpus']:.2f} | {fine['cer_corpus'] - base['cer_corpus']:+.2f} |",
    ]
    table = "\n".join(lines)
    print(table)

    with open("results/comparison.md", "w", encoding="utf-8") as fh:
        fh.write(table + "\n")
    print("\nSaved: results/comparison.md")


if __name__ == "__main__":
    main()