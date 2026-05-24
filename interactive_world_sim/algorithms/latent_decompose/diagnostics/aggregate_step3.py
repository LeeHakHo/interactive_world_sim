"""Step 3 fungibility-test aggregator + Go/No-Go decision.

Per PHASE0_IWS_PLAN_v3.md §3.4:
  gap = (M_{0.5R_1.0H} - M_{1.5R_0H}) / M_{1.5R_0H}  (lower-better metric)
  gap < 0.20   → STRONG, scale up
  gap < 0.50   → WEAK,   consider Phase 4
  gap >= 0.50  → DIAGNOSE
Required sanity: 0R_1.0H must be worse than 1.5R_0H, else ABORT.
"""
from __future__ import annotations

import argparse
import enum
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class Decision(str, enum.Enum):
    STRONG    = "STRONG"
    WEAK      = "WEAK"
    DIAGNOSE  = "DIAGNOSE"
    ABORT     = "ABORT"


@dataclass
class DecisionResult:
    outcome: Decision
    gap: float
    primary_metric: str
    reason: str


def summarise(
    rows: list[dict[str, Any]],
    metric: str,
) -> dict[str, dict[str, float]]:
    """Group rows by `config`, compute mean ± std of `metric`."""
    by_cfg: dict[str, list[float]] = {}
    for r in rows:
        by_cfg.setdefault(r["config"], []).append(float(r[metric]))
    out = {}
    for c, vs in by_cfg.items():
        mean = statistics.fmean(vs)
        std  = statistics.pstdev(vs) if len(vs) > 1 else 0.0
        out[c] = {"mean": mean, "std": std, "n": len(vs)}
    return out


def decide(
    summary: dict[str, dict[str, float]],
    metric_is_lower_better: bool = True,
) -> DecisionResult:
    need = ("1.5R_0H", "0.5R_1.0H", "0R_1.0H")
    missing = [k for k in need if k not in summary]
    if missing:
        return DecisionResult(
            Decision.ABORT, float("nan"), "n/a",
            f"missing required configs: {missing}",
        )
    M_ref = summary["1.5R_0H"]["mean"]
    M_mix = summary["0.5R_1.0H"]["mean"]
    M_hum = summary["0R_1.0H"]["mean"]

    # Sanity: human-only must be worse than robot-only on the same total budget.
    sanity_ok = (M_hum > M_ref) if metric_is_lower_better else (M_hum < M_ref)
    if not sanity_ok:
        return DecisionResult(
            Decision.ABORT, float("nan"), "primary",
            f"sanity failed: 0R_1.0H not worse than 1.5R_0H "
            f"(M_hum={M_hum:.4f}, M_ref={M_ref:.4f})",
        )

    gap = (M_mix - M_ref) / M_ref if metric_is_lower_better else (M_ref - M_mix) / M_ref
    if gap < 0.20:
        return DecisionResult(Decision.STRONG, gap, "primary",
            "gap < 20% — paper-strong, scale up")
    if gap < 0.50:
        return DecisionResult(Decision.WEAK, gap, "primary",
            "20% ≤ gap < 50% — consider Phase 4 anchor")
    return DecisionResult(Decision.DIAGNOSE, gap, "primary",
        "gap ≥ 50% — diagnose before iterating")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rows-jsonl", type=Path, required=True,
        help="JSONL where each line is {config, seed, fvd, psnr, ...}.",
    )
    ap.add_argument("--metric", default="fvd")
    ap.add_argument(
        "--higher-better", action="store_true",
        help="Set if your primary metric is higher-better (e.g. PSNR).",
    )
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in args.rows_jsonl.read_text().splitlines() if line.strip()]
    summary = summarise(rows, args.metric)
    print(f"[step3] metric={args.metric}, configs:")
    for c, s in sorted(summary.items()):
        print(f"  {c:>12}  {s['mean']:.4f} ± {s['std']:.4f}  (n={s['n']})")
    d = decide(summary, metric_is_lower_better=not args.higher_better)
    print(f"[step3] gap = {d.gap:.4f}")
    print(f"[step3] {d.outcome.value} — {d.reason}")
    return 0 if d.outcome in (Decision.STRONG, Decision.WEAK) else 1


if __name__ == "__main__":
    sys.exit(main())
