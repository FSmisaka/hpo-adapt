"""Aggregate experiment results into tidy per-figure data files.

The paper figures themselves are drawn by hand; this script exports the exact
data each figure needs (plus optional quick-look PNG previews with --render).

Inputs : experiments/results/<tag>/runs/*.json + summary.csv
Outputs: experiments/results/<tag>/figs/
    anytime.csv             Fig. "anytime": best-so-far curves per task/method
    main_summary.csv        Tab. "main": final best (mean+-std), rounds, overhead
    actions.csv             action-frequency distribution per task/method
    episodes.csv            Fig. "learning": per-episode best (cross-episode)
    episodes_summary.csv    mean+-std per episode index
    case_study.json         Fig. "case": one full adaptation episode
    overhead.csv            observation+decision layer overhead shares
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "experiments" / "results"

METHOD_ORDER = [
    "static-random",
    "static-tpe",
    "adapt-random-action",
    "adapt-rule-only",
    "adapt-rl-default",
    "adapt-rl-guided",
    "adapt-rl",
]


def load_runs(tag: str, phase: str) -> list[dict]:
    runs = []
    for f in sorted((RESULTS_ROOT / tag / "runs").glob(f"{phase}__*.json")):
        data = json.loads(f.read_text())
        runs.append(data)
    return runs


def anytime_rows(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        res = run["result"]
        direction = -1.0 if res["best_value_raw"] is not None and _is_min(run) else 1.0
        cum = 0
        best = None
        for step in res["steps"]:
            for t in step["trials"]:
                cum += 1
                v = t["value"]
                if v is None:
                    continue
                if best is None or (direction < 0 and v < best) or (direction > 0 and v > best):
                    best = v
                rows.append(
                    {
                        "task": run["task"],
                        "method": run["method_name"],
                        "seed": run["seed"],
                        "trial": cum,
                        "best_so_far": best,
                    }
                )
    return rows


def _is_min(run: dict) -> bool:
    return run["task"] == "branin"  # the only minimize task in the suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="full")
    parser.add_argument("--render", action="store_true", help="quick-look PNG previews")
    args = parser.parse_args()

    out = RESULTS_ROOT / args.tag / "figs"
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- anytime
    main_runs = load_runs(args.tag, "main")
    rows = anytime_rows(main_runs)
    _write_csv(out / "anytime.csv", rows)
    print(f"anytime.csv: {len(rows)} rows")

    # ------------------------------------------------------------- main table
    by: dict[tuple, list[float]] = defaultdict(list)
    rounds_by: dict[tuple, list[int]] = defaultdict(list)
    trials_by: dict[tuple, list[int]] = defaultdict(list)
    for run in main_runs:
        res = run["result"]
        key = (run["task"], run["method_name"])
        by[key].append(res["best_value_raw"])
        rounds_by[key].append(len(res["steps"]))
        trials_by[key].append(res["total_trials_used"])
    summary = []
    for (task, method), vals in sorted(by.items()):
        summary.append(
            {
                "task": task,
                "method": method,
                "n_seeds": len(vals),
                "best_raw_mean": statistics.mean(vals),
                "best_raw_std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "rounds_mean": statistics.mean(rounds_by[(task, method)]),
                "trials_used_mean": statistics.mean(trials_by[(task, method)]),
            }
        )
    _write_csv(out / "main_summary.csv", summary)
    print(f"main_summary.csv: {len(summary)} rows")

    # ---------------------------------------------------------------- actions
    arows = []
    for run in main_runs:
        if run["method_name"].startswith("static"):
            continue
        counts = Counter()
        for step in run["result"]["steps"]:
            a = (step.get("action") or {}).get("action")
            counts[a or "NONE"] += 1
        for action, c in counts.items():
            arows.append(
                {"task": run["task"], "method": run["method_name"], "action": action, "count": c}
            )
    _write_csv(out / "actions.csv", arows)
    print(f"actions.csv: {len(arows)} rows")

    # --------------------------------------------------------------- overhead
    orows = []
    for run in main_runs:
        t = run["timings"]
        total = t["total"] or 1e-9
        orows.append(
            {
                "task": run["task"],
                "method": run["method_name"],
                "hpo_time": t["hpo"],
                "adapt_time": t["adapt"],
                "total_time": total,
                "adapt_share": t["adapt"] / total,
            }
        )
    _write_csv(out / "overhead.csv", orows)
    agg = defaultdict(list)
    for r in orows:
        agg[(r["task"], r["method"])].append(r["adapt_share"])
    osum = [
        {
            "task": k[0],
            "method": k[1],
            "adapt_share_mean": statistics.mean(v),
            "adapt_share_max": max(v),
        }
        for k, v in sorted(agg.items())
    ]
    _write_csv(out / "overhead_summary.csv", osum)
    print(f"overhead_summary.csv: {len(osum)} rows")

    # --------------------------------------------------------------- episodes
    ep_runs = load_runs(args.tag, "episodes")
    erows = []
    for run in ep_runs:
        for ep in run["episodes_data"]:
            erows.append(
                {
                    "task": run["task"],
                    "method": run["method_name"],
                    "seed": run["seed"],
                    "episode": ep["episode"],
                    "best_raw": ep["result"]["best_value_raw"],
                }
            )
    _write_csv(out / "episodes.csv", erows)
    esum = []
    grouped = defaultdict(list)
    for r in erows:
        grouped[(r["task"], r["method"], r["episode"])].append(r["best_raw"])
    for (task, method, episode), vals in sorted(grouped.items()):
        esum.append(
            {
                "task": task,
                "method": method,
                "episode": episode,
                "best_raw_mean": statistics.mean(vals),
                "best_raw_std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            }
        )
    _write_csv(out / "episodes_summary.csv", esum)
    print(f"episodes_summary.csv: {len(esum)} rows")

    # -------------------------------------------------------------- case study
    case = _pick_case(main_runs)
    if case is not None:
        (out / "case_study.json").write_text(json.dumps(case, indent=1))
        print("case_study.json written")
    else:
        print("WARNING: no case-study run found (no EXPAND of learning_rate)")

    if args.render:
        _render(out)
    return 0


def _pick_case(main_runs: list[dict]) -> dict | None:
    """Pick the most instructive synthetic_mlp adaptive episode: one where the
    controller repaired the truncated learning_rate interval (EXPAND or SHIFT)
    and reached the best final objective."""
    candidates = []
    for pref in ("adapt-rule-only", "adapt-rl-guided", "adapt-rl-default", "adapt-rl"):
        for run in main_runs:
            if run["task"] != "synthetic_mlp" or run["method_name"] != pref:
                continue
            repaired = any(
                (s.get("action") or {}).get("action")
                in ("EXPAND_SEARCH_SPACE", "SHIFT_SEARCH_REGION")
                and (s.get("action") or {}).get("candidate_hyperparameter") == "learning_rate"
                for s in run["result"]["steps"]
            )
            if repaired:
                candidates.append(run)
        if candidates:
            break
    if not candidates:
        return None
    best = max(candidates, key=lambda r: r["result"]["best_value_raw"] or -1)
    steps_out = []
    for s in best["result"]["steps"]:
        steps_out.append(
            {
                "round": s["round_index"],
                "search_space": s["search_space"],
                "diagnosis": s["diagnosis"],
                "action": s.get("action"),
                "changes": s.get("changes"),
                "phase_best": s["phase_best"],
                "cumulative_best": s["cumulative_best"],
                "reward": s["reward"],
                "diagnostic_highlights": {
                    "stagnation": s["diagnostic_state"]["optimization"]["stagnation"],
                    "importance": {
                        k: v.get("global_importance")
                        for k, v in s["diagnostic_state"]["search_space"]["importance"].items()
                    },
                    "boundary_concentration_good": {
                        k: v.get("good")
                        for k, v in s["diagnostic_state"]["search_space"][
                            "boundary_concentration"
                        ].items()
                    },
                },
                "trials": s["trials"],
            }
        )
    return {
        "task": best["task"],
        "method": best["method_name"],
        "seed": best["seed"],
        "best_raw": best["result"]["best_value_raw"],
        "best_params": best["result"]["best_params"],
        "steps": steps_out,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _render(out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping previews")
        return
    import csv as _csv

    def read(name):
        with open(out / name) as fh:
            return list(_csv.DictReader(fh))

    # anytime previews
    rows = read("anytime.csv")
    for task in sorted({r["task"] for r in rows}):
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for method in METHOD_ORDER:
            pts = [
                (int(r["trial"]), float(r["best_so_far"]))
                for r in rows
                if r["task"] == task and r["method"] == method and r["seed"] == "0"
            ]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, label=method, linewidth=1.2)
        ax.set_title(task)
        ax.set_xlabel("trials")
        ax.set_ylabel("best-so-far (raw)")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / f"preview_anytime_{task}.png", dpi=140)
        plt.close(fig)
    # episodes preview
    rows = read("episodes_summary.csv")
    for task in sorted({r["task"] for r in rows}):
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for method in sorted({r["method"] for r in rows}):
            pts = [
                (int(r["episode"]), float(r["best_raw_mean"]))
                for r in rows
                if r["task"] == task and r["method"] == method
            ]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", markersize=3, label=method, linewidth=1.2)
        ax.set_title(f"{task} (per-episode best, mean over seeds)")
        ax.set_xlabel("episode")
        ax.set_ylabel("best raw")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"preview_episodes_{task}.png", dpi=140)
        plt.close(fig)
    print("previews written")


if __name__ == "__main__":
    sys.exit(main())
