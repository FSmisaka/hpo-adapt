"""Aggregate experiment results into tidy per-figure data files.

The paper figures themselves are drawn by hand; this script exports the exact
data each figure needs (plus optional quick-look PNG previews with --render).

Inputs : experiments/results/<tag>/runs/*.json — runs produced by the four
         experiment scripts (exp1..exp4), all of which default to tag "full"
Outputs: experiments/results/<tags[0]>/figs/
    anytime.csv             Fig. "anytime": best-so-far curves per task/method
    main_summary.csv        Tab. "main": final best (mean+-std), rounds, overhead
    actions.csv             action-frequency distribution per task/method
    episodes.csv            Fig. "learning": per-episode best (cross-episode)
    episodes_summary.csv    mean+-std per episode index
    case_study.json         Fig. "case": one full adaptation episode
    overhead.csv            observation+decision layer overhead shares

Usage
-----
python experiments/make_figures.py                     # tag "full"
python experiments/make_figures.py --tags full         # equivalent
python experiments/make_figures.py --tags smoke        # after --smoke runs
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parent / "results"

METHOD_ORDER = [
    "static-random",
    "static-tpe",
    "adapt-random-action",
    "adapt-rule-only",
    "adapt-rl-default",
    "adapt-rl-guided",
    "adapt-rl",
]


def load_runs(tags: list[str], phase: str) -> list[dict]:
    """Load <phase>__*.json from every tag; identical filenames (produced
    deterministically by more than one experiment) are deduplicated."""
    runs, seen = [], set()
    for tag in tags:
        for f in sorted((RESULTS_ROOT / tag / "runs").glob(f"{phase}__*.json")):
            if f.name in seen:
                continue
            seen.add(f.name)
            runs.append(json.loads(f.read_text()))
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
    parser.add_argument(
        "--tags", nargs="+", default=["full"], help="results tags to aggregate (dedup by filename)"
    )
    parser.add_argument("--render", action="store_true", help="quick-look PNG previews")
    args = parser.parse_args()

    tags = args.tags
    out = RESULTS_ROOT / tags[0] / "figs"
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- anytime
    main_runs = load_runs(tags, "main")
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
    ep_runs = load_runs(tags, "episodes")
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


METHOD_STYLE = {
    "static-random": dict(color="#BBBBBB", linestyle="--", linewidth=1.1),
    "static-tpe": dict(color="#000000", linestyle="--", linewidth=1.1),
    "adapt-random-action": dict(color="#E69F00", linestyle="-", linewidth=1.3),
    "adapt-rule-only": dict(color="#0072B2", linestyle="-", linewidth=1.5),
    "adapt-rl-default": dict(color="#CC79A7", linestyle=":", linewidth=1.3),
    "adapt-rl-guided": dict(color="#D55E00", linestyle="-", linewidth=1.5),
    "adapt-rl": dict(color="#0072B2", linestyle="-", linewidth=1.5),
}

TASK_PANELS = [
    ("synthetic_mlp", "synthetic_mlp"),
    ("branin", "Branin"),
    ("rf_digits", "rf_digits"),
]

# Methods hidden in the anytime preview: adapt-rl-default coincides with the
# unguided control within a single episode, so the two curves would overplot.
ANYTIME_HIDDEN = {"adapt-rl-default"}

RENDER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
}


def _render(out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping previews")
        return
    import csv as _csv

    plt.rcParams.update(RENDER_RC)

    def read(name):
        with open(out / name) as fh:
            return list(_csv.DictReader(fh))

    def save(fig, stem):
        fig.savefig(out / f"{stem}.png", dpi=200)
        fig.savefig(out / f"{stem}.pdf")
        plt.close(fig)

    # ---------------------------------------------------- anytime (mean+-std)
    rows = read("anytime.csv")
    tasks = [t for t, _ in TASK_PANELS if any(r["task"] == t for r in rows)]
    fig, axes = plt.subplots(
        1, len(tasks), figsize=(5.5, 1.95), sharey=False, layout="constrained"
    )
    axes = [axes] if len(tasks) == 1 else axes
    for ax, (task, label) in zip(axes, [(t, l) for t, l in TASK_PANELS if t in tasks]):
        by_method = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["task"] == task:
                by_method[r["method"]][int(r["trial"])].append(float(r["best_so_far"]))
        for method in METHOD_ORDER:
            if method not in by_method or method in ANYTIME_HIDDEN:
                continue
            trials = sorted(by_method[method])
            mean = [statistics.mean(by_method[method][t]) for t in trials]
            std = [statistics.stdev(by_method[method][t]) if len(by_method[method][t]) > 1 else 0.0
                   for t in trials]
            style = METHOD_STYLE[method]
            ax.plot(trials, mean, label=method, **style)
            ax.fill_between(
                trials, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)],
                color=style["color"], alpha=0.13, linewidth=0,
            )
        ax.set_title(label)
        ax.set_xlabel("trials")
    axes[0].set_ylabel("best-so-far")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    save(fig, "preview_anytime")

    # ------------------------------------------------- episodes (mean+-std)
    rows = read("episodes_summary.csv")
    tasks = [t for t, _ in TASK_PANELS if any(r["task"] == t for r in rows)]
    fig, axes = plt.subplots(
        1, len(tasks), figsize=(5.5, 1.95), layout="constrained"
    )
    axes = [axes] if len(tasks) == 1 else axes
    for ax, (task, label) in zip(axes, [(t, l) for t, l in TASK_PANELS if t in tasks]):
        for method in METHOD_ORDER:
            pts = [
                (int(r["episode"]), float(r["best_raw_mean"]), float(r["best_raw_std"]))
                for r in rows
                if r["task"] == task and r["method"] == method
            ]
            if not pts:
                continue
            xs, ys, std = zip(*pts)
            style = METHOD_STYLE[method]
            ax.plot(xs, ys, marker="o", markersize=2.5, label=method, **style)
            ax.fill_between(xs, [y - s for y, s in zip(ys, std)], [y + s for y, s in zip(ys, std)],
                            color=style["color"], alpha=0.13, linewidth=0)
        ax.set_title(label)
        ax.set_xlabel("episode")
    axes[0].set_ylabel("per-episode best")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    save(fig, "preview_episodes")

    # ------------------------------------- case-study (3 stacked mini-panels)
    case_path = out / "case_study.json"
    if case_path.exists():
        _render_case(case_path, out, save)
    print("previews written")


def _line_label_below_xaxis(fig, ax, x_data, parts, fontsize=7.5, gap_pt=4.5):
    """One horizontal label in several colours, just below the x axis.

    parts: list of (text, color), drawn left to right as a single line.
    The line is centered near x_data (data coords) and shifted sideways to
    clear the x tick labels; gap_pt sets the distance from the axis line.
    """
    fig.canvas.draw()
    tmp = [ax.text(0, 0, s, fontsize=fontsize, alpha=0, transform=ax.transAxes)
           for s, _ in parts]
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    widths = [t.get_window_extent(rend).width for t in tmp]
    text_h = max(t.get_window_extent(rend).height for t in tmp)
    for t in tmp:
        t.remove()
    total = sum(widths)
    px = fig.dpi / 72.0
    x_v = ax.transData.transform((x_data, 0.0))[0]
    y_spine = ax.transAxes.transform((0.0, 0.0))[1]
    x_left = x_v - total / 2.0
    y_top = y_spine - gap_pt * px
    y_bot = y_top - text_h
    ticks = [lb.get_window_extent(rend) for lb in ax.get_xticklabels() if lb.get_text()]
    for b in sorted(ticks, key=lambda b: b.x0):
        same_band = b.y1 > y_bot and b.y0 < y_top
        if same_band and x_left + total > b.x0 and x_left < b.x1:
            x_left = (b.x0 - total - 2.0 * px) if b.x0 >= x_v else (b.x1 + 2.0 * px)
    ax_x0 = ax.transAxes.transform((0.0, 0.0))[0]
    ax_x1 = ax.transAxes.transform((1.0, 0.0))[0]
    x_left = min(max(x_left, ax_x0 + 1.0), ax_x1 - total - 1.0)
    inv = ax.transAxes.inverted()
    cur = x_left
    for (s, color), w in zip(parts, widths):
        xf, yf = inv.transform((cur, y_top))
        ax.text(xf, yf, s, fontsize=fontsize, color=color, ha="left", va="top",
                transform=ax.transAxes)
        cur += w


def _render_case(case_path: Path, out: Path, save) -> None:
    import matplotlib.pyplot as plt

    case = json.loads(case_path.read_text())
    steps = case["steps"]
    lr_bounds = []
    for s in steps:
        lr = next(spec for spec in s["search_space"] if spec["name"] == "learning_rate")
        lr_bounds.append((float(lr["low"]), float(lr["high"])))

    fig, axes = plt.subplots(
        3, 1, figsize=(5.5, 3.0), sharex=False, layout="constrained",
        gridspec_kw={"height_ratios": [1.35, 1.0, 0.7]},
    )

    def tag(ax, s):
        ax.text(0.01, 0.96, s, transform=ax.transAxes, fontsize=8, va="top", color="#333333")

    # (a) trial objectives + best-so-far, round boundaries, actions on top
    ax = axes[0]
    tx, ty, bx, by, best, cum = [], [], [], [], None, 0
    for s in steps:
        for t in s["trials"]:
            cum += 1
            v = t["value"]
            if v is None:
                continue
            tx.append(cum)
            ty.append(v)
            if best is None or v > best:
                best = v
            bx.append(cum)
            by.append(best)
    ax.scatter(tx, ty, s=5, color="#999999", alpha=0.55, linewidths=0, zorder=2)
    ax.plot(bx, by, color="#0072B2", linewidth=1.6, zorder=3)
    lo, hi = min(ty), max(ty)
    span = hi - lo if hi > lo else 1.0
    ax.set_ylim(lo - 0.05 * span, hi + 0.36 * span)
    cum = 0
    for s in steps:
        n = len(s["trials"])
        cum += n
        ax.axvline(cum, color="#CCCCCC", linestyle="--", linewidth=0.7, zorder=1)
        act = (s.get("action") or {}).get("action")
        if act:
            tgt = (s.get("action") or {}).get("candidate_hyperparameter") or "-"
            short = "SHIFT" if "SHIFT" in act else "EXPAND" if "EXPAND" in act else act.split("_")[0]
            ax.text(cum - n / 2, hi + 0.18 * span, f"{short}@{tgt}",
                    fontsize=7.5, ha="center", va="bottom", color="#333333")
    ax.set_ylabel("objective")
    tag(ax, "(a)")

    # (b) learning-rate search interval per round (log scale) + sampled points
    ax = axes[1]
    for i, ((lo, hi), s) in enumerate(zip(lr_bounds, steps)):
        ax.hlines(i, lo, hi, linewidth=7, alpha=0.22, color="#0072B2", zorder=2)
        lrs = [t["params"]["learning_rate"] for t in s["trials"] if t["value"] is not None]
        good = [t["params"]["learning_rate"] for t in s["trials"]
                if t["value"] is not None and t["value"] >= 0.9 * max(
                    (tt["value"] for tt in s["trials"] if tt["value"] is not None), default=0)]
        ax.scatter(lrs, [i] * len(lrs), s=7, color="#999999", zorder=3, linewidths=0)
        ax.scatter(good, [i] * len(good), s=16, color="#E69F00", zorder=4, linewidths=0)
    ax.axvline(0.45, color="#D55E00", linestyle=":", linewidth=1.1)
    ax.set_xscale("log")
    ax.set_ylim(-0.55, len(steps) - 0.5)
    ax.set_yticks(range(len(steps)), [f"round {i}" for i in range(len(steps))])
    ax.xaxis.tick_top()
    ax.set_xlabel("learning-rate search interval (log scale)")
    ax.xaxis.labelpad = 16
    tag(ax, "(b)")

    # (c) reward per round, colored by sign
    ax = axes[2]
    rewards = [s.get("reward") for s in steps]
    xs_r = [i for i, r in enumerate(rewards) if r is not None]
    vals = [rewards[i] for i in xs_r]
    ax.bar(xs_r, vals, width=0.5,
           color=["#0072B2" if v >= 0 else "#D55E00" for v in vals], zorder=2)
    ax.axhline(0, color="#999999", linewidth=0.6)
    for i in range(len(steps)):
        if i not in xs_r:
            ax.text(i, 0, "no action", ha="center", va="bottom",
                    fontsize=7, color="#999999")
    flo = min(min(vals), 0) if vals else 0.0
    fhi = max(max(vals), 0) if vals else 1.0
    pad = 0.15 * (fhi - flo) if fhi > flo else 0.1
    ax.set_ylim(flo - pad, fhi + pad)
    ax.set_xlim(-0.6, len(steps) - 0.4)
    ax.set_xticks(range(len(steps)), [f"round {i}" for i in range(len(steps))])
    ax.set_ylabel("reward")
    tag(ax, "(c)")

    # deferred until every artist exists, so the layout below (b) is final
    _line_label_below_xaxis(fig, axes[1], 0.45, [
        ("ground truth ", "#000000"), ("lr", "#D55E00"), (" = 0.45", "#000000"),
    ])

    save(fig, "preview_case_study")


if __name__ == "__main__":
    sys.exit(main())
