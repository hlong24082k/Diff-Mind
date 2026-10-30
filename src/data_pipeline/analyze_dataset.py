"""Dataset analysis charts for the M1 report: reads data/{raw_diffs,labeled,balanced}.jsonl, writes
PNGs + a markdown table view of every chart's numbers to docs/figures/.

Run from the repo root: `python -m src.data_pipeline.analyze_dataset`

Colour rules (dataviz skill): defective = blue accent, clean = gray; data sources keep one categorical colour in
every chart (blue/orange/aqua, first 3 slots validated for all-pairs CVD safety); severity is a one-hue ramp."""
import collections
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_pipeline.curate_dataset import primary_category, read_jsonl

OUT = Path("docs/figures")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#c9c8c3"
SOURCE_COLOR = {"cybernative": BLUE, "commitpackft": ORANGE, "commitpackft_fix": AQUA}
SOURCE_NAME = {"cybernative": "CyberNative (synthetic)", "commitpackft": "CommitPackFT (real commits)",
               "commitpackft_fix": "CommitPackFT fixes (reversed)"}
SEV_COLOR = {"HIGH": "#184f95", "MEDIUM": "#3987e5", "LOW": "#86b6ef"}  # ordinal blue ramp, darker = worse
CATS = ["BUG", "SECURITY", "RESOURCE_LEAK", "PERFORMANCE"]
FLOOR = 0.10  # plan: no category below ~10% of defective rows

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "font.family": "sans-serif", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK, "axes.edgecolor": GRID, "font.size": 10})
TABLES: dict[str, tuple[str, list[str], list[list]]] = {}  # name -> (title, header, rows): the "table view"


def start(title: str, subtitle: str, w=9.0, h=4.6):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.text(0.02, 0.97, title, fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 0.915, subtitle, fontsize=9.5, color=INK2, va="top")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    return fig, ax


def finish(fig, name: str, top=0.80, left=0.27):
    fig.subplots_adjust(left=left, right=0.96, bottom=0.14, top=top)
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def stacked(ax, rows: list[str], series: list[tuple[str, list[float], str]], fmt, min_label: float):
    """Horizontal stacked bars: 2px surface gap between segments, label inside only when the segment is wide enough."""
    left = [0.0] * len(rows)
    y = range(len(rows) - 1, -1, -1)  # first row on top
    for name, vals, color in series:
        ax.barh(list(y), vals, left=left, height=0.55, color=color, edgecolor=SURFACE, linewidth=1.5, label=name)
        for yi, v, l in zip(y, vals, left):
            if v >= min_label:
                dark = color in (GRAY, "#86b6ef", "#eda100")
                ax.text(l + v / 2, yi, fmt(v), ha="center", va="center", fontsize=9,
                        color=INK if dark else "white", fontweight="bold")
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(list(y))
    ax.set_yticklabels(rows)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    return left


def legend(ax, ncol):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=ncol, frameon=False, fontsize=9, labelcolor=INK2,
              handlelength=1, handleheight=1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw, lab, bal = (read_jsonl(f"data/{n}.jsonl") for n in ("raw_diffs", "labeled", "balanced"))
    valid = [r for r in lab if r["label_json"]]
    bal_def = [r for r in bal if r["is_defective"]]
    clean_pool = [r for r in valid if primary_category(r["label_json"]) is None and r["source"] == "commitpackft"]
    n_def_all = sum(primary_category(r["label_json"]) is not None for r in valid)
    srcs = list(SOURCE_COLOR)

    # 1. pipeline stages ---------------------------------------------------------------------------------
    stages = [("Candidate diffs curated", len(raw)), ("Valid teacher labels", len(valid)),
              ("Teacher says defective", n_def_all), ("Teacher says clean (real commits)", len(clean_pool)),
              ("Final balanced dataset", len(bal))]
    fig, ax = start("Only 801 of 4,000 diffs ended up labeled defective",
                    "Rows at each stage of the M1 data pipeline (plan target: 1,500 balanced rows)")
    ys = list(range(len(stages) - 1, -1, -1))
    ax.barh(ys, [n for _, n in stages], height=0.55, color=BLUE)
    for y, (_, n) in zip(ys, stages):
        ax.text(n + 40, y, f"{n:,}", va="center", color=INK, fontsize=10, fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels([s for s, _ in stages])
    ax.set_xlim(0, 4600)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    finish(fig, "01_pipeline_stages", left=0.31)
    TABLES["01"] = ("Pipeline stages", ["stage", "rows"], [[s, n] for s, n in stages])

    # 2. teacher verdict by source -----------------------------------------------------------------------
    cnt = {s: collections.Counter(primary_category(r["label_json"]) is not None for r in valid if r["source"] == s) for s in srcs}
    fig, ax = start("The teacher flags 80% of CyberNative diffs but only 9% of real commits",
                    "Teacher verdict per source, valid labels only (% of each source's diffs)")
    totals = [sum(cnt[s].values()) for s in srcs]
    d = [100 * cnt[s][True] / t for s, t in zip(srcs, totals)]
    stacked(ax, [f"{SOURCE_NAME[s]}\nn={t:,}" for s, t in zip(srcs, totals)],
            [("Defective", d, BLUE), ("Clean", [100 - x for x in d], GRAY)], lambda v: f"{v:.0f}%", 6)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    legend(ax, 2)
    finish(fig, "02_teacher_verdict_by_source", top=0.80)
    TABLES["02"] = ("Teacher verdict by source", ["source", "n", "defective", "clean"],
                    [[s, t, cnt[s][True], cnt[s][False]] for s, t in zip(srcs, totals)])

    # 3. defective categories vs the 10% floor -----------------------------------------------------------
    cc = collections.Counter(r["category"] for r in bal_def)
    cats = sorted(CATS, key=lambda c: -cc[c])
    floor_n = FLOOR * len(bal_def)
    fig, ax = start("PERFORMANCE and RESOURCE_LEAK are far below the 10% floor",
                    f"Defective rows in the balanced dataset by category (n={len(bal_def)}); orange = under the floor")
    ys = list(range(len(cats) - 1, -1, -1))
    ax.barh(ys, [cc[c] for c in cats], height=0.55, color=[ORANGE if cc[c] < floor_n else BLUE for c in cats])
    for y, c in zip(ys, cats):
        ax.text(cc[c] + 6, y, f"{cc[c]}  ({cc[c] / len(bal_def):.0%})", va="center", fontsize=10, fontweight="bold",
                zorder=3, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2))
    ax.axvline(floor_n, color=INK2, linewidth=1, zorder=1)
    ax.text(floor_n + 4, len(cats) - 0.45, f"10% floor = {floor_n:.0f} rows", color=INK2, fontsize=9, va="bottom")
    ax.set_yticks(ys)
    ax.set_yticklabels(cats)
    ax.set_xlim(0, max(cc.values()) * 1.2)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    finish(fig, "03_category_balance")
    TABLES["03"] = ("Defective rows per category", ["category", "rows", "share"], [[c, cc[c], f"{cc[c] / len(bal_def):.1%}"] for c in cats])

    # 4. category x source (shortcut risk) ---------------------------------------------------------------
    cs = collections.Counter((r["category"], r["source"]) for r in bal_def)
    fig, ax = start("SECURITY examples are almost all synthetic CyberNative code",
                    "Defective rows per category, split by source (a model could learn the source style instead of the bug)")
    ends = stacked(ax, cats, [(SOURCE_NAME[s], [cs[(c, s)] for c in cats], SOURCE_COLOR[s]) for s in srcs], lambda v: f"{v:.0f}", 25)
    for y, c, e in zip(range(len(cats) - 1, -1, -1), cats, ends):
        ax.text(e + 6, y, f"{e:.0f}", va="center", fontsize=10, fontweight="bold")  # segments too thin to label inline
    legend(ax, 3)
    finish(fig, "04_category_by_source")
    TABLES["04"] = ("Defective rows: category x source", ["category"] + srcs, [[c] + [cs[(c, s)] for s in srcs] for c in cats])

    # 5. severity per category (all review comments) -----------------------------------------------------
    sv = collections.Counter()
    for r in bal_def:
        for x in json.loads(r["label_json"])["reviews"]:
            sv[(x["category"], x["severity"])] += 1
    tot = {c: sum(sv[(c, s)] for s in SEV_COLOR) for c in cats}
    fig, ax = start("Severity is mostly HIGH, except for RESOURCE_LEAK",
                    "Severity of each review comment in the balanced dataset, % per category (a rubric the teacher followed)")
    stacked(ax, [f"{c}\nn={tot[c]}" for c in cats],
            [(s.title(), [100 * sv[(c, s)] / tot[c] for c in cats], col) for s, col in SEV_COLOR.items()], lambda v: f"{v:.0f}%", 8)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    legend(ax, 3)
    finish(fig, "05_severity_by_category")
    TABLES["05"] = ("Review comments: category x severity", ["category"] + list(SEV_COLOR), [[c] + [sv[(c, s)] for s in SEV_COLOR] for c in cats])

    # 6. diff length by class and by source (length-shortcut check) --------------------------------------
    ln = lambda r: r["diff"].count("\n") + 1
    bins = list(range(10, 151, 10))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    md_def, md_clean = (statistics.median(ln(r) for r in bal if r["is_defective"] is k) for k in (True, False))
    fig.text(0.02, 0.97, f"Defective diffs run longer than clean ones (median {md_def:.0f} vs {md_clean:.0f} lines)",
             fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 0.915, "Share of rows per diff length (lines), balanced dataset; a length gap is a possible shortcut the model could learn", fontsize=9.5, color=INK2, va="top")
    groups = [(a1, [("Defective", [ln(r) for r in bal_def], BLUE), ("Clean", [ln(r) for r in bal if not r["is_defective"]], INK2)]),
              (a2, [(SOURCE_NAME[s].split(" (")[0], [ln(r) for r in bal if r["source"] == s], SOURCE_COLOR[s]) for s in srcs])]
    rows6 = []
    for ax, series in groups:
        for name, vals, color in series:
            h = [sum(b <= v < b + 10 for v in vals) / len(vals) * 100 for b in bins[:-1]]
            ax.plot([b + 5 for b in bins[:-1]], h, color=color, linewidth=2, solid_capstyle="round",
                    label=f"{name} (median {statistics.median(vals):.0f})")
            rows6.append([name, len(vals), statistics.median(vals)])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
        ax.set_xlabel("diff lines")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    a1.set_ylabel("% of rows")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.14, top=0.80, wspace=0.08)
    fig.savefig(OUT / "06_diff_length.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / "06_diff_length.png")
    TABLES["06"] = ("Diff length (lines)", ["group", "rows", "median"], rows6)

    # 7. hand review of 30 teacher labels (numbers from docs/experiments.md, Phase 3.4) ------------------
    review = [("CyberNative flagged", 8, 0, 0), ("CommitPackFT flagged", 3, 2, 3),
              ("CyberNative passed as clean", 1, 0, 6), ("CommitPackFT passed as clean", 7, 0, 0)]
    fig, ax = start("The teacher is unreliable on CyberNative 'clean' and speculative on real commits",
                    "30 labels reviewed by hand (small sample: 8 / 8 / 7 / 7); counts per group", h=4.4)
    stacked(ax, [r[0] for r in review],
            [("Label correct", [r[1] for r in review], BLUE), ("Arguable", [r[2] for r in review], GRAY),
             ("Label wrong", [r[3] for r in review], ORANGE)], lambda v: f"{v:.0f}", 1)
    ax.set_xlim(0, 8)
    ax.set_xticks(range(0, 9, 2))
    legend(ax, 3)
    finish(fig, "07_teacher_hand_review")
    TABLES["07"] = ("Hand review of 30 teacher labels", ["group", "correct", "arguable", "wrong"], [list(r) for r in review])

    # table view -----------------------------------------------------------------------------------------
    md = ["# Dataset analysis: data behind each chart\n", "Generated by `python -m src.data_pipeline.analyze_dataset`.\n"]
    for k, (title, head, rows) in TABLES.items():
        md += [f"\n## {k}. {title}\n", "| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
        md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    (OUT / "data_tables.md").write_text("\n".join(md) + "\n")
    print("wrote", OUT / "data_tables.md")


if __name__ == "__main__":
    main()
