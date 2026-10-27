"""Score AgentRx judge outputs against root-cause labels, with Wilson 95% intervals.

Accepts one or more judge run1.json files (a single multi-trajectory run, or one
file per trajectory as produced by the Magentic-One per-file runs) and reports
step and category accuracy, the confident-misattribution and inconclusive rates
(proposal Section 5), and breakdowns by root-cause category and failed agent.

Root-cause ids are compared as strings: one Magentic-One label stores its id as
a string while the failure ids are integers.

Usage:
  python baselines/score_runs.py --ground-truth GT.json --out out.json RUN1.json [RUN1.json ...]
"""
import argparse
import collections
import json
import math

CATEGORY_NAMES = {1: "Instruction Adherence Failure", 2: "Invention of New Information",
                  3: "Invalid Invocation", 4: "Misinterpretation of Tool Output",
                  5: "Intent-Plan Misalignment", 6: "Underspecified User Intent",
                  7: "Intent Not Supported", 8: "Guardrails Triggered", 9: "System Failure",
                  10: "Inconclusive"}
_KEYS = [("instruction", 1), ("invention", 2), ("invalid", 3), ("misinterpretation", 4),
         ("intent-plan", 5), ("intent plan", 5), ("underspecified", 6), ("under-specified", 6),
         ("not supported", 7), ("guardrail", 8), ("system", 9), ("inconclusive", 10)]


def cat_id(value):
    if isinstance(value, int):
        return value
    s = str(value).strip().lower()
    if s.isdigit():
        return int(s)
    for key, cid in _KEYS:
        if key in s:
            return cid
    return None


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def root_cause(item):
    rc = str(item["root_cause"]["failure_id"])
    for f in item["failures"]:
        if str(f["failure_id"]) == rc:
            return f
    return item["failures"][0]


def rate(k, n):
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else 0.0, "ci95": [lo, hi]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt = {str(x["trajectory_id"]): x for x in json.load(open(args.ground_truth))}
    rows = []
    for path in args.runs:
        for r in json.load(open(path))["detailed_results"]:
            tid = str(r["task_id"])
            if tid not in gt or not r.get("failures"):
                continue
            p = r["failures"][0]
            g = root_cause(gt[tid])
            pc, gc = cat_id(p.get("failure_case")), cat_id(g["failure_category"])
            try:
                ps = int(p.get("step_number"))
            except (TypeError, ValueError):
                ps = None
            rows.append({"trajectory_id": tid, "pred_category": pc, "pred_step": ps,
                         "gt_category": gc, "gt_step": int(g["step_number"]),
                         "gt_agent": (g.get("failed_agent") or "").strip().lower() or None})

    n = len(rows)
    step_ok = [r for r in rows if r["pred_step"] == r["gt_step"]]
    cat_ok = [r for r in rows if r["pred_category"] == r["gt_category"]]
    inconclusive = [r for r in rows if r["pred_category"] == 10]
    confident_wrong = [r for r in rows if r["pred_category"] != 10
                       and (r["pred_step"] != r["gt_step"] or r["pred_category"] != r["gt_category"])]
    dist = [abs(r["pred_step"] - r["gt_step"]) for r in rows if r["pred_step"] is not None]

    by_cat = collections.defaultdict(lambda: [0, 0])
    by_agent = collections.defaultdict(lambda: [0, 0, 0])
    predicted = collections.Counter(CATEGORY_NAMES.get(r["pred_category"], str(r["pred_category"])) for r in rows)
    for r in rows:
        c = CATEGORY_NAMES.get(r["gt_category"], str(r["gt_category"]))
        by_cat[c][1] += 1
        by_cat[c][0] += r["pred_category"] == r["gt_category"]
        if r["gt_agent"]:
            a = by_agent[r["gt_agent"]]
            a[2] += 1
            a[0] += r["pred_step"] == r["gt_step"]
            a[1] += r["pred_category"] == r["gt_category"]

    summary = {
        "n": n,
        "step_accuracy": rate(len(step_ok), n),
        "category_accuracy": rate(len(cat_ok), n),
        "both_correct": rate(sum(1 for r in rows if r in step_ok and r in cat_ok), n),
        "confident_misattribution": rate(len(confident_wrong), n),
        "inconclusive": rate(len(inconclusive), n),
        **{f"step_within_{k}": rate(sum(1 for d in dist if d <= k), n) for k in (1, 3, 5)},
        "mean_step_distance": sum(dist) / len(dist) if dist else None,
        "by_root_cause_category": {k: {"correct": v[0], "total": v[1]} for k, v in sorted(by_cat.items())},
        "predicted_categories": dict(predicted.most_common()),
        "by_failed_agent": {k: {"step_correct": v[0], "category_correct": v[1], "total": v[2]}
                            for k, v in sorted(by_agent.items())},
    }
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
