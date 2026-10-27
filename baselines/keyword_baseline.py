"""Fixed keyword-rule baseline (non-AI) for AgentRx failure localization.

Proposal, Sections 3 and 4.5: flag the first step that carries an explicit error
signal, using a keyword list written before any Magentic-One trajectory is
inspected. The list below was fixed on 2026-10-27 before the Magentic-One
reproduction and must not be edited after test data are inspected.

Prediction per trajectory:
  step     = index of the first step (1-based IR ``index``) whose non-system,
             non-user content contains a keyword; if none, the last step.
  category = category mapped from the keyword group that fired; if none,
             10 (Inconclusive).

Usage:
  python baselines/keyword_baseline.py <trajectory_ir.json> [more IR files...] \
      --ground-truth data/ground_truth/tau_ground_truth.json --out results.json
"""
import argparse
import json
import re

# Keyword groups, checked in this order. FIXED 2026-10-27; do not edit.
KEYWORD_GROUPS = [
    # (category id, category name, keywords)
    (9, "System Failure", [
        "timeout", "timed out", "time out", "504", "503", "502", "500 internal",
        "service unavailable", "connection refused", "connection reset",
        "rate limit", "too many requests", "server error", "crashed",
    ]),
    (8, "Guardrails Triggered", [
        "access denied", "permission denied", "forbidden", "403",
        "content policy", "safety policy", "blocked by", "captcha",
    ]),
    (3, "Invalid Invocation", [
        "error", "exception", "traceback", "invalid", "not found", "404",
        "failed", "failure", "unknown argument", "missing required",
        "must be", "unexpected keyword", "does not exist",
    ]),
]

FIRST_PASS_ROLES_EXCLUDED = {"system", "user"}

_PATTERNS = [
    (cid, name, re.compile("|".join(re.escape(k) for k in kws), re.IGNORECASE))
    for cid, name, kws in KEYWORD_GROUPS
]


def predict(trajectory):
    steps = trajectory.get("steps", [])
    for step in steps:
        for sub in step.get("substeps", []):
            if str(sub.get("role", "")).lower() in FIRST_PASS_ROLES_EXCLUDED:
                continue
            text = sub.get("content", "")
            if not isinstance(text, str):
                text = json.dumps(text)
            for cid, name, pat in _PATTERNS:
                m = pat.search(text)
                if m:
                    return {"step_number": step.get("index"), "failure_case": cid,
                            "category": name, "matched": m.group(0)}
    last = steps[-1].get("index") if steps else None
    return {"step_number": last, "failure_case": 10, "category": "Inconclusive", "matched": None}


def _cat_id(name):
    s = str(name).lower()
    for key, cid in [("instruction", 1), ("invention", 2), ("invalid", 3), ("misinterpretation", 4),
                     ("intent-plan", 5), ("intent plan", 5), ("underspecified", 6), ("under-specified", 6),
                     ("not supported", 7), ("guardrail", 8), ("system", 9)]:
        if key in s:
            return cid
    return None


def root_cause(gt_item):
    rc = gt_item["root_cause"]["failure_id"]
    for f in gt_item["failures"]:
        if str(f["failure_id"]) == str(rc):
            return _cat_id(f["failure_category"]), f["step_number"]
    f = gt_item["failures"][0]
    return _cat_id(f["failure_category"]), f["step_number"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ir_files", nargs="+")
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt = {str(x["trajectory_id"]): x for x in json.load(open(args.ground_truth))}
    rows = []
    for path in args.ir_files:
        data = json.load(open(path))
        for traj in (data if isinstance(data, list) else [data]):
            tid = str(traj.get("trajectory_id"))
            if tid not in gt:
                continue
            p = predict(traj)
            gc, gs = root_cause(gt[tid])
            rows.append({"trajectory_id": tid, **p, "gt_failure_case": gc, "gt_step_number": gs})

    n = len(rows)
    step_ok = sum(1 for r in rows if r["step_number"] == r["gt_step_number"])
    cat_ok = sum(1 for r in rows if r["failure_case"] == r["gt_failure_case"])
    within = {k: sum(1 for r in rows if r["step_number"] is not None
                     and abs(r["step_number"] - r["gt_step_number"]) <= k) for k in (1, 3, 5)}
    summary = {"n": n, "step_accuracy": step_ok / n if n else 0, "category_accuracy": cat_ok / n if n else 0,
               "correct_steps": step_ok, "correct_categories": cat_ok,
               **{f"step_within_{k}": v / n if n else 0 for k, v in within.items()},
               "fired": sum(1 for r in rows if r["matched"]), "inconclusive": sum(1 for r in rows if not r["matched"])}
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
