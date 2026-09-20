#!/usr/bin/env python3
"""Measure GATE_QUESTION against hand-labelled lines from a session log.

Three stages, run in order:

    extract   session.jsonl            -> cases.jsonl with a blank `label`
    (you)     hand-label each case     -> "question" | "instruction" | "skip"
    score     cases.jsonl + live vLLM  -> adds `p_question` to each case
    report    cases.jsonl              -> separation, AUC, threshold sweep

The point of stage 1 is fidelity: each case carries the conversation exactly as
`render_reply` would have rendered it on the tick the line arrived, so the score
is the score the live loop would have produced. Re-deriving the context by hand
would measure a different prompt than the one that runs.

Only 4 p_question values exist in session.jsonl -- the gate is newer than the
log -- so every case is re-scored here rather than read back out.
"""
import argparse
import json
import sys
from typing import Optional

from openai import OpenAI

import interaction_model as m

QUESTION, INSTRUCTION, SKIP = "question", "instruction", "skip"
LABELS = (QUESTION, INSTRUCTION, SKIP)

# An instruction scored as a question is silently dropped -- no task is assigned,
# so no goal is ever sent and the agent just stands there (the white-house bug).
# A question scored as an instruction gets planned, which is visible and
# recoverable. The two errors are not worth the same, hence the weight.
INSTRUCTION_ERROR_WEIGHT = 3.0


def load_events(path: str) -> list[m.Event]:
    """session.jsonl -> Events, text kinds only.

    Frames are dropped: `render_reply` is text-only, and the log stores a
    placeholder string for the image anyway.
    """
    events = []
    for n, line in enumerate(open(path), 1):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            print(f"[extract] {path}:{n} unparseable; skipped", file=sys.stderr)
            continue
        if d.get("kind") in (m.SPEECH_FINAL, m.AGENT_SAID):
            events.append(m.Event(kind=d["kind"], t=d.get("t", 0.0),
                                  text=d.get("text", ""), meta=d.get("meta", {})))
    return events


def extract(args) -> None:
    events = load_events(args.session)
    cases, seen = [], {}
    for i, ev in enumerate(events):
        if ev.kind != m.SPEECH_FINAL:
            continue
        # Up to and including this line: the gate runs after the line is
        # appended, so the line itself is the last one in the context.
        context = m.recent_dialogue(events[: i + 1])
        case = {"t": round(ev.t, 3), "line": ev.text, "context": context,
                "label": "", "p_question": None}
        # Same line AND same context = the same decision twice; label it once.
        key = (ev.text, context)
        if key in seen:
            seen[key]["duplicates"] += 1
            continue
        case["duplicates"] = 0
        seen[key] = case
        cases.append(case)

    with open(args.cases, "w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    total = len(events)
    spoken = sum(1 for e in events if e.kind == m.SPEECH_FINAL)
    print(f"[extract] {spoken} user lines ({total} dialogue events) -> "
          f"{len(cases)} unique cases in {args.cases}")
    print(f"[extract] now set \"label\" on each to one of {LABELS}, then run `score`.")
    print('[extract]   question    = only asks for information (gate should say yes)')
    print('[extract]   instruction = asks you to do something (gate should say no)')
    print('[extract]   skip        = neither, or not addressed to the agent')


def read_cases(path: str) -> list[dict]:
    cases = [json.loads(l) for l in open(path) if l.strip()]
    bad = {c["label"] for c in cases} - set(LABELS) - {""}
    if bad:
        sys.exit(f"[error] unknown label(s) {sorted(bad)}; use one of {LABELS}")
    return cases


def score(args) -> None:
    cases = read_cases(args.cases)
    todo = [c for c in cases if c["label"] in (QUESTION, INSTRUCTION)
            and (c.get("p_question") is None or args.rescore)]
    unlabelled = sum(1 for c in cases if not c["label"])
    if unlabelled:
        print(f"[score] {unlabelled} case(s) still unlabelled; they are not scored",
              file=sys.stderr)
    if not todo:
        sys.exit("[score] nothing to score -- label some cases first, "
                 "or pass --rescore to redo them.")

    client = OpenAI(base_url=args.url, api_key="none")
    model = m.InteractionModel(client, args.model, max_tokens=64)
    gate = args.gate_question or m.GATE_QUESTION

    for i, c in enumerate(todo, 1):
        # Rebuilt, not stored: the prompt must track edits to GATE_QUESTION and
        # REPLY_SYSTEM, so a rerun after a prompt change measures the new prompt.
        messages = [{"role": "system", "content": m.REPLY_SYSTEM},
                    {"role": "user", "content": "[recent conversation]\n" + c["context"]}]
        try:
            c["p_question"] = round(model.gate(messages, gate), 4)
        except Exception as e:
            sys.exit(f"[score] gate failed on case {i} ({c['line']!r}): {e}\n"
                     f"[score] is vLLM up at {args.url} serving {args.model!r}?")
        print(f"[score] {i}/{len(todo)} p={c['p_question']:.3f} "
              f"[{c['label']}] {c['line']!r}", file=sys.stderr)

    with open(args.cases, "w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    print(f"[score] wrote {len(todo)} score(s) to {args.cases}")


def auc(qs: list[float], ins: list[float]) -> Optional[float]:
    """P(a random question outscores a random instruction). Ties count half."""
    if not qs or not ins:
        return None
    wins = sum((q > i) + 0.5 * (q == i) for q in qs for i in ins)
    return wins / (len(qs) * len(ins))


def report(args) -> None:
    cases = [c for c in read_cases(args.cases)
             if c["label"] in (QUESTION, INSTRUCTION) and c.get("p_question") is not None]
    if not cases:
        sys.exit("[report] no scored cases -- run `score` first.")
    qs = [c["p_question"] for c in cases if c["label"] == QUESTION]
    ins = [c["p_question"] for c in cases if c["label"] == INSTRUCTION]

    print(f"\n=== GATE_QUESTION on {len(cases)} labelled cases "
          f"({len(qs)} question, {len(ins)} instruction) ===\n")
    if not qs or not ins:
        print("[report] need both classes to say anything about separation.")
        return

    mean_q, mean_i = sum(qs) / len(qs), sum(ins) / len(ins)
    print(f"mean p(question class)    {mean_q:.3f}   range {min(qs):.3f}-{max(qs):.3f}")
    print(f"mean p(instruction class) {mean_i:.3f}   range {min(ins):.3f}-{max(ins):.3f}")
    print(f"separation                {mean_q - mean_i:+.3f}   "
          f"(compare: GATE_REPLY -0.272, GATE_DELEGATE +0.069)")
    print(f"AUC                       {auc(qs, ins):.3f}   "
          f"(1.0 = perfectly ordered, 0.5 = chance)\n")

    # Every midpoint between adjacent observed scores; those are the only
    # thresholds that can change a decision.
    pts = sorted({c["p_question"] for c in cases})
    cands = sorted({0.0, 1.01} | {(a + b) / 2 for a, b in zip(pts, pts[1:])})
    print("threshold   dropped-instructions   planned-questions   weighted-errors")
    best, best_cost = None, None
    for thr in cands:
        # An instruction at or above the threshold is routed to the answerer and
        # lost; a question below it is routed to the planner.
        dropped = sum(1 for p in ins if p >= thr)
        planned = sum(1 for p in qs if p < thr)
        cost = INSTRUCTION_ERROR_WEIGHT * dropped + planned
        if best_cost is None or cost < best_cost:
            best, best_cost = thr, cost
        print(f"  {thr:5.3f}     {dropped:3d}/{len(ins):<18d} {planned:3d}/{len(qs):<15d} {cost:6.1f}")

    clean = [t for t in cands if not any(p >= t for p in ins)]
    print(f"\nrecommended --question-threshold {best:.3f} "
          f"(weighted cost {best_cost:.1f}, instruction errors x{INSTRUCTION_ERROR_WEIGHT:g})")
    if clean:
        t = min(clean)
        print(f"lowest threshold that drops no instruction at all: {t:.3f} "
              f"(plans {sum(1 for p in qs if p < t)}/{len(qs)} questions)")
    else:
        print("no threshold drops zero instructions -- the classes overlap; "
              "fix the prompt before trusting any threshold.")
    print(f"\ncurrent default is 0.5 (interaction_model.py --question-threshold)")

    worst = sorted(cases, key=lambda c: c["p_question"] if c["label"] == INSTRUCTION
                   else -c["p_question"], reverse=True)[:args.show]
    print(f"\n--- {min(args.show, len(worst))} most-confusing cases ---")
    for c in worst:
        print(f"  p={c['p_question']:.3f}  [{c['label']:11s}] {c['line']!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("extract", "score", "report"))
    ap.add_argument("--session", default="session.jsonl")
    ap.add_argument("--cases", default="question_cases.jsonl")
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="interaction",
                    help="--served-model-name from serve_interaction.sh")
    ap.add_argument("--gate-question", default=None,
                    help="override the prompt text, to A/B a rewrite without editing the module")
    ap.add_argument("--rescore", action="store_true",
                    help="re-run cases that already have a score (use after a prompt change)")
    ap.add_argument("--show", type=int, default=10, help="how many confusing cases to print")
    args = ap.parse_args()
    {"extract": extract, "score": score, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
