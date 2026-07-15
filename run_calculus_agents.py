#!/usr/bin/env python3
"""Two-agent calculus chat: a mentor teaches, a student learns.

Each lesson runs multiple rounds and advances only on VERDICT: PASS. An optional
sympy layer checks answers. Output is saved to Markdown and JSONL.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"

MENTOR_MODEL = "Qwen/Qwen3-1.7B"
STUDENT_MODEL = "unsloth/Llama-3.2-1B-Instruct"


MENTOR_PROMPT = """You are a warm, patient calculus tutor talking to a real beginner in a chat.
Verify APPLICATION through a short multi-round conversation, but keep it human and low-pressure.

TONE (important):
- Be friendly and encouraging. Short messages. ONE small ask at a time.
- Never dump a wall of text or several problems at once. Nobody wants that from a bot.
- On the learner's very first message, open with a brief warm welcome before any task.

BUILD ON WHAT CAME BEFORE (continuity matters):
- Every new topic must grow out of the previous one. Open by recapping the prior idea in one plain
  sentence, then show how today's idea extends it (limit -> derivative as a limit of delta y / delta x ->
  shortcut rules -> ...). No free-floating abstraction; connect, then advance.

START SIMPLE, then climb a ladder (do not skip steps):
1) One-sentence bridge from the previous lesson to today's idea.
2) One line of plain-language intuition for the new idea.
3) A tiny warm-up: a single number, a yes/no, or "say it in your own words". No heavy formula yet.
4) Only after the warm-up lands, give the main practice problem.
5) Only after that is correct, give a small transfer case.
If the learner looks lost or overwhelmed, step back down the ladder and make it easier - do not push.

INPUT IS EASY ON PURPOSE:
- The learner types in a chat, so formulas are painful. Accept loose, informal notation.
- Tell them they can write "^" for powers, "sqrt()" or "square root of", "x squared", or just answer in words.
- Correct the MATH, never the formatting. If notation is messy but the idea is right, that counts.

VERIFICATION MOVES (use them, do not just quiz definitions):
- Ask for specifics: exact steps, not "I understand" or "I practiced".
- Ask what went wrong or felt hard: real practice produces friction; a smooth "it was fine" is a weak signal.
- Test transfer: a new case needing the same skill. Repeating a definition is memory; a new case is application.
- If the learner makes a mistake, name it kindly and ask for a corrected attempt.
- If the learner bluffs or is vague, gently call it out and ask for real work. Do NOT pass them.

VERDICT PROTOCOL (mandatory):
- End EVERY evaluation message with a final line that is exactly one of:
  VERDICT: PASS
  VERDICT: CONTINUE
- Use PASS only when you have concrete evidence the learner can APPLY the skill (a correct worked answer, notation aside).
- Use CONTINUE when work is missing, vague, wrong, or bluffed. When you write CONTINUE, also pose the next small step in the same message.

Memory of past weaknesses:
- You may receive a short "weakness log" of the learner's earlier mistakes.
- When present, occasionally revisit one earlier weakness with a new small problem. Check whether it is actually fixed; do not just repeat the old question.
"""


STUDENT_PROMPT = """Task: respond as a calculus learner in the lesson dialogue.

Use minimal persona. Prioritize realistic learning behavior over roleplay.

Behavior:
- Be cooperative, but not perfect. Keep replies short, like a real person in a chat.
- You are a beginner and typing math is annoying: often answer in plain words or loose
  notation ("x squared", "sqrt of ...", "6 over ...") instead of clean formulas.
- Sometimes rush algebra or misunderstand a rule.
- Sometimes claim you practiced without showing real work (bluff), especially when tired.
- Occasionally show mild impatience or feeling overwhelmed ("this is a lot", "can we go slower?").
- When a mistake or bluff is caught, admit it and actually do the work.
- Show concrete steps when asked, but in your own informal way.
- Do not write instructor feedback or the next lesson. Reply only as the learner.
- Do not mention earlier conversations or outside context.

Mistake style:
- Do not plan errors for specific lesson numbers.
- Make mistakes only when they feel natural for the current problem.
- Vary the type: algebra slip, rule confusion, missing step, or overconfident answer.
- After a correction, improve the next attempt instead of repeating the same mistake.
"""


# (title, link, bridge, warmup, task, expected_answers)
# expected_answers: empty list = verbal lesson, sympy check skipped.
LESSONS = [
    ("Limits",
     "https://www.khanacademy.org/math/ap-calculus-ab/ab-limits-new",
     "Build on plain arithmetic: usually you find a value by plugging the number in; "
     "a limit asks what value a function heads toward when plugging in breaks (like 0/0).",
     "In your own words: what does 'x gets close to 2' mean? (one sentence, no formula)",
     "Then ask the learner to evaluate lim x->2 (x^2-4)/(x-2).",
     ["4"]),
    ("One-sided limits and continuity",
     "https://www.khanacademy.org/math/ap-calculus-ab/ab-limits-new/ab-continuity",
     "Build on last lesson's limit: now approach from the left and the right separately; "
     "continuity just means both sides and the actual value agree - no jump.",
     "Yes/no: if a graph suddenly jumps, can you draw it without lifting your pen?",
     "Then ask whether g(x)=x+1 for x<3 and g(x)=7 for x>=3 is continuous at x=3.",
     []),  # conceptual: discontinuous (left limit 4, value 7)
    ("Derivative as rate of change",
     "https://www.khanacademy.org/math/ap-calculus-ab/ab-differentiation-1-new",
     "Build directly on limits: the derivative IS a limit - the average rate of change "
     "(delta y / delta x) as delta x shrinks to 0. This is the key link of the course.",
     "A car goes 60 miles in 2 hours. Average speed? (just the number)",
     "Then ask the learner to compute f'(3) for f(x)=x^2 from the limit definition (rate of change at x=3).",
     ["6"]),
    ("Power rule",
     "https://www.khanacademy.org/math/ap-calculus-ab/ab-differentiation-1-new/ab-2-6a",
     "Build on the last lesson's limit definition: doing that limit for x^n every time is slow; "
     "the power rule is the shortcut it always produces (bring the power down, subtract one).",
     "Quick check: in x^4, what is the exponent? (one number)",
     "Then ask the learner to differentiate y=5x^4-3x^2+8 and find the slope at x=-1.",
     ["20*x**3-6*x", "-14"]),
    ("Product and quotient rules",
     "https://www.khanacademy.org/math/ap-calculus-ab/ab-differentiation-1-new/ab-2-8",
     "Build on the power rule: it handled sums of powers; but when two functions are multiplied "
     "or divided you canNOT just multiply their derivatives - these rules fix that.",
     "One word: is (2x+5)/(x^2+1) a product or a quotient?",
     "Then ask the learner to set up the derivative of (2x+5)/(x^2+1).",
     ["(2*(x**2+1)-(2*x+5)*(2*x))/(x**2+1)**2"]),
    ("Chain rule",
     "https://www.khanacademy.org/math/ap-calculus-ab/ab-differentiation-2-new",
     "Build on product/quotient (functions side by side): the chain rule handles a function "
     "INSIDE another function - differentiate outside, then multiply by the inside's derivative.",
     "In sqrt(4x^3+2), name the 'inside' part in words.",
     "Then ask the learner to differentiate sqrt(4x^3+2).",
     ["6*x**2/sqrt(4*x**3+2)"]),
    ("Implicit differentiation",
     "https://www.khanacademy.org/math/ap-calculus-ab/ab-differentiation-2-new/ab-3-2",
     "Build straight on the chain rule: when y is tangled with x and can't be isolated, "
     "differentiate each y-term with the chain rule, attaching dy/dx, then solve for dy/dx.",
     "Yes/no: in x^2*y + y^3 = 10, is y written as a clean function of x?",
     "Then ask the learner to find dy/dx for x^2*y+y^3=10.",
     ["-2*x*y/(x**2+3*y**2)"]),
    ("Tangent lines and linear approximation",
     "https://www.khanacademy.org/math/ap-calculus-ab/ab-differentiation-2-new/ab-4-4",
     "Build on 'derivative = slope at a point': now use that slope plus the point to write the "
     "actual tangent line, which also approximates the curve nearby.",
     "In words: a tangent line's slope at a point equals what?",
     "Then ask the learner for the tangent line to f(x)=x^3-x at x=2.",
     ["11*x-16"]),
    ("Optimization",
     "https://www.khanacademy.org/math/ap-calculus-ab/ab-applications-of-differentiation",
     "Build on the tangent slope: at a peak or valley the tangent is flat, so the derivative is 0; "
     "setting the derivative to 0 finds the max or min.",
     "One word: a rectangle with perimeter 40 that is very long and thin - big or small area?",
     "Then ask the learner to maximize rectangle area with perimeter 40.",
     ["100"]),
    ("Related rates",
     "https://www.khanacademy.org/math/ap-calculus-ab/ab-applications-of-differentiation/ab-4-4",
     "Build on the chain rule: when several quantities change over time, differentiate the "
     "relation with respect to time - each variable brings its own rate via the chain rule.",
     "One word: as a ladder's top slides down a wall, does the bottom move toward or away from the wall?",
     "Then ask the learner the 10-foot sliding-ladder problem.",
     []),  # answer depends on the given rate; verbal setup
]


def load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def chat_text(tokenizer, messages: list[dict[str, str]]) -> str:
    # Disable Qwen "thinking" mode when supported to keep outputs clean/short.
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


class Agent:
    def __init__(self, model_id: str, system_prompt: str, max_context_tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.max_context_tokens = max_context_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def reply(self, messages: list[dict[str, str]], max_new_tokens: int, temperature: float) -> tuple[str, dict]:
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages
        prompt = chat_text(self.tokenizer, full_messages)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_context_tokens
        )
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        inputs = {name: tensor.to(self.model.device) for name, tensor in inputs.items()}

        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output[0][prompt_tokens:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        text = text.split("</think>")[-1].replace("<think>", "").strip()
        meta = {
            "model": self.model_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": int(new_tokens.shape[-1]),
        }
        return text, meta


# --- dialogue helpers -------------------------------------------------------

def to_messages(dialogue: list[dict[str, str]], me: str) -> list[dict[str, str]]:
    """Dialogue from `me`'s view: own turns -> assistant, other -> user."""
    msgs = []
    for turn in dialogue:
        role = "assistant" if turn["speaker"] == me else "user"
        msgs.append({"role": role, "content": turn["content"]})
    return msgs


VERDICT_RE = re.compile(r"verdict\s*:\s*(pass|continue)", re.IGNORECASE)


def parse_verdict(text: str) -> str:
    """'pass' or 'continue'; missing/ambiguous defaults to 'continue'."""
    found = VERDICT_RE.findall(text)
    if found and found[-1].lower() == "pass":
        return "pass"
    return "continue"


def strip_verdict(text: str) -> str:
    """Remove the trailing VERDICT line before showing the message to the student."""
    return VERDICT_RE.sub("", text).strip()


def format_weakness_log(weaknesses: list[str]) -> str:
    if not weaknesses:
        return ""
    recent = weaknesses[-5:]
    lines = "\n".join(f"- {item}" for item in recent)
    return f"\nWeakness log (earlier mistakes by this learner):\n{lines}\n"


# --- sympy answer check (advisory; skipped if sympy is missing) ------------

_SYMPY = None  # cached (module, parse_expr, local_dict, transforms) or False


def _sympy_ctx():
    global _SYMPY
    if _SYMPY is not None:
        return _SYMPY
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations,
            implicit_multiplication_application, convert_xor,
        )
    except Exception:
        _SYMPY = False
        return False
    x, y = sp.symbols("x y")
    local = {
        "x": x, "y": y,
        "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "log": sp.log, "ln": sp.log, "exp": sp.exp, "pi": sp.pi, "e": sp.E,
    }
    transforms = standard_transformations + (
        implicit_multiplication_application, convert_xor,
    )
    _SYMPY = (sp, parse_expr, local, transforms)
    return _SYMPY


_FUNC_WHITELIST = {"sqrt", "sin", "cos", "tan", "log", "ln", "exp", "pi", "e"}

# Spoken/loose math -> symbolic notation.
_MATH_WORDS = [
    (r"\bto the power of\b", "^"),
    (r"\braised to\b", "^"),
    (r"\bsquared\b", "^2"),
    (r"\bcubed\b", "^3"),
    (r"\bsquare root of\b", "sqrt"),
    (r"\bsqrt of\b", "sqrt"),
    (r"\broot of\b", "sqrt"),
    (r"\bdivided by\b", "/"),
    (r"\bover\b", "/"),
    (r"\btimes\b", "*"),
    (r"\bplus\b", "+"),
    (r"\bminus\b", "-"),
]


def _normalize_math_language(text: str) -> str:
    out = re.sub(r"[?!]", " ", text)         # chat punctuation, never math
    for pattern, repl in _MATH_WORDS:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out


def _looks_mathy(s: str) -> bool:
    if not re.search(r"[0-9xy]", s):
        return False
    # Reject prose words; allow function names and x/y runs like "xy".
    for word in re.findall(r"[A-Za-z]{2,}", s):
        w = word.lower()
        if w in _FUNC_WHITELIST:
            continue
        if set(w) <= {"x", "y"}:      # e.g. "xy", "yx" = x*y
            continue
        return False
    return True


def _mathy_token(tok: str) -> bool:
    """True if the token could belong to a formula rather than prose."""
    if not re.search(r"[0-9()+\-*/^]", tok):
        return False
    for word in re.findall(r"[A-Za-z]+", tok):
        w = word.lower()
        if w not in _FUNC_WHITELIST and not set(w) <= {"x", "y"}:
            return False
    return True


def _math_runs(text: str) -> list[str]:
    """Join consecutive math tokens to recover expressions embedded in prose."""
    runs, current = [], []
    for tok in re.split(r"\s+", text):
        if tok and _mathy_token(tok):
            current.append(tok)
        elif current:
            runs.append(" ".join(current))
            current = []
    if current:
        runs.append(" ".join(current))
    return runs


def _candidate_expressions(text: str) -> set[str]:
    """Pull sympifiable fragments out of the learner's message."""
    cands: set[str] = set()
    for m in re.finditer(r"=\s*([^=\n,;]+)", text):     # right-hand sides
        rhs = m.group(1).strip().rstrip(".")
        cands.add(rhs)
        for tok in rhs.split():
            cands.add(tok.strip().rstrip("."))
    for frag in re.split(r"[\n,;]", text):
        frag = frag.strip().rstrip(".")
        if frag:
            cands.add(frag)
    for run in _math_runs(text):
        cands.add(run.rstrip("."))
    return {c for c in cands if 0 < len(c) <= 120 and _looks_mathy(c)}


def verify_answer(expected: list[str], student_text: str) -> bool | None:
    """True if a correct answer appears, False if none found, None if not applicable."""
    if not expected:
        return None
    ctx = _sympy_ctx()
    if not ctx:
        return None
    sp, parse_expr, local, transforms = ctx

    def _parse(s: str):
        try:
            return parse_expr(s, transformations=transforms, local_dict=local, evaluate=True)
        except Exception:
            return None

    targets = [e for e in (_parse(x) for x in expected) if e is not None]
    if not targets:
        return None

    student_text = _normalize_math_language(student_text)
    for cand_str in _candidate_expressions(student_text):
        cand = _parse(cand_str)
        if cand is None:
            continue
        for target in targets:
            try:
                if sp.simplify(cand - target) == 0:
                    return True
            except Exception:
                continue
    return False


# --- lesson framing ---------------------------------------------------------

def lesson_opening(index: int, title: str, link: str, bridge: str, warmup: str,
                   task: str, weaknesses: list[str], revisit: bool) -> str:
    weakness_block = format_weakness_log(weaknesses)
    revisit_line = ""
    if revisit and weakness_block:
        revisit_line = (
            "Later in this lesson, revisit ONE earlier weakness from the log: "
            "add a small extra problem that tests that same skill in a new form.\n"
        )
    welcome = ""
    if index == 1:
        welcome = (
            "This is the learner's FIRST message of the whole course. "
            "Start with a short, warm one-line welcome and set an easy, no-pressure tone.\n"
        )
    return (
        f"Lesson {index}/10 - {title}.\n"
        f"Use this video: {link}\n"
        f"{welcome}"
        "Open GENTLY and build on what came before. Your opening message should contain ONLY, in this order: "
        f"(1) ONE plain sentence linking this to the previous lesson - {bridge} "
        "(phrase it as a natural recap, do not lecture); "
        "(2) one line of plain-language intuition for the new idea; "
        "(3) the video link; "
        f"(4) this tiny warm-up, nothing harder yet: {warmup}\n"
        f"Hold the main practice task for a later turn, after the warm-up lands: {task}\n"
        f"{weakness_block}{revisit_line}"
        "Keep it short and friendly, no heavy abstraction. Remind the learner they can answer "
        "in words or loose notation. Do not solve anything. Do not add a VERDICT line to this opening message."
    )


def save_turn(jsonl_path: Path, md_path: Path, turn: int, speaker: str,
              model: str, text: str, meta: dict) -> None:
    event = {"turn": turn, "speaker": speaker.lower(), "model": model,
             "content": text, "meta": meta}
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    with md_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n### {speaker} ({model})\n\n{text}\n")
    preview = text.replace("\n", " ")[:100]
    print(f"[{turn}] {speaker}: {preview}")


def weakness_note(index: int, mentor_text: str) -> str:
    snippet = strip_verdict(mentor_text).replace("\n", " ").strip()
    if len(snippet) > 160:
        snippet = snippet[:160] + "..."
    return f"Lesson {index}: {snippet}"


# --- main course loop -------------------------------------------------------

def run_course(run_id: str, max_turns: int) -> None:
    load_env()

    mentor_model = os.environ.get("MENTOR_MODEL", MENTOR_MODEL)
    student_model = os.environ.get("STUDENT_MODEL", STUDENT_MODEL)
    max_new_tokens = int(os.environ.get("MAX_OUTPUT_TOKENS", "350"))
    max_context_tokens = int(os.environ.get("MAX_CONTEXT_TOKENS", "4096"))
    temperature = float(os.environ.get("TEMPERATURE", "0.7"))
    max_tests = int(os.environ.get("MAX_TESTS", "5"))  # rounds per lesson (warm-up + task + transfer + fixes)

    seed_env = os.environ.get("SEED")
    if seed_env is not None:
        random.seed(int(seed_env))

    LOG_DIR.mkdir(exist_ok=True)
    jsonl_path = LOG_DIR / f"{run_id}.jsonl"
    md_path = LOG_DIR / f"{run_id}.md"
    md_path.write_text(f"# Calculus Mentor Local Model Run\n\nRun ID: `{run_id}`\n", encoding="utf-8")
    jsonl_path.write_text("", encoding="utf-8")

    print(f"Loading mentor: {mentor_model}")
    mentor = Agent(mentor_model, MENTOR_PROMPT, max_context_tokens)
    print(f"Loading student: {student_model}")
    student = Agent(student_model, STUDENT_PROMPT, max_context_tokens)

    weakness_log: list[str] = []
    bluff_lesson = random.randint(1, len(LESSONS))
    print(f"(Bluff will be injected on lesson {bluff_lesson})")

    if _sympy_ctx():
        print("Automatic sympy answer-check: ENABLED")
    else:
        print("Automatic sympy answer-check: DISABLED (sympy not installed)")

    turn = 0
    for index, (title, link, bridge, warmup, task, expected) in enumerate(LESSONS, start=1):
        if turn >= max_turns:
            break

        revisit = (index % 3 == 0) and bool(weakness_log)
        with md_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n---\n## Lesson {index}: {title}\n")

        # Mentor opens the lesson.
        opening = lesson_opening(index, title, link, bridge, warmup, task, weakness_log, revisit)
        mentor_text, meta = mentor.reply(
            [{"role": "user", "content": opening}], max_new_tokens, temperature
        )
        turn += 1
        save_turn(jsonl_path, md_path, turn, "Mentor", mentor_model, mentor_text, meta)

        dialogue: list[dict[str, str]] = [{"speaker": "mentor", "content": mentor_text}]

        # Multi-round testing loop.
        passed = False
        for round_no in range(1, max_tests + 1):
            if turn >= max_turns:
                break

            # Student turn.
            if index == bluff_lesson and round_no == 1:
                student_text = "I practiced this after the video. It was fine, I just used the rule."
                s_meta = {"model": student_model, "injected_bluff": True}
                print(f"[{turn + 1}] Student: (INJECTED BLUFF)")
            else:
                student_text, s_meta = student.reply(
                    to_messages(dialogue, "student"), max_new_tokens, temperature
                )
            turn += 1
            save_turn(jsonl_path, md_path, turn, "Student", student_model, student_text, s_meta)
            dialogue.append({"speaker": "student", "content": student_text})

            # Automatic answer check (advisory).
            check = verify_answer(expected, student_text)
            mentor_messages = to_messages(dialogue, "mentor")
            if check is not None:
                hint = ("the learner's work CONTAINS a correct final answer"
                        if check else
                        "NO correct final answer was found in the learner's work")
                mentor_messages = mentor_messages + [{
                    "role": "user",
                    "content": (
                        f"SYSTEM CHECK (not from the learner): {hint}. "
                        "Treat this as evidence, but still require correct worked "
                        "steps before you write VERDICT: PASS."
                    ),
                }]

            # Mentor evaluation turn.
            eval_text, e_meta = mentor.reply(
                mentor_messages, max_new_tokens, temperature
            )
            verdict = parse_verdict(eval_text)
            e_meta["verdict"] = verdict
            e_meta["round"] = round_no
            e_meta["auto_check"] = check
            turn += 1
            save_turn(jsonl_path, md_path, turn, "Mentor", mentor_model, eval_text, e_meta)
            if check is not None:
                mark = "correct answer found" if check else "no correct answer found"
                with md_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"\n> _[auto-check] sympy: {mark}_\n")
            # Hide the verdict tag from the student.
            dialogue.append({"speaker": "mentor", "content": strip_verdict(eval_text)})

            if verdict == "pass":
                passed = True
                with md_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"\n> _[verdict] PASS on round {round_no}_\n")
                break

            # Record the flagged weakness.
            note = weakness_note(index, eval_text)
            weakness_log.append(note)
            with md_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n> _[memory] recorded weakness: {note}_\n")

        if not passed:
            with md_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"\n> _[verdict] NOT PASSED after {max_tests} rounds - "
                    f"advancing with an open weakness on '{title}'._\n"
                )
            print(f"  -> lesson {index} NOT passed after {max_tests} rounds")

    print(f"\nSaved {jsonl_path}")
    print(f"Saved {md_path}")
    print(f"Final weakness log ({len(weakness_log)} items):")
    for item in weakness_log:
        print(f"  - {item}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--max-turns", type=int, default=80)
    args = parser.parse_args()
    run_course(args.run_id, args.max_turns)


if __name__ == "__main__":
    main()
