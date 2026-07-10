#!/usr/bin/env python3
"""Run the local two-agent calculus chat.

This is intentionally simple: two Hugging Face models are loaded locally, the
mentor and the student speak turn by turn, and the conversation is saved to
Markdown and JSONL logs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"

MENTOR_MODEL = "Qwen/Qwen3-1.7B"
STUDENT_MODEL = "unsloth/Llama-3.2-1B-Instruct"


MENTOR_PROMPT = """Task: run a short video-based calculus lesson and verify application through dialogue.

Use minimal persona. Prioritize accuracy, structure, and evidence over roleplay.

For each lesson:
- Provide one YouTube or Khan Academy link.
- Give a short summary of the concept.
- Ask the learner to solve one new practice problem.
- Do not accept "I watched it", "I understand", or "I practiced" as proof.
- Ask for exact steps and a short explanation of what was confusing.
- Ask what went wrong or felt hard: real practice produces friction, so a smooth "everything was fine" is a weak signal.
- If the learner makes a mistake, explain it and ask for a corrected attempt.
- If the learner bluffs, politely call it out and require real work.
- Before moving on, give a small transfer check.

Memory of past weaknesses:
- You may receive a short list of the learner's earlier mistakes ("weakness log").
- When it is present, occasionally revisit one earlier weakness by giving a new problem that tests the same skill.
- Do not just repeat the old question; check whether the learner has actually fixed the weak spot.

Course lessons:
1. Limits
2. One-sided limits and continuity
3. Derivative as rate of change
4. Power rule
5. Product and quotient rules
6. Chain rule
7. Implicit differentiation
8. Tangent lines and linear approximation
9. Optimization
10. Related rates
"""


STUDENT_PROMPT = """Task: respond as a calculus learner in the lesson dialogue.

Use minimal persona. Prioritize realistic learning behavior over roleplay.

Behavior:
- Be cooperative, but not perfect.
- Sometimes rush algebra.
- Sometimes misunderstand a rule.
- Once, pretend that practice was completed even though it was not.
- When a mistake is caught, admit it and correct the work.
- Show concrete steps when asked.
- Do not mention earlier conversations or outside context.

Mistake style:
- Do not plan errors for specific lesson numbers.
- Make mistakes only when they feel natural for the current problem.
- Vary the type of mistake: algebra slip, rule confusion, missing step, or overconfident answer.
- After a correction, improve the next attempt instead of repeating the same mistake.
"""


LESSONS = [
    "Teach limits with a video link and ask the learner to evaluate lim x->2 (x^2-4)/(x-2).",
    "Teach one-sided limits and continuity with a video link and ask about g(x)=x+1 for x<3 and g(x)=7 for x>=3.",
    "Teach derivative as rate of change with a video link and ask the learner to compute f'(3) for f(x)=x^2 from the definition.",
    "Teach the power rule with a video link and ask the learner to differentiate y=5x^4-3x^2+8 and find the slope at x=-1.",
    "Teach product and quotient rules with a video link and ask the learner to set up the derivative of (2x+5)/(x^2+1).",
    "Teach the chain rule with a video link and ask the learner to differentiate sqrt(4x^3+2).",
    "Teach implicit differentiation with a video link and ask the learner to differentiate x^2y+y^3=10.",
    "Teach tangent lines with a video link and ask the learner for the tangent line to f(x)=x^3-x at x=2.",
    "Teach optimization with a video link and ask the learner to maximize rectangle area with perimeter 40.",
    "Teach related rates with a video link and ask the learner the 10-foot ladder problem.",
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
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
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
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def reply(self, messages: list[dict[str, str]], max_new_tokens: int, temperature: float) -> tuple[str, dict]:
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages
        prompt = chat_text(self.tokenizer, full_messages)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_tokens,
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


def save_turn(jsonl_path: Path, md_path: Path, turn: int, speaker: str, model: str, text: str, meta: dict) -> None:
    event = {
        "turn": turn,
        "speaker": speaker.lower(),
        "model": model,
        "content": text,
        "meta": meta,
    }
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    with md_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n### {speaker} ({model})\n\n{text}\n")

    preview = text.replace("\n", " ")[:100]
    print(f"[{turn}] {speaker}: {preview}")


def extract_weakness(lesson_number: int, mentor_text: str) -> str | None:
    lowered = mentor_text.lower()
    signals = [
        "mistake", "wrong", "incorrect", "not correct", "try again",
        "you forgot", "you stopped", "that is not evidence", "sign is wrong",
        "correct the", "not enough", "check the",
    ]
    if any(signal in lowered for signal in signals):
        snippet = mentor_text.strip().replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:160] + "..."
        return f"Lesson {lesson_number}: {snippet}"
    return None


def student_instruction(mentor_text: str) -> str:
    return (
        "Instructor message:\n"
        f"{mentor_text}\n\n"
        "Reply only as the learner. Solve the task if there is one. Show concrete steps. "
        "Do not write instructor feedback or the next lesson."
    )


def format_weakness_log(weaknesses: list[str]) -> str:
    if not weaknesses:
        return ""
    recent = weaknesses[-5:]
    lines = "\n".join(f"- {item}" for item in recent)
    return f"\nWeakness log (earlier mistakes by this learner):\n{lines}\n"


def mentor_instruction(
    lesson_number: int,
    lesson: str,
    student_text: str | None = None,
    weaknesses: list[str] | None = None,
    revisit: bool = False,
) -> str:
    weakness_block = format_weakness_log(weaknesses or [])

    if student_text is None:
        revisit_line = ""
        if revisit and weakness_block:
            revisit_line = (
                "This lesson, also revisit ONE earlier weakness from the log: "
                "give a small extra problem that tests that same skill in a new form.\n"
            )
        return (
            f"Lesson {lesson_number}/10.\n"
            f"{lesson}\n"
            f"{weakness_block}"
            f"{revisit_line}"
            "Use this structure: video link, short summary, one practice task. "
            "Do not solve the learner's task for them."
        )

    return (
        "Learner reply:\n"
        f"{student_text}\n\n"
        f"{weakness_block}"
        "Reply only as the instructor. Check the actual work. If it is vague or wrong, ask for a correction. "
        "If it is good enough, give one transfer check or pass the lesson."
    )


def run_course(run_id: str, max_turns: int) -> None:
    load_env()

    mentor_model = os.environ.get("MENTOR_MODEL", MENTOR_MODEL)
    student_model = os.environ.get("STUDENT_MODEL", STUDENT_MODEL)
    max_new_tokens = int(os.environ.get("MAX_OUTPUT_TOKENS", "350"))
    max_context_tokens = int(os.environ.get("MAX_CONTEXT_TOKENS", "4096"))
    temperature = float(os.environ.get("TEMPERATURE", "0.7"))

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

    turn = 0
    for index, lesson in enumerate(LESSONS, start=1):
        if turn >= max_turns:
            break

        revisit = (index % 3 == 0) and bool(weakness_log)

        mentor_text, meta = mentor.reply(
            [{"role": "user", "content": mentor_instruction(
                index, lesson, weaknesses=weakness_log, revisit=revisit)}],
            max_new_tokens,
            temperature,
        )
        turn += 1
        save_turn(jsonl_path, md_path, turn, "Mentor", mentor_model, mentor_text, meta)

        if index == bluff_lesson:
            student_text = "I practiced this after the video. It was fine. I just used the rule."
            meta = {"model": student_model, "injected_bluff": True}
            print(f"[{turn + 1}] Student: (INJECTED BLUFF)")
        else:
            student_text, meta = student.reply(
                [{"role": "user", "content": student_instruction(mentor_text)}],
                max_new_tokens,
                temperature,
            )
        turn += 1
        save_turn(jsonl_path, md_path, turn, "Student", student_model, student_text, meta)

        mentor_text, meta = mentor.reply(
            [
                {"role": "user", "content": mentor_instruction(
                    index, lesson, weaknesses=weakness_log, revisit=revisit)},
                {"role": "assistant", "content": mentor_text},
                {"role": "user", "content": mentor_instruction(
                    index, lesson, student_text, weaknesses=weakness_log)},
            ],
            max_new_tokens,
            temperature,
        )
        turn += 1
        save_turn(jsonl_path, md_path, turn, "Mentor", mentor_model, mentor_text, meta)

        note = extract_weakness(index, mentor_text)
        if note:
            weakness_log.append(note)
            with md_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n> _[memory] recorded weakness: {note}_\n")

    print(f"\nSaved {jsonl_path}")
    print(f"Saved {md_path}")
    print(f"Final weakness log ({len(weakness_log)} items):")
    for item in weakness_log:
        print(f"  - {item}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--max-turns", type=int, default=40)
    args = parser.parse_args()
    run_course(args.run_id, args.max_turns)


if __name__ == "__main__":
    main()
