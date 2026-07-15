# Changes to `run_calculus_agents.py`

Improvements over the original version: **multi-round testing with verdict
gating**, a **sympy layer for automatic answer checking**, and fixes from the
instructor's feedback (gentle onboarding, easy formula entry, topic continuity).

## 1. Multi-round testing inside a lesson

Previously a lesson was a rigid 3 turns (intro -> answer -> feedback), and the
loop advanced to the next lesson unconditionally - so the assignment's core
requirement, "do not advance the student without evidence of application," was
not actually enforced in code.

Now:

- **The mentor holds a coherent conversation.** A running history of turns
  (`dialogue`) is kept within the lesson, and every model call receives it in
  full via `to_messages()` - previously the agents saw only the last message.
- **Rounds of `test -> answer -> evaluate`.** The lesson loops up to `MAX_TESTS`
  rounds. Each round the student answers, the mentor checks the work and poses
  the next test.
- **Verdict gating.** The mentor must end every evaluation with a line
  `VERDICT: PASS` or `VERDICT: CONTINUE` (protocol lives in the system prompt).
  `parse_verdict()` reads it; the course advances **only** on `PASS`. A
  missing/ambiguous verdict is treated as `CONTINUE`, so the mentor never
  advances without an explicit PASS.
- **A bluff has consequences.** Without real work there is no PASS, and the loop
  keeps the student on the same lesson.
- **An open weakness instead of a silent pass.** If PASS is not reached after
  `MAX_TESTS` rounds, the lesson is marked `NOT PASSED` in the log, which is more
  transparent for the self-evaluation section.
- **Weakness memory** is recorded on a `CONTINUE` verdict, not by keyword
  grepping as before.

Other: Qwen thinking mode disabled (`enable_thinking=False`) to remove
`</think>` noise; video links hardcoded in `LESSONS` so the model doesn't
hallucinate them; a `SEED` added for reproducibility; the verdict line is
stripped from the message the student sees (`strip_verdict`).

## 2. Sympy layer for automatic answer checking

Problem: correctness was judged only by the mentor model (1.7B), which can be
wrong. Now the answer is also checked programmatically.

- Each lesson in `LESSONS` carries a list of reference answers
  (`expected_answers`). An empty list means the lesson is verbal/conceptual
  (lessons 2 and 10), and the auto-check is skipped for it.
- `verify_answer()` pulls math fragments out of the student's free text
  (`_candidate_expressions`: right-hand sides of equations, individual tokens,
  and maximal "math" runs embedded in prose) and compares them to the reference
  **symbolically**: `sympy.simplify(cand - target) == 0`. This catches
  equivalent forms, e.g. `20x^3-6x` and `2x(10x^2-3)` are treated as equal.
- The result is passed to the mentor as a **`SYSTEM CHECK`** - evidence that a
  correct answer is/isn't present in the work - but the final verdict is still
  the mentor's (verbal lessons simply get no signal). False positives are not
  critical.
- Returns `True` / `False` / `None` (`None` = check not applicable). If `sympy`
  is not installed, the layer silently disables itself and the script runs as
  before.
- The check result is logged to Markdown (`[auto-check] sympy: …`) and to the
  JSONL meta (`auto_check`).

`sympy` added to `requirements.txt`.

## 3. Usability refinements

Two points guided this section: (a) typing formulas in a chat is awkward for a
learner, and (b) the mentor should start simple, since a beginner is otherwise
lost immediately.

**Lower friction for formula entry:**

- The mentor prompt now explicitly accepts *informal notation* and corrects only
  the math, not the formatting: you can write `^` for powers, `sqrt()` or
  "square root of", "x squared", or just answer in words.
- The sympy layer understands "spoken" math: `_normalize_math_language()`
  converts "squared/cubed", "to the power of", "over/divided by", "times",
  "plus/minus", "sqrt of/root of" into symbolic form and strips chat punctuation
  (`?`, `!`). Now "6 x squared over sqrt(4x^3+2)" or "slope is minus 14" are
  recognized as correct answers.
- The student prompt itself answers in words or loose notation more often, as a
  beginner realistically would.

**Gentle onboarding and a difficulty ladder:**

- The mentor prompt is rewritten into a warm, patient, *short* tone: one small
  ask per message, no walls of text and no batch of problems at once.
- The ladder: bridge from the previous topic -> one line of intuition -> a tiny
  warm-up -> the main task -> transfer. It escalates only after a success; if the
  student gets lost, the mentor steps back down instead of pushing.
- Each lesson now has a warm-up (`warmup`) - a number, yes/no, or "in your own
  words" question, no heavy formula. The lesson starts with it.
- On the very first message of the course the mentor gives a short, friendly
  welcome.
- The student sometimes shows mild impatience or overwhelm ("this is a lot", "can
  we go slower?"), which tests the mentor's patience.
- `MAX_TESTS` raised from 3 to 5: the ladder (warm-up + task + transfer + fixes)
  needs more rounds.

**Topic continuity (each topic builds on the previous one):**

- Each lesson gains a "bridge" (`bridge`) - one plain sentence linking the new
  topic to the last one, with no free-floating abstraction: limit -> derivative
  as a limit of the increment (Δy/Δx as Δx->0) -> power rule as the shortcut for
  that limit -> product/quotient -> chain rule -> implicit differentiation (the
  same chain rule) -> tangent line (derivative = slope) -> optimization (slope = 0
  at an extremum) -> related rates (the chain rule again, over time).
- The mentor prompt requires opening each lesson with this bridge: first a brief
  recap of the prior idea, then how the new one grows out of it, and only then
  the intuition and warm-up. The ladder was extended with a "bridge" step.

## Tests

The sympy layer's logic is covered by unit tests on 16 realistic cases (correct
answers in various forms, a wrong sign, a bluff with no answer, a verbal lesson,
an expression inside prose, "spoken" input like "x squared over ..." and "minus
14") - all pass. Syntax verified with `py_compile`.

> Note: no end-to-end run with the real models was performed (it needs torch and
> the model weights - a heavy environment); syntax and all non-model logic were
> verified. Before submitting, run the full course locally and rebuild the final
> document (prompts + transcript + reflection).
