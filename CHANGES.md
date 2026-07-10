# Changes to run_calculus_agents.py

Three improvements that address weak points in the first version. All changes live in a
single file, and the default run behavior stays compatible with the original.

## 1. Mentor memory of past weaknesses (weakness log)

**Problem:** in the first version the mentor received only the current lesson each time.
It never saw earlier lessons, so it could not "circle back to weak spots" - one of the
criteria for a strong submission in the assignment.

**Solution:** added a `weakness_log` list. After each lesson, `extract_weakness()` checks
the mentor's reply for error signals (mistake, wrong, try again, etc.) and, if there was
a mistake, records a short note. At the start of each following lesson this log is passed
to the mentor via `mentor_instruction(..., weaknesses=...)`. Roughly every third lesson
the mentor is explicitly asked to revisit one earlier weakness with a new problem
(`revisit=True`).

This is the lightweight "memory tool" the assignment mentions.

## 2. A real bluff (not hand-written)

**Problem:** in the first version the bluff moment was effectively edited in by hand. The
model could not produce it on its own, because the student was never given history and
did not know whether it had already bluffed.

**Solution:** at the start of a run, one lesson is chosen at random (`bluff_lesson`). On
that lesson the student's real answer is replaced with an empty claim, "I practiced, it
was fine," with no details. This tests the system's real behavior: does the mentor catch
the bluff by itself? The injected reply is marked in the log with an `injected_bluff` flag.

## 3. An honest raw log

**Problem:** only the cleaned transcript went into the submission, while the assignment
asks us not to trim the dialogue down to the good parts.

**Solution:** each run writes a full raw log to `logs/<run_id>.md` and
`logs/<run_id>.jsonl`, including markers for where the injected bluff happened and what
was added to the mentor's memory. It's recommended to attach this raw log next to the
cleaned version.

## How to run

Unchanged:

```
python run_calculus_agents.py
```

Logs appear in the `logs/` folder. The final weakness log is printed to the console at
the end.

## Note on models

Qwen3-1.7B and Llama-3.2-1B are very small and often slip on arithmetic. These changes
make the system more honest (a real bluff, real memory), but for reliably clean math it's
worth trying larger models or an API.
