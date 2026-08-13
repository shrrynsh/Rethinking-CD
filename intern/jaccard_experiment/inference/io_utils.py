"""Question sharding + resumable answer writing.

Differs from jaccard_on_qwen, which opens answer files with "w" and therefore
restarts from zero every time. Lightning Studios auto-stop on an idle timer that
GPU activity does not reset, so a multi-hour sweep can lose everything at any
moment. Runs here are append-only and skip question_ids already on disk, making
re-running any script idempotent.
"""
import json
import math
import os


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


def _read_valid_records(path):
    """Parse a JSONL file, stopping at the first unparseable line.

    A process killed mid-write leaves a truncated final line. Appending after it
    would corrupt the file, so the caller truncates back to the last complete
    record. Stopping at the first bad line (rather than skipping it) is
    deliberate: everything after a torn write is not trustworthy either.
    """
    records, valid_bytes = [], 0
    with open(path, "rb") as f:
        for raw in f:
            try:
                rec = json.loads(raw.decode("utf-8"))
                if "question_id" not in rec:
                    break
            except (json.JSONDecodeError, UnicodeDecodeError):
                break
            records.append(rec)
            valid_bytes += len(raw)
    return records, valid_bytes


def prepare_answers_file(answers_file, resume=True):
    """Open the answers file for writing and return (handle, done_ids).

    resume=True  -> repair any torn tail, keep complete records, append
    resume=False -> truncate and start over
    """
    path = os.path.expanduser(answers_file)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if not resume or not os.path.exists(path):
        return open(path, "w"), set()

    records, valid_bytes = _read_valid_records(path)
    done = {r["question_id"] for r in records}

    # Drop a partially written trailing record so the append starts clean.
    if valid_bytes != os.path.getsize(path):
        with open(path, "r+b") as f:
            f.truncate(valid_bytes)
        print(f"[resume] repaired truncated tail of {os.path.basename(path)}")

    if done:
        print(f"[resume] {len(done)} answers already present in "
              f"{os.path.basename(path)} — skipping those")

    return open(path, "a"), done


def load_questions(question_file, num_chunks, chunk_idx, done_ids=frozenset()):
    with open(os.path.expanduser(question_file)) as f:
        questions = [json.loads(line) for line in f]
    questions = get_chunk(questions, num_chunks, chunk_idx)
    if done_ids:
        questions = [q for q in questions if q["question_id"] not in done_ids]
    return questions
