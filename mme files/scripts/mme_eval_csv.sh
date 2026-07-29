#!/usr/bin/env bash
# MME evaluation: render the per-subtask results table (Yes% | Accuracy | Score |
# F1) plus Perception/Cognition group totals for each method's answer file.
#
# Run the inference scripts first (mme_infer_base.sh / mme_infer_cd.sh /
# mme_infer_spurious.sh) to populate ./outputs/mme, then run this script.
#
# Each answer file is evaluated against the combined MME reference. Only answer
# files that exist on disk are evaluated, so this works after running any
# subset of the inference scripts.
#
# Results are written to:
#   - ${OUT_ROOT}/mme_eval_log.txt    (full raw stdout per method, for inspection)
#   - ${OUT_ROOT}/mme_eval_results.csv (long format: one row per method+subtask,
#     columns: method, answer_file, subtask, yes_pct, accuracy, score, f1;
#     plus a perception_total/cognition_total row per method, with totals in
#     the score column)
set -euo pipefail

REF_FILE=${REF_FILE:-./data/mme/mme_reference.jsonl}
OUT_ROOT=${OUT_ROOT:-./outputs/mme}
LOG_FILE=${LOG_FILE:-${OUT_ROOT}/mme_eval_log.txt}
CSV_FILE=${CSV_FILE:-${OUT_ROOT}/mme_eval_results_qwen.csv}

mkdir -p "${OUT_ROOT}"
: > "${LOG_FILE}"

# label -> answer file (mirrors the inference output paths)
RESULTS=(
    "baseline (greedy)|${OUT_ROOT}/baseline/qwen25-7b-mme-greedy.jsonl"
    "baseline (sample)|${OUT_ROOT}/baseline/qwen25-7b-mme-sample.jsonl"
    "vcd (greedy)|${OUT_ROOT}/vcd/qwen25-7b-mme-greedy.jsonl"
    "vcd (sample)|${OUT_ROOT}/vcd/qwen25-7b-mme-sample.jsonl"
    "icd (greedy)|${OUT_ROOT}/icd/qwen25-7b-mme-greedy.jsonl"
    "icd (sample)|${OUT_ROOT}/icd/qwen25-7b-mme-sample.jsonl"
    "sid (greedy)|${OUT_ROOT}/sid/qwen25-7b-mme-greedy.jsonl"
    "sid (sample)|${OUT_ROOT}/sid/qwen25-7b-mme-sample.jsonl"
    "pba (greedy)|${OUT_ROOT}/pba/qwen25-7b-mme-greedy.jsonl"
    "olm (greedy)|${OUT_ROOT}/olm/qwen25-7b-mme-greedy.jsonl"
    "apc (sample)|${OUT_ROOT}/apc/qwen25-7b-mme-sample.jsonl"
)

found=0
# Temp file holding raw stdout for the current method, used for CSV parsing.
tmp_out=$(mktemp)
trap 'rm -f "${tmp_out}"' EXIT

for entry in "${RESULTS[@]}"; do
    label=${entry%%|*}
    res_file=${entry##*|}

    if [[ ! -f "${res_file}" ]]; then
        continue
    fi
    found=1

    {
        echo "========================================================================"
        echo "Method: ${label}"
        echo "Answer file: ${res_file}"
        echo "========================================================================"
    } | tee -a "${LOG_FILE}"

    python ./eval/mme_eval.py \
        --ref-files "${REF_FILE}" \
        --res-files "${res_file}" \
        | tee "${tmp_out}" | tee -a "${LOG_FILE}"
    echo | tee -a "${LOG_FILE}"

    # Parse the per-subtask table plus Perception/Cognition totals from this
    # run's output into long-format CSV rows (one row per method+subtask, plus
    # two summary rows per method for the group totals).
    METHOD="${label}" RES_FILE="${res_file}" python3 - "${tmp_out}" "${CSV_FILE}" <<'PYEOF'
import csv
import os
import re
import sys

raw_path, csv_path = sys.argv[1], sys.argv[2]
method = os.environ["METHOD"]
res_file = os.environ["RES_FILE"]

fieldnames = ["method", "answer_file", "subtask", "yes_pct", "accuracy", "score", "f1"]

# Per-subtask rows, e.g.:
# existence            48.33    0.9500  185.00   0.9492
subtask_pattern = re.compile(
    r"^([A-Za-z_]+)\s+"
    r"([-+]?[0-9]*\.?[0-9]+)\s+"
    r"([-+]?[0-9]*\.?[0-9]+)\s+"
    r"([-+]?[0-9]*\.?[0-9]+)\s+"
    r"([-+]?[0-9]*\.?[0-9]+)\s*$"
)
# Group totals, e.g.: "Perception Total          1297.37"
total_pattern = re.compile(r"^(Perception|Cognition) Total\s+([-+]?[0-9]*\.?[0-9]+)\s*$")

new_rows = []
with open(raw_path) as f:
    for line in f:
        line = line.rstrip("\n")
        m = subtask_pattern.match(line)
        if m:
            subtask, yes_pct, accuracy, score, f1 = m.groups()
            if subtask.lower() == "subtask":
                continue  # header line
            new_rows.append({
                "method": method,
                "answer_file": res_file,
                "subtask": subtask,
                "yes_pct": yes_pct,
                "accuracy": accuracy,
                "score": score,
                "f1": f1,
            })
            continue
        m = total_pattern.match(line)
        if m:
            group, total = m.groups()
            new_rows.append({
                "method": method,
                "answer_file": res_file,
                "subtask": f"{group.lower()}_total",
                "yes_pct": "",
                "accuracy": "",
                "score": total,
                "f1": "",
            })

existing_rows = []
if os.path.exists(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)

existing_rows.extend(new_rows)

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in existing_rows:
        writer.writerow(r)
PYEOF

done

if [[ "${found}" -eq 0 ]]; then
    echo "No MME answer files found under ${OUT_ROOT}." >&2
    echo "Run the inference scripts first, e.g.: bash scripts/mme_infer_base.sh" >&2
    exit 1
fi

echo "Full log written to: ${LOG_FILE}"
echo "CSV results written to: ${CSV_FILE}"