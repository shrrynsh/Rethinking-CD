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
set -euo pipefail

REF_FILE=${REF_FILE:-./data/mme/mme_reference.jsonl}
OUT_ROOT=${OUT_ROOT:-./outputs/mme}

# label -> answer file (mirrors the inference output paths)
RESULTS=(
    "baseline (greedy)|${OUT_ROOT}/baseline/llava-7b-mme-greedy.jsonl"
    "baseline (sample)|${OUT_ROOT}/baseline/llava-7b-mme-sample.jsonl"
    "vcd (greedy)|${OUT_ROOT}/vcd/llava-7b-mme-greedy.jsonl"
    "vcd (sample)|${OUT_ROOT}/vcd/llava-7b-mme-sample.jsonl"
    "icd (greedy)|${OUT_ROOT}/icd/llava-7b-mme-greedy.jsonl"
    "icd (sample)|${OUT_ROOT}/icd/llava-7b-mme-sample.jsonl"
    "sid (greedy)|${OUT_ROOT}/sid/llava-7b-mme-greedy.jsonl"
    "sid (sample)|${OUT_ROOT}/sid/llava-7b-mme-sample.jsonl"
    "pba (greedy)|${OUT_ROOT}/pba/llava-7b-mme-greedy.jsonl"
    "olm (greedy)|${OUT_ROOT}/olm/llava-7b-mme-greedy.jsonl"
    "apc (sample)|${OUT_ROOT}/apc/llava-7b-mme-sample.jsonl"
)

found=0
for entry in "${RESULTS[@]}"; do
    label=${entry%%|*}
    res_file=${entry##*|}

    if [[ ! -f "${res_file}" ]]; then
        continue
    fi
    found=1

    echo "========================================================================"
    echo "Method: ${label}"
    echo "Answer file: ${res_file}"
    echo "========================================================================"
    python ./eval/mme_eval.py \
        --ref-files "${REF_FILE}" \
        --res-files "${res_file}"
    echo
done

if [[ "${found}" -eq 0 ]]; then
    echo "No MME answer files found under ${OUT_ROOT}." >&2
    echo "Run the inference scripts first, e.g.: bash scripts/mme_infer_base.sh" >&2
    exit 1
fi
