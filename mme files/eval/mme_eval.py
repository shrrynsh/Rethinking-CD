"""MME benchmark evaluation script.

This module evaluates LLaVA answer files against MME reference files and renders
a per-subtask results table (Yes%, Accuracy, Score, F1) plus Perception and
Cognition group totals.

Only the answer-classification primitives (`classify_answer` and `is_correct`)
are implemented here for now. The remaining components (data loading, alignment
validation, subtask/image grouping, per-subtask metrics, group totals, results
table rendering, and the argparse entry point) are added by later tasks.
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple

from mme_constants import PERCEPTION, COGNITION


def classify_answer(text: str) -> str:
    """Classify a generated answer as ``"yes"``, ``"no"``, or ``"unknown"``.

    Matching is case-insensitive and ignores surrounding whitespace. The text is
    trimmed and lower-cased, then checked for the substrings ``"yes"`` and
    ``"no"``. ``"yes"`` takes precedence over ``"no"`` when both are present.

    Args:
        text: The generated answer text.

    Returns:
        Exactly one of ``"yes"``, ``"no"``, or ``"unknown"``.
    """
    normalized = text.strip().lower()
    if "yes" in normalized:
        return "yes"
    if "no" in normalized:
        return "no"
    return "unknown"


def is_correct(label: str, prediction: str) -> bool:
    """Return ``True`` iff the prediction matches the ground-truth label.

    The label is normalized (stripped and lower-cased) and compared against the
    classification of the prediction. Because ``classify_answer`` only ever
    returns ``"yes"``, ``"no"``, or ``"unknown"`` and labels are expected to be
    ``"yes"`` or ``"no"``, a prediction that classifies as ``"unknown"`` is
    always incorrect.

    A prediction that classifies as ``"unknown"`` is always incorrect,
    regardless of the label (Requirement 5.5).

    Args:
        label: The ground-truth label (expected ``"yes"`` or ``"no"``).
        prediction: The generated answer text.

    Returns:
        ``True`` if the predicted classification equals the normalized label
        and is not ``"unknown"``, otherwise ``False``.
    """
    predicted = classify_answer(prediction)
    if predicted == "unknown":
        return False
    normalized_label = label.strip().lower()
    return predicted == normalized_label


def load_json_lines(file_path: str) -> List[Dict]:
    """Read a JSONL file, returning one dict per non-empty line.

    Mirrors :func:`eval.pope_eval_base.load_json_lines`. The path is expanded
    with :func:`os.path.expanduser`; each non-empty line is parsed as JSON.

    A missing path is detected up front so the caller gets a descriptive error
    naming the path before any processing begins (Requirement 1.5).

    Args:
        file_path: Path to a file where each line is a JSON object.

    Returns:
        A list of dicts, one per non-empty line, in file order.

    Raises:
        FileNotFoundError: If the expanded path does not exist.
    """
    path = os.path.expanduser(file_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data path not found: {path}")

    data: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def validate_alignment(ref_data: List[Dict], res_data: List[Dict]) -> None:
    """Validate that reference and result entries are positionally aligned.

    Two checks are performed:

    1. The two sequences must have the same length; otherwise a ``ValueError``
       reporting both counts is raised (Requirement 9.5).
    2. The ``question_id`` of each reference entry must equal the
       ``question_id`` of the result entry at the same position; otherwise a
       ``ValueError`` naming the mismatched ids and the index is raised
       (Requirement 9.6).

    Args:
        ref_data: Reference (ground-truth) entries.
        res_data: Result (generated answer) entries.

    Raises:
        ValueError: If the lengths differ, or if any positional
            ``question_id`` differs between ``ref_data`` and ``res_data``.
    """
    if len(ref_data) != len(res_data):
        raise ValueError(
            f"REF length ({len(ref_data)}) and RES length ({len(res_data)}) do not match"
        )

    for index, (ref_item, res_item) in enumerate(zip(ref_data, res_data)):
        ref_id, res_id = ref_item["question_id"], res_item["question_id"]
        if ref_id != res_id:
            raise ValueError(
                f"ID mismatch at index {index}: REF={ref_id}, RES={res_id}"
            )


def group_by_subtask(
    ref_data: List[Dict], res_data: List[Dict]
) -> Dict[str, List[Tuple[Dict, Dict]]]:
    """Group aligned ``(ref, res)`` pairs by subtask.

    Each positional reference entry is paired with the answer entry at the same
    index and attributed to the subtask named by the reference entry's
    ``category`` field. Subtask keys appear in first-seen order.

    Args:
        ref_data: Reference entries, each carrying a ``category`` field.
        res_data: Answer entries, positionally aligned with ``ref_data``.

    Returns:
        A mapping from subtask name to the list of ``(ref, res)`` pairs
        attributed to that subtask, preserving input order.
    """
    grouped: Dict[str, List[Tuple[Dict, Dict]]] = {}
    for ref_item, res_item in zip(ref_data, res_data):
        subtask = ref_item["category"]
        grouped.setdefault(subtask, []).append((ref_item, res_item))
    return grouped


def group_by_image(
    pairs: List[Tuple[Dict, Dict]]
) -> Dict[str, List[Tuple[Dict, Dict]]]:
    """Group a subtask's ``(ref, res)`` pairs by reference image identifier.

    Each image in an MME subtask must carry exactly two questions. If any image
    is associated with a number of questions other than two, a ``ValueError`` is
    raised naming the affected image identifier and the count found.

    Args:
        pairs: The ``(ref, res)`` pairs of a single subtask.

    Returns:
        A mapping from image identifier to the list of ``(ref, res)`` pairs for
        that image, preserving first-seen order.

    Raises:
        ValueError: If any image is associated with a number of questions other
            than two.
    """
    grouped: Dict[str, List[Tuple[Dict, Dict]]] = {}
    for ref_item, res_item in pairs:
        image = ref_item["image"]
        grouped.setdefault(image, []).append((ref_item, res_item))

    for image, image_pairs in grouped.items():
        count = len(image_pairs)
        if count != 2:
            raise ValueError(
                f"Image {image} has {count} questions, expected exactly 2"
            )

    return grouped


@dataclass
class SubtaskMetrics:
    """Per-subtask evaluation metrics for the MME results table.

    Attributes:
        subtask: The subtask name (the reference ``category``).
        total_questions: Number of questions in the subtask.
        total_images: Number of distinct images in the subtask (each carrying
            exactly two questions).
        yes_percentage: Percentage of answers classified ``"yes"``, in ``[0, 100]``.
        accuracy: Fraction of correctly answered questions, in ``[0, 1]``.
        accuracy_plus: Fraction of images whose two questions are both correct,
            in ``[0, 1]``.
        score: ``(accuracy + accuracy_plus) * 100``, in ``[0, 200]``.
        f1: Harmonic mean of precision and recall with ``"yes"`` as the positive
            class, in ``[0, 1]``.
    """

    subtask: str
    total_questions: int
    total_images: int
    yes_percentage: float  # 0..100
    accuracy: float  # 0..1
    accuracy_plus: float  # 0..1
    score: float  # 0..200 == (accuracy + accuracy_plus) * 100
    f1: float  # 0..1


def compute_subtask_metrics(
    subtask: str, pairs: List[Tuple[Dict, Dict]]
) -> SubtaskMetrics:
    """Compute the per-subtask metrics for one MME subtask.

    For each ``(ref, res)`` pair, per-question correctness is determined by
    :func:`is_correct` and yes-classification by :func:`classify_answer`. The
    metrics are (with ``"yes"`` as the positive class):

    * **Accuracy** = correct questions / total questions (Req 6.2).
    * **Yes%** = answers classified ``"yes"`` / total questions × 100 (Req 6.3).
    * **F1** = ``2·precision·recall / (precision + recall)`` where
      ``precision = tp/(tp+fp)`` and ``recall = tp/(tp+fn)``; F1 is ``0.0`` when
      its denominator is ``0`` (Req 6.4). A prediction that classifies as
      ``"unknown"`` is never counted as ``"yes"``.
    * **Accuracy_Plus** = images where BOTH questions are correct / total images
      (Req 7.1, 7.2). Images are grouped via :func:`group_by_image`, which
      raises if any image is not associated with exactly two questions.
    * **Score** = ``(accuracy + accuracy_plus) * 100`` (Req 7.3).

    All divisions are guarded so an empty subtask yields ``0.0`` rates.

    Args:
        subtask: The subtask name to record on the result.
        pairs: The ``(ref, res)`` pairs of this subtask, where each ``ref`` has
            keys ``question_id``, ``image``, ``text``, ``label``, ``category``
            and each ``res`` has key ``text``.

    Returns:
        A :class:`SubtaskMetrics` describing the subtask.

    Raises:
        ValueError: If any image is associated with a number of questions other
            than two (propagated from :func:`group_by_image`).
    """
    total_questions = len(pairs)

    correct_questions = 0
    yes_count = 0
    tp = fp = fn = 0
    for ref_item, res_item in pairs:
        label = ref_item["label"]
        prediction = res_item["text"]

        if is_correct(label, prediction):
            correct_questions += 1

        predicted_yes = classify_answer(prediction) == "yes"
        if predicted_yes:
            yes_count += 1

        label_yes = label.strip().lower() == "yes"
        if label_yes and predicted_yes:
            tp += 1
        elif (not label_yes) and predicted_yes:
            fp += 1
        elif label_yes and (not predicted_yes):
            fn += 1
        # else: label no & predicted not-yes -> true negative (not tracked)

    accuracy = correct_questions / total_questions if total_questions else 0.0
    yes_percentage = (
        yes_count / total_questions * 100 if total_questions else 0.0
    )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    images = group_by_image(pairs)
    total_images = len(images)
    correct_images = 0
    for image_pairs in images.values():
        if all(
            is_correct(ref_item["label"], res_item["text"])
            for ref_item, res_item in image_pairs
        ):
            correct_images += 1
    accuracy_plus = correct_images / total_images if total_images else 0.0

    score = (accuracy + accuracy_plus) * 100

    return SubtaskMetrics(
        subtask=subtask,
        total_questions=total_questions,
        total_images=total_images,
        yes_percentage=yes_percentage,
        accuracy=accuracy,
        accuracy_plus=accuracy_plus,
        score=score,
        f1=f1,
    )


def compute_group_total(
    metrics_by_subtask: Dict[str, SubtaskMetrics], group: List[str]
) -> float:
    """Sum :attr:`SubtaskMetrics.score` over the subtasks in ``group`` present.

    The total for a group (Perception or Cognition) is the sum of the
    per-subtask Score over exactly those subtasks named in ``group`` that exist
    as keys in ``metrics_by_subtask`` (Requirements 8.1, 8.2). Subtasks listed in
    ``group`` but absent from the metrics are skipped (contributing nothing), so
    an empty intersection yields ``0.0``.

    Args:
        metrics_by_subtask: Mapping from subtask name to its
            :class:`SubtaskMetrics`.
        group: The subtask names of a group (e.g. ``PERCEPTION`` or
            ``COGNITION`` from :mod:`eval.mme_constants`).

    Returns:
        The sum of ``score`` over the subtasks in ``group`` present in
        ``metrics_by_subtask``; ``0.0`` when none are present.
    """
    return sum(
        (
            metrics_by_subtask[subtask].score
            for subtask in group
            if subtask in metrics_by_subtask
        ),
        0.0,
    )


# Column layout for the rendered results table. The first column holds the
# subtask (or group-total) label; the remaining four hold the metrics.
_LABEL_WIDTH = 24
_COL_WIDTH = 9


def render_results_table(
    metrics_by_subtask: Dict[str, SubtaskMetrics],
    perception_total: float,
    cognition_total: float,
) -> str:
    """Render the MME per-subtask results table as a multi-line string.

    The table has a header row with a ``Subtask`` label column followed by the
    four metric columns ``Yes%``, ``Accuracy``, ``Score``, and ``F1``. It then
    has one row per subtask present in ``metrics_by_subtask``, ordered with all
    Perception-group subtasks first (in ``PERCEPTION`` order) followed by all
    Cognition-group subtasks (in ``COGNITION`` order) (Requirements 9.1, 9.3).
    Only subtasks present in ``metrics_by_subtask`` are emitted.

    After a separator, a ``Perception Total`` row and a ``Cognition Total`` row
    populate the ``Score`` column with ``perception_total`` and
    ``cognition_total`` respectively, leaving the other metric columns blank
    (Requirements 8.3, 9.2).

    Float formatting follows the design example: ``Yes%`` and ``Score`` to two
    decimals, ``Accuracy`` and ``F1`` to four decimals.

    Args:
        metrics_by_subtask: Mapping from subtask name to its
            :class:`SubtaskMetrics`.
        perception_total: Summed Score over the Perception-group subtasks.
        cognition_total: Summed Score over the Cognition-group subtasks.

    Returns:
        A multi-line string containing the header, per-subtask rows, a
        separator, and the two group-total rows.
    """

    def _row(label: str, yes_pct: str, acc: str, score: str, f1: str) -> str:
        return (
            f"{label:<{_LABEL_WIDTH}}"
            f"{yes_pct:<{_COL_WIDTH}}"
            f"{acc:<{_COL_WIDTH}}"
            f"{score:<{_COL_WIDTH}}"
            f"{f1:<{_COL_WIDTH}}"
        ).rstrip()

    lines: List[str] = []
    lines.append(_row("Subtask", "Yes%", "Accuracy", "Score", "F1"))

    for subtask in PERCEPTION + COGNITION:
        if subtask not in metrics_by_subtask:
            continue
        m = metrics_by_subtask[subtask]
        lines.append(
            _row(
                subtask,
                f"{m.yes_percentage:.2f}",
                f"{m.accuracy:.4f}",
                f"{m.score:.2f}",
                f"{m.f1:.4f}",
            )
        )

    separator_width = _LABEL_WIDTH + _COL_WIDTH * 4
    lines.append("-" * separator_width)

    lines.append(_row("Perception Total", "", "", f"{perception_total:.2f}", ""))
    lines.append(_row("Cognition Total", "", "", f"{cognition_total:.2f}", ""))

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the MME evaluator.

    Mirrors :func:`eval.pope_eval_base.parse_args`: both ``--ref-files`` (the
    MME reference JSONL) and ``--res-files`` (the generated answer JSONL) are
    required. argparse converts the hyphenated flags to ``args.ref_files`` and
    ``args.res_files``.

    Returns:
        The parsed arguments namespace with ``ref_files`` and ``res_files``.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate MME answers and render a per-subtask results table."
    )
    parser.add_argument(
        "--ref-files",
        type=str,
        required=True,
        help="Path to MME reference (GT) file (one JSON per line).",
    )
    parser.add_argument(
        "--res-files",
        type=str,
        required=True,
        help="Path to result (generated answer) file (one JSON per line).",
    )
    return parser.parse_args()


def main() -> None:
    """Load, validate, evaluate, and print the MME results table.

    Loads the reference and answer JSONL files, validates positional alignment
    by ``question_id``, groups pairs by subtask, computes per-subtask metrics
    (each subtask grouping by image inside :func:`compute_subtask_metrics`),
    computes the Perception and Cognition group totals, and prints the rendered
    results table (Requirements 9.1, 9.4, 9.5, 9.6).
    """
    args = parse_args()
    ref_data = load_json_lines(args.ref_files)
    res_data = load_json_lines(args.res_files)
    validate_alignment(ref_data, res_data)
    by_subtask = group_by_subtask(ref_data, res_data)
    metrics_by_subtask: Dict[str, SubtaskMetrics] = {}
    for subtask, pairs in by_subtask.items():
        metrics_by_subtask[subtask] = compute_subtask_metrics(subtask, pairs)
    perception_total = compute_group_total(metrics_by_subtask, PERCEPTION)
    cognition_total = compute_group_total(metrics_by_subtask, COGNITION)
    print(render_results_table(metrics_by_subtask, perception_total, cognition_total))


if __name__ == "__main__":
    main()
