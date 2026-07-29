"""MME raw-tree data-preparation converter.

This module converts the committed raw MME release tree
(``data/mme/MME_Benchmark_release_version/MME_Benchmark/{subtask}/``) into the
exact JSONL + image layout the already-built MME inference and evaluation
pipeline consumes. It is a one-time preparation step: the inference modules
(``inference/mme_infer_*.py``) and the evaluator (``eval/mme_eval.py``,
``eval/mme_constants.py``) are treated as fixed downstream consumers.

The converter:

* parses each raw two-line ``question<TAB>answer`` ``.txt`` annotation into
  reference records,
* auto-detects each subtask's on-disk layout (FLAT vs SPLIT) and reconciles the
  two layouts into a single uniform image view under ``data/mme/images/``,
* derives the combined inference-input file (``mme_questions.jsonl``) from the
  combined reference (``mme_reference.jsonl``), and
* self-validates its output so the downstream evaluator's invariants hold by
  construction.

It follows the repo convention of pure, testable helper functions plus a thin
``argparse`` entry point, with a sibling ``eval/test_mme_prepare.py`` carrying
the pytest + Hypothesis tests.

The pure helpers, the I/O-bound functions, and the ``main`` entry point are
added by later tasks; this module currently only establishes the imports and
the core constants.
"""

import os
import json
import argparse
import collections
import shutil

from mme_constants import SUBTASKS, PERCEPTION, COGNITION


# Per-subtask raw sub-paths for the SPLIT layout: images live in ``images/`` and
# annotations live in ``questions_answers_YN/`` inside the subtask folder.
SPLIT_IMAGE_DIR = "images"
SPLIT_TEXT_DIR = "questions_answers_YN"

# Recognized image file extensions (matched case-insensitively by callers).
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# The only two valid (normalized, lower-cased) ground-truth labels.
VALID_LABELS = {"yes", "no"}

def parse_annotation(raw_text: str) -> list[tuple[str, str]]:
    """Parse one raw ``.txt`` body into exactly two ``(question, label)`` pairs.

    The body is expected to contain exactly two non-empty lines (a line with at
    least one non-whitespace character), each of the form
    ``question<TAB>answer`` where ``answer`` is ``Yes`` or ``No`` in any letter
    casing. Lines that are empty or whitespace-only are ignored.

    Args:
        raw_text: The full contents of the annotation file.

    Returns:
        A two-element list ``[(question, label), (question, label)]`` in
        top-to-bottom line order. Each ``question`` is the text left of the
        first TAB with surrounding whitespace removed; each ``label`` is the
        text right of the first TAB, whitespace-stripped and lower-cased, and is
        exactly ``"yes"`` or ``"no"``.

    Raises:
        ValueError: If the body does not contain exactly two non-empty lines; if
            a non-empty line lacks a TAB separator; if a question is empty after
            whitespace removal; or if an answer is not ``Yes``/``No``
            (case-insensitive). The message names the offending line count,
            line, or value so the caller can identify the bad file.
    """
    lines = [ln for ln in raw_text.splitlines() if ln.strip()]
    if len(lines) != 2:
        raise ValueError(
            f"expected exactly 2 non-empty lines, found {len(lines)}"
        )
    pairs: list[tuple[str, str]] = []
    for ln in lines:
        if "\t" not in ln:
            raise ValueError(f"missing TAB separator in line: {ln!r}")
        question, _, answer = ln.partition("\t")
        question = question.strip()
        label = answer.strip().lower()
        if not question:
            raise ValueError(f"empty question in line: {ln!r}")
        if label not in VALID_LABELS:
            raise ValueError(f"label must be Yes/No, got: {answer.strip()!r}")
        pairs.append((question, label))
    return pairs


def make_question_id(subtask: str, stem: str, line_index: int) -> str:
    """Build the deterministic, globally-unique id for one question.

    The scheme is ``f"{subtask}/{stem}_{line_index}"`` with
    ``line_index`` in ``{0, 1}``. It depends only on its inputs (no wall-clock,
    randomness, or iteration nondeterminism), so it is deterministic and stable.
    Subtask folders are distinct and stems are unique within a subtask, and
    ``line_index`` distinguishes the two questions of one image, so the id is
    globally unique with two distinct ids (``_0`` and ``_1``) per image.

    Args:
        subtask: The subtask folder name (a member of ``SUBTASKS``).
        stem: The image filename with its extension removed.
        line_index: The question's line index within the annotation, ``0`` or
            ``1``.

    Returns:
        The ``question_id`` string ``f"{subtask}/{stem}_{line_index}"``.
    """
    return f"{subtask}/{stem}_{line_index}"


def build_image_field(subtask: str, image_filename: str) -> str:
    """Build the uniform ``image`` field for a reference/question record.

    The ``image`` field is always ``f"{subtask}/{filename}"`` so it resolves
    under ``--image-folder ./data/mme/images`` regardless of the raw on-disk
    layout (FLAT or SPLIT). The real on-disk extension is preserved by the
    caller passing the actual ``image_filename`` (``.jpg`` vs ``.png``), so the
    ``image`` field and the materialized file always agree.

    Args:
        subtask: The subtask folder name (a member of ``SUBTASKS``).
        image_filename: The image's on-disk filename including its real
            extension.

    Returns:
        The ``image`` field string ``f"{subtask}/{image_filename}"``.
    """
    return f"{subtask}/{image_filename}"


def build_records_for_image(
    subtask: str, image_filename: str, raw_text: str
) -> list[dict]:
    """Build the two reference records for one image.

    Combines :func:`parse_annotation`, :func:`make_question_id`, and
    :func:`build_image_field` into the two ``Reference_Record`` dicts for a
    single image. The image stem is derived from ``image_filename`` via
    :func:`os.path.splitext` (so the real on-disk extension is dropped only for
    the ``question_id`` stem, while the ``image`` field keeps the full filename).

    Args:
        subtask: The subtask folder name the image came from (a member of
            ``SUBTASKS``); becomes each record's ``category``.
        image_filename: The image's on-disk filename including its real
            extension.
        raw_text: The full contents of the image's ``.txt`` annotation file.

    Returns:
        A two-element list of reference-record dicts, each with exactly the keys
        ``{question_id, image, text, label, category}``. Both records share the
        same ``image`` (``f"{subtask}/{image_filename}"``) and ``category``
        (``== subtask``); their ``question_id``s end in ``_0`` and ``_1``
        respectively; and their ``text``/``label`` come from the first and
        second parsed ``(question, label)`` pairs in top-to-bottom order.

    Raises:
        ValueError: Propagated from :func:`parse_annotation` if ``raw_text`` is
            malformed (wrong non-empty line count, missing TAB, empty question,
            or a label that is not ``Yes``/``No``).
    """
    stem = os.path.splitext(image_filename)[0]
    image_field = build_image_field(subtask, image_filename)
    records: list[dict] = []
    for line_index, (question, label) in enumerate(parse_annotation(raw_text)):
        records.append(
            {
                "question_id": make_question_id(subtask, stem, line_index),
                "image": image_field,
                "text": question,
                "label": label,
                "category": subtask,
            }
        )
    return records


def validate_reference(records: list) -> None:
    """Re-assert the evaluator's invariants on the combined reference.

    This self-validation runs *before* any output file is created, opened, or
    written, so a failure never leaves a half-written reference that the
    downstream evaluator (``eval/mme_eval.py``) would reject. All four checks
    are performed in full so the raised error names every offender (rather than
    aborting on the first bad record), giving the operator a complete picture to
    fix the raw data and re-run.

    The invariants mirror what ``eval/mme_eval.py`` consumes:

    * every ``question_id`` occurs exactly once across the combined reference,
    * every distinct ``image`` value is associated with exactly two records,
    * every ``label`` is exactly ``"yes"`` or ``"no"``, and
    * every ``category`` is one of the fourteen ``SUBTASKS`` names.

    Args:
        records: The combined list of ``Reference_Record`` dicts to validate.

    Returns:
        ``None`` if every invariant holds.

    Raises:
        ValueError: If any ``question_id`` is duplicated (the message names the
            duplicated ids and their counts); if any ``image`` does not have
            exactly two records (the message names the affected images and their
            record counts); if any ``label`` is not ``"yes"``/``"no"`` (the
            message names the offending values); or if any ``category`` is not a
            member of ``SUBTASKS`` (the message names the offending values). No
            output file is created, opened, or written when this raises.
    """
    # All checks complete before any write happens (caller invokes this prior to
    # opening any output file). Each check scans the full record list so the
    # error message can name every offender at once.

    # 1. question_id uniqueness: every id occurs exactly once.
    id_counts = collections.Counter(r["question_id"] for r in records)
    duplicate_ids = {qid: n for qid, n in id_counts.items() if n > 1}
    if duplicate_ids:
        raise ValueError(
            f"duplicate question_id(s) detected: {duplicate_ids}"
        )

    # 2. Two records per image: every distinct image has exactly two records.
    per_image = collections.Counter(r["image"] for r in records)
    bad_images = {img: n for img, n in per_image.items() if n != 2}
    if bad_images:
        raise ValueError(
            f"images without exactly 2 questions: {bad_images}"
        )

    # 3. Label validity: every label is exactly "yes" or "no".
    bad_labels = sorted(
        {r["label"] for r in records if r["label"] not in VALID_LABELS}
    )
    if bad_labels:
        raise ValueError(f"invalid label value(s): {bad_labels}")

    # 4. Category validity: every category is one of the fourteen SUBTASKS.
    bad_categories = sorted(
        {r["category"] for r in records if r["category"] not in SUBTASKS}
    )
    if bad_categories:
        raise ValueError(f"unknown subtask category(ies): {bad_categories}")


def derive_questions(reference: list) -> list:
    """Project the combined reference into the inference-input question records.

    Derives ``mme_questions.jsonl``'s records from ``mme_reference.jsonl``'s
    records by projecting each ``Reference_Record`` to exactly the three keys
    ``question_id``, ``image``, and ``text``, copying each value unchanged from
    the source record (``label`` and ``category`` are dropped). The projection
    is order-preserving: exactly one ``Question_Record`` is emitted per
    ``Reference_Record`` in the same order, so the record at each line position
    has the same ``question_id`` as the reference record at that position and
    the two files contain an equal number of records. This guarantees the
    positional alignment the evaluator's ``validate_alignment`` checks holds by
    construction.

    Args:
        reference: The combined list of ``Reference_Record`` dicts (each with
            keys ``{question_id, image, text, label, category}``).

    Returns:
        A list of ``Question_Record`` dicts, one per input record in the same
        order, each with exactly the keys ``{question_id, image, text}`` and
        values copied unchanged from the corresponding reference record.
    """
    return [
        {key: record[key] for key in ("question_id", "image", "text")}
        for record in reference
    ]


def detect_layout(subtask_dir: str) -> str:
    """Auto-detect a subtask folder's on-disk layout (FLAT vs SPLIT).

    Two raw layouts exist in the MME release tree and both must be handled:

    * **SPLIT** (e.g. ``artwork``, ``celebrity``): images live in a
      ``images/`` sub-folder and annotations live in a
      ``questions_answers_YN/`` sub-folder inside the subtask folder.
    * **FLAT** (e.g. ``OCR``, ``code_reasoning``): ``{stem}.jpg|.png`` image
      files and ``{stem}.txt`` annotation files are siblings directly inside
      the subtask folder.

    The classification is decided purely by the observable structure of the
    folder: a subtask is SPLIT only when *both* the ``images/`` and
    ``questions_answers_YN/`` sub-folders are present; otherwise it is FLAT.

    Args:
        subtask_dir: The path to the subtask folder inside the raw MME tree.

    Returns:
        ``"split"`` when both the ``images/`` (``SPLIT_IMAGE_DIR``) and
        ``questions_answers_YN/`` (``SPLIT_TEXT_DIR``) sub-folders exist;
        otherwise ``"flat"``.
    """
    if os.path.isdir(os.path.join(subtask_dir, SPLIT_IMAGE_DIR)) and os.path.isdir(
        os.path.join(subtask_dir, SPLIT_TEXT_DIR)
    ):
        return "split"
    return "flat"


def enumerate_pairs(subtask_dir: str, layout: str) -> list[tuple[str, str]]:
    """Enumerate ``(image_path, txt_path)`` pairs for one subtask, stem-sorted.

    Walks a subtask folder according to its detected ``layout`` and pairs each
    image file with the annotation file sharing its stem. For the FLAT layout
    image and annotation files are siblings directly inside ``subtask_dir``; for
    the SPLIT layout images are read from the ``images/`` (``SPLIT_IMAGE_DIR``)
    sub-folder and annotations from the ``questions_answers_YN/``
    (``SPLIT_TEXT_DIR``) sub-folder.

    A file is treated as an image when its extension is one of
    ``IMAGE_EXTENSIONS`` (``.jpg``/``.jpeg``/``.png``, matched
    case-insensitively) and as an annotation when its extension is ``.txt``
    (matched case-insensitively). Images and annotations are paired by an exact,
    case-sensitive match of their stems (the filename with its extension
    removed). The returned pairs are ordered by stem in ascending lexicographic
    (Unicode code point) order so the produced output is deterministic.

    Args:
        subtask_dir: Path to the subtask folder inside the raw MME tree.
        layout: The detected layout, ``"flat"`` or ``"split"`` (as returned by
            :func:`detect_layout`).

    Returns:
        A list of ``(image_path, txt_path)`` tuples, sorted by stem in ascending
        Unicode order. Each path is an absolute/joined path under the
        appropriate image or annotation directory.

    Raises:
        ValueError: If a single stem maps to more than one image file (for
            example both ``.jpg`` and ``.png``), naming the ambiguous stem(s);
            if any image file has no annotation file sharing its stem, or any
            annotation file has no image file sharing its stem, listing the
            unpaired stems; or if the subtask folder contains zero
            image/annotation pairs, naming the empty folder.
    """
    if layout == "split":
        img_dir = os.path.join(subtask_dir, SPLIT_IMAGE_DIR)
        txt_dir = os.path.join(subtask_dir, SPLIT_TEXT_DIR)
    else:
        img_dir = txt_dir = subtask_dir

    # Collect images: stem -> list of filenames (so duplicate-extension stems,
    # e.g. both "0001.jpg" and "0001.png", can be detected rather than silently
    # overwritten).
    images_by_stem: dict[str, list[str]] = collections.defaultdict(list)
    for name in os.listdir(img_dir):
        if not os.path.isfile(os.path.join(img_dir, name)):
            continue
        stem, ext = os.path.splitext(name)
        if ext.lower() in IMAGE_EXTENSIONS:
            images_by_stem[stem].append(name)

    # Collect annotations: stem -> filename.
    texts: dict[str, str] = {}
    for name in os.listdir(txt_dir):
        if not os.path.isfile(os.path.join(txt_dir, name)):
            continue
        stem, ext = os.path.splitext(name)
        if ext.lower() == ".txt":
            texts[stem] = name

    # A stem that maps to more than one image file is ambiguous.
    ambiguous = {
        stem: sorted(names)
        for stem, names in images_by_stem.items()
        if len(names) > 1
    }
    if ambiguous:
        raise ValueError(
            f"ambiguous image stem(s) in {subtask_dir}: {ambiguous}"
        )

    images = {stem: names[0] for stem, names in images_by_stem.items()}

    # Every image must have a matching .txt and vice versa.
    missing_txt = set(images) - set(texts)
    missing_img = set(texts) - set(images)
    if missing_txt or missing_img:
        raise ValueError(
            f"unpaired files in {subtask_dir}: "
            f"no-txt={sorted(missing_txt)}, no-image={sorted(missing_img)}"
        )

    pairs = [
        (
            os.path.join(img_dir, images[stem]),
            os.path.join(txt_dir, texts[stem]),
        )
        for stem in sorted(images)
    ]

    # A subtask folder with zero pairs is an error.
    if not pairs:
        raise ValueError(f"no image/annotation pairs found in {subtask_dir}")

    return pairs


def convert_subtask(raw_root: str, subtask: str) -> list[dict]:
    """Convert one subtask folder into its list of reference records.

    Joins ``raw_root`` with ``subtask`` to locate the subtask folder inside the
    raw MME tree, auto-detects its on-disk layout via :func:`detect_layout`,
    enumerates its stem-sorted ``(image_path, txt_path)`` pairs via
    :func:`enumerate_pairs`, reads each annotation file as UTF-8, and
    accumulates the two reference records each image produces (via
    :func:`build_records_for_image`) in ascending stem order. Because
    :func:`enumerate_pairs` returns pairs sorted by stem and the records for a
    FLAT layout and a SPLIT layout with the same logical content are built from
    identical inputs, the produced records are deterministic and layout-invariant.

    Args:
        raw_root: Path to the raw MME release tree root containing one folder
            per subtask
            (``data/mme/MME_Benchmark_release_version/MME_Benchmark``).
        subtask: The subtask folder name to convert (a member of ``SUBTASKS``).

    Returns:
        A list of ``Reference_Record`` dicts for the subtask, in ascending stem
        order, two per image (``question_id`` ending in ``_0``/``_1``), each
        with keys ``{question_id, image, text, label, category}`` and
        ``category == subtask``.

    Raises:
        ValueError: Propagated from :func:`enumerate_pairs` (unpaired or
            ambiguous files, or an empty subtask folder) or from
            :func:`build_records_for_image` / :func:`parse_annotation` (a
            malformed annotation file).
        OSError: If the subtask folder or an annotation file cannot be read.
    """
    subtask_dir = os.path.join(raw_root, subtask)
    layout = detect_layout(subtask_dir)
    records: list[dict] = []
    for image_path, txt_path in enumerate_pairs(subtask_dir, layout):
        image_filename = os.path.basename(image_path)
        with open(txt_path, encoding="utf-8") as f:
            raw_text = f.read()
        records.extend(
            build_records_for_image(subtask, image_filename, raw_text)
        )
    return records


def schedule_subtasks(raw_root: str, subset: list[str] | None = None) -> list[str]:
    """Resolve the ordered list of subtask names ``main`` should process.

    Determines exactly which subtasks the converter will walk, enforcing the
    fixed subtask vocabulary and presence on disk *before* any conversion or
    output is produced. This is the single gate that lets ``main`` ignore stray
    folders in the raw tree while still reporting an operator's typo or a
    genuinely missing subtask folder.

    The resolution rules are:

    * **No subset (``subset is None``):** schedule exactly the fourteen
      ``SUBTASKS`` names in their canonical ``SUBTASKS`` order.
    * **Explicit subset:** schedule only the names in ``subset``, but reordered
      to follow the canonical ``SUBTASKS`` order (so the combined output is
      always ordered by ``SUBTASKS`` regardless of the order the operator typed
      the names). Duplicate names in ``subset`` are collapsed to one entry.
    * **Unrecognized name:** if ``subset`` contains any name that is not one of
      the fourteen ``SUBTASKS`` names, raise ``ValueError`` and schedule
      nothing.
    * **Stray raw-tree folders:** folders in ``raw_root`` whose names are not
      scheduled are simply never returned, so they are excluded from processing
      without error.
    * **Missing scheduled folder:** if any scheduled subtask has no
      exactly-named (case-sensitive) sub-folder in ``raw_root``, raise
      ``ValueError`` naming the missing folder(s) and schedule nothing (so the
      converter produces no output for any subtask).

    Args:
        raw_root: Path to the raw MME release tree root containing one folder
            per subtask
            (``data/mme/MME_Benchmark_release_version/MME_Benchmark``).
        subset: An optional explicit list of subtask names to restrict
            processing to. ``None`` (the default) schedules all fourteen
            ``SUBTASKS``.

    Returns:
        The ordered list of subtask names to process, in canonical ``SUBTASKS``
        order, every one of which is guaranteed to have an exactly-named
        sub-folder present under ``raw_root``.

    Raises:
        ValueError: If ``subset`` contains a name that is not one of the
            fourteen ``SUBTASKS`` names (the message names the unrecognized
            name(s)); or if a scheduled subtask has no exactly-named sub-folder
            in ``raw_root`` (the message names the missing folder(s)). In either
            case no subtask is scheduled, so the caller produces no output.
    """
    if subset is None:
        scheduled = list(SUBTASKS)
    else:
        unrecognized = sorted(set(subset) - set(SUBTASKS))
        if unrecognized:
            raise ValueError(
                f"unrecognized subtask name(s): {unrecognized}; "
                f"valid subtasks are {SUBTASKS}"
            )
        requested = set(subset)
        scheduled = [name for name in SUBTASKS if name in requested]

    missing = [
        name
        for name in scheduled
        if not os.path.isdir(os.path.join(raw_root, name))
    ]
    if missing:
        raise ValueError(
            f"missing subtask folder(s) under {raw_root}: {missing}"
        )

    return scheduled


def _assert_symlinks_supported(probe_dir: str, subtask: str) -> None:
    """Verify the filesystem under ``probe_dir`` supports symlinks.

    Attempts to create (and immediately remove) a throwaway probe symlink inside
    ``probe_dir``. This is used by :func:`reconcile_images` in symlink mode to
    fail *before* any image entry is materialized when the destination
    filesystem cannot create symlinks (for example certain Windows setups or
    restrictive/archived filesystems).

    Args:
        probe_dir: An existing directory to probe (the subtask's output image
            directory).
        subtask: The subtask name, used only to make the error message specific.

    Raises:
        ValueError: If a symlink cannot be created under ``probe_dir``. The
            message identifies the unsupported-symlink condition and directs the
            operator to re-run with the ``--copy`` option.
    """
    probe_link = os.path.join(probe_dir, ".mme_prepare_symlink_probe")
    # Clean up any stale probe left by a previous interrupted run.
    if os.path.lexists(probe_link):
        try:
            os.remove(probe_link)
        except OSError:
            pass
    try:
        os.symlink("mme_prepare_symlink_probe_target", probe_link)
    except OSError as exc:
        raise ValueError(
            "filesystem does not support symlinks, which are required to "
            f"reconcile images for subtask {subtask!r}; re-run with the "
            "--copy option to materialize byte-for-byte image copies instead"
        ) from exc
    # Probe succeeded; remove it so no stray entry is left behind.
    os.remove(probe_link)


def reconcile_images(
    raw_root: str,
    subtask: str,
    layout: str,
    out_images_root: str,
    copy: bool = False,
) -> None:
    """Materialize a subtask's images into the uniform ``images/`` view.

    Reconciles the two raw on-disk layouts (FLAT and SPLIT) into a single
    uniform image view by materializing every source image for ``subtask`` at
    ``{out_images_root}/{subtask}/{filename}`` so the records' ``image`` field
    (``{subtask}/{filename}``) resolves under ``--image-folder
    {out_images_root}``. The source images are exactly the image side of the
    stem-sorted ``(image_path, txt_path)`` pairs returned by
    :func:`enumerate_pairs` for ``subtask`` under the given ``layout``, so the
    real on-disk extension (``.jpg`` vs ``.png``) is preserved by construction.

    The subtask's output directory is created if absent. Each image is
    materialized either as a relative symlink (the default) whose target
    resolves to the corresponding source file in the raw tree, or, when
    ``copy=True``, as a byte-for-byte physical copy of the source file. Any
    pre-existing entry at a destination path (a regular file, a copy, or a
    symlink, including a broken one) is removed and re-created, so re-running
    against an image view previously materialized from the same raw tree
    overwrites each entry in place and yields exactly the same set of paths with
    identical symlink targets (symlink mode) or identical contents (copy mode),
    leaving no leftover or duplicate entries. This makes reconciliation
    idempotent.

    Args:
        raw_root: Path to the raw MME release tree root containing one folder
            per subtask
            (``data/mme/MME_Benchmark_release_version/MME_Benchmark``).
        subtask: The subtask folder name to reconcile (a member of
            ``SUBTASKS``).
        layout: The detected layout for ``subtask``, ``"flat"`` or ``"split"``
            (as returned by :func:`detect_layout`).
        out_images_root: The root of the uniform image view (typically
            ``data/mme/images``); the subtask's images are materialized under
            ``{out_images_root}/{subtask}/``.
        copy: When ``False`` (default), materialize each image as a relative
            symlink to its raw source; when ``True``, materialize each image as
            a byte-for-byte physical copy.

    Returns:
        ``None``.

    Raises:
        ValueError: If symlink mode is selected but the destination filesystem
            does not support symlinks (the message directs the operator to
            ``--copy`` and no entry is materialized); or if a source image is
            absent from the raw tree when it is about to be materialized (the
            message names the missing ``{subtask}/{filename}`` and no further
            entries are materialized). Also propagated from
            :func:`enumerate_pairs` (unpaired/ambiguous files or an empty
            subtask folder).
        OSError: If a directory or destination entry cannot be created, removed,
            or copied for reasons unrelated to symlink support.
    """
    subtask_dir = os.path.join(raw_root, subtask)
    pairs = enumerate_pairs(subtask_dir, layout)

    out_subtask_dir = os.path.join(out_images_root, subtask)
    os.makedirs(out_subtask_dir, exist_ok=True)

    # In symlink mode, confirm the filesystem supports symlinks BEFORE any image
    # entry is materialized, so an unsupported filesystem aborts cleanly with an
    # actionable error rather than after a partial materialization.
    if not copy:
        _assert_symlinks_supported(out_subtask_dir, subtask)

    for image_path, _txt_path in pairs:
        filename = os.path.basename(image_path)

        # A source image absent from the raw tree aborts reconciliation without
        # materializing any further entries (the already-materialized entries of
        # this run remain).
        if not os.path.isfile(image_path):
            raise ValueError(
                f"missing source image for reconciliation: {subtask}/{filename}"
            )

        dest = os.path.join(out_subtask_dir, filename)

        # Overwrite any existing entry (regular file, copy, or symlink including
        # a broken one) so a re-run is idempotent with no leftovers/duplicates.
        if os.path.lexists(dest):
            os.remove(dest)

        if copy:
            # Byte-for-byte physical copy of the source file's contents.
            shutil.copyfile(image_path, dest)
        else:
            # Relative symlink whose target resolves to the raw source. The
            # target is computed relative to the directory that holds the link
            # (``out_subtask_dir``) from absolute paths, so it is independent of
            # the current working directory and therefore deterministic.
            rel_target = os.path.relpath(
                os.path.abspath(image_path), os.path.abspath(out_subtask_dir)
            )
            os.symlink(rel_target, dest)


def write_jsonl(path: str, records: list[dict]) -> None:
    """Write ``records`` to ``path`` as JSONL, one JSON object per line.

    Each record is serialized with :func:`json.dumps` and written on its own
    line terminated by a single ``\\n`` newline, including the final line, so the
    file always ends with exactly one trailing newline. The parent directory of
    ``path`` is created if it does not already exist, so callers need not
    pre-create the output directory.

    Args:
        path: The destination file path. Its parent directory is created if
            absent.
        records: The list of JSON-serializable dicts to write, one per line, in
            the given order.

    Returns:
        ``None``.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record))
            f.write("\n")


def write_per_subtask(reference_dir: str, records_by_subtask: dict) -> None:
    """Write each subtask's reference records to ``{reference_dir}/{subtask}.jsonl``.

    Writes one per-subtask reference file per entry in ``records_by_subtask``,
    each as JSONL (via :func:`write_jsonl`, so each line including the last is
    terminated by a single ``\\n`` and the parent directory is created as
    needed). Each subtask's records are written in the order they appear in the
    mapping's value, which the caller supplies in ascending stem order.

    Args:
        reference_dir: The directory under which the per-subtask files
            (``{subtask}.jsonl``) are written (typically
            ``data/mme/reference``).
        records_by_subtask: A mapping from subtask name to its list of
            ``Reference_Record`` dicts.

    Returns:
        ``None``.
    """
    for subtask, records in records_by_subtask.items():
        write_jsonl(os.path.join(reference_dir, f"{subtask}.jsonl"), records)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the converter's command-line arguments.

    Defines the converter's flags, mirroring the design's Example Usage and the
    ``scripts/mme_prepare.sh`` wrapper (which passes ``--raw-root``,
    ``--out-root``, and ``--per-subtask``).

    Args:
        argv: Optional argument vector to parse (defaults to ``sys.argv[1:]``
            when ``None``); accepted to make the entry point testable.

    Returns:
        The parsed :class:`argparse.Namespace` with attributes ``raw_root``,
        ``out_root``, ``copy`` (bool), ``per_subtask`` (bool), and ``subtasks``
        (``list[str]`` or ``None``).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert the raw MME release tree into the evaluator-ready "
            "mme_reference.jsonl / mme_questions.jsonl files (plus optional "
            "per-subtask references) and a reconciled images/ view."
        )
    )
    parser.add_argument(
        "--raw-root",
        required=True,
        help=(
            "Path to the raw MME release tree root containing one folder per "
            "subtask "
            "(data/mme/MME_Benchmark_release_version/MME_Benchmark)."
        ),
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help=(
            "Output root under which mme_reference.jsonl, mme_questions.jsonl, "
            "reference/{subtask}.jsonl, and images/{subtask}/ are written "
            "(typically data/mme)."
        ),
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help=(
            "Materialize each reconciled image as a byte-for-byte physical copy "
            "instead of a relative symlink (use on symlink-averse filesystems)."
        ),
    )
    parser.add_argument(
        "--per-subtask",
        action="store_true",
        help=(
            "Also write per-subtask reference files to "
            "{out-root}/reference/{subtask}.jsonl."
        ),
    )
    parser.add_argument(
        "--subtasks",
        nargs="+",
        default=None,
        help=(
            "Optional explicit subset of subtask names to process (in any "
            "order). Defaults to all fourteen SUBTASKS."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the end-to-end conversion: schedule, convert, reconcile, validate, write.

    Orchestrates the converter:

    1. Resolve the ordered subtasks to process via :func:`schedule_subtasks`
       (canonical ``SUBTASKS`` order; raises before any output on an unknown or
       missing subtask folder).
    2. For each scheduled subtask, accumulate its reference records (via
       :func:`convert_subtask`) into the combined reference in
       ``SUBTASKS``-then-stem order, and reconcile its images into the uniform
       ``{out_root}/images/{subtask}/`` view (via :func:`detect_layout` +
       :func:`reconcile_images`).
    3. Run :func:`validate_reference` on the combined reference *before* any
       output file is created, opened, or written, so a validation failure
       leaves no output files written.
    4. Write ``{out_root}/mme_reference.jsonl``, the optional per-subtask
       ``{out_root}/reference/{subtask}.jsonl`` files (when ``--per-subtask`` is
       given), and the derived ``{out_root}/mme_questions.jsonl`` (via
       :func:`derive_questions`).

    Args:
        argv: Optional argument vector forwarded to :func:`parse_args` (defaults
            to ``sys.argv[1:]`` when ``None``).

    Returns:
        ``None``.

    Raises:
        ValueError: Propagated from :func:`schedule_subtasks`,
            :func:`convert_subtask`, :func:`reconcile_images`, or
            :func:`validate_reference` (no JSONL output file is written when
            :func:`validate_reference` raises).
        OSError: If a subtask folder, annotation file, or output path cannot be
            read or written.
    """
    args = parse_args(argv)

    scheduled = schedule_subtasks(args.raw_root, args.subtasks)
    out_images_root = os.path.join(args.out_root, "images")

    reference: list[dict] = []
    records_by_subtask: dict[str, list[dict]] = {}
    for subtask in scheduled:
        records = convert_subtask(args.raw_root, subtask)
        records_by_subtask[subtask] = records
        reference.extend(records)

        layout = detect_layout(os.path.join(args.raw_root, subtask))
        reconcile_images(
            args.raw_root, subtask, layout, out_images_root, copy=args.copy
        )

    # Self-validate the combined reference BEFORE creating/opening/writing any
    # output file, so a failure never leaves a half-written reference.
    validate_reference(reference)

    write_jsonl(os.path.join(args.out_root, "mme_reference.jsonl"), reference)
    if args.per_subtask:
        write_per_subtask(
            os.path.join(args.out_root, "reference"), records_by_subtask
        )
    write_jsonl(
        os.path.join(args.out_root, "mme_questions.jsonl"),
        derive_questions(reference),
    )


if __name__ == "__main__":
    main()
