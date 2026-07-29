"""Property-based tests for the MME data-preparation converter (eval/mme_prepare.py).

Mirrors the structure and conventions of ``eval/test_mme_eval.py``: pytest +
Hypothesis, no GPU or model required, importing the module under test by bare
name (pytest's default prepend import mode puts ``eval/`` on ``sys.path``).

This file currently provides only the shared Hypothesis strategy section that
synthesizes raw MME trees (a random subtask, image stems, single-line question
strings, mixed-case ``Yes``/``No`` labels, a FLAT or SPLIT layout, and a
``.jpg``/``.png`` extension per image). Later tasks add the individual unit and
property-based tests that consume these strategies.
"""

import os
import string

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import mme_prepare
from mme_constants import SUBTASKS, PERCEPTION, COGNITION


# ---------------------------------------------------------------------------
# Shared Hypothesis strategies: synthesize raw MME trees.
#
# A generated "raw dataset" is a plain dict describing one subtask folder before
# it is materialized on disk:
#
#     {
#         "subtask": <one of SUBTASKS>,
#         "layout":  "flat" | "split",
#         "images": [
#             {
#                 "stem": <filename stem, no extension>,
#                 "ext":  ".jpg" | ".png",
#                 "pairs": [(question0, label0), (question1, label1)],
#             },
#             ...
#         ],
#     }
#
# The label values are emitted in mixed casing (e.g. "Yes", "no", "YES") so the
# property tests exercise the converter's case-insensitive normalization. The
# question strings are single-line and TAB-free so they round-trip cleanly
# through the two-line ``question<TAB>answer`` annotation format.
# ---------------------------------------------------------------------------

# Characters that must not appear inside generated questions, so each annotation
# line stays a single ``question<TAB>answer`` line. This must cover every
# character ``str.splitlines()`` treats as a line boundary (the converter parses
# annotations with ``raw_text.splitlines()``); otherwise a generated "single-line"
# question could split into multiple lines and break the two-line format. The
# full set is \n \r \v \f \x1c \x1d \x1e \x85 \u2028 \u2029, plus \t (the
# question/answer field separator), which must also stay excluded.
_LINE_WHITESPACE = "\t\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"

# Characters allowed in synthetic filename stems: no path separators, no dot,
# no whitespace, so each stem maps cleanly to an on-disk ``{stem}.{ext}`` name.
_STEM_ALPHABET = string.ascii_letters + string.digits + "_-"

# Image extensions the converter recognizes (the SPLIT/FLAT fixtures use these).
_IMAGE_EXTS = st.sampled_from([".jpg", ".png"])

# The two on-disk layouts a subtask folder can take.
_LAYOUTS = st.sampled_from(["flat", "split"])

# Mixed-case spellings of the two valid answers, to exercise normalization.
_YESNO = st.sampled_from(
    ["Yes", "No", "yes", "no", "YES", "NO", "yEs", "nO"]
)


def _stem_strategy() -> st.SearchStrategy:
    """Generate a non-empty filename stem from the safe stem alphabet."""
    return st.text(alphabet=_STEM_ALPHABET, min_size=1, max_size=8)


def _question_strategy() -> st.SearchStrategy:
    """Generate a non-empty, single-line, TAB-free question string.

    The value is stripped and required to remain non-empty so it equals the
    question the converter parses back out of an annotation line.
    """
    return (
        st.text(
            alphabet=st.characters(
                blacklist_categories=("Cs",),
                blacklist_characters=_LINE_WHITESPACE,
            ),
            min_size=1,
            max_size=40,
        )
        .map(str.strip)
        .filter(lambda q: len(q) > 0)
    )


@st.composite
def _image_strategy(draw) -> dict:
    """Generate one image entry: stem, extension, and two (question, label) pairs."""
    return {
        "stem": draw(_stem_strategy()),
        "ext": draw(_IMAGE_EXTS),
        "pairs": [
            (draw(_question_strategy()), draw(_YESNO)),
            (draw(_question_strategy()), draw(_YESNO)),
        ],
    }


@st.composite
def raw_datasets(draw, min_images: int = 1, max_images: int = 5) -> dict:
    """Generate a synthetic raw-tree dataset for a single subtask.

    Stems are unique within the dataset so each maps to exactly one image and
    one annotation file. The returned dict is layout-agnostic; helpers
    materialize it as either a FLAT or a SPLIT folder on disk.
    """
    subtask = draw(st.sampled_from(SUBTASKS))
    layout = draw(_LAYOUTS)
    stems = draw(
        st.lists(
            _stem_strategy(),
            min_size=min_images,
            max_size=max_images,
            unique=True,
        )
    )
    images = []
    for stem in stems:
        image = draw(_image_strategy())
        image["stem"] = stem
        images.append(image)
    return {"subtask": subtask, "layout": layout, "images": images}


def _annotation_body(pairs) -> str:
    """Render two ``(question, label)`` pairs into a raw annotation file body.

    Produces exactly two ``question<TAB>answer`` lines (newline-separated), the
    on-disk form a synthetic ``{stem}.txt`` annotation file would contain.
    """
    return "\n".join(f"{question}\t{label}" for question, label in pairs)


# ---------------------------------------------------------------------------
# Property 3: Label validity
#
# Every label produced by parsing a valid two-line annotation body is exactly
# one of the strings "yes" or "no", regardless of the source letter casing.
#
# Validates: Requirements 1.3, 1.4, 8.3
# ---------------------------------------------------------------------------


@given(
    pairs=st.lists(
        st.tuples(_question_strategy(), _YESNO),
        min_size=2,
        max_size=2,
    )
)
@settings(max_examples=200)
def test_parsed_labels_are_valid(pairs):
    """Property 3: every parsed label is in {"yes", "no"} (mixed casing).

    Validates: Requirements 1.3, 1.4, 8.3
    """
    raw_text = _annotation_body(pairs)
    parsed = mme_prepare.parse_annotation(raw_text)

    assert len(parsed) == 2
    for (_, source_label), (question, label) in zip(pairs, parsed):
        # Requirement 1.3/1.4: label is the lower-cased text right of the TAB.
        assert label == source_label.strip().lower()
        # Requirement 8.3: label is exactly one of the two valid strings.
        assert label in mme_prepare.VALID_LABELS
        assert label in {"yes", "no"}


# ---------------------------------------------------------------------------
# Example / edge-case tests: parse_annotation error branches and known bodies.
#
# These are concrete examples (not Hypothesis property tests) that pin down the
# specific behavior of parse_annotation against the converter's error contract
# and against real raw annotation bodies from the committed MME tree.
#
# Validates: Requirements 1.1, 1.6, 1.7, 1.8, 1.9
# ---------------------------------------------------------------------------


# A known FLAT-style body, byte-for-byte from
# data/mme/.../MME_Benchmark/OCR/0001.txt (note the trailing blank line, which
# must be ignored as a whitespace-only line).
_FLAT_BODY = (
    'Is the word in the logo "angie\'s"? Please answer yes or no.\tYes\n'
    'Is the word in the logo "angle\'s"? Please answer yes or no.\tNo\n'
)
_FLAT_EXPECTED = [
    ('Is the word in the logo "angie\'s"? Please answer yes or no.', "yes"),
    ('Is the word in the logo "angle\'s"? Please answer yes or no.', "no"),
]

# A known SPLIT-style body, byte-for-byte from
# data/mme/.../MME_Benchmark/artwork/questions_answers_YN/10002.txt.
_SPLIT_BODY = (
    "Does this artwork exist in the form of painting? Please answer yes or no.\tYes\n"
    "Does this artwork exist in the form of glassware? Please answer yes or no.\tNo\n"
)
_SPLIT_EXPECTED = [
    ("Does this artwork exist in the form of painting? Please answer yes or no.", "yes"),
    ("Does this artwork exist in the form of glassware? Please answer yes or no.", "no"),
]


def test_parse_annotation_known_flat_body():
    """A known FLAT-style body yields the expected (question, label) pairs.

    Validates: Requirement 1.1 (trailing blank line ignored), Requirement 1.2.
    """
    assert mme_prepare.parse_annotation(_FLAT_BODY) == _FLAT_EXPECTED


def test_parse_annotation_known_split_body():
    """A known SPLIT-style body yields the expected (question, label) pairs.

    Validates: Requirement 1.1 (trailing blank line ignored), Requirement 1.2.
    """
    assert mme_prepare.parse_annotation(_SPLIT_BODY) == _SPLIT_EXPECTED


def test_parse_annotation_ignores_whitespace_only_lines():
    """Empty and whitespace-only lines are ignored when counting non-empty lines.

    A body padded with blank, space-only, and tab/space-only lines around the
    two real annotation lines still parses to exactly the two expected pairs.

    Validates: Requirement 1.1
    """
    padded = (
        "\n"
        "   \n"
        "Is the cat sleeping? Please answer yes or no.\tYes\n"
        "\t  \n"
        "Is the dog running? Please answer yes or no.\tNo\n"
        "    \n"
    )
    assert mme_prepare.parse_annotation(padded) == [
        ("Is the cat sleeping? Please answer yes or no.", "yes"),
        ("Is the dog running? Please answer yes or no.", "no"),
    ]


@pytest.mark.parametrize(
    "body",
    [
        pytest.param("", id="zero-nonempty-lines"),
        pytest.param("Only one question?\tYes\n", id="one-nonempty-line"),
        pytest.param(
            "Q1?\tYes\nQ2?\tNo\nQ3?\tYes\n", id="three-nonempty-lines"
        ),
        pytest.param("   \n\t\n  \n", id="all-whitespace-lines"),
    ],
)
def test_parse_annotation_rejects_wrong_line_count(body):
    """A body without exactly two non-empty lines raises ValueError.

    Validates: Requirement 1.6
    """
    with pytest.raises(ValueError):
        mme_prepare.parse_annotation(body)


def test_parse_annotation_rejects_missing_tab():
    """A non-empty line lacking a TAB separator raises ValueError.

    Validates: Requirement 1.7
    """
    body = (
        "Is the sky blue? Please answer yes or no. Yes\n"  # space, not TAB
        "Is the grass green? Please answer yes or no.\tNo\n"
    )
    with pytest.raises(ValueError):
        mme_prepare.parse_annotation(body)


def test_parse_annotation_rejects_empty_question():
    """A line whose question is empty after stripping raises ValueError.

    Validates: Requirement 1.8
    """
    body = (
        "\tYes\n"  # nothing (or only whitespace) left of the TAB
        "Is the grass green? Please answer yes or no.\tNo\n"
    )
    with pytest.raises(ValueError):
        mme_prepare.parse_annotation(body)


@pytest.mark.parametrize(
    "bad_label",
    ["Maybe", "y", "true", "Yesno", "1", ""],
)
def test_parse_annotation_rejects_non_yesno_label(bad_label):
    """A label that is not Yes/No (case-insensitive) raises ValueError.

    Validates: Requirement 1.9
    """
    body = (
        f"Is the answer valid? Please answer yes or no.\t{bad_label}\n"
        "Is the grass green? Please answer yes or no.\tNo\n"
    )
    with pytest.raises(ValueError):
        mme_prepare.parse_annotation(body)


# ---------------------------------------------------------------------------
# Property 2: Schema completeness
#
# Every reference record built by build_records_for_image has exactly the keys
# {question_id, image, text, label, category} and no additional or missing keys.
#
# Validates: Requirements 2.2, 7.4
# ---------------------------------------------------------------------------

# The exact, complete key set every Reference_Record must carry (Requirement 2.2).
_REFERENCE_KEYS = {"question_id", "image", "text", "label", "category"}


@given(
    subtask=st.sampled_from(SUBTASKS),
    image=_image_strategy(),
)
@settings(max_examples=200)
def test_reference_records_have_complete_schema(subtask, image):
    """Property 2: every reference record has exactly the five required keys.

    Builds the two reference records for a generated image via
    build_records_for_image and asserts each record's key set is exactly
    {question_id, image, text, label, category} with no extra or missing keys.

    Validates: Requirements 2.2, 7.4
    """
    image_filename = f"{image['stem']}{image['ext']}"
    raw_text = _annotation_body(image["pairs"])

    records = mme_prepare.build_records_for_image(
        subtask, image_filename, raw_text
    )

    # build_records_for_image always emits exactly two records per image.
    assert len(records) == 2
    for record in records:
        # Requirement 2.2: exactly the five keys, no additional keys.
        assert set(record.keys()) == _REFERENCE_KEYS
        # Each key maps to a present (non-None) value.
        for key in _REFERENCE_KEYS:
            assert record[key] is not None


# ---------------------------------------------------------------------------
# Property 5: Image-field shape
#
# Every reference record built by build_records_for_image has an ``image`` field
# equal to f"{category}/{filename}", where ``filename`` is the image's on-disk
# filename including its real extension (.jpg/.png). In particular the record's
# ``image`` is consistent with its ``category`` (the subtask folder) and the
# extension of ``image`` is the actual on-disk image extension.
#
# Validates: Requirements 1.2, 2.4, 4.3, 6.1
# ---------------------------------------------------------------------------


@given(
    subtask=st.sampled_from(SUBTASKS),
    image=_image_strategy(),
)
@settings(max_examples=200)
def test_image_field_shape(subtask, image):
    """Property 5: every record's image == f"{category}/{filename}" with real ext.

    Builds the two reference records for a generated image via
    build_records_for_image and asserts each record's ``image`` field is exactly
    f"{subtask}/{stem}{ext}", that it agrees with the record's ``category``, and
    that its extension is the real on-disk extension drawn for the image.

    Validates: Requirements 1.2, 2.4, 4.3, 6.1
    """
    stem = image["stem"]
    ext = image["ext"]
    image_filename = f"{stem}{ext}"
    raw_text = _annotation_body(image["pairs"])

    records = mme_prepare.build_records_for_image(
        subtask, image_filename, raw_text
    )

    assert len(records) == 2
    expected_image = f"{subtask}/{image_filename}"
    for record in records:
        # Requirement 2.4: image is exactly "{subtask}/{filename}".
        assert record["image"] == expected_image
        # Requirement 4.3 / 6.1: the extension is the real on-disk extension.
        assert os.path.splitext(record["image"])[1] == ext
        assert record["image"].endswith(ext)
        # Requirement 1.2 / 2.4: image is consistent with the record's category.
        assert record["image"] == f"{record['category']}/{image_filename}"
        # The component before the slash is the subtask/category folder.
        assert record["image"].split("/", 1)[0] == subtask


# ---------------------------------------------------------------------------
# Property 1: Two lines per image
#
# build_records_for_image always returns exactly two reference records for one
# image, and both records share the same ``image`` field (the single image they
# were built from). This is the per-image foundation of the evaluator's
# "exactly two questions per image" invariant.
#
# Validates: Requirements 1.1, 2.1, 8.1
# ---------------------------------------------------------------------------


@given(
    subtask=st.sampled_from(SUBTASKS),
    image=_image_strategy(),
)
@settings(max_examples=200)
def test_two_records_per_image(subtask, image):
    """Property 1: build_records_for_image returns exactly two records sharing one image.

    Builds the reference records for a generated image via
    build_records_for_image and asserts there are exactly two of them and that
    they both carry the same ``image`` field equal to f"{subtask}/{filename}".

    Validates: Requirements 1.1, 2.1, 8.1
    """
    image_filename = f"{image['stem']}{image['ext']}"
    raw_text = _annotation_body(image["pairs"])

    records = mme_prepare.build_records_for_image(
        subtask, image_filename, raw_text
    )

    # Requirement 2.1: exactly two reference records are produced for one image.
    assert len(records) == 2

    # Requirement 1.1 / 8.1: both records belong to the single source image, so
    # they share one identical ``image`` value (== f"{subtask}/{filename}").
    images = {record["image"] for record in records}
    assert images == {f"{subtask}/{image_filename}"}


# ---------------------------------------------------------------------------
# Property 6: question_id uniqueness
#
# For any valid generated reference -- the combined list of reference records
# built for every image in a generated raw_datasets dataset -- every
# ``question_id`` is unique across the whole reference, i.e.
# ``len({r.question_id for r in R}) == len(R)``. Because a generated dataset has
# unique stems within its subtask and build_records_for_image emits ``_0``/``_1``
# suffixed ids, no two records collide.
#
# Validates: Requirements 2.5, 8.2
# ---------------------------------------------------------------------------


@given(dataset=raw_datasets())
@settings(max_examples=200)
def test_question_ids_are_unique(dataset):
    """Property 6: all question_ids in a generated reference are unique.

    Builds the full reference R for every image in a generated raw_datasets
    dataset (via build_records_for_image) and asserts that the number of
    distinct question_id values equals the total number of records, i.e. every
    question_id is unique.

    Validates: Requirements 2.5, 8.2
    """
    subtask = dataset["subtask"]

    reference = []
    for image in dataset["images"]:
        image_filename = f"{image['stem']}{image['ext']}"
        raw_text = _annotation_body(image["pairs"])
        reference.extend(
            mme_prepare.build_records_for_image(
                subtask, image_filename, raw_text
            )
        )

    question_ids = [record["question_id"] for record in reference]

    # Property 6: len({r.question_id}) == len(R) -- all question_ids are unique.
    assert len(set(question_ids)) == len(reference)
    # Each image contributes exactly two records, so R has 2 * #images records.
    assert len(reference) == 2 * len(dataset["images"])


# ---------------------------------------------------------------------------
# Property 7: Positional alignment (round-trip)
#
# For any valid generated reference R -- the combined list of reference records
# built for every image in a generated raw_datasets dataset -- the derived
# questions Q = derive_questions(R) is positionally aligned with R:
# ``len(Q) == len(R)`` and ``Q[i].question_id == R[i].question_id`` for all i.
# Equivalently, the evaluator's ``validate_alignment(R, Q)`` never raises, which
# is the alignment invariant eval/mme_eval.py relies on by construction.
#
# Validates: Requirements 7.5, 10.1
# ---------------------------------------------------------------------------

import mme_eval


@given(dataset=raw_datasets())
@settings(max_examples=200)
def test_positional_alignment_round_trip(dataset):
    """Property 7: derive_questions(R) is positionally aligned with R.

    Builds the full reference R for every image in a generated raw_datasets
    dataset, derives Q = derive_questions(R), and asserts len(Q) == len(R) and
    Q[i].question_id == R[i].question_id for all i. Also asserts the evaluator's
    validate_alignment(R, Q) never raises (it takes reference first, result
    second).

    Validates: Requirements 7.5, 10.1
    """
    subtask = dataset["subtask"]

    reference = []
    for image in dataset["images"]:
        image_filename = f"{image['stem']}{image['ext']}"
        raw_text = _annotation_body(image["pairs"])
        reference.extend(
            mme_prepare.build_records_for_image(
                subtask, image_filename, raw_text
            )
        )

    questions = mme_prepare.derive_questions(reference)

    # Property 7: equal length and matching question_id at every position.
    assert len(questions) == len(reference)
    for ref_record, question_record in zip(reference, questions):
        assert question_record["question_id"] == ref_record["question_id"]

    # The evaluator's alignment invariant holds by construction (no raise).
    mme_eval.validate_alignment(reference, questions)


# ---------------------------------------------------------------------------
# Property 10: Downstream acceptance
#
# For any valid generated reference R -- the combined list of reference records
# built for every image in a generated raw_datasets dataset -- the fixed
# evaluator's invariants hold by construction:
#
#   * mme_eval.group_by_image, applied to R's (ref, res) pairs, never raises and
#     groups every distinct image to exactly two records, and
#   * mme_eval.validate_alignment(R, Q), with Q = derive_questions(R), never
#     raises.
#
# group_by_image takes a list of (ref, res) pairs and reads only ref["image"],
# so we pair each reference record with its derived question record (Q is the
# positional projection of R). validate_alignment takes (reference, result) and
# checks equal length plus matching question_id at each position.
#
# Validates: Requirements 7.1, 10.2, 10.3
# ---------------------------------------------------------------------------


@given(dataset=raw_datasets())
@settings(max_examples=200)
def test_downstream_acceptance(dataset):
    """Property 10: group_by_image(R) and validate_alignment(R, Q) never raise.

    Builds the full reference R for every image in a generated raw_datasets
    dataset, derives Q = derive_questions(R), and asserts the fixed evaluator's
    invariants hold by construction: mme_eval.group_by_image (over R's
    (ref, res) pairs) groups every image to exactly two records without raising,
    and mme_eval.validate_alignment(R, Q) does not raise.

    Validates: Requirements 7.1, 10.2, 10.3
    """
    subtask = dataset["subtask"]

    reference = []
    for image in dataset["images"]:
        image_filename = f"{image['stem']}{image['ext']}"
        raw_text = _annotation_body(image["pairs"])
        reference.extend(
            mme_prepare.build_records_for_image(
                subtask, image_filename, raw_text
            )
        )

    questions = mme_prepare.derive_questions(reference)

    # Requirement 10.3: group_by_image takes (ref, res) pairs and reads only
    # ref["image"]; pairing R with its derived Q gives well-formed pairs. It
    # must not raise and must group every distinct image to exactly two records.
    pairs = list(zip(reference, questions))
    grouped = mme_eval.group_by_image(pairs)
    assert all(len(image_pairs) == 2 for image_pairs in grouped.values())

    # Requirement 7.1 / 10.2: the evaluator's alignment invariant holds by
    # construction (validate_alignment takes reference first, result second).
    mme_eval.validate_alignment(reference, questions)


# ---------------------------------------------------------------------------
# Example / edge-case tests: detect_layout and enumerate_pairs (I/O helpers).
#
# These are concrete examples (not Hypothesis property tests) that pin down the
# layout auto-detection against the committed raw MME tree and the deterministic
# pairing / error contract of enumerate_pairs against tmp_path fixtures.
#
# Validates: Requirements 3.1, 3.2, 4.4, 4.5, 4.6, 4.7, 4.8
# ---------------------------------------------------------------------------

# The committed raw MME release tree root (sibling of eval/ under the repo root).
_RAW_ROOT = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "mme",
        "MME_Benchmark_release_version",
        "MME_Benchmark",
    )
)


def _make_flat_subtask(tmp_path, stems_exts):
    """Materialize a FLAT-layout subtask folder under tmp_path.

    ``stems_exts`` is an iterable of ``(stem, ext)`` image entries; for each a
    sibling ``{stem}{ext}`` image file and ``{stem}.txt`` annotation file are
    written directly inside the subtask folder. Returns the subtask dir path.
    """
    subtask_dir = tmp_path / "flat_subtask"
    subtask_dir.mkdir()
    for stem, ext in stems_exts:
        (subtask_dir / f"{stem}{ext}").write_bytes(b"\xff\xd8\xff")  # dummy image
        (subtask_dir / f"{stem}.txt").write_text(
            "Is it real? Please answer yes or no.\tYes\n"
            "Is it fake? Please answer yes or no.\tNo\n",
            encoding="utf-8",
        )
    return subtask_dir


# --- detect_layout against the committed raw tree --------------------------


@pytest.mark.parametrize("subtask", ["OCR", "code_reasoning"])
def test_detect_layout_flat_committed_tree(subtask):
    """FLAT subtasks in the committed raw tree are detected as "flat".

    OCR and code_reasoning store {stem}.jpg|.png and {stem}.txt as siblings, so
    neither images/ nor questions_answers_YN/ sub-folders are present.

    Validates: Requirement 3.2
    """
    subtask_dir = os.path.join(_RAW_ROOT, subtask)
    if not os.path.isdir(subtask_dir):
        pytest.skip(f"committed raw tree missing subtask folder: {subtask}")
    assert mme_prepare.detect_layout(subtask_dir) == "flat"


@pytest.mark.parametrize("subtask", ["artwork", "celebrity"])
def test_detect_layout_split_committed_tree(subtask):
    """SPLIT subtasks in the committed raw tree are detected as "split".

    artwork and celebrity store images under images/ and annotations under
    questions_answers_YN/, so both required sub-folders are present.

    Validates: Requirement 3.1
    """
    subtask_dir = os.path.join(_RAW_ROOT, subtask)
    if not os.path.isdir(subtask_dir):
        pytest.skip(f"committed raw tree missing subtask folder: {subtask}")
    assert mme_prepare.detect_layout(subtask_dir) == "split"


# --- detect_layout against tmp_path fixtures -------------------------------


def test_detect_layout_flat_tmp_fixture(tmp_path):
    """A folder with sibling image/txt files (no sub-folders) is "flat".

    Validates: Requirement 3.2
    """
    subtask_dir = _make_flat_subtask(tmp_path, [("0001", ".jpg")])
    assert mme_prepare.detect_layout(str(subtask_dir)) == "flat"


def test_detect_layout_split_tmp_fixture(tmp_path):
    """A folder with both images/ and questions_answers_YN/ is "split".

    Validates: Requirement 3.1
    """
    subtask_dir = tmp_path / "split_subtask"
    (subtask_dir / mme_prepare.SPLIT_IMAGE_DIR).mkdir(parents=True)
    (subtask_dir / mme_prepare.SPLIT_TEXT_DIR).mkdir()
    assert mme_prepare.detect_layout(str(subtask_dir)) == "split"


def test_detect_layout_only_images_subfolder_is_flat(tmp_path):
    """A folder with only an images/ sub-folder (no questions_answers_YN/) is "flat".

    SPLIT requires BOTH sub-folders, so a partial structure falls back to FLAT.

    Validates: Requirement 3.2
    """
    subtask_dir = tmp_path / "partial_subtask"
    (subtask_dir / mme_prepare.SPLIT_IMAGE_DIR).mkdir(parents=True)
    assert mme_prepare.detect_layout(str(subtask_dir)) == "flat"


def test_detect_layout_only_text_subfolder_is_flat(tmp_path):
    """A folder with only a questions_answers_YN/ sub-folder is "flat".

    Validates: Requirement 3.2
    """
    subtask_dir = tmp_path / "partial_subtask2"
    (subtask_dir / mme_prepare.SPLIT_TEXT_DIR).mkdir(parents=True)
    assert mme_prepare.detect_layout(str(subtask_dir)) == "flat"


# --- enumerate_pairs happy path --------------------------------------------


def test_enumerate_pairs_flat_happy_path_stem_sorted(tmp_path):
    """FLAT happy path returns one (image, txt) pair per stem, sorted by stem.

    The fixture is created out of stem order to prove the result is sorted in
    ascending Unicode order, and each returned path points under the subtask dir.

    Validates: Requirement 4.4
    """
    subtask_dir = _make_flat_subtask(
        tmp_path, [("0003", ".jpg"), ("0001", ".png"), ("0002", ".jpg")]
    )
    pairs = mme_prepare.enumerate_pairs(str(subtask_dir), "flat")

    stems = [os.path.splitext(os.path.basename(img))[0] for img, _ in pairs]
    assert stems == ["0001", "0002", "0003"]
    # Each pair points at the matching image and txt for that stem.
    for img_path, txt_path in pairs:
        img_stem = os.path.splitext(os.path.basename(img_path))[0]
        txt_stem = os.path.splitext(os.path.basename(txt_path))[0]
        assert img_stem == txt_stem
        assert os.path.isfile(img_path)
        assert os.path.isfile(txt_path)


def test_enumerate_pairs_split_happy_path(tmp_path):
    """SPLIT happy path reads images from images/ and annotations from questions_answers_YN/.

    Validates: Requirements 4.4
    """
    subtask_dir = tmp_path / "split_subtask"
    img_dir = subtask_dir / mme_prepare.SPLIT_IMAGE_DIR
    txt_dir = subtask_dir / mme_prepare.SPLIT_TEXT_DIR
    img_dir.mkdir(parents=True)
    txt_dir.mkdir()
    for stem in ("0002", "0001"):
        (img_dir / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff")
        (txt_dir / f"{stem}.txt").write_text(
            "Is it real? Please answer yes or no.\tYes\n"
            "Is it fake? Please answer yes or no.\tNo\n",
            encoding="utf-8",
        )
    pairs = mme_prepare.enumerate_pairs(str(subtask_dir), "split")

    stems = [os.path.splitext(os.path.basename(img))[0] for img, _ in pairs]
    assert stems == ["0001", "0002"]
    for img_path, txt_path in pairs:
        # Image side lives under images/, annotation side under questions_answers_YN/.
        assert os.path.basename(os.path.dirname(img_path)) == mme_prepare.SPLIT_IMAGE_DIR
        assert os.path.basename(os.path.dirname(txt_path)) == mme_prepare.SPLIT_TEXT_DIR


# --- enumerate_pairs error branches ----------------------------------------


def test_enumerate_pairs_unpaired_image_raises(tmp_path):
    """An image with no matching .txt raises ValueError.

    Validates: Requirement 4.5
    """
    subtask_dir = _make_flat_subtask(tmp_path, [("0001", ".jpg")])
    # Add a second image with no annotation file sharing its stem.
    (subtask_dir / "0002.jpg").write_bytes(b"\xff\xd8\xff")
    with pytest.raises(ValueError):
        mme_prepare.enumerate_pairs(str(subtask_dir), "flat")


def test_enumerate_pairs_unpaired_txt_raises(tmp_path):
    """A .txt with no matching image raises ValueError.

    Validates: Requirement 4.6
    """
    subtask_dir = _make_flat_subtask(tmp_path, [("0001", ".jpg")])
    # Add a second annotation file with no image sharing its stem.
    (subtask_dir / "0002.txt").write_text(
        "Is it real? Please answer yes or no.\tYes\n"
        "Is it fake? Please answer yes or no.\tNo\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        mme_prepare.enumerate_pairs(str(subtask_dir), "flat")


def test_enumerate_pairs_duplicate_extension_stem_raises(tmp_path):
    """A single stem mapping to more than one image (.jpg + .png) raises ValueError.

    Validates: Requirement 4.7
    """
    subtask_dir = _make_flat_subtask(tmp_path, [("0001", ".jpg")])
    # Add a second image file for the same stem with a different extension.
    (subtask_dir / "0001.png").write_bytes(b"\x89PNG")
    with pytest.raises(ValueError):
        mme_prepare.enumerate_pairs(str(subtask_dir), "flat")


def test_enumerate_pairs_empty_folder_raises(tmp_path):
    """A subtask folder containing zero image/annotation pairs raises ValueError.

    Validates: Requirement 4.8
    """
    subtask_dir = tmp_path / "empty_subtask"
    subtask_dir.mkdir()
    with pytest.raises(ValueError):
        mme_prepare.enumerate_pairs(str(subtask_dir), "flat")


# ---------------------------------------------------------------------------
# Property 8: Layout invariance
#
# A FLAT raw tree and a SPLIT raw tree carrying the same logical content (the
# same set of stems, the same question strings, and the same Yes/No labels)
# produce identical reference records R -- same records, in the same order --
# when each is run through convert_subtask. The on-disk shape (siblings vs the
# images/ + questions_answers_YN/ split) is an implementation detail the
# converter abstracts away; only the logical content drives R.
#
# Both trees are materialized from ONE generated raw_datasets dataset so they
# are guaranteed to carry identical logical content; the dataset's own "layout"
# field is ignored here (we always build BOTH a flat and a split tree from it).
#
# Hypothesis runs the body many times, so we materialize the trees with
# tempfile.mkdtemp (cleaned up in a finally) rather than the pytest tmp_path
# fixture, which is created once per test function and is not refreshed across
# Hypothesis examples.
#
# Validates: Requirements 3.5, 4.1, 4.2, 7.2
# ---------------------------------------------------------------------------

import shutil
import tempfile


def _materialize_flat(dataset) -> str:
    """Materialize ``dataset`` as a FLAT raw tree; return the raw-root path.

    Creates a fresh temp directory holding a single ``{subtask}/`` folder with
    ``{stem}{ext}`` image files and ``{stem}.txt`` annotation files as siblings
    directly inside it. The caller is responsible for removing the returned
    root.
    """
    raw_root = tempfile.mkdtemp(prefix="mme_flat_")
    subtask_dir = os.path.join(raw_root, dataset["subtask"])
    os.makedirs(subtask_dir)
    for image in dataset["images"]:
        stem, ext = image["stem"], image["ext"]
        with open(os.path.join(subtask_dir, f"{stem}{ext}"), "wb") as f:
            f.write(b"\xff\xd8\xff")  # dummy image bytes
        body = _annotation_body(image["pairs"])
        with open(
            os.path.join(subtask_dir, f"{stem}.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(body)
    return raw_root


def _materialize_split(dataset) -> str:
    """Materialize ``dataset`` as a SPLIT raw tree; return the raw-root path.

    Creates a fresh temp directory holding a single ``{subtask}/`` folder with
    images under ``images/`` and annotations under ``questions_answers_YN/``.
    The logical content (stems, questions, labels, extensions) is identical to
    :func:`_materialize_flat` for the same dataset. The caller is responsible
    for removing the returned root.
    """
    raw_root = tempfile.mkdtemp(prefix="mme_split_")
    subtask_dir = os.path.join(raw_root, dataset["subtask"])
    img_dir = os.path.join(subtask_dir, mme_prepare.SPLIT_IMAGE_DIR)
    txt_dir = os.path.join(subtask_dir, mme_prepare.SPLIT_TEXT_DIR)
    os.makedirs(img_dir)
    os.makedirs(txt_dir)
    for image in dataset["images"]:
        stem, ext = image["stem"], image["ext"]
        with open(os.path.join(img_dir, f"{stem}{ext}"), "wb") as f:
            f.write(b"\xff\xd8\xff")  # dummy image bytes
        body = _annotation_body(image["pairs"])
        with open(
            os.path.join(txt_dir, f"{stem}.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(body)
    return raw_root


@given(dataset=raw_datasets())
@settings(max_examples=200, deadline=None)
def test_layout_invariance(dataset):
    """Property 8: a FLAT tree and a SPLIT tree with the same logical content
    produce identical reference records (same records, same order).

    Materializes ONE generated dataset both as a FLAT raw tree (image/txt
    siblings) and as a SPLIT raw tree (images/ + questions_answers_YN/), runs
    convert_subtask on each, and asserts the two reference-record lists are
    exactly equal. detect_layout classifies each tree correctly, so the only
    thing that can differ is the layout handling -- which Property 8 requires to
    be invisible in R.

    Validates: Requirements 3.5, 4.1, 4.2, 7.2
    """
    subtask = dataset["subtask"]
    flat_root = _materialize_flat(dataset)
    split_root = _materialize_split(dataset)
    try:
        # Sanity: the two trees are genuinely the two different layouts.
        assert (
            mme_prepare.detect_layout(os.path.join(flat_root, subtask)) == "flat"
        )
        assert (
            mme_prepare.detect_layout(os.path.join(split_root, subtask))
            == "split"
        )

        flat_records = mme_prepare.convert_subtask(flat_root, subtask)
        split_records = mme_prepare.convert_subtask(split_root, subtask)

        # Property 8: identical records in identical order, regardless of layout.
        assert flat_records == split_records
    finally:
        shutil.rmtree(flat_root, ignore_errors=True)
        shutil.rmtree(split_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Example / edge-case tests: schedule_subtasks (the main() scheduling gate).
#
# These are concrete examples (not Hypothesis property tests) that pin down the
# subtask-scheduling contract against tmp_path raw-root fixtures: it enforces
# the fixed SUBTASKS vocabulary and folder presence BEFORE any conversion, while
# reordering an explicit subset into canonical SUBTASKS order and silently
# ignoring stray (non-subtask) folders in the raw tree.
#
# Validates: Requirements 5.2, 5.3, 5.4, 5.5
# ---------------------------------------------------------------------------


def _make_raw_root(tmp_path, names, *, stray=()):
    """Materialize a raw-root directory holding one folder per name in ``names``.

    Each entry in ``names`` becomes an (empty) subtask sub-folder under a fresh
    raw-root directory; each entry in ``stray`` becomes a sub-folder whose name
    is not one of the scheduled subtasks. Returns the raw-root path as a string.
    schedule_subtasks only checks for folder presence (via os.path.isdir), so the
    folders need no contents for these scheduling tests.
    """
    raw_root = tmp_path / "raw_root"
    raw_root.mkdir()
    for name in names:
        (raw_root / name).mkdir()
    for name in stray:
        (raw_root / name).mkdir()
    return str(raw_root)


def test_schedule_subtasks_no_subset_returns_all_in_subtasks_order(tmp_path):
    """No subset schedules exactly the fourteen SUBTASKS in canonical order.

    Validates: Requirement 5.2 (canonical SUBTASKS order)
    """
    raw_root = _make_raw_root(tmp_path, SUBTASKS)
    assert mme_prepare.schedule_subtasks(raw_root) == list(SUBTASKS)


def test_schedule_subtasks_subset_reordered_into_subtasks_order(tmp_path):
    """An explicit subset is scheduled in canonical SUBTASKS order, not input order.

    The subset is supplied deliberately out of (and reversed from) SUBTASKS order
    to prove the result follows SUBTASKS regardless of how the operator typed it.

    Validates: Requirement 5.2
    """
    # A handful of names spanning both groups, supplied out of order.
    subset = ["OCR", "existence", "code_reasoning", "color"]
    raw_root = _make_raw_root(tmp_path, subset)

    scheduled = mme_prepare.schedule_subtasks(raw_root, subset=subset)

    # Result is exactly the subset, but ordered to follow SUBTASKS.
    expected = [name for name in SUBTASKS if name in set(subset)]
    assert scheduled == expected
    # Sanity: the expected order differs from the (reversed-ish) input order.
    assert scheduled != subset


def test_schedule_subtasks_duplicate_subset_names_collapsed(tmp_path):
    """Duplicate names in the subset collapse to a single scheduled entry.

    Validates: Requirement 5.2
    """
    subset = ["color", "OCR", "color", "OCR", "color"]
    raw_root = _make_raw_root(tmp_path, ["color", "OCR"])

    scheduled = mme_prepare.schedule_subtasks(raw_root, subset=subset)

    assert scheduled == [name for name in SUBTASKS if name in {"color", "OCR"}]
    # Each name appears exactly once (no duplicates carried through).
    assert len(scheduled) == len(set(scheduled))


def test_schedule_subtasks_unrecognized_name_raises(tmp_path):
    """An unrecognized name in the subset raises ValueError (schedules nothing).

    The bad name's folder is present on disk to prove the rejection is purely a
    vocabulary check, independent of folder presence.

    Validates: Requirement 5.3
    """
    raw_root = _make_raw_root(tmp_path, ["OCR", "not_a_real_subtask"])
    with pytest.raises(ValueError):
        mme_prepare.schedule_subtasks(
            raw_root, subset=["OCR", "not_a_real_subtask"]
        )


def test_schedule_subtasks_stray_folder_ignored(tmp_path):
    """A stray (non-subtask) folder in the raw tree is excluded, not an error.

    The raw tree holds every scheduled subtask folder plus two stray folders;
    scheduling succeeds and returns exactly SUBTASKS, with the stray folders
    silently excluded.

    Validates: Requirement 5.4
    """
    raw_root = _make_raw_root(
        tmp_path, SUBTASKS, stray=["README_files", "MME_Benchmark_extra"]
    )

    scheduled = mme_prepare.schedule_subtasks(raw_root)

    assert scheduled == list(SUBTASKS)
    assert "README_files" not in scheduled
    assert "MME_Benchmark_extra" not in scheduled


def test_schedule_subtasks_missing_folder_raises(tmp_path):
    """A scheduled subtask with no matching folder raises ValueError (no output).

    All SUBTASKS folders are present EXCEPT one ("OCR"), so the default schedule
    (all fourteen) must raise because that scheduled subtask folder is absent.

    Validates: Requirement 5.5
    """
    present = [name for name in SUBTASKS if name != "OCR"]
    raw_root = _make_raw_root(tmp_path, present)
    with pytest.raises(ValueError):
        mme_prepare.schedule_subtasks(raw_root)


def test_schedule_subtasks_subset_missing_folder_raises(tmp_path):
    """A subset naming a recognized subtask whose folder is absent raises.

    The name is a valid SUBTASKS member (so it passes the vocabulary check) but
    has no folder on disk, so the folder-presence check must raise.

    Validates: Requirement 5.5
    """
    raw_root = _make_raw_root(tmp_path, ["OCR"])  # "color" folder absent
    with pytest.raises(ValueError):
        mme_prepare.schedule_subtasks(raw_root, subset=["OCR", "color"])


def test_schedule_subtasks_case_sensitive_folder_match_raises(tmp_path):
    """Folder matching is case-sensitive: a differently-cased folder is missing.

    The raw tree holds "ocr" (lower-case) but the scheduled subtask is "OCR";
    since the match is case-sensitive, the scheduled "OCR" folder is considered
    absent and scheduling raises.

    Validates: Requirements 5.4, 5.5
    """
    raw_root = _make_raw_root(tmp_path, ["ocr"])  # wrong casing vs "OCR"
    with pytest.raises(ValueError):
        mme_prepare.schedule_subtasks(raw_root, subset=["OCR"])


# ---------------------------------------------------------------------------
# Example / edge-case tests: reconcile_images (the uniform image-view writer).
#
# These are concrete examples (not Hypothesis property tests) that pin down the
# image-reconciliation contract against tmp_path raw-root + output fixtures:
#   * symlink mode (default) materializes relative links resolving to the raw
#     source file,
#   * --copy (copy=True) materializes byte-for-byte physical copies,
#   * a re-run is idempotent (same set of paths, same symlink targets / same
#     contents, no leftover or duplicate entries), and
#   * a source image absent at materialization time aborts with a ValueError
#     naming the missing {subtask}/{filename}, without materializing further
#     entries.
#
# Validates: Requirements 6.2, 6.3, 6.4, 6.5, 6.6
# ---------------------------------------------------------------------------


def _make_reconcile_raw(tmp_path, subtask, stems_exts, *, contents=None):
    """Materialize a FLAT raw-root holding one subtask folder for reconciliation.

    Builds ``{raw_root}/{subtask}/`` with a ``{stem}{ext}`` image file and a
    matching ``{stem}.txt`` annotation file (so :func:`enumerate_pairs` pairs
    them) for each ``(stem, ext)`` in ``stems_exts``. ``contents`` optionally
    maps ``(stem, ext)`` (or ``stem``) to the exact image bytes to write, so a
    copy-mode test can assert byte-for-byte equality against known data; when a
    key is absent a per-stem default of ``b"img-" + stem`` is used so distinct
    images carry distinct bytes. Returns ``(raw_root, subtask_dir)`` as strings.
    """
    raw_root = tmp_path / "raw"
    subtask_dir = raw_root / subtask
    subtask_dir.mkdir(parents=True)
    contents = contents or {}
    for stem, ext in stems_exts:
        data = contents.get((stem, ext), contents.get(stem))
        if data is None:
            data = b"img-" + stem.encode("utf-8")
        (subtask_dir / f"{stem}{ext}").write_bytes(data)
        (subtask_dir / f"{stem}.txt").write_text(
            "Is it real? Please answer yes or no.\tYes\n"
            "Is it fake? Please answer yes or no.\tNo\n",
            encoding="utf-8",
        )
    return str(raw_root), str(subtask_dir)


# --- symlink mode (default) ------------------------------------------------


def test_reconcile_images_symlink_creates_relative_links_to_raw_source(tmp_path):
    """Default (symlink) mode materializes relative links resolving to the raw source.

    Each materialized entry under images/{subtask}/{filename} is a symlink whose
    stored target is relative (not absolute) and whose resolution points at the
    corresponding source file in the raw tree, with the same bytes.

    Validates: Requirements 6.2, 6.3
    """
    stems_exts = [("0001", ".jpg"), ("0002", ".png")]
    raw_root, subtask_dir = _make_reconcile_raw(tmp_path, "OCR", stems_exts)
    out_images_root = str(tmp_path / "images")

    mme_prepare.reconcile_images(raw_root, "OCR", "flat", out_images_root)

    out_subtask_dir = os.path.join(out_images_root, "OCR")
    for stem, ext in stems_exts:
        filename = f"{stem}{ext}"
        dest = os.path.join(out_subtask_dir, filename)
        source = os.path.join(subtask_dir, filename)

        # Requirement 6.2: materialized at images/{subtask}/{filename} with the
        # real on-disk extension preserved.
        assert os.path.basename(dest) == filename
        # Requirement 6.3: the entry is a symlink with a *relative* target.
        assert os.path.islink(dest)
        assert not os.path.isabs(os.readlink(dest))
        # Requirement 6.3: the link resolves to the raw source file...
        assert os.path.realpath(dest) == os.path.realpath(source)
        # ...and therefore reads back the source bytes.
        with open(dest, "rb") as f:
            assert f.read() == open(source, "rb").read()


# --- copy mode (--copy) ----------------------------------------------------


def test_reconcile_images_copy_creates_byte_identical_copies(tmp_path):
    """--copy (copy=True) materializes byte-for-byte physical copies, not links.

    Each materialized entry is a real (non-symlink) file whose bytes exactly
    equal the corresponding raw source image's bytes.

    Validates: Requirements 6.2, 6.4
    """
    contents = {"0001": b"\x89PNG\r\n\x1a\n-distinct-A", "0002": b"\xff\xd8\xff-distinct-B"}
    stems_exts = [("0001", ".png"), ("0002", ".jpg")]
    raw_root, subtask_dir = _make_reconcile_raw(
        tmp_path, "OCR", stems_exts, contents=contents
    )
    out_images_root = str(tmp_path / "images")

    mme_prepare.reconcile_images(raw_root, "OCR", "flat", out_images_root, copy=True)

    out_subtask_dir = os.path.join(out_images_root, "OCR")
    for stem, ext in stems_exts:
        filename = f"{stem}{ext}"
        dest = os.path.join(out_subtask_dir, filename)
        source = os.path.join(subtask_dir, filename)

        # Requirement 6.4: a physical copy, not a symlink.
        assert os.path.isfile(dest)
        assert not os.path.islink(dest)
        # Requirement 6.4: byte-for-byte identical to the raw source.
        with open(dest, "rb") as f_dest, open(source, "rb") as f_src:
            assert f_dest.read() == f_src.read()
        assert open(dest, "rb").read() == contents[stem]


# --- idempotency (re-run) --------------------------------------------------


def test_reconcile_images_symlink_rerun_is_idempotent(tmp_path):
    """Re-running symlink reconciliation yields the same paths and link targets.

    Two consecutive runs produce exactly the same set of entries, each still a
    symlink with an identical (relative) target, and no leftover or duplicate
    entries are left behind.

    Validates: Requirements 6.4, 6.5
    """
    stems_exts = [("0001", ".jpg"), ("0002", ".png")]
    raw_root, _ = _make_reconcile_raw(tmp_path, "OCR", stems_exts)
    out_images_root = str(tmp_path / "images")
    out_subtask_dir = os.path.join(out_images_root, "OCR")

    mme_prepare.reconcile_images(raw_root, "OCR", "flat", out_images_root)
    first_entries = sorted(os.listdir(out_subtask_dir))
    first_targets = {n: os.readlink(os.path.join(out_subtask_dir, n)) for n in first_entries}

    # Re-run against the previously materialized view.
    mme_prepare.reconcile_images(raw_root, "OCR", "flat", out_images_root)
    second_entries = sorted(os.listdir(out_subtask_dir))
    second_targets = {n: os.readlink(os.path.join(out_subtask_dir, n)) for n in second_entries}

    # Requirement 6.5: same set of paths (no leftovers/duplicates).
    assert second_entries == first_entries
    assert second_entries == sorted(f"{s}{e}" for s, e in stems_exts)
    # Requirement 6.4/6.5: identical (relative) symlink targets on both runs.
    assert second_targets == first_targets
    for name in second_entries:
        assert os.path.islink(os.path.join(out_subtask_dir, name))


def test_reconcile_images_copy_rerun_is_idempotent(tmp_path):
    """Re-running copy reconciliation yields the same paths and identical contents.

    Validates: Requirements 6.4, 6.5
    """
    contents = {"0001": b"copy-A", "0002": b"copy-B"}
    stems_exts = [("0001", ".jpg"), ("0002", ".png")]
    raw_root, _ = _make_reconcile_raw(
        tmp_path, "OCR", stems_exts, contents=contents
    )
    out_images_root = str(tmp_path / "images")
    out_subtask_dir = os.path.join(out_images_root, "OCR")

    mme_prepare.reconcile_images(raw_root, "OCR", "flat", out_images_root, copy=True)
    first_entries = sorted(os.listdir(out_subtask_dir))
    first_blobs = {n: open(os.path.join(out_subtask_dir, n), "rb").read() for n in first_entries}

    mme_prepare.reconcile_images(raw_root, "OCR", "flat", out_images_root, copy=True)
    second_entries = sorted(os.listdir(out_subtask_dir))
    second_blobs = {n: open(os.path.join(out_subtask_dir, n), "rb").read() for n in second_entries}

    # Requirement 6.5: same set of paths and identical contents, no duplicates.
    assert second_entries == first_entries
    assert second_blobs == first_blobs
    for name in second_entries:
        assert not os.path.islink(os.path.join(out_subtask_dir, name))


def test_reconcile_images_rerun_removes_stale_entry(tmp_path):
    """A re-run overwrites an existing entry in place (no stale/duplicate left).

    Pre-seeds the destination with a stale regular file at an entry's path; after
    reconciliation that path is a correct symlink to the raw source, proving the
    existing entry was overwritten rather than duplicated or left stale.

    Validates: Requirement 6.5
    """
    stems_exts = [("0001", ".jpg")]
    raw_root, subtask_dir = _make_reconcile_raw(tmp_path, "OCR", stems_exts)
    out_images_root = str(tmp_path / "images")
    out_subtask_dir = os.path.join(out_images_root, "OCR")
    os.makedirs(out_subtask_dir)
    # Pre-seed a stale plain file where the symlink should go.
    stale_dest = os.path.join(out_subtask_dir, "0001.jpg")
    with open(stale_dest, "wb") as f:
        f.write(b"stale-content")

    mme_prepare.reconcile_images(raw_root, "OCR", "flat", out_images_root)

    # The stale plain file was overwritten by a correct relative symlink.
    assert os.path.islink(stale_dest)
    assert os.path.realpath(stale_dest) == os.path.realpath(
        os.path.join(subtask_dir, "0001.jpg")
    )
    # Only the single expected entry exists (no leftovers/duplicates).
    assert sorted(os.listdir(out_subtask_dir)) == ["0001.jpg"]


# --- missing source aborts -------------------------------------------------


def test_reconcile_images_missing_source_aborts_with_naming_error(
    tmp_path, monkeypatch
):
    """A source image absent at materialization time aborts with a naming error.

    ``enumerate_pairs`` validates pairing up front, so by the time
    ``reconcile_images`` walks its pairs the listed sources normally exist. To
    exercise the missing-source guard (a source removed between enumeration and
    materialization), ``enumerate_pairs`` is patched to yield a pair whose image
    path does not exist on disk; ``reconcile_images`` must raise ``ValueError``
    naming the missing ``{subtask}/{filename}`` and materialize no entry for it.

    Validates: Requirement 6.6
    """
    raw_root = str(tmp_path / "raw")
    os.makedirs(os.path.join(raw_root, "OCR"))
    out_images_root = str(tmp_path / "images")

    missing_image = os.path.join(raw_root, "OCR", "0001.jpg")  # never created
    monkeypatch.setattr(
        mme_prepare,
        "enumerate_pairs",
        lambda subtask_dir, layout: [(missing_image, "irrelevant.txt")],
    )

    with pytest.raises(ValueError) as excinfo:
        # copy mode so the symlink-support probe is skipped and we reach the
        # per-image missing-source guard directly.
        mme_prepare.reconcile_images(
            raw_root, "OCR", "flat", out_images_root, copy=True
        )

    # Requirement 6.6: the error identifies the missing {subtask}/{filename}.
    assert "OCR/0001.jpg" in str(excinfo.value)
    # No entry was materialized for the missing source.
    dest = os.path.join(out_images_root, "OCR", "0001.jpg")
    assert not os.path.lexists(dest)


def test_reconcile_images_missing_source_aborts_without_further_entries(
    tmp_path, monkeypatch
):
    """On a missing source, reconciliation stops without materializing later entries.

    Two pairs are yielded: the first source exists, the second does not. The
    first entry is materialized, then the missing second source aborts the run,
    so no entry is materialized for the second (later) pair.

    Validates: Requirement 6.6
    """
    raw_root = str(tmp_path / "raw")
    subtask_dir = os.path.join(raw_root, "OCR")
    os.makedirs(subtask_dir)
    present_image = os.path.join(subtask_dir, "0001.jpg")
    with open(present_image, "wb") as f:
        f.write(b"present")
    missing_image = os.path.join(subtask_dir, "0002.jpg")  # never created
    out_images_root = str(tmp_path / "images")

    monkeypatch.setattr(
        mme_prepare,
        "enumerate_pairs",
        lambda subtask_dir, layout: [
            (present_image, "a.txt"),
            (missing_image, "b.txt"),
        ],
    )

    with pytest.raises(ValueError) as excinfo:
        mme_prepare.reconcile_images(
            raw_root, "OCR", "flat", out_images_root, copy=True
        )

    assert "OCR/0002.jpg" in str(excinfo.value)
    out_subtask_dir = os.path.join(out_images_root, "OCR")
    # The first (valid) entry was materialized before the abort...
    assert os.path.isfile(os.path.join(out_subtask_dir, "0001.jpg"))
    # ...but no entry exists for the missing, later pair.
    assert not os.path.lexists(os.path.join(out_subtask_dir, "0002.jpg"))


# ---------------------------------------------------------------------------
# Property 9: Idempotency
#
# Running the converter's main() twice on the same generated raw tree with
# identical command-line options yields a byte-identical mme_reference.jsonl and
# mme_questions.jsonl (and byte-identical per-subtask reference/{subtask}.jsonl),
# plus a stable images/ view: the same set of relative image paths, each
# resolving to the same Raw_Tree source via an identical relative symlink
# target. This is the end-to-end determinism/idempotency guarantee the operator
# relies on to safely re-run preparation.
#
# A single generated raw_datasets dataset is materialized on disk (FLAT or SPLIT
# per the dataset's own layout) and main() is invoked with explicit
# --raw-root/--out-root/--subtasks (the dataset's subtask) and --per-subtask, in
# the default (symlink) mode so the image-view comparison can assert identical
# relative symlink targets. main() is then run a second time against the same
# raw tree and out-root and the two output snapshots are compared.
#
# Hypothesis re-runs the body many times doing real filesystem I/O, so the trees
# are materialized with tempfile.mkdtemp (cleaned up in a finally) and a modest
# max_examples with deadline=None is used.
#
# Validates: Requirements 6.4, 9.1, 9.2, 9.3
# ---------------------------------------------------------------------------


def _snapshot_prepared_outputs(out_root: str, subtask: str) -> dict:
    """Snapshot main()'s outputs under ``out_root`` for idempotency comparison.

    Captures the exact bytes of ``mme_reference.jsonl``, ``mme_questions.jsonl``,
    and the per-subtask ``reference/{subtask}.jsonl``, plus the images/ view as a
    mapping of every relative image path to its (relative) symlink target. The
    image view is read with :func:`os.readlink` because the converter is run in
    its default symlink mode, so two idempotent runs must produce the same set of
    relative paths each resolving via the same relative target.
    """
    snapshot: dict = {}
    for name in ("mme_reference.jsonl", "mme_questions.jsonl"):
        with open(os.path.join(out_root, name), "rb") as f:
            snapshot[name] = f.read()
    per_subtask = os.path.join(out_root, "reference", f"{subtask}.jsonl")
    with open(per_subtask, "rb") as f:
        snapshot[f"reference/{subtask}.jsonl"] = f.read()

    images_root = os.path.join(out_root, "images")
    image_view: dict = {}
    for dirpath, _dirnames, filenames in os.walk(images_root):
        for filename in filenames:
            full = os.path.join(dirpath, filename)
            rel = os.path.relpath(full, images_root)
            # Default (symlink) mode: store the relative symlink target so the
            # view comparison asserts identical paths AND identical targets.
            image_view[rel] = os.readlink(full)
    snapshot["__images__"] = image_view
    return snapshot


@given(dataset=raw_datasets())
@settings(max_examples=30, deadline=None)
def test_main_is_idempotent(dataset):
    """Property 9: running main() twice on the same raw tree is byte-idempotent.

    Materializes a generated dataset as a raw tree (FLAT or SPLIT per its own
    layout), runs ``mme_prepare.main`` with explicit
    --raw-root/--out-root/--subtasks/--per-subtask in the default symlink mode,
    snapshots the outputs, runs main() a second time against the same raw tree
    and out-root, and asserts the second run produced byte-identical
    mme_reference.jsonl / mme_questions.jsonl / reference/{subtask}.jsonl and an
    identical images/ view (same relative paths, same relative symlink targets).

    Validates: Requirements 6.4, 9.1, 9.2, 9.3
    """
    subtask = dataset["subtask"]
    if dataset["layout"] == "split":
        raw_root = _materialize_split(dataset)
    else:
        raw_root = _materialize_flat(dataset)
    out_root = tempfile.mkdtemp(prefix="mme_out_")
    try:
        argv = [
            "--raw-root", raw_root,
            "--out-root", out_root,
            "--subtasks", subtask,
            "--per-subtask",
        ]

        # First run, then snapshot the prepared outputs + image view.
        mme_prepare.main(argv)
        first = _snapshot_prepared_outputs(out_root, subtask)

        # Second run against the same raw tree and out-root with identical opts.
        mme_prepare.main(argv)
        second = _snapshot_prepared_outputs(out_root, subtask)

        # Requirement 9.1: byte-identical combined reference on both runs.
        assert first["mme_reference.jsonl"] == second["mme_reference.jsonl"]
        # Requirement 9.2: byte-identical derived questions on both runs.
        assert first["mme_questions.jsonl"] == second["mme_questions.jsonl"]
        # Requirement 9.3: byte-identical per-subtask reference on both runs.
        assert (
            first[f"reference/{subtask}.jsonl"]
            == second[f"reference/{subtask}.jsonl"]
        )
        # Requirement 6.4: stable images/ view -- same set of relative paths,
        # each resolving to the same raw source via an identical relative target.
        assert first["__images__"] == second["__images__"]
        # Sanity: the image view actually covers every generated image.
        assert len(first["__images__"]) == len(dataset["images"])
    finally:
        shutil.rmtree(raw_root, ignore_errors=True)
        shutil.rmtree(out_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Example / edge-case tests: writer behavior and validation gating (Task 9.3).
#
# These are concrete examples (not Hypothesis property tests) that pin down the
# JSONL writer's trailing-newline contract, the combined reference's
# SUBTASKS-then-stem ordering, the questions file as a positional projection of
# the reference, and the all-or-nothing write gate: when validate_reference
# fails, main() leaves none of mme_reference.jsonl / mme_questions.jsonl /
# reference/{subtask}.jsonl written.
#
# Validates: Requirements 7.1, 7.2, 7.5, 7.7, 7.8, 9.4, 9.5
# ---------------------------------------------------------------------------


def _make_multi_subtask_flat(tmp_path, subtask_stems):
    """Materialize a FLAT raw-root spanning several subtasks under tmp_path.

    ``subtask_stems`` maps each subtask name to an iterable of image stems; for
    every stem a ``{stem}.jpg`` image file and a matching ``{stem}.txt``
    annotation file are written as siblings inside ``{raw_root}/{subtask}/``.
    Each annotation carries the same two valid ``question<TAB>answer`` lines (a
    distinct question text per subtask/stem so the records are easy to tell
    apart). Returns the raw-root path as a string.
    """
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    for subtask, stems in subtask_stems.items():
        subtask_dir = raw_root / subtask
        subtask_dir.mkdir()
        for stem in stems:
            (subtask_dir / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff")
            (subtask_dir / f"{stem}.txt").write_text(
                f"Is {subtask}/{stem} real? Please answer yes or no.\tYes\n"
                f"Is {subtask}/{stem} fake? Please answer yes or no.\tNo\n",
                encoding="utf-8",
            )
    return str(raw_root)


def _read_jsonl_records(path):
    """Read a JSONL file into a list of parsed dicts (one per line)."""
    import json

    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- write_jsonl trailing-newline contract ---------------------------------


def test_write_jsonl_ends_with_exactly_one_trailing_newline(tmp_path):
    """write_jsonl terminates every line (including the last) with a single \\n.

    The raw bytes end with exactly one ``\\n`` (not zero, not two), each line is
    a valid JSON object, and there are exactly as many lines as records.

    Validates: Requirement 7.1
    """
    records = [
        {"question_id": "OCR/0001_0", "image": "OCR/0001.jpg", "text": "a"},
        {"question_id": "OCR/0001_1", "image": "OCR/0001.jpg", "text": "b"},
        {"question_id": "OCR/0002_0", "image": "OCR/0002.jpg", "text": "c"},
    ]
    path = str(tmp_path / "out.jsonl")

    mme_prepare.write_jsonl(path, records)

    with open(path, "rb") as f:
        raw = f.read()

    # Requirement 7.1: the file ends with exactly one trailing newline.
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    # Every line including the last is terminated by \n, so splitting on \n
    # yields one trailing empty element and exactly len(records) content lines.
    lines = raw.decode("utf-8").split("\n")
    assert lines[-1] == ""
    content_lines = lines[:-1]
    assert len(content_lines) == len(records)
    for line, record in zip(content_lines, records):
        import json

        assert json.loads(line) == record


def test_write_jsonl_creates_parent_directory(tmp_path):
    """write_jsonl creates a missing parent directory before writing.

    Validates: Requirement 7.8
    """
    path = str(tmp_path / "nested" / "deeper" / "out.jsonl")
    record = {"question_id": "OCR/0001_0", "image": "OCR/0001.jpg", "text": "q"}

    mme_prepare.write_jsonl(path, [record])

    assert os.path.isfile(path)
    with open(path, "rb") as f:
        raw = f.read()
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert _read_jsonl_records(path) == [record]


# --- combined order: SUBTASKS-then-stem ------------------------------------


def test_main_combined_reference_ordered_by_subtasks_then_stem(tmp_path):
    """main() orders the combined reference by SUBTASKS order, then ascending stem.

    A small multi-subtask raw tree (existence + OCR, two stems each) is converted
    with the subtasks deliberately supplied out of SUBTASKS order; the resulting
    mme_reference.jsonl must group all of the SUBTASKS-earlier subtask's records
    (existence) before the later one's (OCR), and within each subtask order the
    records by ascending stem (each stem contributing its _0 then _1 record).

    Validates: Requirements 7.2
    """
    # existence precedes OCR in SUBTASKS; supply them reversed to prove the
    # output order follows SUBTASKS, not the operator's input order.
    raw_root = _make_multi_subtask_flat(
        tmp_path, {"existence": ["e002", "e001"], "OCR": ["0002", "0001"]}
    )
    out_root = str(tmp_path / "out")

    mme_prepare.main(
        ["--raw-root", raw_root, "--out-root", out_root,
         "--subtasks", "OCR", "existence"]
    )

    records = _read_jsonl_records(os.path.join(out_root, "mme_reference.jsonl"))

    # The expected order: SUBTASKS-earlier subtask (existence) first with its
    # stems ascending, then OCR with its stems ascending; two records per stem.
    expected_question_ids = [
        "existence/e001_0", "existence/e001_1",
        "existence/e002_0", "existence/e002_1",
        "OCR/0001_0", "OCR/0001_1",
        "OCR/0002_0", "OCR/0002_1",
    ]
    assert [r["question_id"] for r in records] == expected_question_ids

    # The categories appear in SUBTASKS order (all existence before all OCR).
    categories = [r["category"] for r in records]
    assert categories == ["existence"] * 4 + ["OCR"] * 4
    # Within each subtask, stems are ascending (Unicode order).
    existence_stems = [
        r["image"].split("/", 1)[1] for r in records if r["category"] == "existence"
    ]
    assert existence_stems == ["e001.jpg", "e001.jpg", "e002.jpg", "e002.jpg"]


# --- questions file is the positional projection of the reference ----------


def test_main_questions_file_is_positional_projection(tmp_path):
    """mme_questions.jsonl is a line-for-line projection of mme_reference.jsonl.

    The two files have the same number of lines; the question_id at each line
    position is identical between them; and every question record has exactly the
    three keys {question_id, image, text} (label/category stripped), with image
    and text copied unchanged from the reference at that position.

    Validates: Requirements 7.5, 9.4, 9.5
    """
    raw_root = _make_multi_subtask_flat(
        tmp_path, {"existence": ["e001"], "OCR": ["0001", "0002"]}
    )
    out_root = str(tmp_path / "out")

    mme_prepare.main(
        ["--raw-root", raw_root, "--out-root", out_root,
         "--subtasks", "existence", "OCR"]
    )

    reference = _read_jsonl_records(os.path.join(out_root, "mme_reference.jsonl"))
    questions = _read_jsonl_records(os.path.join(out_root, "mme_questions.jsonl"))

    # Equal number of lines.
    assert len(questions) == len(reference)
    # 3 images total => 6 records.
    assert len(reference) == 6

    for ref_record, q_record in zip(reference, questions):
        # Positional question_id match.
        assert q_record["question_id"] == ref_record["question_id"]
        # Exactly the three projected keys, no label/category.
        assert set(q_record.keys()) == {"question_id", "image", "text"}
        # image and text copied unchanged from the reference at that position.
        assert q_record["image"] == ref_record["image"]
        assert q_record["text"] == ref_record["text"]


# --- validation failure leaves no output files written ---------------------


def test_main_validation_failure_leaves_no_files_written(tmp_path, monkeypatch):
    """A validate_reference failure leaves none of the JSONL outputs written.

    validate_reference is patched to raise (simulating a reference that violates
    an evaluator invariant). main() must propagate the error and, because
    validation runs before any output file is created/opened/written, leave
    mme_reference.jsonl, mme_questions.jsonl, and reference/{subtask}.jsonl
    absent afterward.

    Validates: Requirements 7.7
    """
    raw_root = _make_multi_subtask_flat(
        tmp_path, {"existence": ["e001"], "OCR": ["0001"]}
    )
    out_root = str(tmp_path / "out")

    def _boom(records):
        raise ValueError("simulated validation failure")

    monkeypatch.setattr(mme_prepare, "validate_reference", _boom)

    with pytest.raises(ValueError):
        mme_prepare.main(
            ["--raw-root", raw_root, "--out-root", out_root,
             "--subtasks", "existence", "OCR", "--per-subtask"]
        )

    # Requirement 7.7: none of the output JSONL files exist.
    assert not os.path.exists(os.path.join(out_root, "mme_reference.jsonl"))
    assert not os.path.exists(os.path.join(out_root, "mme_questions.jsonl"))
    reference_dir = os.path.join(out_root, "reference")
    existing_per_subtask = (
        sorted(
            name
            for name in os.listdir(reference_dir)
            if name.endswith(".jsonl")
        )
        if os.path.isdir(reference_dir)
        else []
    )
    assert existing_per_subtask == []


# ---------------------------------------------------------------------------
# Integration / smoke test: converter output -> evaluator.
#
# Runs the full converter (mme_prepare.main) against the committed raw MME tree
# restricted to a single real subtask (OCR), then feeds the produced
# mme_reference.jsonl back through eval/mme_eval.py as a perfect-answer result
# file (every predicted answer equals its record's label) and asserts the
# results table renders for the processed subtask with no alignment or grouping
# error. This exercises the end-to-end RAW -> {reference, questions, images}
# -> evaluator path and confirms the evaluator's invariants hold by
# construction (Requirement 10).
#
# Validates: Requirements 10.1, 10.2, 10.3, 10.4
# ---------------------------------------------------------------------------

import json


def _build_perfect_answer_results(reference):
    """Build perfect-answer result records from a list of reference records.

    Produces, for each Reference_Record, a result record in the schema
    ``eval/mme_eval.py`` consumes: a ``question_id`` identical to the reference
    record's (so ``validate_alignment`` sees positional alignment) and a
    ``text`` set to the record's ground-truth ``label`` (``"yes"``/``"no"``), so
    every predicted answer equals its record's label and the evaluator scores a
    perfect run. The ``image`` is copied through for completeness.
    """
    return [
        {
            "question_id": record["question_id"],
            "image": record["image"],
            "text": record["label"],
        }
        for record in reference
    ]


def test_integration_converter_output_feeds_evaluator(tmp_path):
    """End-to-end: converter output is accepted by mme_eval as a perfect-answer run.

    Runs ``mme_prepare.main`` against the committed raw tree restricted to the
    ``OCR`` subtask (writing to a tmp out-root with ``--copy`` so no symlink
    support is required), then derives a perfect-answer result file from the
    produced ``mme_reference.jsonl`` (each predicted answer equal to its
    record's ``label``) and drives the full ``eval/mme_eval.py`` pipeline. The
    reference and result are positionally aligned (10.1, 10.2), every image
    groups to exactly two questions (10.3), and the results table renders for
    the processed subtask without raising an alignment or grouping error (10.4).

    Validates: Requirements 10.1, 10.2, 10.3, 10.4
    """
    subtask = "OCR"
    subtask_dir = os.path.join(_RAW_ROOT, subtask)
    if not os.path.isdir(subtask_dir):
        pytest.skip(
            f"committed raw tree missing subtask folder: {subtask} "
            f"(expected under {_RAW_ROOT})"
        )

    out_root = tmp_path / "out"

    # 1. Run the converter against the committed raw tree (single subtask).
    #    --copy avoids any symlink-support concerns in the tmp out-root.
    mme_prepare.main(
        [
            "--raw-root",
            _RAW_ROOT,
            "--out-root",
            str(out_root),
            "--subtasks",
            subtask,
            "--copy",
            "--per-subtask",
        ]
    )

    reference_path = out_root / "mme_reference.jsonl"
    questions_path = out_root / "mme_questions.jsonl"
    assert reference_path.is_file(), "converter did not write mme_reference.jsonl"
    assert questions_path.is_file(), "converter did not write mme_questions.jsonl"

    # 2. Build a perfect-answer result file from the produced reference: each
    #    predicted answer (text) equals its record's ground-truth label.
    reference = mme_eval.load_json_lines(str(reference_path))
    assert reference, "converter produced an empty reference"
    results = _build_perfect_answer_results(reference)

    result_path = out_root / "mme_result_perfect.jsonl"
    with open(result_path, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record))
            f.write("\n")

    # 3. Feed the reference + perfect-answer result through eval/mme_eval.py.
    ref_data = mme_eval.load_json_lines(str(reference_path))
    res_data = mme_eval.load_json_lines(str(result_path))

    # Requirements 10.1 / 10.2: equal record count and positional question_id
    # alignment -- validate_alignment must not raise.
    mme_eval.validate_alignment(ref_data, res_data)

    # Group aligned pairs by subtask and compute per-subtask metrics. The
    # per-subtask metric computation invokes group_by_image internally, which
    # raises if any image is not associated with exactly two questions
    # (Requirement 10.3).
    by_subtask = mme_eval.group_by_subtask(ref_data, res_data)
    assert subtask in by_subtask

    metrics_by_subtask = {}
    for name, pairs in by_subtask.items():
        # Requirement 10.3: group_by_image (called inside) must not raise.
        grouped = mme_eval.group_by_image(pairs)
        assert all(len(image_pairs) == 2 for image_pairs in grouped.values())
        metrics_by_subtask[name] = mme_eval.compute_subtask_metrics(name, pairs)

    # Only the processed subtask is present in the output.
    assert set(metrics_by_subtask) == {subtask}

    # Perfect answers => perfect accuracy/score for the processed subtask.
    ocr_metrics = metrics_by_subtask[subtask]
    assert ocr_metrics.accuracy == 1.0
    assert ocr_metrics.accuracy_plus == 1.0
    assert ocr_metrics.score == 200.0

    # 4. The results table renders for the processed subtask without raising.
    perception_total = mme_eval.compute_group_total(metrics_by_subtask, PERCEPTION)
    cognition_total = mme_eval.compute_group_total(metrics_by_subtask, COGNITION)
    table = mme_eval.render_results_table(
        metrics_by_subtask, perception_total, cognition_total
    )

    # Requirement 10.4: the table renders for the processed subtask and carries
    # the group-total rows.
    assert subtask in table
    assert "Perception Total" in table
    assert "Cognition Total" in table
