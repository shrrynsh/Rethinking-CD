"""Shared MME benchmark constants.

Defines the fourteen fixed MME subtask names and the Perception/Cognition
group split. These are imported by the MME evaluator (``eval/mme_eval.py``)
and may be reused by the inference scripts so the subtask vocabulary and the
group membership have a single source of truth.

The fourteen subtasks are partitioned into two groups:

- ``PERCEPTION`` (10 subtasks): perception-oriented yes/no questions.
- ``COGNITION`` (4 subtasks): reasoning-oriented yes/no questions.

Each MME image is associated with exactly two yes/no questions.
"""

# Perception group: 10 subtasks.
PERCEPTION = [
    "existence",
    "count",
    "position",
    "color",
    "posters",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "OCR",
]

# Cognition group: 4 subtasks.
COGNITION = [
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
]

# All fourteen subtasks, Perception subtasks first then Cognition subtasks.
SUBTASKS = PERCEPTION + COGNITION
