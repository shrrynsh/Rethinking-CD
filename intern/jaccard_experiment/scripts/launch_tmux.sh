#!/usr/bin/env bash
# Launch the full 17-condition sweep inside a detached tmux session.
#
# The sweep runs ~7 hours. A plain `bash scripts/run_all.sh` dies with its
# parent shell — a dropped SSH connection, a closed browser tab, or an IDE
# reload takes the whole run with it. tmux keeps it attached to the Studio
# machine instead of to your terminal.
#
# Usage:
#   bash scripts/launch_tmux.sh              # start the full sweep, detached
#   bash scripts/launch_tmux.sh run_vcd.sh   # start one condition instead
#
# Then:
#   tmux attach -t internvl        # watch it
#   Ctrl-b d                       # detach again, leaving it running
#   tmux ls                        # list sessions
#   tail -f logs/<session>.log     # follow without attaching
#   tmux kill-session -t internvl  # stop it

set -euo pipefail
cd "$(dirname "$0")/.."

SESSION="${SESSION:-internvl}"
TARGET="${1:-run_all.sh}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux is not installed." >&2
    exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists." >&2
    echo "  attach:  tmux attach -t $SESSION" >&2
    echo "  kill:    tmux kill-session -t $SESSION" >&2
    echo "  or set SESSION=<other-name> to run alongside it." >&2
    exit 1
fi

mkdir -p logs
# Timestamp comes from the shell, not hardcoded, so reruns don't clobber logs.
LOG="logs/${SESSION}-$(date +%Y%m%d-%H%M%S).log"

echo "Session : $SESSION"
echo "Running : scripts/$TARGET"
echo "Log     : $LOG"

# `bash -lc` so the login shell sets up conda/PATH the same way an interactive
# terminal would. `tee` keeps output visible when attached AND on disk.
# Trailing `exec bash` holds the pane open after the run so a crash message
# stays readable instead of the session vanishing.
tmux new-session -d -s "$SESSION" \
    "bash -lc 'bash scripts/$TARGET 2>&1 | tee $LOG; echo; echo \"=== finished (exit \$?) ===\"; exec bash'"

echo
echo "Started detached. Attach with:"
echo "  tmux attach -t $SESSION"
echo "Or follow the log without attaching:"
echo "  tail -f $(pwd)/$LOG"
