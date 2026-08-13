"""Submit the 17-condition sweep as a Lightning Job.

Why a Job and not tmux:
    A Studio auto-stops on an idle timer (auto_shutdown_time, 600 s here) that
    GPU activity does NOT reset. The first attempt at this sweep died 3 minutes
    in when the Studio suspended, taking the whole tmux server with it. tmux
    survives a dropped terminal; it does not survive the host being stopped.
    A Job runs on its own machine, independent of the Studio session.

Output persistence:
    A Job gets a snapshot of the Studio filesystem on a separate machine, so
    anything it writes into its own copy is lost when it ends. Outputs are
    therefore directed at $LIGHTNING_ARTIFACTS_DIR, which is persisted and
    surfaces under /teamspace/jobs/<job-name>/ where the Studio can read it.

Usage:
    python scripts/launch_job.py                 # submit
    python scripts/launch_job.py --name my-run   # custom job name
    python scripts/launch_job.py --status        # check submitted jobs
"""
import argparse
import sys

MACHINE = "L40S"
REPO_SUBDIR = "Rethinking-CD/intern/jaccard_experiment"
MODEL_PATH = "/teamspace/studios/this_studio/models/InternVL3-8B-hf"

# Runs inside the job. Notes on the pieces:
#   - OUT_ROOT -> artifacts dir, so results survive the job ending.
#   - The weights guard re-downloads if the snapshot did not carry the 15.9 GB
#     models/ directory; wget is resumable so a partial copy is fine too.
#   - set -e so a failure surfaces as a failed job rather than a silent partial.
JOB_COMMAND = f"""
set -e
# HOME may not resolve to the Studio root inside a job, so try both.
cd ~/{REPO_SUBDIR} 2>/dev/null \\
  || cd /teamspace/studios/this_studio/{REPO_SUBDIR}
echo "workdir: $PWD"

export MODEL_PATH={MODEL_PATH}
export OUT_ROOT="${{LIGHTNING_ARTIFACTS_DIR:-$PWD}}/outputs/llava_bench"
mkdir -p "$OUT_ROOT"

echo "=== job environment ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "OUT_ROOT=$OUT_ROOT"

if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "=== weights absent from snapshot; downloading ==="
    STAGING="$MODEL_PATH" bash scripts/download_internvl3_wget.sh
fi

echo "=== starting sweep ==="
bash scripts/run_all.sh
echo "=== sweep complete ==="
ls -R "$OUT_ROOT"
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="internvl-jaccard")
    ap.add_argument("--machine", default=MACHINE)
    ap.add_argument("--status", action="store_true",
                    help="List this teamspace's jobs and exit.")
    ap.add_argument("--interruptible", action="store_true",
                    help="Cheaper spot instance, but can be preempted. Safe "
                         "here: the run resumes from partial outputs.")
    args = ap.parse_args()

    from lightning_sdk import Job, Studio

    studio = Studio()

    if args.status:
        jobs = Job.list(teamspace=studio.teamspace)
        if not jobs:
            print("no jobs in this teamspace")
            return
        for j in jobs:
            print(f"  {j.name:30s} {j.status}")
        return

    print(f"Submitting job '{args.name}' on {args.machine} …")
    job = studio.run_job(
        name=args.name,
        machine=args.machine,
        command=JOB_COMMAND,
        interruptible=args.interruptible,
    )
    print(f"  submitted: {job.name}")
    print(f"  status   : {job.status}")
    print()
    print("Monitor with:")
    print(f"  python scripts/launch_job.py --status")
    print(f"  lightning job logs {job.name}")
    print(f"Results will appear under /teamspace/jobs/{job.name}/outputs/")


if __name__ == "__main__":
    main()
