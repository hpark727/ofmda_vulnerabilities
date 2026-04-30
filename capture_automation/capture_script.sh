#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USAGE=$(cat <<'EOF'
Usage:
  capture_script.sh [ssh sync options] <duration_seconds> <output_dir> <iterations> <offset_seconds>

SSH sync options:
  --site-file FILE             Shared site list file (default: sites.txt next to this script)
  --ssh-host HOST              Remote host running playwright_script.py
  --ssh-user USER              Optional SSH username
  --ssh-port PORT              SSH port (default: 22)
  --remote-dir DIR             Repo directory on the remote laptop
  --remote-python PATH         Remote Python executable (default: python3)
  --remote-script PATH         Remote playwright script path relative to remote-dir (default: playwright_script.py)
  --remote-scroll-seconds SEC  Value for playwright_script.py --scroll-seconds (default: duration_seconds)
  --lead-seconds SEC           How far in the future to schedule each coordinated start (default: 12)
  --offset-samples N           Number of SSH clock samples to take per run (default: 3)
  -h, --help                   Show this help text
EOF
)

CAPTURE_INTERFACE="wlx9cefd5f63c20"
SITE_FILE="$SCRIPT_DIR/sites.txt"
SSH_HOST=""
SSH_USER="hhy-a@192.168.0.15"
SSH_PORT="22"
REMOTE_DIR="/Users/haelpark/ofmda_vulnerabilities/.venv/bin/python"
REMOTE_PYTHON="python3"
REMOTE_SCRIPT="playwright_script.py"
REMOTE_SCROLL_SECONDS=""
LEAD_SECONDS="12"
OFFSET_SAMPLES="3"
REMOTE_JOB_PID=""
FLOAT_RE='^([0-9]+([.][0-9]+)?|[.][0-9]+)$'

timestamp_now() {
    python3 - <<'PY'
import time
print(f"{time.time():.6f}")
PY
}

wait_until_epoch() {
    python3 - "$1" <<'PY'
import sys
import time

target = float(sys.argv[1])
while True:
    remaining = target - time.time()
    if remaining <= 0:
        break
    time.sleep(min(remaining, 0.25))
PY
}

format_epoch() {
    python3 - "$1" <<'PY'
from datetime import datetime
import sys

print(datetime.fromtimestamp(float(sys.argv[1])).isoformat(timespec="seconds"))
PY
}

sanitize_site_name() {
    python3 - "$1" <<'PY'
import re
import sys
from urllib.parse import urlsplit

site = sys.argv[1].strip()
candidate = site if "://" in site else f"https://{site}"
parsed = urlsplit(candidate)
hostname = parsed.netloc or parsed.path.split("/", 1)[0]
label_source = hostname or site
label = re.sub(r"[^A-Za-z0-9._-]+", "_", label_source).strip("._-")
print(label or "site")
PY
}

float_lt() {
    python3 - "$1" "$2" <<'PY'
import sys

print("true" if float(sys.argv[1]) < float(sys.argv[2]) else "false")
PY
}

load_sites() {
    python3 - "$SITE_FILE" <<'PY'
from pathlib import Path
import sys

site_file = Path(sys.argv[1])
for line in site_file.read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if stripped and not stripped.startswith("#"):
        print(stripped)
PY
}

cleanup() {
    if [[ -n "$REMOTE_JOB_PID" ]] && kill -0 "$REMOTE_JOB_PID" 2>/dev/null; then
        kill "$REMOTE_JOB_PID" 2>/dev/null || true
    fi
}

measure_remote_clock_offset() {
    local best_offset=""
    local best_rtt=""
    local sample
    local current_offset
    local current_rtt
    local local_before
    local local_after
    local remote_now

    for ((sample = 1; sample <= OFFSET_SAMPLES; sample++)); do
        local_before=$(timestamp_now)
        remote_now=$("${SSH_BASE[@]}" "$SSH_TARGET" bash -s -- "$REMOTE_PYTHON" <<'EOF'
set -euo pipefail
REMOTE_PYTHON="$1"
"$REMOTE_PYTHON" - <<'PY'
import time
print(f"{time.time():.6f}")
PY
EOF
)
        local_after=$(timestamp_now)

        read -r current_offset current_rtt < <(
            python3 - "$local_before" "$remote_now" "$local_after" <<'PY'
import sys

local_before = float(sys.argv[1])
remote_now = float(sys.argv[2])
local_after = float(sys.argv[3])
midpoint = (local_before + local_after) / 2.0
rtt = local_after - local_before
offset = remote_now - midpoint
print(f"{offset:.6f} {rtt:.6f}")
PY
        )

        if [[ -z "$best_rtt" || "$(float_lt "$current_rtt" "$best_rtt")" == "true" ]]; then
            best_offset="$current_offset"
            best_rtt="$current_rtt"
        fi
    done

    echo "$best_offset"
}

while (($#)); do
    case "$1" in
        --site-file)
            SITE_FILE="${2:?Missing value for --site-file}"
            shift 2
            ;;
        --ssh-host)
            SSH_HOST="${2:?Missing value for --ssh-host}"
            shift 2
            ;;
        --ssh-user)
            SSH_USER="${2:?Missing value for --ssh-user}"
            shift 2
            ;;
        --ssh-port)
            SSH_PORT="${2:?Missing value for --ssh-port}"
            shift 2
            ;;
        --remote-dir)
            REMOTE_DIR="${2:?Missing value for --remote-dir}"
            shift 2
            ;;
        --remote-python)
            REMOTE_PYTHON="${2:?Missing value for --remote-python}"
            shift 2
            ;;
        --remote-script)
            REMOTE_SCRIPT="${2:?Missing value for --remote-script}"
            shift 2
            ;;
        --remote-scroll-seconds)
            REMOTE_SCROLL_SECONDS="${2:?Missing value for --remote-scroll-seconds}"
            shift 2
            ;;
        --lead-seconds)
            LEAD_SECONDS="${2:?Missing value for --lead-seconds}"
            shift 2
            ;;
        --offset-samples)
            OFFSET_SAMPLES="${2:?Missing value for --offset-samples}"
            shift 2
            ;;
        -h|--help)
            echo "$USAGE"
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Error: unknown option $1" >&2
            echo "$USAGE" >&2
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if (( $# != 4 )); then
    echo "$USAGE" >&2
    exit 1
fi

trap cleanup EXIT

DURATION="$1"
OUTDIR="$2"
ITERATIONS="$3"
OFFSET="$4"

if [[ -z "$REMOTE_SCROLL_SECONDS" ]]; then
    REMOTE_SCROLL_SECONDS="$DURATION"
fi

mkdir -p "$OUTDIR"

if ! touch "$OUTDIR/.capture_write_test" 2>/dev/null; then
    echo "Error: cannot write to output directory: $OUTDIR" >&2
    exit 1
fi
rm -f "$OUTDIR/.capture_write_test"

if ! [[ -f "$SITE_FILE" ]]; then
    echo "Error: site file not found: $SITE_FILE" >&2
    exit 1
fi

if [[ -z "$CAPTURE_INTERFACE" ]]; then
    echo "Error: CAPTURE_INTERFACE is empty in capture_script.sh" >&2
    exit 1
fi

if ! [[ "$DURATION" =~ ^[0-9]+$ && "$ITERATIONS" =~ ^[1-9][0-9]*$ && "$OFFSET" =~ ^[0-9]+$ ]]; then
    echo "Error: duration and offset must be non-negative integers, and iterations must be a positive integer." >&2
    exit 1
fi

if ! [[ "$SSH_PORT" =~ ^[1-9][0-9]*$ && "$OFFSET_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --ssh-port and --offset-samples must be positive integers." >&2
    exit 1
fi

if ! [[ "$LEAD_SECONDS" =~ $FLOAT_RE && "$REMOTE_SCROLL_SECONDS" =~ $FLOAT_RE ]]; then
    echo "Error: --lead-seconds and --remote-scroll-seconds must be non-negative numbers." >&2
    exit 1
fi

mapfile -t SITES < <(load_sites)

if (( ${#SITES[@]} == 0 )); then
    echo "Error: no sites found in $SITE_FILE" >&2
    exit 1
fi

SSH_BASE=(ssh)
if [[ "$SSH_PORT" != "22" ]]; then
    SSH_BASE+=(-p "$SSH_PORT")
fi

SCP_BASE=(scp)
if [[ "$SSH_PORT" != "22" ]]; then
    SCP_BASE+=(-P "$SSH_PORT")
fi

if [[ -n "$SSH_HOST" ]]; then
    if [[ -z "$REMOTE_DIR" ]]; then
        echo "Error: --remote-dir is required when --ssh-host is set." >&2
        exit 1
    fi
fi

SSH_TARGET="$SSH_HOST"
if [[ -n "$SSH_USER" && -n "$SSH_HOST" ]]; then
    SSH_TARGET="${SSH_USER}@${SSH_HOST}"
fi

REMOTE_LOG_DIR="$OUTDIR/_remote_logs"
mkdir -p "$REMOTE_LOG_DIR"

TOTAL_CAPTURES=$(( ${#SITES[@]} * ITERATIONS ))
CAPTURE_NUM=0

echo "Capturing on interface: $CAPTURE_INTERFACE"
echo "Duration: ${DURATION}s"
echo "Iterations per site: $ITERATIONS"
echo "Offset after each run: ${OFFSET}s"
echo "Site count: ${#SITES[@]}"

if [[ -n "$SSH_HOST" ]]; then
    echo "SSH sync enabled: $SSH_TARGET"
    echo "Remote repo dir: $REMOTE_DIR"
    echo "Remote script: $REMOTE_SCRIPT"

    "${SSH_BASE[@]}" "$SSH_TARGET" "mkdir -p '$REMOTE_DIR'"
    "${SCP_BASE[@]}" "$SCRIPT_DIR/playwright_script.py" "$SCRIPT_DIR/sites.txt" "$SSH_TARGET:$REMOTE_DIR/"
fi

for ((CURRENT_ITER = 1; CURRENT_ITER <= ITERATIONS; CURRENT_ITER++)); do
    for SITE in "${SITES[@]}"; do
        CAPTURE_NUM=$((CAPTURE_NUM + 1))
        SITE_KEY=$(sanitize_site_name "$SITE")
        SITE_OUTDIR="$OUTDIR/$SITE_KEY"
        OUTFILE="$SITE_OUTDIR/run_${CURRENT_ITER}.pcap"
        REMOTE_LOG_FILE="$REMOTE_LOG_DIR/$SITE_KEY/run_${CURRENT_ITER}.log"

        mkdir -p "$SITE_OUTDIR" "$(dirname "$REMOTE_LOG_FILE")"

        LOCAL_START_EPOCH=$(timestamp_now)
        REMOTE_START_EPOCH=""
        REMOTE_JOB_PID=""

        if [[ -n "$SSH_HOST" ]]; then
            REMOTE_CLOCK_OFFSET=$(measure_remote_clock_offset)

            read -r LOCAL_START_EPOCH REMOTE_START_EPOCH < <(
                python3 - "$LEAD_SECONDS" "$REMOTE_CLOCK_OFFSET" <<'PY'
import sys
import time

lead_seconds = float(sys.argv[1])
remote_clock_offset = float(sys.argv[2])
local_start = time.time() + lead_seconds
remote_start = local_start + remote_clock_offset
print(f"{local_start:.6f} {remote_start:.6f}")
PY
            )

            "${SSH_BASE[@]}" "$SSH_TARGET" bash -s -- \
                "$REMOTE_DIR" \
                "$REMOTE_PYTHON" \
                "$REMOTE_SCRIPT" \
                "$SITE" \
                "$CURRENT_ITER" \
                "$REMOTE_START_EPOCH" \
                "$REMOTE_SCROLL_SECONDS" <<'EOF' >"$REMOTE_LOG_FILE" 2>&1 &
set -euo pipefail

REMOTE_DIR="$1"
REMOTE_PYTHON="$2"
REMOTE_SCRIPT="$3"
SITE="$4"
CURRENT_ITER="$5"
REMOTE_START_EPOCH="$6"
REMOTE_SCROLL_SECONDS="$7"

cd "$REMOTE_DIR"
REMOTE_ARGS=(
    "$REMOTE_PYTHON"
    "$REMOTE_SCRIPT"
    --site "$SITE"
    --run-label "site=${SITE} iteration=${CURRENT_ITER}"
    --scroll-seconds "$REMOTE_SCROLL_SECONDS"
    --start-epoch "$REMOTE_START_EPOCH"
)

if [[ "$(uname -s)" == "Darwin" ]]; then
    CONSOLE_USER=$(stat -f %Su /dev/console)

    if [[ "$CONSOLE_USER" == "root" ]]; then
        echo "Error: no logged-in macOS console user was found; cannot launch a headed browser window over SSH" >&2
        exit 1
    fi

    # Use the logged-in user's HOME so Playwright and Chromium run in that user's normal profile context.
    env PYTHONUNBUFFERED=1 HOME="/Users/$CONSOLE_USER" "${REMOTE_ARGS[@]}"
else
    PYTHONUNBUFFERED=1 "${REMOTE_ARGS[@]}"
fi
EOF
            REMOTE_JOB_PID=$!
        fi

        echo "Capture ${CAPTURE_NUM}/${TOTAL_CAPTURES}"
        echo "Site: $SITE"
        echo "Iteration: ${CURRENT_ITER}/${ITERATIONS}"
        echo "Saving to: $OUTFILE"
        echo "Scheduled start: $(format_epoch "$LOCAL_START_EPOCH")"
        if [[ -n "$SSH_HOST" ]]; then
            echo "Remote log: $REMOTE_LOG_FILE"
        fi

        wait_until_epoch "$LOCAL_START_EPOCH"
        tshark -i "$CAPTURE_INTERFACE" -F pcap -a "duration:${DURATION}" -w "$OUTFILE"

        if [[ -n "$REMOTE_JOB_PID" ]]; then
            if ! wait "$REMOTE_JOB_PID"; then
                echo "Error: remote Playwright run failed for site $SITE iteration $CURRENT_ITER. See $REMOTE_LOG_FILE" >&2
                exit 1
            fi
            REMOTE_JOB_PID=""
        fi

        if (( CAPTURE_NUM < TOTAL_CAPTURES )) && (( OFFSET > 0 )); then
            echo "Waiting ${OFFSET}s before the next synchronized run..."
            sleep "$OFFSET"
        fi
    done
done

echo "Done."
