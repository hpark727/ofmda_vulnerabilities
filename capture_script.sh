#!/usr/bin/env bash

set -euo pipefail

IFACE="${1:?Usage: $0 <interface> <duration_seconds> <output_dir> [prefix]}"
DURATION="${2:?Usage: $0 <interface> <duration_seconds> <output_dir> [prefix]}"
OUTDIR="${3:?Usage: $0 <interface> <duration_seconds> <output_dir> [prefix]}"
PREFIX="${4:-capture}"

mkdir -p "$OUTDIR"

TS="$(date +%Y%m%d_%H%M%S)"
OUTFILE="$OUTDIR/${PREFIX}_${TS}.pcap"

if ! touch "$OUTDIR/.capture_write_test" 2>/dev/null; then
    echo "Error: cannot write to output directory: $OUTDIR" >&2
    exit 1
fi
rm -f "$OUTDIR/.capture_write_test"

echo "Capturing on interface: $IFACE"
echo "Duration: ${DURATION}s"
echo "Saving to: $OUTFILE"

tshark -i "$IFACE" -F pcap -a "duration:${DURATION}" -w "$OUTFILE"

echo "Done."
