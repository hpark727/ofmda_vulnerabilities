#!/usr/bin/env bash

set -euo pipefail

IFACE="${1:?Usage: $0 <interface> <duration_seconds> <output_dir> <i> <j>}"
DURATION="${2:?Usage: $0 <interface> <duration_seconds> <output_dir> <i> <j>}"
OUTDIR="${3:?Usage: $0 <interface> <duration_seconds> <output_dir> <i> <j>}"
I="${4:?Usage: $0 <interface> <duration_seconds> <output_dir> <i> <j>}"
J="${5:?Usage: $0 <interface> <duration_seconds> <output_dir> <i> <j>}"

mkdir -p "$OUTDIR"

OUTFILE="$OUTDIR/${I}_${J}.pcap"

if ! touch "$OUTDIR/.capture_write_test" 2>/dev/null; then
    echo "Error: cannot write to output directory: $OUTDIR" >&2
    exit 1
fi
rm -f "$OUTDIR/.capture_write_test"

echo "Capturing on interface: $IFACE"
echo "Duration: ${DURATION}s"
echo "Indices: ${I}, ${J}"
echo "Saving to: $OUTFILE"

tshark -i "$IFACE" -F pcap -a "duration:${DURATION}" -w "$OUTFILE"

echo "Done."
