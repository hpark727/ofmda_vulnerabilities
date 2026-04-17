#!/usr/bin/env bash

set -euo pipefail

IFACE="${1:?Usage: $0 <interface> <duration_seconds> <output_dir> [prefix]}"
DURATION="${2:?Usage: $0 <interface> <duration_seconds> <output_dir> [prefix]}"
OUTDIR="${3:?Usage: $0 <interface> <duration_seconds> <output_dir> [prefix]}"
PREFIX="${4:-capture}"

mkdir -p "$OUTDIR"

TS="$(date +%Y%m%d_%H%M%S)"
OUTFILE="$OUTDIR/${PREFIX}_${TS}.pcap"

echo "Capturing on interface: $IFACE"
echo "Duration: ${DURATION}s"
echo "Saving to: $OUTFILE"

sudo tshark -i "$IFACE" -F pcap -a "duration:${DURATION}" -w "$OUTFILE"

echo "Done."