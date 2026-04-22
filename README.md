# ofmda_vulnerabilities

This repo now uses per-site synchronization so each packet capture gets its own fresh SSH timing handshake with the remote Playwright laptop.

## How it works

`capture_script.sh` is now the controller:

1. Loads the shared site order from `sites.txt`.
2. Loops through every site for `N` iterations.
3. Before each run, measures remote clock offset over SSH.
4. Schedules one near-future start time for both machines.
5. Launches one remote Playwright site visit.
6. Starts one local `tshark` capture for that same site visit.

This avoids cumulative drift from a long pre-scheduled browser batch.

## Example

```bash
./capture_script.sh \
  --ssh-host 192.168.1.50 \
  --ssh-user hael \
  --remote-dir /home/hael/ofmda_vulnerabilities \
  --remote-scroll-seconds 12 \
  12 ./captures 2 2
```

That command means:

- capture on the hardcoded `CAPTURE_INTERFACE` inside `capture_script.sh`
- capture each run for `12` seconds
- save pcaps under `./captures`
- visit every site in `sites.txt` for `2` iterations
- wait `2` seconds between synchronized runs

## Files

- `sites.txt`: shared site order used by both scripts
- `capture_script.sh`: local controller and packet capture runner
- `playwright_script.py`: remote single-site worker, with optional batch mode still available

## Notes

- Increase `--lead-seconds` if the remote laptop needs more time to open SSH, start Python, and launch Chromium before the scheduled start.
- Remote stdout and stderr are saved locally under `output_dir/_remote_logs/`.
- The default output layout is `output_dir/<site>/run_<iteration>.pcap`.
- `playwright_script.py --print-sites` prints the shared site list from `sites.txt`.
