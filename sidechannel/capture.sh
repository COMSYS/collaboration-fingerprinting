#!/usr/bin/env bash
# Passive capture of this laptop's own HTTPS traffic to the ki-connect
# endpoint during an MDAgents run. Ciphertext only -- no decryption, no
# MITM, no key material touched. Must be run with sudo (raw packet capture
# requires root on macOS; /dev/bpf* is root-only on this machine and
# ChmodBPF is not installed).
#
# Usage:
#   ./capture.sh [host] [outfile]
#   ./capture.sh                                # host is required
#   sudo ./capture.sh api.example.com run.pcap
#
# Stop with Ctrl-C (SIGINT) -- tcpdump flushes and closes the pcap cleanly
# on SIGINT, so don't kill -9 it.

set -euo pipefail

HOST="${1:?usage: capture.sh <host> [outfile] -- the hostname of the LLM endpoint to capture}"
OUTFILE="${2:-run_$(date +%Y%m%dT%H%M%S).pcap}"

if [[ "$(id -u)" -ne 0 ]]; then
    echo "This must be run as root (sudo ./capture.sh ...) -- raw packet capture needs it." >&2
    exit 1
fi

# --- Resolve the endpoint IP(s). Linux would use `getent hosts`; macOS
# doesn't ship it by default, so prefer `dig`/`dscacheutil` and fall back
# to getent if this is ever run on Linux. NOTE: this is a snapshot -- if
# the endpoint sits behind a CDN or DNS changes mid-run, packets from an
# IP not in this list will be silently excluded. Re-resolve if a run spans
# many hours.
if command -v getent >/dev/null 2>&1; then
    IPS=$(getent hosts "$HOST" | awk '{print $1}' | sort -u)
elif command -v dig >/dev/null 2>&1; then
    IPS=$(dig +short "$HOST" | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' | sort -u)
else
    echo "Neither getent nor dig available to resolve $HOST" >&2
    exit 1
fi

if [[ -z "$IPS" ]]; then
    echo "Could not resolve $HOST to any IPv4 address." >&2
    exit 1
fi

echo "Resolved $HOST to:"
echo "$IPS" | sed 's/^/  /'
N_IPS=$(echo "$IPS" | wc -l | tr -d ' ')
if [[ "$N_IPS" -gt 1 ]]; then
    echo "NOTE: multiple IPs returned -- likely a CDN/load balancer. All are included in the filter." >&2
fi

HOST_FILTER=$(echo "$IPS" | awk '{printf "host %s or ", $1}' | sed 's/ or $//')

# --- Identify the active interface (the one carrying the default route)
# rather than -i any, to cut unrelated noise (VPN, loopback, other
# adapters). Falls back to -i any with a warning if this can't be
# determined.
IFACE=$(route -n get default 2>/dev/null | awk '/interface:/{print $2}')
if [[ -z "$IFACE" ]]; then
    echo "Could not determine the default-route interface; falling back to -i any (noisier)." >&2
    IFACE="any"
else
    echo "Using interface: $IFACE"
fi

FILTER="($HOST_FILTER) and tcp port 443"
echo "tcpdump filter: $FILTER"
echo "Writing to: $OUTFILE"
echo "$OUTFILE" > .last_capture_path

# --- Timestamp basis: tcpdump's default packet timestamps are wall-clock
# (gettimeofday, microsecond precision) from this same machine's clock --
# the same clock time.time_ns() reads in the Python process. Since both
# are the same host, there is no cross-machine sync problem, only a
# precision one (pcap: microseconds: Python: nanoseconds) -- slice_pcap.py
# truncates ns -> us when comparing, which is the right direction (no
# false precision). Cross-check: after capture, slice_pcap.py will report
# whether each call's [call_start_ts, call_end_ts] window actually
# contains packets to/from the endpoint; an empty window for every call
# would indicate a clock or filter problem worth investigating by hand.
#
# -s 0: capture full packets, no truncation (needed for TLS record sizes)
# -w: write pcap to disk as captured (not just printed)
echo "Starting capture. Press Ctrl-C to stop cleanly."
exec tcpdump -i "$IFACE" -s 0 -w "$OUTFILE" "$FILTER"
