"""Slice a packet capture into per-call wire-level measurements, keyed by
the call_start_ts/call_end_ts (nanosecond wall-clock) columns already in
the logging CSV (see utils.py's _log_api_call).

We never decrypt anything. TLS record *headers* (content type, version,
length) are plaintext by design in the TLS record-layer spec -- only the
record *payload* is opaque -- so tls.record.length/content_type are
available from tshark without touching key material.

Usage (from the repository root):
    python sidechannel/slice_pcap.py --csv runs/logs/smoketest_log_test.csv --pcap run.pcap \
        --out-csv runs/captures/sliced.csv --out-json runs/captures/sliced_chunks.json

Requires tshark on PATH (reading an existing .pcap does not need root;
only live capture does).
"""
import argparse
import csv
import json
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--csv', required=True, help='logging CSV with call_start_ts/call_end_ts')
parser.add_argument('--pcap', required=True)
parser.add_argument('--out-csv', default='sliced.csv')
parser.add_argument('--out-json', default='sliced_chunks.json')
parser.add_argument('--server-port', type=int, default=443,
                     help='packets with this dst port are "up" (request), src port are "down" (response)')
args = parser.parse_args()


HANDSHAKE_TYPES = {20, 21, 22}  # change_cipher_spec, alert, handshake
APPLICATION_DATA_TYPE = 23

def load_packets(pcap_path):
    # One row per TCP segment on port 443. tls.record.length/content_type
    # can have multiple values per segment (several TLS records coalesced
    # into one TCP payload) -- tshark joins repeated occurrences with '|'.
    # A record fragmented across multiple TCP segments is only annotated
    # with content_type/length on the segment where reassembly completes;
    # earlier fragments show empty tls fields even though they carry real
    # payload bytes (tcp.len is always populated). We backward-fill: a
    # type-less segment belongs to whatever type is revealed by the next
    # informative segment in the same flow, since fragments always precede
    # the completing segment, never follow it.
    cmd = [
        'tshark', '-r', pcap_path, '-Y', f'tcp.port=={args.server_port}', '-T', 'fields',
        '-e', 'frame.time_epoch', '-e', 'ip.src', '-e', 'ip.dst',
        '-e', 'tcp.srcport', '-e', 'tcp.dstport', '-e', 'tcp.len',
        '-e', 'tls.record.length', '-e', 'tls.record.content_type',
        '-E', 'separator=,', '-E', 'occurrence=a', '-E', 'aggregator=|',
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)

    packets = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split(',')
        if len(fields) != 8:
            continue  # malformed/reassembly line; skip rather than guess
        time_epoch, ip_src, ip_dst, srcport, dstport, tcp_len, tls_lens, tls_types = fields
        direction = 'up' if dstport == str(args.server_port) else 'down'
        local_port = srcport if dstport == str(args.server_port) else dstport
        tls_record_lengths = [int(x) for x in tls_lens.split('|') if x]
        tls_record_types = [int(x) for x in tls_types.split('|') if x]
        packets.append({
            'ts': float(time_epoch),
            'direction': direction,
            'flow': local_port,  # groups packets belonging to one TCP connection
            'tcp_len': int(tcp_len) if tcp_len else 0,
            'tls_record_lengths': tls_record_lengths,
            'tls_record_types': tls_record_types,
            'category': None,  # filled in below: 'handshake' / 'application_data' / 'other'
        })
    packets.sort(key=lambda p: p['ts'])

    # Backward-fill content-type category per flow, scanning each flow's
    # packets in reverse time order.
    by_flow = {}
    for p in packets:
        by_flow.setdefault(p['flow'], []).append(p)
    for flow_packets in by_flow.values():
        flow_packets.sort(key=lambda p: p['ts'])
        next_known_type = None
        for p in reversed(flow_packets):
            if p['tls_record_types']:
                # The first record in this segment is the one any preceding
                # (type-less) fragments belong to.
                effective_type = p['tls_record_types'][0]
                next_known_type = effective_type
            elif p['tcp_len'] > 0:
                effective_type = next_known_type
            else:
                effective_type = None  # pure ACK, no payload either way
            if effective_type in HANDSHAKE_TYPES:
                p['category'] = 'handshake'
            elif effective_type == APPLICATION_DATA_TYPE:
                p['category'] = 'application_data'
            else:
                p['category'] = 'other'

    return packets


def slice_call(packets, start_ts_s, end_ts_s):
    window = [p for p in packets if start_ts_s <= p['ts'] <= end_ts_s]
    up = [p for p in window if p['direction'] == 'up']
    down = [p for p in window if p['direction'] == 'down']

    def summarize(pkts):
        total_bytes = sum(p['tcp_len'] for p in pkts)
        handshake_bytes = sum(p['tcp_len'] for p in pkts if p['category'] == 'handshake')
        appdata_bytes = sum(p['tcp_len'] for p in pkts if p['category'] == 'application_data')
        chunk_sizes = [{'ts': p['ts'], 'tcp_len': p['tcp_len'], 'category': p['category'],
                         'tls_record_lengths': p['tls_record_lengths'],
                         'tls_record_types': p['tls_record_types']} for p in pkts]
        gaps = [round(b['ts'] - a['ts'], 6) for a, b in zip(pkts, pkts[1:])]
        return total_bytes, handshake_bytes, appdata_bytes, chunk_sizes, gaps

    up_total, up_handshake, up_appdata, up_chunks, up_gaps = summarize(up)
    down_total, down_handshake, down_appdata, down_chunks, down_gaps = summarize(down)
    return {
        # Total wire bytes, everything on the connection during this call's
        # window -- includes TLS handshake bytes if this call opened (or is
        # sharing a window with) a fresh connection. This is the true total
        # an observer would see, but is NOT directly comparable to the
        # app-layer request_bytes/response_bytes.
        'request_bytes_wire_total': up_total,
        'response_bytes_wire_total': down_total,
        # Application-data-only subset (TLS content-type 23) -- the fair,
        # apples-to-apples comparison basis against request_bytes/response_bytes.
        'request_bytes_wire_appdata': up_appdata,
        'response_bytes_wire_appdata': down_appdata,
        # Handshake-only subset (content-types 20/21/22) -- connection-setup
        # cost, itself an observable signal (reveals "fresh connection?").
        'request_bytes_wire_handshake': up_handshake,
        'response_bytes_wire_handshake': down_handshake,
        'n_up_packets': len(up),
        'n_down_packets': len(down),
        'up_chunks': up_chunks,
        'down_chunks': down_chunks,
        'down_inter_arrival_gaps_s': down_gaps,
        'up_inter_arrival_gaps_s': up_gaps,
    }


def main():
    with open(args.csv, newline='') as f:
        rows = list(csv.DictReader(f))

    packets = load_packets(args.pcap)
    print(f"Loaded {len(packets)} TCP segments on port {args.server_port} from {args.pcap}")

    merged_rows = []
    chunk_records = {}
    empty_windows = 0

    for row in rows:
        start_ns = int(row['call_start_ts'])
        end_ns = int(row['call_end_ts'])
        start_s, end_s = start_ns / 1e9, end_ns / 1e9

        sliced = slice_call(packets, start_s, end_s)
        if sliced['n_up_packets'] == 0 and sliced['n_down_packets'] == 0:
            empty_windows += 1

        merged = {
            'call_index': row['call_index'],
            'question_id': row['question_id'],
            'role': row['role'],
            'assigned_tier': row['assigned_tier'],
            'prompt_tokens': row['prompt_tokens'],
            'completion_tokens': row['completion_tokens'],
            'request_bytes': row['request_bytes'],       # app-layer, already logged
            'response_bytes': row['response_bytes'],     # app-layer, already logged
            'request_bytes_wire_appdata': sliced['request_bytes_wire_appdata'],
            'response_bytes_wire_appdata': sliced['response_bytes_wire_appdata'],
            'request_bytes_wire_handshake': sliced['request_bytes_wire_handshake'],
            'response_bytes_wire_handshake': sliced['response_bytes_wire_handshake'],
            'request_bytes_wire_total': sliced['request_bytes_wire_total'],
            'response_bytes_wire_total': sliced['response_bytes_wire_total'],
            'n_up_packets': sliced['n_up_packets'],
            'n_down_packets': sliced['n_down_packets'],
        }
        merged_rows.append(merged)
        chunk_records[row['call_index']] = {
            'up_chunks': sliced['up_chunks'],
            'down_chunks': sliced['down_chunks'],
            'down_inter_arrival_gaps_s': sliced['down_inter_arrival_gaps_s'],
            'up_inter_arrival_gaps_s': sliced['up_inter_arrival_gaps_s'],
        }

    if empty_windows:
        print(f"WARNING: {empty_windows}/{len(rows)} call windows had zero packets in the capture "
              f"-- check clock alignment, filter IPs, or that the capture was running during those calls.")

    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(merged_rows[0].keys()))
        writer.writeheader()
        writer.writerows(merged_rows)
    print(f"Wrote {len(merged_rows)} rows to {args.out_csv}")

    with open(args.out_json, 'w') as f:
        json.dump(chunk_records, f, indent=2)
    print(f"Wrote per-call chunk sequences to {args.out_json}")

    return merged_rows


if __name__ == '__main__':
    main()
