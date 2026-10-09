"""Per-call wire log: one CSV row per API call, plus prompt/completion text.

This is the schema the side-channel analysis consumes. It is reproduced here
unchanged from the instrumented run that produced the published traces -- the
column set, the byte-counting convention and the latency rounding all have to
match, or previously collected traces and newly collected ones stop being
comparable.
"""
import csv
import json
import os
import time

SMOKETEST_LOG_PATH = os.environ.get('MDA_LOG_PATH', 'runs/logs/smoketest_log.csv')
TEXT_LOG_DIR = os.environ.get('MDA_TEXT_DIR', 'runs/logs/text')

LOG_FIELDNAMES = [
    'call_index', 'question_id', 'role', 'assigned_tier',
    'call_start_ts', 'timestamp', 'call_end_ts', 'latency_seconds',
    'prompt_text', 'prompt_tokens', 'request_bytes',
    'completion_text', 'completion_tokens', 'response_bytes',
]

_call_index = 0
_current_question_id = None
_current_tier = None
_log_header_checked = False

_UNSET = object()


def set_log_context(question_id=_UNSET, tier=_UNSET):
    global _current_question_id, _current_tier
    if question_id is not _UNSET:
        _current_question_id = question_id
    if tier is not _UNSET:
        _current_tier = tier


def _ensure_log_header_compatible():
    # If a log file from the old (or any other) schema already exists at
    # SMOKETEST_LOG_PATH, archive it instead of silently interleaving
    # mismatched rows under one header.
    global _log_header_checked
    if _log_header_checked:
        return
    _log_header_checked = True
    if not os.path.isfile(SMOKETEST_LOG_PATH):
        return
    with open(SMOKETEST_LOG_PATH, 'r', newline='') as f:
        first_line = f.readline().rstrip('\n')
    if first_line and first_line.split(',') != LOG_FIELDNAMES:
        backup_path = f"{SMOKETEST_LOG_PATH}.{int(time.time())}.bak"
        os.rename(SMOKETEST_LOG_PATH, backup_path)
        print(f"[log] {SMOKETEST_LOG_PATH} used an older schema; archived existing file to {backup_path}")


def _write_text(call_index, suffix, text):
    if not text:
        return ''
    os.makedirs(TEXT_LOG_DIR, exist_ok=True)
    path = os.path.join(TEXT_LOG_DIR, f'{call_index}_{suffix}.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return path


def serialize_for_byte_count(obj):
    # Compact, non-ASCII-preserving JSON so request-side byte counts use the
    # same encoding convention as pydantic's model_dump_json() on the
    # response side (identical separators, identical Unicode handling) --
    # required for request_bytes/response_bytes to be directly comparable.
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))


def log_api_call(role, call_start_ts, call_end_ts, request_payload,
                 response_obj=None, response_bytes=None, request_bytes=None):
    # Writes one row per API call attempt, success or failure. Callers must
    # invoke this strictly AFTER call_end_ts has been captured: all
    # serialization, file I/O, and CSV writing here happens outside the
    # timed network-call window, since call_start_ts/call_end_ts are used to
    # slice a packet capture and must bracket only the .create() call.
    #
    # response_bytes must be the literal byte length of the HTTP response
    # body (e.g. len(raw.content) from `.with_raw_response.create()`), NOT
    # derived from response_obj.model_dump_json(). The parsed/typed object
    # is a reconstruction: the openai SDK's Pydantic models fill in optional
    # fields as `null` even when the server never sent them, which silently
    # pads the re-serialized size by an amount that was never on the wire.
    global _call_index
    _ensure_log_header_compatible()
    _call_index += 1
    idx = _call_index

    request_json = serialize_for_byte_count(request_payload)
    if request_bytes is None:
        request_bytes = len(request_json.encode('utf-8'))
    prompt_text_path = _write_text(idx, 'prompt', request_json)

    if response_obj is not None:
        assert response_bytes is not None, "response_bytes (literal wire byte count) is required when response_obj is given"
        usage = response_obj.usage
        completion_text = response_obj.choices[0].message.content
        completion_text_path = _write_text(idx, 'completion', completion_text)
        prompt_tokens = usage.prompt_tokens if usage else ''
        completion_tokens = usage.completion_tokens if usage else ''
    else:
        completion_text_path = ''
        prompt_tokens = ''
        completion_tokens = ''
        response_bytes = ''

    latency_seconds = round((call_end_ts - call_start_ts) / 1e9, 4)

    row = {
        'call_index': idx,
        'question_id': _current_question_id,
        'role': role,
        'assigned_tier': _current_tier,
        'call_start_ts': call_start_ts,
        'timestamp': call_start_ts,  # alias of call_start_ts, kept for downstream scripts
        'call_end_ts': call_end_ts,
        'latency_seconds': latency_seconds,
        'prompt_text': prompt_text_path,
        'prompt_tokens': prompt_tokens,
        'request_bytes': request_bytes,
        'completion_text': completion_text_path,
        'completion_tokens': completion_tokens,
        'response_bytes': response_bytes,
    }

    os.makedirs(os.path.dirname(SMOKETEST_LOG_PATH), exist_ok=True)
    file_exists = os.path.isfile(SMOKETEST_LOG_PATH)
    with open(SMOKETEST_LOG_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()

    status = 'ok' if response_obj is not None else 'FAILED'
    print(f"[api call #{idx}] role={role} tier={_current_tier} latency={latency_seconds:.2f}s status={status}")
    return idx
