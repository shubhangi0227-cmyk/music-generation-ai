"""
midi_utils.py — Pure-Python MIDI read/write (no external libs required).
Supports Type-0 and Type-1 MIDI files.
"""

import struct
import os


# ---------------------------------------------------------------------------
# Variable-length quantity helpers
# ---------------------------------------------------------------------------

def _read_vlq(data: bytes, pos: int):
    """Read a variable-length quantity from bytes at pos. Returns (value, new_pos)."""
    value = 0
    while True:
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not (byte & 0x80):
            break
    return value, pos


def _write_vlq(value: int) -> bytes:
    """Encode an integer as a MIDI variable-length quantity."""
    result = [value & 0x7F]
    value >>= 7
    while value:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(result))


# ---------------------------------------------------------------------------
# Note dataclass (plain dict for zero dependencies)
# ---------------------------------------------------------------------------

def make_note(pitch: int, velocity: int, start_tick: int, end_tick: int) -> dict:
    return {"pitch": pitch, "velocity": velocity,
            "start_tick": start_tick, "end_tick": end_tick}


# ---------------------------------------------------------------------------
# MIDI parser
# ---------------------------------------------------------------------------

def parse_midi(path: str):
    """
    Parse a MIDI file and return a dict:
        {
          "ticks_per_beat": int,
          "tempo": int (microseconds per beat, default 500000 = 120 BPM),
          "notes": [{"pitch", "velocity", "start_tick", "end_tick"}, ...]
        }
    """
    with open(path, "rb") as f:
        data = f.read()

    # --- Header chunk ---
    if data[:4] != b"MThd":
        raise ValueError("Not a MIDI file")
    header_len = struct.unpack(">I", data[4:8])[0]
    fmt = struct.unpack(">H", data[8:10])[0]
    num_tracks = struct.unpack(">H", data[10:12])[0]
    ticks_per_beat = struct.unpack(">H", data[12:14])[0]

    pos = 8 + header_len  # skip past header data

    tempo = 500000  # default 120 BPM
    all_notes = []

    for _track in range(num_tracks):
        if data[pos:pos+4] != b"MTrk":
            break
        track_len = struct.unpack(">I", data[pos+4:pos+8])[0]
        track_data = data[pos+8 : pos+8+track_len]
        pos += 8 + track_len

        # Parse track events
        tp = 0  # position in track_data
        current_tick = 0
        running_status = 0
        open_notes = {}  # (channel, pitch) -> start_tick

        while tp < len(track_data):
            delta, tp = _read_vlq(track_data, tp)
            current_tick += delta

            if tp >= len(track_data):
                break

            byte = track_data[tp]

            # Meta event
            if byte == 0xFF:
                tp += 1
                meta_type = track_data[tp]; tp += 1
                meta_len, tp = _read_vlq(track_data, tp)
                meta_data = track_data[tp:tp+meta_len]; tp += meta_len
                if meta_type == 0x51 and meta_len == 3:
                    tempo = struct.unpack(">I", b"\x00" + meta_data)[0]
                continue

            # SysEx
            if byte in (0xF0, 0xF7):
                tp += 1
                sysex_len, tp = _read_vlq(track_data, tp)
                tp += sysex_len
                continue

            # MIDI event
            if byte & 0x80:
                running_status = byte
                tp += 1

            status = running_status
            msg_type = status & 0xF0
            channel = status & 0x0F

            if msg_type in (0x80, 0x90):  # note off / note on
                if tp + 1 >= len(track_data):
                    break
                pitch = track_data[tp]; tp += 1
                velocity = track_data[tp]; tp += 1
                key = (channel, pitch)
                if msg_type == 0x90 and velocity > 0:
                    open_notes[key] = current_tick
                else:
                    if key in open_notes:
                        all_notes.append(make_note(
                            pitch, velocity,
                            open_notes.pop(key), current_tick
                        ))
            elif msg_type in (0xA0, 0xB0, 0xE0):  # aftertouch / CC / pitch bend
                tp += 2
            elif msg_type in (0xC0, 0xD0):  # program change / channel pressure
                tp += 1
            else:
                tp += 1  # unknown, skip

    all_notes.sort(key=lambda n: n["start_tick"])
    return {"ticks_per_beat": ticks_per_beat, "tempo": tempo, "notes": all_notes}


# ---------------------------------------------------------------------------
# MIDI writer
# ---------------------------------------------------------------------------

def write_midi(path: str, notes: list, ticks_per_beat: int = 480, tempo: int = 500000):
    """
    Write a list of note dicts to a Type-0 MIDI file.
    Each note: {"pitch", "velocity", "start_tick", "end_tick"}
    """
    # Build event list: (tick, type, pitch, velocity)
    events = []
    for n in notes:
        events.append((n["start_tick"], "on",  n["pitch"], n.get("velocity", 80)))
        events.append((n["end_tick"],   "off", n["pitch"], 0))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1))

    # Encode track bytes
    track_bytes = bytearray()

    # Tempo meta event at tick 0
    track_bytes += _write_vlq(0)                  # delta
    track_bytes += bytes([0xFF, 0x51, 0x03])
    track_bytes += struct.pack(">I", tempo)[1:]   # 3 bytes

    prev_tick = 0
    for tick, etype, pitch, vel in events:
        delta = tick - prev_tick
        prev_tick = tick
        track_bytes += _write_vlq(delta)
        status = 0x90 if etype == "on" else 0x80
        track_bytes += bytes([status, pitch & 0x7F, vel & 0x7F])

    # End-of-track meta event
    track_bytes += bytes([0x00, 0xFF, 0x2F, 0x00])

    # Header chunk
    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks_per_beat)

    # Track chunk
    track_chunk = b"MTrk" + struct.pack(">I", len(track_bytes)) + bytes(track_bytes)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header + track_chunk)


# ---------------------------------------------------------------------------
# Tick ↔ seconds helpers
# ---------------------------------------------------------------------------

def ticks_to_seconds(ticks: int, ticks_per_beat: int, tempo: int) -> float:
    """Convert MIDI ticks to seconds."""
    beats = ticks / ticks_per_beat
    return beats * (tempo / 1_000_000)
