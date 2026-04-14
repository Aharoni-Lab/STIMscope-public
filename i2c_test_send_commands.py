#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from i2c_send_custom_cmd import execute_i2c_transfer, format_hex_list, parse_int_token


DEFAULT_TIMING_DATA = [
    0x03, 0x01, 0x04, 0xF8, 0x2A, 0x00, 0x00, 0x98,
    0x08, 0x00, 0x00, 0x88, 0x13, 0x00, 0x00,
]
DEFAULT_TRIGGER_DATA = [0x00, 0x00, 0x00, 0x00, 0x64, 0x00]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Configure the DMD trigger/pattern sequence over I2C")
    parser.add_argument("--bus", default="1", help="I2C bus number, e.g. 1")
    parser.add_argument("--addr", default="0x1B", help="7-bit I2C address, e.g. 0x1B")
    parser.add_argument("--seq-first", default="0x03", help="Sequence type byte, e.g. 0x03")
    parser.add_argument("--led", default="0x03", help="LED selection byte, e.g. 0x03")
    parser.add_argument("--delay-ms", default="40", help="Delay between commands in milliseconds")
    parser.add_argument("--no-start", action="store_true", help="Skip the final sequence-start command")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        bus_num = parse_int_token(args.bus, bits=16)
        addr = parse_int_token(args.addr, bits=8)
        seq_first = parse_int_token(args.seq_first, bits=8)
        led_byte = parse_int_token(args.led, bits=8)
        delay_s = parse_int_token(args.delay_ms, bits=16) / 1000.0
    except Exception as exc:
        print(f"[I2C] Argument error: {exc}", file=sys.stderr)
        return 2

    commands: List[Tuple[str, int, List[int]]] = [
        ("pattern-config", 0x92, [seq_first, 0x00, 0x00, 0x00, 0x00]),
        ("timing-config", 0x96, list(DEFAULT_TIMING_DATA)),
        ("trigger-config", 0x54, list(DEFAULT_TRIGGER_DATA)),
        ("led-select", 0x05, [led_byte]),
    ]
    if not args.no_start:
        commands.append(("sequence-start", 0x07, [0x02]))

    print(
        f"[I2C] starting projector trigger setup on bus={bus_num} "
        f"addr=0x{addr:02X} seq_first=0x{seq_first:02X} led=0x{led_byte:02X}"
    )

    for idx, (label, cmd, data) in enumerate(commands, start=1):
        print(f"[I2C] step {idx}/{len(commands)} {label}: cmd=0x{cmd:02X} data={format_hex_list(data)}")
        try:
            execute_i2c_transfer(bus_num, addr, cmd, data, 0)
        except Exception as exc:
            print(f"[I2C] step failed ({label}): {exc}", file=sys.stderr)
            return 1
        if delay_s > 0:
            time.sleep(delay_s)

    print("[I2C] projector trigger configuration complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
