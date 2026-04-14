#!/usr/bin/env python3
import argparse
import subprocess
import sys
from typing import Iterable, List, Optional, Sequence

try:
    from smbus2 import SMBus, i2c_msg  # type: ignore
    _HAS_RDWR = True
except Exception:
    try:
        from smbus import SMBus  # type: ignore
    except Exception as exc:
        print(f"[I2C] Could not import an SMBus backend: {exc}", file=sys.stderr)
        sys.exit(2)
    i2c_msg = None
    _HAS_RDWR = False


def parse_int_token(token: str, *, bits: int = 8) -> int:
    value = int(str(token).strip(), 0)
    lo = 0
    hi = (1 << bits) - 1
    if not (lo <= value <= hi):
        raise ValueError(f"value {token!r} out of range for {bits}-bit field")
    return value


def parse_byte_list(values: Sequence[str]) -> List[int]:
    if not values:
        return []
    text = " ".join(str(v) for v in values)
    parts = [p for p in text.replace(",", " ").split() if p]
    return [parse_int_token(p, bits=8) for p in parts]


def format_hex_list(values: Iterable[int]) -> str:
    vals = list(values)
    return " ".join(f"0x{v:02X}" for v in vals) if vals else "<none>"


def _run_i2ctransfer(bus_num: int, addr: int, write_payload: List[int], read_len: int) -> List[int]:
    cmd = ["i2ctransfer", "-y", str(bus_num), f"w{len(write_payload)}@0x{addr:02X}"]
    cmd.extend(f"0x{b:02X}" for b in write_payload)
    if read_len > 0:
        cmd.append(f"r{read_len}")
    try:
        res = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("i2ctransfer not found; install i2c-tools") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("i2ctransfer timed out") from exc

    if res.returncode != 0:
        err = (res.stderr or res.stdout or "").strip()
        raise RuntimeError(err or f"i2ctransfer failed with exit code {res.returncode}")

    if read_len <= 0:
        return []

    tokens = [t for t in (res.stdout or "").replace(",", " ").split() if t]
    out: List[int] = []
    for tok in tokens:
        out.append(parse_int_token(tok, bits=8))
    return out


def execute_i2c_transfer(bus_num: int, addr: int, cmd: int, data: Optional[Sequence[int]] = None, read_len: int = 0) -> List[int]:
    data = list(data or [])
    payload = [cmd] + data

    if _HAS_RDWR and i2c_msg is not None:
        with SMBus(bus_num) as bus:
            if read_len > 0:
                write_msg = i2c_msg.write(addr, payload)
                read_msg = i2c_msg.read(addr, read_len)
                bus.i2c_rdwr(write_msg, read_msg)
                return list(read_msg)
            if data:
                bus.i2c_rdwr(i2c_msg.write(addr, payload))
            else:
                bus.write_byte(addr, cmd)
        return []

    if read_len > 0:
        return _run_i2ctransfer(bus_num, addr, payload, read_len)

    with SMBus(bus_num) as bus:
        if data:
            bus.write_i2c_block_data(addr, cmd, data)
        else:
            bus.write_byte(addr, cmd)
    return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send a custom I2C command to the DMD")
    parser.add_argument("--bus", default="1", help="I2C bus number, e.g. 1")
    parser.add_argument("--addr", default="0x1B", help="7-bit I2C address, e.g. 0x1B")
    parser.add_argument("--cmd", required=True, help="Command/register byte, e.g. 0x05")
    parser.add_argument(
        "--data",
        nargs="*",
        default=[],
        help="Optional data bytes; accepts space- or comma-separated hex/decimal values",
    )
    parser.add_argument("--read-len", default="0", help="Optional number of bytes to read back")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        bus_num = parse_int_token(args.bus, bits=16)
        addr = parse_int_token(args.addr, bits=8)
        cmd = parse_int_token(args.cmd, bits=8)
        data = parse_byte_list(args.data)
        read_len = parse_int_token(args.read_len, bits=16)
    except Exception as exc:
        print(f"[I2C] Argument error: {exc}", file=sys.stderr)
        return 2

    print(
        f"[I2C] bus={bus_num} addr=0x{addr:02X} "
        f"cmd=0x{cmd:02X} data={format_hex_list(data)} read_len={read_len}"
    )

    try:
        response = execute_i2c_transfer(bus_num, addr, cmd, data, read_len)
    except PermissionError as exc:
        print(f"[I2C] Permission error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"[I2C] Device not found: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"[I2C] Bus error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[I2C] Transfer failed: {exc}", file=sys.stderr)
        return 1

    if read_len > 0:
        print(f"[I2C] read={format_hex_list(response)}")
    else:
        print("[I2C] write complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
