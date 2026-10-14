#!/usr/bin/env python3
"""Reject unbounded Docker logs and desktop entrypoints before collection."""
import json
import re
import subprocess
import sys


def validate(info):
    log = info["HostConfig"]["LogConfig"]
    config = log.get("Config", {})
    match = re.fullmatch(r"(\d+)([kmg]?)", str(config.get("max-size", "")).lower())
    if log.get("Type") not in ("json-file", "local") or not match:
        raise ValueError("explicit rotating Docker logs required (max-size=10m, max-file=3)")
    size = int(match[1]) * {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3}[match[2]]
    count = int(config.get("max-file", "0"))
    if not (0 < size <= 10 * 1024**2 and 1 <= count <= 3):
        raise ValueError("Docker log cap must be <=10m per file, <=3 files")
    entry = info["Config"].get("Entrypoint") or []
    cmd = info["Config"].get("Cmd") or []
    if entry != ["/bin/sleep"] or cmd != ["infinity"]:
        raise ValueError("collection requires headless entrypoint /bin/sleep infinity; no VNC")
    if not info.get("State", {}).get("Running"):
        raise ValueError("collection container is not running")


if __name__ == "__main__":
    try:
        info = json.loads(subprocess.check_output(["docker", "inspect", sys.argv[1]], text=True))[0]
        validate(info)
        print("[container] headless; Docker logs capped at 10m x 3")
    except (ValueError, KeyError, IndexError, subprocess.CalledProcessError) as exc:
        sys.exit("[container] BLOCKED: " + str(exc))
