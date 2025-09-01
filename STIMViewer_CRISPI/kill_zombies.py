
import os
import sys
import psutil
from typing import Iterable, Optional, Tuple, List

ENV_DISABLE = "STIM_ALLOW_MULTI"      
ENV_TARGETS = "STIM_KILL_TARGETS"      
DEFAULT_TARGETS = (
    "main_gui.pyw",
    "main_gui.py",
    "main_gui",
    "STIMViewer_CRISPI",
    "STIMViewer",
    "stimviewer",
)

def _env_true(name: str) -> bool:
    v = os.getenv(name)
    return bool(v) and v.strip().lower() in ("1", "true", "yes", "on")

def _gather_targets(explicit: Optional[Iterable[str]] = None) -> Tuple[str, ...]:
    targets: List[str] = []

    if explicit:
        targets.extend(t.strip() for t in explicit if t and t.strip())

    try:
        if len(sys.argv) > 0 and sys.argv[0]:
            base = os.path.basename(sys.argv[0])
            targets.append(base)
            root, _ = os.path.splitext(base)
            if root and root not in targets:
                targets.append(root)
    except Exception:
        pass

    for t in DEFAULT_TARGETS:
        if t not in targets:
            targets.append(t)

    env_extra = os.getenv(ENV_TARGETS)
    if env_extra:
        for t in env_extra.split(","):
            t = t.strip()
            if t and t not in targets:
                targets.append(t)


    seen = set()
    uniq = [t for t in targets if not (t in seen or seen.add(t))]
    return tuple(uniq)

def _cmdline_matches_targets(cmdline_parts: Optional[Iterable[str]], targets: Tuple[str, ...]) -> bool:
    if not cmdline_parts:
        return False
    for part in cmdline_parts:
        if not part:
            continue
        part_basename = os.path.basename(part).lower()
        for t in targets:
            if t.lower() in part_basename:
                return True
    return False

def _same_user(proc: psutil.Process) -> bool:
    try:
        if hasattr(proc, "uids"):
            u = proc.uids()
            return u and hasattr(u, "real") and u.real == os.getuid()
        else:
            return proc.username() == psutil.Process().username()
    except Exception:
        return False

def kill_other_instances(targets: Optional[Iterable[str]] = None, timeout: float = 3.0) -> None:
    if _env_true(ENV_DISABLE):
        print("INFO: Multi-instance allowed by env; skipping zombie-kill stage")
        return

    try:
        me = psutil.Process(os.getpid())
        my_ctime = me.create_time()
    except Exception:
        me = None
        my_ctime = None

    match_targets = _gather_targets(targets)
    print("INFO: kill_zombies targets =", match_targets)

    victims: List[psutil.Process] = []

    for proc in psutil.process_iter(attrs=["pid", "name", "cmdline", "create_time"]):
        try:
            if proc.pid == os.getpid():
                continue
            if not _same_user(proc):
                continue
            if not _cmdline_matches_targets(proc.info.get("cmdline"), match_targets):
                continue
            p_ctime = proc.info.get("create_time")
            if my_ctime and p_ctime and p_ctime >= my_ctime:
                continue
            print(f"INFO: Marking older instance for termination: PID {proc.pid}")
            victims.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception as e:
            print(f"WARN: inspect error on PID {proc.pid}: {e}")

    if not victims:
        print("INFO: No zombie processes found.")
        return


    for p in victims:
        try:
            p.terminate()
        except Exception:
            pass

    try:
        gone, alive = psutil.wait_procs(victims, timeout=timeout)
    except Exception:
        alive = [p for p in victims if p.is_running()]


    for p in alive:
        try:
            print(f"INFO: Escalating to kill(): PID {p.pid}")
            p.kill()
        except Exception:
            pass

if __name__ == "__main__":
    kill_other_instances()
