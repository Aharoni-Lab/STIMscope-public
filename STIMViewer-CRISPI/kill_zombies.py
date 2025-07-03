# kill_zombies.py
import os
import psutil


def kill_other_instances():
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['pid'] == current_pid:
                continue
            if (
                'python' in proc.info['name']
                and any('main_gui.pyw' in part for part in proc.info['cmdline'])
            ):
                print(f"[DEBUG] Killing older instance: PID {proc.info['pid']}")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

kill_other_instances()