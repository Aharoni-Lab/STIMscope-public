


import os


def setup_opengl_safety():
    
    try:

        os.environ.setdefault("QT_OPENGL", "software")
        os.environ.setdefault("QT_OPENGL_BUGLIST", "disable")
        

        os.environ.setdefault("QT_WIDGETS_RHI", "0")
        os.environ.setdefault("QT_QUICK_BACKEND", "software")
        

        os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
        

        os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "0")
        os.environ.setdefault("QT_SCALE_FACTOR", "1")
        

        os.environ.setdefault("QT_X11_NO_MITSHM", "1")
        
        print("✅ OpenGL safety environment configured for Jetson AGX Orin")
    except Exception as e:
        print(f"⚠️ OpenGL safety setup failed: {e}")

setup_opengl_safety()

import sys
import gc
import signal
import psutil
import threading
import shutil
import traceback
import time
from time import monotonic
import stat
import subprocess
import re
import faulthandler; faulthandler.enable()
import atexit
import multiprocessing as mp
from collections import deque
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Deque
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json



USE_TRACEMALLOC = os.getenv("STIMVIEWER_TRACE", "0") == "1"
if USE_TRACEMALLOC:
    import tracemalloc
    tracemalloc.start()

try:
    mp.set_start_method("spawn", force=False)
except RuntimeError:
    pass


try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    print("Warning: ZeroMQ not available. Install with: pip install pyzmq")


try:
    import pynvml
    from pynvml import NVMLError_NotSupported
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    class NVMLError_NotSupported(Exception):  
        pass

_last_ex_hook = {"t": 0.0}

def _soft_excepthook(exc_type, exc, tb):
    msg = "".join(traceback.format_exception(exc_type, exc, tb))
    now = time.monotonic()
    print(msg)
    if now - _last_ex_hook["t"] > 10.0:
        _last_ex_hook["t"] = now
        try:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(None, "Error", "An error occurred. See log for details.")
        except Exception:
            pass

sys.excepthook = _soft_excepthook


@dataclass
class SystemConfig:
    target_fps: int = 30
    max_memory_mb: int = 4096
    cpu_reserved_cores: int = 2
    gpu_memory_limit_mb: int = 2048


    cpu_processes: int = 8
    io_threads: int = 4
    camera_threads: int = 2


    zmq_pub_port: int = 5555
    zmq_control_port: int = 5557
    zmq_pub_host: str = "127.0.0.1" 
    zmq_transport: str = "ipc"  
    zmq_ipc_dir: str = f"/run/user/{os.getuid()}/stimviewer" 


    camera_buffer_size: int = 15
    processing_queue_size: int = 100
    perf_history_len: int = 1000


    enable_gpu_acceleration: bool = True
    enable_zero_mq: bool = True
    enable_performance_monitoring: bool = True
    enable_multiprocessing: bool = True
    safe_process_pool_on_jetson: bool = False


    enable_jetson_optimizations: bool = True
    enable_cuda_streams: bool = True
    enable_memory_pinning: bool = True

    tegrastats_interval_ms: int = 1000


@dataclass
class PerformanceMetrics:
    timestamp: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_usage: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    frame_rate: float = 0.0
    frame_latency_ms: float = 0.0
    queue_size: int = 0
    error_count: int = 0
    gil_contention: float = 0.0
    process_count: int = 0
    thread_count: int = 0


class GILAwareThreadManager:
   
    def __init__(self, config: SystemConfig):
        self.config = config
        self._cpu_pool: Optional[ProcessPoolExecutor] = None
        self.io_pool: Optional[ThreadPoolExecutor] = None
        self._camera_pool: Optional[ThreadPoolExecutor] = None
        self._running = False
        self._lock = threading.RLock()

    def initialize(self) -> bool:
        try:
            with self._lock:
                is_jetson = os.path.exists("/etc/nv_tegra_release")
                has_qt = "PyQt5" in sys.modules
                has_cuda = any(k in os.environ for k in ("CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING"))
                use_pool = (self.config.enable_multiprocessing and
                            (not is_jetson or self.config.safe_process_pool_on_jetson) and
                            not has_qt and not has_cuda)
                if not use_pool:
                    print("ProcessPool disabled (Jetson/Qt/CUDA fork-safety)")
                elif use_pool:
                    def _mp_init(reserved_cores=self.config.cpu_reserved_cores):
                        try:
                            import psutil, os
                            cpu_count = os.cpu_count() or 1
                            reserved = min(reserved_cores, max(1, cpu_count // 2))
                            psutil.Process().cpu_affinity(list(range(reserved, cpu_count)))
                        except Exception:
                            pass
                        os.environ.setdefault("OMP_NUM_THREADS", "1")
                        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
                        os.environ.setdefault("MKL_NUM_THREADS", "1")

                    self._cpu_pool = ProcessPoolExecutor(
                        max_workers=self.config.cpu_processes,
                        mp_context=mp.get_context("spawn"),
                        initializer=_mp_init,
                    )
                    print(f"ProcessPool enabled (workers={self.config.cpu_processes})")
                else:
                    print("ProcessPool disabled (Jetson fork-safety)")

                self.io_pool = ThreadPoolExecutor(max_workers=self.config.io_threads)
                self._camera_pool = ThreadPoolExecutor(max_workers=self.config.camera_threads)
                self._running = True
                return True
        except Exception as e:
            print(f"Failed to initialize thread manager: {e}")
            return False


    def submit_cpu_task(self, func, *args, **kwargs):
        if not self._running or not self._cpu_pool:
            return None
        try:
            return self._cpu_pool.submit(func, *args, **kwargs)
        except Exception as e:
            print(f"Failed to submit CPU task: {e}")
            return None

    def submit_io_task(self, func, *args, **kwargs):
        if not self._running or not self.io_pool:
            return None
        try:
            return self.io_pool.submit(func, *args, **kwargs)
        except Exception as e:
            print(f"Failed to submit I/O task: {e}")
            return None

    def submit_camera_task(self, func, *args, **kwargs):
        if not self._running or not self._camera_pool:
            return None
        try:
            return self._camera_pool.submit(func, *args, **kwargs)
        except Exception as e:
            print(f"Failed to submit camera task: {e}")
            return None

    def _shutdown_executor_with_timeout(self, executor, timeout: float, label: str):
        if not executor:
            return
        done = threading.Event()

        def _do_shutdown():
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except Exception as e:
                print(f"{label} shutdown raised: {e}")
            finally:
                done.set()

        t = threading.Thread(target=_do_shutdown, name=f"{label}-shutdown", daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            print(f"{label} shutdown timed out after {timeout:.1f}s; continuing")


    def cleanup(self):
        with self._lock:
            self._running = False
            self._shutdown_executor_with_timeout(self._cpu_pool, 3.0, "ProcessPool")
            self._cpu_pool = None
            self._shutdown_executor_with_timeout(self.io_pool, 2.0, "IO ThreadPool")
            self.io_pool = None
            self._shutdown_executor_with_timeout(self._camera_pool, 2.0, "Camera ThreadPool")
            self._camera_pool = None


class ZeroMQManager:
   
    def __init__(self, config: SystemConfig):
        self.config = config
        self.context: Optional["zmq.Context"] = None if ZMQ_AVAILABLE else None
        self.publisher = None
        self.control_socket = None
        self._running = False
        self._lock = threading.RLock()
        self._pub_lock = threading.Lock()
        self._io_pool: Optional[ThreadPoolExecutor] = None
        self._io_pool_shutdown_t = 2.0
        self._ctrl_thread: Optional[threading.Thread] = None
        self._ctrl_stop = threading.Event()
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}

    def register_handler(self, command: str, func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        if not callable(func):
            raise TypeError(f"handler for '{command}' must be callable")
        with self._lock:
            self._handlers[command] = func

    @staticmethod
    def _ensure_private_dir(path: str) -> None:
        os.makedirs(path, mode=0o700, exist_ok=True)
        try:
            st = os.stat(path)
            if (st.st_mode & 0o777) != 0o700:
                os.chmod(path, 0o700)
        except Exception:
            pass

    def initialize(self, io_pool: Optional[ThreadPoolExecutor]) -> bool:
        if not ZMQ_AVAILABLE:
            print("ZeroMQ not available, skipping initialization")
            return False
        self._io_pool = io_pool
        try:
            with self._lock:
                self.context = zmq.Context()

                self.publisher = self.context.socket(zmq.PUB)
                self.publisher.setsockopt(zmq.SNDHWM, 1000)
                self.publisher.setsockopt(zmq.LINGER, 0)

                if self.config.zmq_transport == "ipc":
                    self._ensure_private_dir(self.config.zmq_ipc_dir)
                    pub_path = os.path.join(self.config.zmq_ipc_dir, "pub.sock")
                    try:
                        if os.path.exists(pub_path):
                            os.unlink(pub_path)
                    except Exception:
                        pass
                    self.publisher.bind(f"ipc://{pub_path}")

                else:
                    if self.config.zmq_pub_host not in ("127.0.0.1", "localhost"):
                        print(f"Refusing to bind PUB to non-loopback host: {self.config.zmq_pub_host}")
                        return False
                    self.publisher.setsockopt(zmq.TCP_KEEPALIVE, 1)
                    self.publisher.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
                    self.publisher.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 30)
                    self.publisher.setsockopt(zmq.TCP_KEEPALIVE_CNT, 5)
                    self.publisher.bind(f"tcp://{self.config.zmq_pub_host}:{self.config.zmq_pub_port}")
                    print(f"ZeroMQ: PUB bound on {self.config.zmq_pub_host}:{self.config.zmq_pub_port}")

                self.control_socket = self.context.socket(zmq.REP)
                self.control_socket.setsockopt(zmq.LINGER, 0)
                self.control_socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
                self.control_socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
                self.control_socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 30)
                self.control_socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, 5)
                try:
                    self.control_socket.bind(f"tcp://127.0.0.1:{self.config.zmq_control_port}")
                    print(
                        f"ZeroMQ: REP control server bound on tcp://127.0.0.1:{self.config.zmq_control_port}"
                    )
                except Exception as e:
                    print(f"ZeroMQ control bind failed on {self.config.zmq_control_port}: {e}; disabling control.")
                    try:
                        self.control_socket.close(linger=0)
                    except Exception:
                        pass
                    self.control_socket = None

                self._running = True
                return True
        except Exception as e:
            print(f"Failed to initialize ZeroMQ: {e}")
            return False

    def _control_loop(self):
        if not self.control_socket:
            return

        poller = zmq.Poller()
        poller.register(self.control_socket, zmq.POLLIN)
        while not self._ctrl_stop.is_set():
            try:
                events = dict(poller.poll(timeout=100))
                if self.control_socket in events and events[self.control_socket] == zmq.POLLIN:
                    try:
                        data = self.control_socket.recv_json(flags=0)
                    except ValueError:
                        self.control_socket.send_json({"ok": False, "error": "invalid json"})
                        continue
                    cmd = (data or {}).get("command")
                    params = (data or {}).get("params") or {}
                    with self._lock:
                        handler = self._handlers.get(cmd)
                    if handler is None:
                        self.control_socket.send_json({"ok": False, "error": f"unknown command: {cmd}"})
                        continue
                    try:
                        resp = handler(params) or {}
                        self.control_socket.send_json({"ok": True, **resp})
                    except Exception as he:
                        self.control_socket.send_json({"ok": False, "error": str(he)})
            except zmq.error.ContextTerminated:
                break
            except zmq.ZMQError as ze:
                if getattr(ze, "errno", None) == zmq.ETERM:
                    break
                print(f"Control loop ZMQ error: {ze}")
            except Exception as e:
                print(f"Control loop error: {e}")
                time.sleep(0.05)

    def start_control_server(self) -> None:
        if not self.control_socket:
            print("ZeroMQ control socket unavailable; control server not started (PUB-only mode).")
            return
        if self._ctrl_thread and self._ctrl_thread.is_alive():
            return
        self._ctrl_stop.clear()
        self._ctrl_thread = threading.Thread(target=self._control_loop, name="zmq-rep", daemon=True)
        self._ctrl_thread.start()

    def stop_control_server(self):
        self._ctrl_stop.set()
        if self._ctrl_thread:
            self._ctrl_thread.join(timeout=2.0)
            self._ctrl_thread = None

    def publish_message(self, topic: str, data: Dict[str, Any]):
        if not self._running or not self.publisher or not self._io_pool:
            return
        try:
            self._io_pool.submit(self._publish_message_async, topic, data)
        except Exception:
            pass

    def _publish_message_async(self, topic: str, data: Dict[str, Any]):
        try:
            payload = json.dumps({"timestamp": time.time(), "data": data})
            with self._pub_lock:
                self.publisher.send_multipart(
                    [topic.encode("utf-8"), payload.encode("utf-8")]
                )
        except Exception as e:
            print(f"ZMQ PUB send error: {e}")



    def cleanup(self):
        with self._lock:
            self._running = False
            try:
                self.stop_control_server()
            except Exception as e:
                print(f"stop_control_server() error: {e}")
            
            for sock_attr in ("publisher", "control_socket"):
                sock = getattr(self, sock_attr, None)
                if sock is not None:
                    try:
                        sock.close(linger=0)
                    except Exception as e:
                        print(f"ZeroMQ socket '{sock_attr}' close error: {e}")
                    finally:
                        setattr(self, sock_attr, None)
            io_pool = self._io_pool
            self._io_pool = None
            if io_pool:
                try:
                    io_pool.shutdown(wait=True, cancel_futures=True)
                except Exception as e:
                    print(f"ZeroMQ I/O pool shutdown error: {e}")
            ctx = self.context
            self.context = None
            if ctx is not None:
                try:
                    ctx.term()
                except Exception as e:
                    print(f"ZeroMQ context termination error: {e}")


class PerformanceMonitor:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.metrics = PerformanceMetrics()
        self._stop_evt = threading.Event()
        self._lock = threading.RLock()
        self._max_history = self.config.perf_history_len
        self._history: Deque[PerformanceMetrics] = deque(maxlen=self._max_history)
        self._thread: Optional[threading.Thread] = None
        self._use_tegrastats = False
        self._ts_proc = None
        self._ts_thread = None
        self._ts_last_gpu = None
        self.gpu_available = False
        self._nvml_inited = False
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_available = True
                self._nvml_inited = True
            except Exception:
                self.gpu_available = False
        if not self.gpu_available:
            self._start_tegrastats()

    
    def clear_history(self):
        with self._lock:
            self._history.clear()

    def _start_tegrastats(self):
        try:
            cmd = "tegrastats"
            if not shutil.which(cmd):
                alt = "/usr/bin/tegrastats"
                if os.path.exists(alt):
                    cmd = alt
                else:
                    print("tegrastats not found; GPU util fallback disabled")
                    self._use_tegrastats = False
                    return

            self._ts_proc = subprocess.Popen(
                [cmd, "--interval", str(self.config.tegrastats_interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
            self._use_tegrastats = True
            self._ts_thread = threading.Thread(target=self._read_tegrastats, daemon=True)
            self._ts_thread.start()
            atexit.register(self._stop_tegrastats)
            print("Using tegrastats for GPU monitoring")
        except Exception as e:
            print(f"Failed to start tegrastats: {e}")
            self._use_tegrastats = False


    def _read_tegrastats(self):
        proc = self._ts_proc  
        if not proc or not proc.stdout:
            return
        freq_re = re.compile(r"GR3D_FREQ\s+(\d+)%")
        for line in proc.stdout:
            if not line:
                break
            m = freq_re.search(line)
            if m:
                self._ts_last_gpu = float(m.group(1))

    def _stop_tegrastats(self):
        try:
            if self._ts_proc:
                self._ts_proc.terminate()
                try:
                    self._ts_proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        self._ts_proc.kill()
                    except Exception:
                        pass
                finally:
                    try:
                        if self._ts_proc.stdout:
                            self._ts_proc.stdout.close()
                    except Exception:
                        pass
        except Exception:
            pass
        finally:
            self._ts_proc = None
            if self._ts_thread:
                try:
                    self._ts_thread.join(timeout=1.0)
                except Exception:
                    pass
                self._ts_thread = None




    def start_monitoring(self):
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, name="perf-monitor", daemon=True)
        self._thread.start()
        print("Performance monitoring started")

    def stop_monitoring(self):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        if NVML_AVAILABLE and self._nvml_inited:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_inited = False
        self._stop_tegrastats()


    def _loop(self):
        while not self._stop_evt.wait(1.0):
            try:
                self._update_metrics()
            except Exception as e:
                print(f"Performance monitoring error: {e}")

    def _update_metrics(self):
        from time import time as _now, monotonic as _mono

        with self._lock:
            try:
                self.metrics.timestamp = _now()
            except Exception:
                self.metrics.timestamp = _mono()
            self.metrics.cpu_usage = psutil.cpu_percent(interval=None)
            proc = psutil.Process()
            self.metrics.memory_usage_mb = proc.memory_info().rss / (1024 * 1024)

            if self.gpu_available:
                try:
                    self.metrics.gpu_usage = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
                    self.metrics.gpu_memory_mb = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle).used / 1024 / 1024
                except NVMLError_NotSupported:
                    self.gpu_available = False
                    print("NVML not supported; switching to tegrastats")
                    if not self._use_tegrastats:
                        self._start_tegrastats()
                except Exception as e:
                    print(f"GPU metrics error: {e}")
            elif self._use_tegrastats:
                self.metrics.gpu_usage = float(self._ts_last_gpu or 0.0)
                self.metrics.gpu_memory_mb = None


            self.metrics.process_count =0
            self.metrics.thread_count = threading.active_count()

            self._history.append(PerformanceMetrics(**self.metrics.__dict__))
            

    def get_current_metrics(self) -> PerformanceMetrics:
        with self._lock:
            return PerformanceMetrics(**self.metrics.__dict__)

    def get_average_metrics(self, window_seconds: float = 60.0) -> PerformanceMetrics:
        with self._lock:
            cutoff = time.time() - window_seconds
            recent = [m for m in self._history if m.timestamp > cutoff]
            if not recent:
                return PerformanceMetrics()
            avg = PerformanceMetrics()
            avg.cpu_usage = sum(m.cpu_usage for m in recent) / len(recent)
            avg.memory_usage_mb = sum(m.memory_usage_mb for m in recent) / len(recent)
            avg.frame_rate = sum(m.frame_rate for m in recent) / len(recent)
            avg.gil_contention = sum(m.gil_contention for m in recent) / len(recent)
            return avg


class HardwareOptimizer:
    def __init__(self, config: SystemConfig):
        self.config = config

    def setup_jetson_optimizations(self):
        if not self.config.enable_jetson_optimizations:
            return
        try:
            if os.path.exists("/etc/nv_tegra_release"):
                print("Jetson detected: applying performance settings")
                self._run_system_command([ 'jetson_clocks'], "Jetson performance mode")
                self._run_system_command([ 'nvpmodel', '-m', '0'], "MAXN mode")
                for i in range(os.cpu_count() or 1):
                    gov_path = f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor"
                    if os.path.exists(gov_path):
                        self._run_system_command(
                            ['sh', '-c', f'echo performance > {gov_path}'],
                            f"CPU {i} governor performance mode"
                        )

                self._run_system_command(
                    [ 'sh', '-c', 'echo mq-deadline > /sys/block/nvme0n1/queue/scheduler'],
                    "NVMe I/O scheduler optimization"
                )
                if self.config.enable_cuda_streams:
                    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
                print("Jetson optimizations applied")
            else:
                print("Not running on Jetson; skipping Jetson-specific optimizations")
        except Exception as e:
            print(f"Error setting up Jetson optimizations: {e}")

    def setup_cpu_affinity(self):
        try:
            cpu_count = os.cpu_count() or 1
            reserved = min(self.config.cpu_reserved_cores, max(1, cpu_count // 2))
            cores = list(range(reserved, cpu_count))
            psutil.Process().cpu_affinity(cores)
            print(f"CPU affinity set: reserved cores 0-{reserved-1}, using cores {cores}")
        except Exception as e:
            print(f"Could not set CPU affinity: {e}")

    def _run_system_command(self, command: List[str], description: str):
        try:
            if hasattr(os, "geteuid") and os.geteuid() != 0:
                print(f"{description} skipped (not root). Run these manually as root (Optional): {' '.join(command)}")
                return
            subprocess.run(command, check=True, capture_output=True, timeout=10)
            print(f"{description} enabled")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"Could not enable {description}: {e}")



class DisplayManager:
    def __init__(self):
        self.detected_display: Optional[str] = None

    @staticmethod
    def _x_is_available(display: str) -> bool:
        try:
            idx = display.split(':', 1)[1].split('.', 1)[0]
            return Path(f"/tmp/.X11-unix/X{idx}").exists()
        except Exception:
            return False

    def detect_display(self) -> Optional[str]:
        try:
            disp = os.environ.get("DISPLAY") or ""
            if disp and self._x_is_available(disp):
                self.detected_display = disp
                return disp
            for display in (":0", ":1"):
                if self._x_is_available(display):
                    os.environ["DISPLAY"] = display
                    self.detected_display = display
                    return display
        except Exception as e:
            print(f"Display detection error: {e}")
        self.detected_display = None
        return None

    def configure_environment(self):
        if self.detected_display:
            os.environ.update({
                "DISPLAY": self.detected_display,
                "QT_QPA_PLATFORM": "xcb",
                "QT_XCB_GL_INTEGRATION": "xcb_egl",
                "PYOPENGL_PLATFORM": "egl",
            })
        else:
            os.environ.update({
                "QT_QPA_PLATFORM": "offscreen",
                "PYOPENGL_PLATFORM": "egl",
                "QT_QPA_FONTDIR": "/usr/share/fonts",
                "QT_QPA_GENERIC_PLUGINS": "evdevmouse,evdevkeyboard",
            })
        os.environ.pop("MESA_GL_VERSION_OVERRIDE", None)
        os.environ.pop("MESA_GLSL_VERSION_OVERRIDE", None)


        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        os.environ.setdefault("SDL_VIDEODRIVER", "x11" if self.detected_display else "dummy")


class ApplicationManager:
    def __init__(self):
        self.config = SystemConfig()
        self.thread_manager = GILAwareThreadManager(self.config)
        self.zero_mq_manager = ZeroMQManager(self.config)
        self.performance_monitor = PerformanceMonitor(self.config)
        self.hardware_optimizer = HardwareOptimizer(self.config)
        self.display_manager = DisplayManager()
        self._camera_stop_fn = None
        self._ids_initialized = False 
        self.ids_peak = None
        self.app = None
        self.ui = None
        self.log_window = None
        self.memory_monitor_timer = None
        self._running = False
        self._cleanup_lock = threading.RLock()
        self._shutdown_event = threading.Event()

        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print("INFO", f"Signal {signum} received; requesting shutdown")
        self._shutdown_event.set()
        try:
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtCore import QTimer
            app = QApplication.instance()
            if app:
                QTimer.singleShot(0, app.quit) 
        except Exception:
            pass




    def initialize(self) -> bool:
        try:


            self.display_manager.detect_display()
            self.display_manager.configure_environment()


            self.hardware_optimizer.setup_jetson_optimizations()
            self.hardware_optimizer.setup_cpu_affinity()

            self.zero_mq_manager.publish_message("jetson_perf_flags", {
                "maxn": os.geteuid()==0,
                "jetson_clocks": os.geteuid()==0
            })

            if not self.thread_manager.initialize():
                raise RuntimeError("Failed to initialize thread manager")
            print("INFO", f"Pools: cpu={self.config.cpu_processes}, io={self.config.io_threads}, cam={self.config.camera_threads}")

            self._import_modules()

            self._create_qt_application()

            if self.config.enable_zero_mq:
                ok = self.zero_mq_manager.initialize(self.thread_manager.io_pool)
                if ok:
                    if self.zero_mq_manager.control_socket is not None:
                        self.zero_mq_manager.register_handler("ping", lambda p: {"pong": True, "t": time.time()})
                        def handle_shutdown(params: Dict[str, Any]) -> Dict[str, Any]:
                            try:
                                from PyQt5.QtWidgets import QApplication
                                from PyQt5.QtCore import QTimer
                                app = QApplication.instance()
                                if app:
                                    QTimer.singleShot(0, app.quit)
                                return {"scheduled": bool(app)}
                            except Exception as e:
                                return {"scheduled": False, "error": str(e)}

                        self.zero_mq_manager.register_handler("shutdown", handle_shutdown)
                        try:
                            self.zero_mq_manager.start_control_server()
                        except Exception as e:
                            print("WARN", f"ZeroMQ control server not started: {e}")
                    else:
                        print("WARN", "ZeroMQ control disabled (port busy); continuing with PUB only.")
                else:
                    print("WARN", "ZeroMQ init failed; PUB/REP not started")


            if self.config.enable_performance_monitoring:
                self.performance_monitor.start_monitoring()

            if self.config.enable_zero_mq:
                def _pub_perf():
                    m = self.performance_monitor.get_current_metrics().__dict__
                    self.zero_mq_manager.publish_message("perf_metrics", m)
                from PyQt5.QtCore import QTimer, QObject
                self._perf_pub_timer = QTimer(self.ui)
                self._perf_pub_timer.timeout.connect(_pub_perf)
                self._perf_pub_timer.start(1000)



            self._initialize_ids_peak()
            self._create_ui_components()
            self._setup_memory_monitoring()

            self._running = True
            print("INFO", "Initialization complete")
            return True
        except Exception as e:
            print("ERRO", f"Application initialization failed: {e}")
            return False

    def _import_modules(self):
        try:
            from main import main
            from kill_zombies import kill_other_instances
            from qt_interface import Interface
            from ids_peak import ids_peak
            self.main_module = main
            self.kill_zombies_module = kill_other_instances
            self.Interface = Interface
            self.ids_peak = ids_peak
        except ImportError as e:
            print("ERRO", f"Failed to import required modules: {e}")
            raise

    def _create_qt_application(self):
        try:
            from PyQt5.QtCore import QCoreApplication, Qt, QTimer
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtGui import QSurfaceFormat

            fmt = QSurfaceFormat()
            fmt.setRenderableType(QSurfaceFormat.OpenGLES)
            QSurfaceFormat.setDefaultFormat(fmt)

            if os.getenv("STIMVIEWER_SOFTGL", "0") == "1":
                QCoreApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)

            QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
            QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

            self.app = QApplication(sys.argv)
            self.app.setQuitOnLastWindowClosed(True)
            self.QTimer = QTimer
        except Exception as e:
            print("ERRO", f"Failed to create Qt application: {e}")
            raise


    def _initialize_ids_peak(self):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print("INFO", f"IDS Peak init attempt {attempt+1}/{max_retries}")
                if not self.ids_peak:
                    raise RuntimeError("ids_peak not imported")
                self.ids_peak.Library.Initialize()
                self._ids_initialized = True     
                print("INFO", "IDS Peak initialized")
                break
            except Exception as e:
                print("WARN", f"IDS Peak initialization failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1.0)

    def _create_ui_components(self):

    
        try:
            self.kill_zombies_module()
            print("INFO", "Killed zombie processes")

            

            self.ui = self.Interface()
            setattr(self.ui, "_io_pool", self.thread_manager.io_pool)
            setattr(self.ui, "_camera_pool", self.thread_manager._camera_pool)
            setattr(self.ui, "_zmq_pub", self.zero_mq_manager.publish_message if self.config.enable_zero_mq else None)


         

            from PyQt5.QtWidgets import QApplication
            app = QApplication.instance()


            cam = getattr(self.ui, "_camera", None)
            if app and cam:
                quit_fn = (getattr(cam, "shutdown", None) or
                        getattr(cam, "close", None) or
                        getattr(cam, "cleanup_resources", None) or
                        getattr(cam, "stop_realtime_acquisition", None))
                if callable(quit_fn):
                    self._camera_stop_fn = quit_fn
                    app.aboutToQuit.connect(quit_fn)


        except Exception as e:
            print("ERRO", f"Failed to create UI components: {e}")
            raise


    def _setup_memory_monitoring(self):
        try:
            self.memory_monitor_timer = self.QTimer(self.ui)
            self.memory_monitor_timer.timeout.connect(self._memory_monitor)
            self.memory_monitor_timer.start(30000)  # 30s
            print("INFO", "Memory monitoring enabled")
        except Exception as e:
            print("WARN", f"Could not set up memory monitoring: {e}")

    def _memory_monitor(self):
        try:
            rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            if rss_mb > self.config.max_memory_mb:
                print("WARN", f"High memory usage: {rss_mb:.1f} MB")
                gc.collect()
                print("INFO", "Forced garbage collection due to high memory usage")
        except Exception as e:
            print("ERRO", f"Memory monitoring error: {e}")

    def _dump_lingering(self):
        import traceback
        alive = []
        for t in threading.enumerate():
            if t is threading.current_thread():
                continue
            alive.append((t.name, t.ident, t.daemon, t.is_alive()))
        if alive:
            print("WARN", f"Lingering threads: {alive}")
            frames = sys._current_frames()
            for name, ident, daemon, alive_flag in alive:
                f = frames.get(ident)
                if f:
                    stack = "".join(traceback.format_stack(f))
                    print("DBUG", f"Stack for {name} (daemon={daemon}):\n{stack}")
        try:
            kids = psutil.Process().children(recursive=True)
            if kids:
                brief = [(p.pid, p.name(), p.status()) for p in kids if p.is_running()]
                print("WARN", f"Lingering child processes: {brief}")
        except Exception as e:
            print("WARN", f"Could not enumerate child processes: {e}")

    def run(self):
        try:
            print("INFO", "Starting main loop")
            if self.config.enable_zero_mq:
                self.zero_mq_manager.publish_message("app_startup", {
                    "timestamp": time.time(),
                    "config": self.config.__dict__,
                    "python_version": sys.version,
                    "platform": sys.platform
                })
            self.main_module(self.ui)
            print("INFO", "Application exited cleanly")
        except Exception as e:
            print("ERRO", f"Error in main application loop: {e}")
            import traceback
            print("ERRO", f"Stack trace: {traceback.format_exc()}")
            raise

    def cleanup(self):
        with self._cleanup_lock:
            if self._running:
                self._running = False
            print("INFO", "Cleanup starting...")
            try:
                if self.config.enable_performance_monitoring:
                    self.performance_monitor.stop_monitoring()

                if self.memory_monitor_timer:
                    try:
                        self.memory_monitor_timer.stop()
                        self.memory_monitor_timer.deleteLater()
                    except Exception:
                        pass
                    self.memory_monitor_timer = None
                try:
                    cam = getattr(self.ui, "_camera", None)
                    if self.log_window:
                        try:
                            self.log_window.close() 
                        except Exception:
                            pass

                    if cam:
                        for fn_name in ("shutdown", "close", "cleanup_resources", "stop_realtime_acquisition"):
                            fn = getattr(cam, fn_name, None)
                            if callable(fn):
                                try: fn()
                                except Exception as e: print("WARN", f"{fn_name} failed: {e}")
                                break  

                except Exception as e:
                    print("WARN", f"Camera stop/join during cleanup failed: {e}")
                try:
                    from PyQt5.QtWidgets import QApplication
                    app = QApplication.instance()
                    if app and self._camera_stop_fn:
                        try:
                            app.aboutToQuit.disconnect(self._camera_stop_fn)
                        except Exception:
                            pass
                except Exception:
                    pass

                if self.ui:
                    try:
                        self.ui.close()
                        if hasattr(self.ui, "deleteLater"):
                            self.ui.deleteLater()
                    except Exception as e:
                        print("WARN", f"Error closing UI: {e}")
                    self.ui = None

                if self.log_window:
                    try:
                        if hasattr(self.log_window, "shutdown"):
                            self.log_window.shutdown()
                        elif hasattr(self.log_window, "_cleanup"):
                            self.log_window._cleanup()
                        if hasattr(self.log_window, "deleteLater"):
                            self.log_window.deleteLater()
                    except Exception as e:
                        print("WARN", f"Logbook shutdown error: {e}")
                    self.log_window = None

                try:
                    from PyQt5.QtWidgets import QApplication
                    app = QApplication.instance()
                    if app:
                        app.processEvents()
                except Exception:
                    pass


                try:
                    print("INFO", "Closing IDS SDK")
                    if getattr(self, "_ids_initialized", False) and getattr(self, "ids_peak", None):
                        self.ids_peak.Library.Close()
                    else:
                        print("INFO", "IDS SDK not initialized; skip Library.Close()")
                except Exception as e:
                    print("WARN", f"Could not cleanly close IDS SDK: {e}")

                if self.config.enable_zero_mq:
                    try:
                        self.zero_mq_manager.cleanup()
                    except Exception as e:
                        print("WARN", f"ZeroMQ cleanup error: {e}")

                self.thread_manager.cleanup()

                gc.collect()

                try:
                    rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                    print("INFO", f"Final RSS: {rss_mb:.1f} MB")
                except Exception as e:
                    print("WARN", f"Could not get final memory usage: {e}")

                self._dump_lingering()

                if USE_TRACEMALLOC:
                    try:
                        tracemalloc.stop()
                    except Exception:
                        pass

                print("INFO", "Cleanup complete")
            except Exception as e:
                print("ERRO", f"Error during cleanup: {e}")


def main():
    app_manager = ApplicationManager()
    try:
        if app_manager.initialize():
            app_manager.run()
    except KeyboardInterrupt:
        print("INFO", "Application interrupted by user")
    except Exception as e:
        print("ERRO", f"Fatal error: {e}")
        sys.exit(1)
    finally:
        app_manager.cleanup()

if __name__ == "__main__":
    main()
