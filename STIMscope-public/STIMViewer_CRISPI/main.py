
import os
os.environ.setdefault("QT_X11_NO_MITSHM", "1")

import threading
import time
import gc
from typing import TYPE_CHECKING, Union, Optional
import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication

sys.path.append(str(Path(__file__).resolve().parent))



try:
    from stimviewer.utils.error_handler import safe_execute, retry_on_error
    from stimviewer.utils.performance import log_performance, log_memory_usage
    from stimviewer.utils.thread_manager import get_thread_manager
except ImportError:
    def safe_execute(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            return None
    
    def retry_on_error(max_retries=3, delay=1.0):
        def decorator(func):
            def wrapper(*args, **kwargs):
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if attempt < max_retries:
                            print(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                            time.sleep(delay)
                        else:
                            print(f"All {max_retries + 1} attempts failed for {func.__name__}")
                            raise e
                return None
            return wrapper
        return decorator
    
    def log_performance(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                print(f"{func.__name__} executed in {execution_time:.4f} seconds")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                print(f"{func.__name__} failed after {execution_time:.4f} seconds: {e}")
                raise
        return wrapper
    
    def log_memory_usage(func):
        def wrapper(*args, **kwargs):
            try:
                import psutil
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                result = func(*args, **kwargs)
                
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_diff = memory_after - memory_before
                
                print(f"{func.__name__} memory usage: {memory_before:.2f}MB -> {memory_after:.2f}MB (diff: {memory_diff:+.2f}MB)")
                return result
            except Exception as e:
                print(f"Memory logging failed for {func.__name__}: {e}")
                return func(*args, **kwargs)
        return wrapper
    
    def get_thread_manager():
       
        import threading
        return threading

if TYPE_CHECKING:
    from qt_interface import Interface as QtInterface
    Interface = Union[CLIInterface, QtInterface]



ASSET_CLEANUP_INTERVAL = 60  
import camera

@log_performance
def start(camera_device: camera.Camera, ui: 'Interface') -> bool:
    try:
        try:
            camera_device.start(start_rt=True)   
        except Exception as e:
            print(f"Failed to start camera: {e}")
            return False

        try:
            from WhiteBackgroundGen import makeWhite

            makeWhite(1936, 1096)
            from calibration import create_custom_registration_image

            create_custom_registration_image(
                width=1920, height=1080,
                line_color=(255, 255, 255), fill_color=(0, 0, 0)
            )
            print("Assets created successfully")
        except Exception as e:
            print(f"Warning: Failed to create assets: {e}")

        try:
            ui.start_window()
        except Exception as e:
            print(f"Error starting UI window: {e}")
            return False

        ui.acquisition_thread = getattr(camera_device, "acquisition_thread", None)
        print("Camera acquisition thread started")

        return True

    except Exception as e:
        print(f"Critical error in start function: {e}")
        return False



@log_memory_usage
def main(ui: 'Interface') -> int:
    app = QApplication.instance()
    cam = getattr(ui, "_camera", None)

    try:
        if not cam:
            print("Error: No camera available")
            return 1

        if not start(cam, ui):
            print("Failed to start STIMViewer application")
            return 1

        print("Entering Qt event loop…")
        ui.start_interface()
        print("STIMViewer closed")
        return 0

    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        return 0
    except Exception as e:
        print(f"Critical error in main: {e}")
        return 1
    finally:
        try:
            if cam and hasattr(cam, "shutdown"):
                cam.shutdown()
            elif cam and hasattr(cam, "stop_realtime_acquisition"):
                cam.stop_realtime_acquisition()
        except Exception:
            pass

        t = getattr(ui, "acquisition_thread", None)
        if t and t.is_alive():
            try: t.join(timeout=1.0)
            except Exception: pass

        gc.collect()
        print("Application cleanup completed")




def cleanup_resources():
   
    try:

        gc.collect()
        

        import threading
        for thread in threading.enumerate():
            if thread.name.startswith("CameraAcquisition"):
                try:
                    thread.join(timeout=1.0)
                except Exception:
                    pass
        
        print("Resource cleanup completed")
    
    except Exception as e:
        print(f"Error during resource cleanup: {e}")

import atexit
atexit.register(cleanup_resources)
