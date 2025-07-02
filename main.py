# main.py
import threading
import camera

from ids_peak import ids_peak
from WhiteBackgroundGen import makeWhite
from calibration import create_custom_registration_image
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from cli_interface import CLIInterface
    from qt_interface import QtInterface
    Interface = Union[CLIInterface, QtInterface]


def start(camera_device: camera.Camera, ui: 'Interface'):
    if not camera_device.start_realtime_acquisition():
        print("Failed to start acquisition!")
        return
    
    # Assets
    makeWhite(1936, 1096) #resolution    
    create_custom_registration_image(
        width=1920, 
        height=1080, 
        line_color=(255, 255, 255), 
        fill_color=(0, 0, 0)
    )

    ui.start_window()
    thread = threading.Thread(target=camera_device.acquisition_thread, args=())
    thread.start()
    ui.acquisition_thread = thread
    ui.start_interface()


def main(ui: 'Interface'):

    cam = ui._camera
    start(cam, ui)
