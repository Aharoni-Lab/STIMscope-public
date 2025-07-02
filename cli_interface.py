# \file    cli_interface.py
# \author  IDS Imaging Development Systems GmbH
# \date    2024-02-20
#
# \brief   This sample shows how to start and stop acquisition as well as
#          how to capture images using a software trigger
#
# \version 1.0
#
# Copyright (C) 2024, IDS Imaging Development Systems GmbH.
#
# The information in this document is subject to change without notice
# and should not be construed as a commitment by IDS Imaging Development Systems GmbH.
# IDS Imaging Development Systems GmbH does not assume any responsibility for any errors
# that may appear in this document.
#
# This document, or source code, is provided solely as an example of how to utilize
# IDS Imaging Development Systems GmbH software libraries in a sample application.
# IDS Imaging Development Systems GmbH does not assume any responsibility
# for the use or reliability of any portion of this document.
#
# General permission to copy or modify is hereby granted.
from ids_peak import ids_peak
from camera import Camera
from logbook import Logbook


class Interface:
    def __init__(self, cam_module: Camera = None):
        self._camera = cam_module
        self.acquisition_thread = None

    def is_gui(self):
        return False

    def set_camera(self, cam_module: Camera):
        self._camera = cam_module

    def print_help(self):
        Logbook.log_INFO(
            "Available commands:\n"
            "\"trigger\" capture an image.\n"
            "\"start\" start acquisition.\n"
            "\"stop\" stop acquisition.\n"
            "\"save True|False\" wether captured images should be saved to a file.\n"
            "\"pixelformat\" change the pixelformat.\n"
            "\"exit\" close the program\n"
            "\"help\" display this text"
        )

    def acquisition_check_and_set(self):
        if not self._camera.acquisition_running:
            Logbook.log_NOTI("The image acquisition must be running to get an image.")
            choice = input("Start acquisition now?: [Y|n]")
            if choice == "" or choice == "y" or choice == "Y":
                self._camera.start_realtime_acquisition()
                return True
            return False
        return True

    def acquisition_check_and_disable(self):
        if self._camera.acquisition_running:
            Logbook.log_NOTI("Acquisition must NOT be running to set a new pixelformat")
            choice = input("Stop acquisition now?: [Y|n]")
            if choice == "" or choice == "y" or choice == "Y":
                self._camera.stop_realtime_acquisition()
                return True
            if choice == "n" or choice == "N":
                return False
        return True

    def change_pixelformat(self):
        formats = self._camera.node_map.FindNode("PixelFormat").Entries()
        available_options = []
        for idx in formats:
            if (idx.AccessStatus() != ids_peak.NodeAccessStatus_NotAvailable
                    and idx.AccessStatus() != ids_peak.NodeAccessStatus_NotImplemented):
                available_options.append(idx.SymbolicValue())
        Logbook.log_ALRT("Select available option by index:\n")
        counter = 0
        for entry in available_options:
            Logbook.log_INFO(f"[{counter}]: {entry}")
            counter += 1
        selected = -1
        while selected == -1:
            try:
                selected = int(input(" >> "))
            except ValueError:
                selected = -1
            if selected < 0 or selected >= len(available_options):
                Logbook.log_NOTI(
                    f"Please enter a number between 0 and {len(available_options) - 1}")
                selected = -1
        self._camera.change_pixel_format(available_options[selected])

    def start_interface(self):
        self.print_help()
        try:
            while True:
                var = input("> ")

                var = var.split()
                if not var:
                    continue

                if var[0] == "trigger":
                    # trigger an image
                    if not self.acquisition_check_and_set():
                        Logbook.log_ALRT("Acquisition not started... Skipping trigger command!")
                        continue
                    self._camera.make_image = True
                    # wait until image has been made
                    while self._camera.make_image:
                        pass

                elif var[0] == "save":
                    # enable/disable saving to drive
                    if len(var) < 2:
                        Logbook.log_NOTI("Missing argument! Usage: save True|False")
                        continue
                    if var[1] == "True":
                        self._camera.keep_image = True
                        Logbook.log_NOTI("Saving images: Enabled")
                    elif var[1] == "False":
                        self._camera.keep_image = False
                        Logbook.log_NOTI("Saving images: Disabled")

                elif var[0] == "start":
                    self._camera.start_realtime_acquisition()

                elif var[0] == "stop":
                    self._camera.stop_realtime_acquisition()

                elif var[0] == "help":
                    self.print_help()

                elif var[0] == "pixelformat":
                    if not self.acquisition_check_and_disable():
                        continue
                    self.change_pixelformat()

                elif var[0] == "exit":
                    break
                else:
                    Logbook.log_ERRO(f"Unrecognized command: `{var[0]}`")
                    self.print_help()
        finally:
            # make sure to always stop the acquisition_thread, otherwise
            # we'd hang, e.g. on KeyboardInterrupt
            self._camera.killed = True
            self.acquisition_thread.join()

    def start_window(self):
        pass

    def on_image_received(self, image):
        pass

    def warning(self, message: str):
        Logbook.log_WARN(f"Warning: {message}")

    def information(self, message: str):
        Logbook.log_INFO(f"Info: {message}")