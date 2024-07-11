"""main hook to start the video scam detection"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import os
import getpass
import psutil
import argparse
from pathlib import Path
import datetime
import torch

from videoscam_pipeline.colors import TColors

def main(device: str) -> None:
    """
    Main function to start the videoscam detection.

    Parameters:
        device: str - the device to run the computation on (cpu, cuda, mps)

    Returns:
        None
    """

    # set the devices correctly
    if device == "cpu":
        device = torch.device("cpu")
    elif device != "cpu" and device == "cuda" and torch.cuda.is_available():
        device = torch.device(device)
    elif device != "cpu" and device == "mps" and torch.backends.mps.is_available():
        device = torch.device(device)
    else:
        print(f"{TColors.WARNING}Warning{TColors.ENDC}: Device {TColors.OKCYAN}{device} " \
              f"{TColors.ENDC}is not available. Setting device to CPU instead.")
        device = torch.device("cpu")


    print("\n"+f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}System Information" + \
          f"{TColors.ENDC} " + "#"*(os.get_terminal_size().columns-23))
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Date{TColors.ENDC}: " + \
          str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print(f"## {TColors.OKBLUE}{TColors.BOLD}System{TColors.ENDC}: " \
          f"{torch.get_num_threads()} CPU cores with {os.cpu_count()} threads and " \
          f"{torch.cuda.device_count()} GPUs on user: {getpass.getuser()}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Device{TColors.ENDC}: {device}")
    if device == "cuda" and torch.cuda.is_available():
        print(f"## {TColors.OKBLUE}{TColors.BOLD}GPU Memory{TColors.ENDC}: " \
              f"{torch.cuda.mem_get_info()[1] // 1024**2} MB")
    elif device == "mps" and torch.backends.mps.is_available():
        print(f"## {TColors.OKBLUE}{TColors.BOLD}Shared Memory{TColors.ENDC}: " \
              f"{psutil.virtual_memory()[0] // 1024**2} MB")
    else:
        print(f"## {TColors.OKBLUE}{TColors.BOLD}CPU Memory{TColors.ENDC}: " \
              f"{psutil.virtual_memory()[0] // 1024**2} MB")
    print(f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Parameters" + \
          f"{TColors.ENDC} " + "#"*(os.get_terminal_size().columns-14))
    

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="videoscam-detection")
    parser.add_argument("--device", "-dx", type=str, default="cpu",
        help="specifies the device to run the computations on (cpu, cuda, mps)")
    args = parser.parse_args()
    main(**vars(args))
