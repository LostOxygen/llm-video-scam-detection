"""main hook to start the video scam detection"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import os
import getpass
import psutil
import argparse
import datetime
import torch
from typing import Final
import warnings

import yt_dlp

from videoscam_pipeline.colors import TColors
from videoscam_pipeline.pipeline import ScamPipeline

# setting the warnings to ignore
warnings.filterwarnings("ignore")

VIDEO_PATH: Final[str] = "videoscam_pipeline/video_files/"
AUDIO_PATH: Final[str] = "videoscam_pipeline/audio_files/"
TEST_VIDEO_URL: Final[str] = "https://www.youtube.com/watch?v=2paOYObEhoA"

YDL_OPTS: Final[dict] = {
    "format": "mp4",
    # pylint: disable=line-too-long
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36",
    "fragment_requests": 100,  # try increasing this
    "outtmpl": "videoscam_pipeline/video_files/test_video.mp4",
    "output_namplate": "test_video.mp4",
}

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


    # create the folders necessary
    if not os.path.exists(VIDEO_PATH):
        os.makedirs(VIDEO_PATH)
    if not os.path.exists(AUDIO_PATH):
        os.makedirs(AUDIO_PATH)

    # download the youtube test video
    with yt_dlp.YoutubeDL(YDL_OPTS) as ydl:
        ydl.download([TEST_VIDEO_URL])

    # create the pipeline
    pipeline = ScamPipeline(device=device)

    # run the pipeline
    pipeline.run(
        video_file_path=VIDEO_PATH+"test_video.mp4",
        audio_file_path=AUDIO_PATH+"test_audio.wav",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="videoscam-detection")
    parser.add_argument("--device", "-dx", type=str, default="cpu",
        help="specifies the device to run the computations on (cpu, cuda, mps)")
    args = parser.parse_args()
    main(**vars(args))
