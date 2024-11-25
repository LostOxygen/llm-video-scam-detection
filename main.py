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

from videoscam_pipeline.colors import TColors
from videoscam_pipeline.pipeline import ScamPipeline

# setting the warnings to ignore
warnings.filterwarnings("ignore")

STD_VIDEO_PATH: Final[str] = "videoscam_pipeline/video_files/"
STD_AUDIO_PATH: Final[str] = "videoscam_pipeline/audio_files/"
STD_OUTPUT_PATH: Final[str] = "videoscam_pipeline/output_files/"
YOUTUBE_DATA: Final[str] = "datasets/youtube.json"
TIKTOK_DATA: Final[str] = "datasets/tiktok.txt"
TEST_VIDEO_URL: Final[str] = "https://www.youtube.com/shorts/WApQyL-GT1k"

YDL_OPTS: Final[dict] = {
    "format": "mp4",
    # pylint: disable=line-too-long
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36",
    "fragment_requests": 100,  # try increasing this
    "outtmpl": "videoscam_pipeline/video_files/test_video.mp4",
    "output_namplate": "test_video.mp4",
}

def main(device: str, video_path: str, audio_path: str, output_path: str) -> None:
    """
    Main function to start the videoscam detection.

    Parameters:
        device: str - the device to run the computation on (cpu, cuda, mps)
        video_path: str - the path to the video files
        audio_path: str - the path to the audio files
        output_path: str - the path to the output dump files

    Returns:
        None
    """
    # load the po token and visitor data from file
    try:
        with open(file="po_token.txt", mode="r", encoding="utf-8") as f:
            token = f.read().replace("\n", "")
            assert token != "", f"{TColors.FAIL}PO token file is empty.{TColors.ENDC}"

            os.environ["PO_TOKEN"] = token
            print(f"{TColors.OKGREEN}PO token loaded.{TColors.ENDC}")

    except FileNotFoundError:
        print(f"{TColors.FAIL}Please paste your PO token into the po_token.txt file to prevent " \
               "youtube from blocking this script. See: " \
              f"https://github.com/JuanBindez/pytubefix/pull/209{TColors.ENDC}")

    try:
        with open(file="visitor_data.txt", mode="r", encoding="utf-8") as f:
            data = f.read().replace("\n", "")
            assert data != "", f"{TColors.FAIL}PO token file is empty.{TColors.ENDC}"

            os.environ["VISITOR_DATA"] = data
            print(f"{TColors.OKGREEN}Visitor data loaded.{TColors.ENDC}")

    except FileNotFoundError:
        print(f"{TColors.FAIL}Please paste your visitor data into the visitor_data.txt file to " \
               "prevent youtube from blocking this script. See: " \
              f"https://github.com/JuanBindez/pytubefix/pull/209{TColors.ENDC}")


    # set the devices correctly
    if device == "cpu":
        device = torch.device("cpu")
    elif device == "cuda" and torch.cuda.is_available():
        device = torch.device(device)
    elif device == "mps" and torch.backends.mps.is_available():
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
    if (device == "cuda" or torch.device("cuda")) and torch.cuda.is_available():
        print(f"## {TColors.OKBLUE}{TColors.BOLD}GPU Memory{TColors.ENDC}: " \
              f"{torch.cuda.mem_get_info()[1] // 1024**2} MB")
    elif (device == "mps" or torch.device("mps")) and torch.backends.mps.is_available():
        print(f"## {TColors.OKBLUE}{TColors.BOLD}Shared Memory{TColors.ENDC}: " \
              f"{psutil.virtual_memory()[0] // 1024**2} MB")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}CPU Memory{TColors.ENDC}: " \
            f"{psutil.virtual_memory()[0] // 1024**2} MB")
    print(f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Parameters" + \
          f"{TColors.ENDC} " + "#"*(os.get_terminal_size().columns-14))
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Video Path{TColors.ENDC}: {video_path}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Audio Path{TColors.ENDC}: {audio_path}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Output Path{TColors.ENDC}: {output_path}")
    print("#"*os.get_terminal_size().columns+"\n")


    # create the folders necessary
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    if not os.path.exists(audio_path):
        os.makedirs(audio_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # create the pipeline
    pipeline = ScamPipeline(
        video_file_path=video_path,
        audio_file_path=audio_path,
        output_path=output_path,
        video_url_data=YOUTUBE_DATA,
        device=device
    )

    # run the pipeline
    pipeline.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="videoscam-detection")
    parser.add_argument("--device", "-dx", type=str, default="cpu",
        help="specifies the device to run the computations on (cpu, cuda, mps)")
    parser.add_argument("--video_path", "-vp", type=str, default=STD_VIDEO_PATH,
        help="specifies the path to the video files")
    parser.add_argument("--audio_path", "-ap", type=str, default=STD_AUDIO_PATH,
        help="specifies the path to the audio files")
    parser.add_argument("--output_path", "-op", type=str, default=STD_OUTPUT_PATH,
        help="specifies the path to the output dump files")
    args = parser.parse_args()
    main(**vars(args))
