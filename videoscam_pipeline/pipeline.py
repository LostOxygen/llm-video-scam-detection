"""Video Scam Pipeline Class Implementation"""
import os
import json
import whisper
import av
import torch
import numpy as np
from audio_extract import extract_audio
from transformers import (
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoProcessor,
)

from pytubefix import YouTube
from pytubefix.cli import on_progress

from .colors import TColors

class ScamPipeline:
    """Video Scam Pipeline Class"""

    def __init__(self, device: str, video_file_path: str, audio_file_path: str) -> None:
        """Initialize the pipeline with the given configuration"""
        self.device: str = device
        self.whisper_model = None
        self.llava_model = None
        self.llava_processor = None
        self.video_file_path: str = video_file_path
        self.audio_file_path: str = audio_file_path

    def run(self) -> None:
        """
        Run the pipeline
        
        Parameters:
            None
        
        Returns:
            None
        """

        # transcribe and summarize the video
        self.__load_whisper_model()
        self.__load_llava_model()

        # go through every folder in the video file path and process every video
        for account in os.listdir(self.video_file_path):
            account_folder = os.path.join(self.video_file_path, account)
            if os.path.isdir(account_folder):
                for video in os.listdir(account_folder):
                    # set paths for video and audio file
                    video_file = os.path.join(account_folder, video)

                    video_id = video.split("/")[-1]
                    audio_file = os.path.join(self.audio_file_path, video_id)

                    # summarize the video
                    video = self.__read_video_av(video_file)
                    summary = self.__summarize_video(video)

                    # extract audio and transcribe it
                    print(f"audio file: {audio_file}")
                    self.__extract_audio_from_video(video_file, audio_file)
                    transcription = self.__transcribe_audio_file(audio_file)

                    # print results
                    print(f"\n{TColors.HEADER}Transcription{TColors.ENDC}: {transcription}\n")
                    print(f"\n{TColors.HEADER}Video Summary{TColors.ENDC}: {summary}\n")

        self.__delete_whisper_model()
        self.__delete_llava_model()

    def __load_whisper_model(self) -> None:
        """Load the whisper model"""
        self.whisper_model = whisper.load_model(
            name="large",
        )


    def __delete_whisper_model(self) -> None:
        """Delete the whisper model for free memory and cache"""
        del self.whisper_model
        self.whisper_model = None
        torch.cuda.empty_cache()

    def __load_llava_model(self) -> None:
        # enabling apple silicon mps fix
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        self.llava_model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            "llava-hf/LLaVA-NeXT-Video-7B-hf",
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.llava_processor = LlavaNextVideoProcessor.from_pretrained(
            "llava-hf/LLaVA-NeXT-Video-7B-hf",
        )

    def __delete_llava_model(self) -> None:
        """Delete the llava model for free memory and cache"""
        del self.llava_model
        del self.llava_processor
        self.llava_processor = None
        self.llava_model = None
        torch.cuda.empty_cache()

    def __extract_audio_from_video(self, video_path: str, audio_path: str) -> None:
        """Extract audio from video"""
        extract_audio(
            input_path=video_path,
            output_path=audio_path,
            output_format="wav",
            overwrite=True,
        )

    def __transcribe_audio_file(self, audio_path: str) -> str:
        """Transcribe the audio file using whisper. Returns transcription"""
        transcription = self.whisper_model.transcribe(audio_path)["text"]
        print("[transcriber] Audio transcription successful.")
        return transcription

    def __summarize_video(self, video: np.array) -> str:
        """Summarize the video using llava model"""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize the video."},
                    {"type": "video"},
                ],
            },
        ]

        prompt = self.llava_processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )

        inputs = self.llava_processor(
            text=prompt,
            videos=video,
            return_tensors="pt"
        ).to(self.device)

        out = self.llava_model.generate(**inputs, max_new_tokens=4096)

        summary = self.llava_processor.batch_decode(
            out,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        # cut off the prompt from the summary
        summary = summary[0].split("ASSISTANT: ")[1].strip()
        print("[summarizer] Video summarization successful.")
        return summary

    def __read_video_av(self, video_file_path) -> np.array:
        """
        Decode the video with PyAV decoder.

        Parameters:
            None
        
        Returns:
            np.array: The decoded frames. Shape: (num_frames, height, width, 3)
        """
        # load the video file
        container = av.open(video_file_path)
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)

        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        result = np.stack([x.to_ndarray(format="rgb24") for x in frames])
        print("[decoder] Video decoding successful.")
        return result


    def download_youtube_videos(self, video_url_data: dict[tuple[str, str], list[str]]) -> None:
        """
        Downloads all youtube urls from a list of video urls.

        Parameters:
            video_urls (list[str]): List of video urls to download

        Returns:
            None
        """
        print(f"{TColors.HEADER}[INFO]{TColors.ENDC} Download youtube videos")

        # load the json data
        with open(video_url_data, "r", encoding="utf-8") as f:
            video_url_data = json.load(f)
        # every youtube tuple consists of the accountname and a list of video urls
        for youtube_tuple in video_url_data:
            if "channel" not in youtube_tuple["account"]:
                account_name = youtube_tuple["account"].split("https://www.youtube.com/")[-1]
            else:
                account_name = youtube_tuple["account"].split("channel/")[-1].replace("/", "")
            print(f"{TColors.HEADER}[INFO]{TColors.ENDC} Account name: {account_name}")
            video_urls = youtube_tuple["channels"]

            # create the account folder
            account_folder = os.path.join(self.video_file_path, account_name)
            if not os.path.exists(account_folder):
                os.makedirs(account_folder)

            # download the videos
            for url in video_urls:
                try:
                    yt = YouTube(url, on_progress_callback = on_progress)
                    video_title = yt.title
                    ys = yt.streams.get_highest_resolution()
                    ys.download(output_path=account_folder, filename=video_title+".mp4")
                except Exception as e:
                    print(f"{TColors.FAIL}[ERROR]{TColors.ENDC} Could not download video: {e}")
