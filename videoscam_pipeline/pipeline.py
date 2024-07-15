"""Video Scam Pipeline Class Implementation"""
import os
import whisper
import av
import torch
import numpy as np
from audio_extract import extract_audio
from transformers import (
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoProcessor,
)

from .colors import TColors

class ScamPipeline:
    """Video Scam Pipeline Class"""

    def __init__(self, device: str) -> None:
        """Initialize the pipeline with the given configuration"""
        self.device: str = device
        self.whisper_model = None
        self.llava_model = None
        self.llava_processor = None

    def run(self, video_file_path: str, audio_file_path: str) -> None:
        """
        Run the pipeline
        
        Parameters:
            video_file_path (str): The path to the video file.
            audio_file_path (str): The path to the audio file.
        
        Returns:
            None
        """
        # check if video file exists and load the video file
        if not os.path.exists(video_file_path):
            raise FileNotFoundError(
                f"{TColors.FAIL}Error{TColors.ENDC}: " \
                f"Video file {video_file_path} not found."
            )

        # transcribe the video
        self.__load_whisper_model()
        self.__extract_audio_from_video(video_file_path, audio_file_path)
        transcription = self.__transcribe_audio_file(audio_file_path)
        self.__delete_whisper_model()

        # summarize the video
        self.__load_llava_model()
        video = self.__read_video_av(video_file_path)
        summary = self.__summarize_video(video)
        self.__delete_llava_model()

        # print results
        print(f"\n{TColors.HEADER}Transcription{TColors.ENDC}: {transcription}\n")
        print(f"\n{TColors.HEADER}Video Summary{TColors.ENDC}: {summary}\n")

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
