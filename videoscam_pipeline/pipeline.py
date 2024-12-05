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
from sentence_transformers import SentenceTransformer
from lingua import LanguageDetectorBuilder, Language
from bertopic import BERTopic

from pytubefix import YouTube
from pytubefix.cli import on_progress

from .colors import TColors

class ScamPipeline:
    """Video Scam Pipeline Class"""

    def __init__(self,
        device: str,
        video_file_path: str,
        audio_file_path: str,
        output_path: str,
        video_url_data: str,
    ) -> None:
        """Initialize the pipeline with the given configuration"""
        self.device: str = device
        self.language_detector = None
        self.whisper_model = None
        self.topic_model = None
        self.llava_model = None
        self.llava_processor = None
        self.download_videos = True
        self.video_file_path: str = video_file_path
        self.audio_file_path: str = audio_file_path
        self.embedding_model = None
        self.output_path: str = output_path
        self.video_url_data: str = video_url_data

        # init the models
        self.__load_whisper_model()
        self.__load_llava_model()
        self.__load_embedding_model()
        self.__load_topic_model()
        self.__load_language_detector()

    def run(self, download: bool) -> None:
        """
        Run the pipeline
        
        Parameters:
            download: bool: True if the videos should be downloaded, False otherwise
        
        Returns:
            None
        """

        # transcribe and summarize the video
        # load the json data
        with open(self.video_url_data, "r", encoding="utf-8") as f:
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

            # process each video of a channel
            for url in video_urls:
                # download the video (skip if the video was not downloaded successfully)
                video_downloaded, vid_title = self.__download_youtube_video(
                    url,
                    account_folder,
                    download
                )
                if not video_downloaded:
                    continue

                # check if the video is already summarized
                if os.path.exists(os.path.join(self.output_path, f"{vid_title}.json")):
                    print(f"{TColors.WARNING}[WARNING]{TColors.ENDC} Video already summarized")
                    continue

                # set paths for video and audio file
                video_file = os.path.join(account_folder, vid_title+".mp4")

                video_id = video_file.split("/")[-1].replace(".mp4", ".wav")
                audio_file = os.path.join(self.audio_file_path, video_id)

                # read the video av
                try:
                    video = self.__read_video_av(video_file)
                except Exception as e:
                    print(f"{TColors.FAIL}[ERROR]{TColors.ENDC} Could not read video: {e}")
                    continue

                # summarize the video
                try:
                    summary = self.__summarize_video(video)
                except Exception as e:
                    print(f"{TColors.FAIL}[ERROR]{TColors.ENDC} Could not summarize video: {e}")
                    continue

                # extract audio
                try:
                    self.__extract_audio_from_video(video_file, audio_file)
                except Exception as e:
                    print(f"{TColors.FAIL}[ERROR]{TColors.ENDC} Could not extract audio: {e}")
                    continue

                # transcribe audio
                try:
                    transcription = self.__transcribe_audio_file(audio_file)
                except Exception as e:
                    print(f"{TColors.FAIL}[ERROR]{TColors.ENDC} Could not transcribe audio: {e}")
                    continue

                # print and save summary and transcription results
                print(f"\n{TColors.HEADER}Transcription{TColors.ENDC}: {transcription}\n")
                print(f"\n{TColors.HEADER}Video Summary{TColors.ENDC}: {summary}\n")
                self.__dump_data(
                    video_summary=summary,
                    transcription=transcription,
                    video_name=vid_title,
                    video_url=url,
                    channel_url=youtube_tuple["account"],
                    file_path=os.path.join(self.output_path, f"{vid_title}.json"),
                )

        # cluster the video informations, iterating over every output file
        information_list = []
        for file in os.listdir(self.output_path):
            with open(os.path.join(self.output_path, file), "r", encoding="utf-8") as f:
                video_data = json.load(f)
                summary = video_data["video_summary"]
                transcript = video_data["transcription"]
                information = summary + transcript

                # filter the summarizations for english language only using CLD2
                language = self.__get_information_language(information)

                if language != Language.ENGLISH:
                    continue

                information_list.append(information)

        # create the topic clustering with BERTopic
        topics, _ = self.topic_model.fit_transform(information_list)
        print(f"{TColors.OKBLUE}[Topic Model]{TColors.ENDC} Topics: {topics}")

    def __del__(self) -> None:
        self.__delete_whisper_model()
        self.__delete_llava_model()
        self.__delete_embedding_model()
        self.__delete_language_detector()
        self.__delete_topic_model()


    def __dump_data(
        self,
        file_path: str,
        video_summary: str,
        transcription: str,
        video_name: str,
        video_url: str,
        channel_url: str,
    ) -> None:
        """
        Dump the transcription, summary, and other relevant data of a video to a json file
        
        Parameters:
            file_path str: the path to the json file
            video_summary str: the summary of the video
            transcription str: the transcription of the video
            video_name str: the name of the video
            video_url str: the url of the video
            channel_url str: the url of the channel

        Returns:
            None
        """
        data = {
            "channel_url": channel_url,
            "video_name": video_name,
            "video_url": video_url,
            "video_summary": video_summary,
            "transcription": transcription,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)


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


    def __load_language_detector(self) -> None:
        """Load the language detector model"""
        self.language_detector = LanguageDetectorBuilder \
            .from_all_languages() \
            .with_preloaded_language_models() \
            .build()


    def __delete_language_detector(self) -> None:
        """Delete the language detector model for free memory and cache"""
        del self.language_detector
        self.language_detector = None
        torch.cuda.empty_cache()


    def __load_topic_model(self) -> None:
        """Load the topic model"""
        self.topic_model = BERTopic()


    def __delete_topic_model(self) -> None:
        """Delete the topic model for free memory and cache"""
        del self.topic_model
        self.topic_model = None
        torch.cuda.empty_cache()


    def __load_llava_model(self) -> None:
        "Load the llava vision-llm model"
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


    def __load_embedding_model(self) -> None:
        "Load the mpnet-base-v2 embedding model"
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
        self.embedding_model = self.embedding_model.to(self.device)


    def __delete_embedding_model(self) -> None:
        """Delete the llava model for free memory and cache"""
        del self.embedding_model
        self.embedding_model = None
        torch.cuda.empty_cache()


    def __extract_audio_from_video(self, video_path: str, audio_path: str) -> None:
        """Extract audio from video"""
        extract_audio(
            input_path=video_path,
            output_path=audio_path,
            output_format="wav",
            overwrite=True,
        )


    def __get_information_language(self, text) -> Language:
        """Get the language of the text using CLD2"""
        return self.language_detector.detect_language_of(text)


    # pylint: disable=unused-private-member
    def __get_information_embedding(self, text) -> torch.tensor:
        """Get the embedding of the text using the embedding model"""
        return self.embedding_model.encode(text)


    def __transcribe_audio_file(self, audio_path: str) -> str:
        """Transcribe the audio file using whisper. Returns transcription"""
        transcription = self.whisper_model.transcribe(audio_path)["text"]
        print(f"{TColors.OKBLUE}[Transcriber]{TColors.ENDC} Audio transcription successful.")
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
        print(f"{TColors.OKBLUE}[Summarizer]{TColors.ENDC} Video summarization successful.")
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
        print(f"{TColors.OKBLUE}[Decoder]{TColors.ENDC} Video decoding successful.")
        return result


    def __download_youtube_video(self, video_url: str, account_folder: str, download: bool) -> bool:
        """
        Downloads the youtube video from the given url and saves it to the account folder

        Parameters:
            video_url str: url of the video to download
            account_folder str: the folder to save the video
            download bool: True if the video should be downloaded, False otherwise

        Returns:
            tuple[bool, str]: True if the video was downloaded successfully, False otherwise
                Video title if the video was downloaded successfully, "error downloading" otherwise
        """
        print(f"{TColors.HEADER}[INFO]{TColors.ENDC} Download youtube video: {video_url}")

        # download the video
        try:
            yt = YouTube(
                video_url,
                on_progress_callback=on_progress,
                use_po_token=True,
                allow_oauth_cache=True,
                po_token_verifier=self.__load_po_tokens,
                )
            video_title = yt.title

            if not download:
                return True, video_title

            # check if the video is already downloaded
            if os.path.exists(os.path.join(account_folder, video_title+".mp4")):
                print(f"{TColors.WARNING}[WARNING]{TColors.ENDC} Video already " \
                      f"downloaded: {video_title}")
                return True, video_title

            ys = yt.streams.get_highest_resolution()
            ys.download(output_path=account_folder, filename=video_title+".mp4")
            return True, video_title
        except Exception as e:
            if not download:
                return True, str(e) + " (use cached version)"
            print(f"{TColors.FAIL}[ERROR]{TColors.ENDC} Could not download video: {e}")
            return False, "error downloading"


    def __load_po_tokens(self) -> tuple[str, str]:
        """
        Load the po tokens from the environment variables and return as 
        a tuple: tuple[visitorData: str, po_token: str]
        """
        visitor_data = os.environ.get("VISITOR_DATA")
        po_token = os.environ.get("PO_TOKEN")
        return visitor_data, po_token
