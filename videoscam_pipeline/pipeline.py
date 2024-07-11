"""Video Scam Pipeline Class Implementation"""
import os

from .colors import TColors

class ScamPipeline:
    """Video Scam Pipeline Class"""

    def __init__(self, device, video_file_path) -> None:
        """Initialize the pipeline with the given configuration"""
        self.device: str = device
        self.video_file_path: str = video_file_path

        # check if video file exists and load the video file
        if not os.path.exists(self.video_file_path):
            raise FileNotFoundError(
                f"{TColors.FAIL}Error{TColors.ENDC}: " \
                f"Video file {self.video_file_path} not found."
            )

    def run(self):
        """Run the pipeline"""
        pass


    def extract_audio_from_video(self):
        """Extract audio from video"""
        pass
