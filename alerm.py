
from dotenv import load_dotenv
load_dotenv()
from utils import * 


# Example usage

if __name__ == "__main__":
    detector = IntrusionDetector(
        video_path="test_videos/thief_video4.mp4",
        model_path="yolo11n.pt",
        alarm_path="Alarm/alarm.wav",
        from_email=os.getenv("EMAIL_SENDER"),
        email_password=os.getenv("EMAIL_PASSWORD"),
        to_email=os.getenv("EMAIL_RECEIVER"),
        target_classes=[],
        device_list=[0]
    )
    detector.detect()
