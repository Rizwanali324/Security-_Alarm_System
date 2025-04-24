import cv2
import torch
import numpy as np
import pygame
import os
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from ultralytics import YOLO

class IntrusionDetector:
    def __init__(self, video_path, model_path, alarm_path,
                 results_folder="Detected Photos", number_of_photos=3,
                 from_email=None, email_password=None, to_email=None,
                 target_classes=None, device_list=None):
        self.video_path = video_path
        self.model_path = model_path
        self.alarm_path = alarm_path
        self.results_folder = results_folder
        self.number_of_photos = number_of_photos
        self.target_classes = target_classes or ['person']
        self.pts = []
        self.count = 0
        self.email_sent = False

        # Email
        self.from_email = from_email
        self.email_password = email_password
        self.to_email = to_email
        self.smtp_server = None

                # Device handling with device print
        print("🔍 Available devices:")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  ✅ CUDA:{i} - {torch.cuda.get_device_name(i)}")
            if device_list:
                self.device = f"cuda:{device_list[0]}"
            else:
                self.device = "cuda:0"
        else:
            print("  ⚠️ No CUDA devices found. Using CPU.")
            self.device = "cpu"

        print(f"🚀 Using device: {self.device}")

        os.makedirs(self.results_folder, exist_ok=True)
        self._init_alarm()
        self._init_email()

        self.model = YOLO(self.model_path).to(self.device)
        self.cap = cv2.VideoCapture(self.video_path)

        cv2.namedWindow('Video')
        cv2.setMouseCallback('Video', self.draw_polygon)

    def _init_alarm(self):
        pygame.init()
        pygame.mixer.music.load(self.alarm_path)

    def _init_email(self):
        if self.from_email and self.email_password and self.to_email:
            try:
                self.smtp_server = smtplib.SMTP("smtp.gmail.com", 587)
                self.smtp_server.starttls()
                self.smtp_server.login(self.from_email, self.email_password)
                print("✅ Email authenticated successfully.")
            except Exception as e:
                print(f"❌ Failed to authenticate email: {e}")

    def draw_polygon(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"📍 Point added: ({x}, {y})")
            self.pts.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            print("❌ Polygon reset")
            self.pts = []

    def inside_polygon(self, point, polygon):
        return cv2.pointPolygonTest(polygon, (point[0], point[1]), False) >= 0

    def preprocess(self, img):
        height, width = img.shape[:2]
        ratio = height / width
        return cv2.resize(img, (640, int(640 * ratio)))

    def send_email_alert(self, image_path, records=1):
        if self.smtp_server is None or self.email_sent:
            return

        with open(image_path, "rb") as img_file:
            img_data = img_file.read()

        message = MIMEMultipart()
        message["From"] = self.from_email
        message["To"] = self.to_email
        message["Subject"] = "🚨 Intrusion Alert!"

        body = f"{records} suspicious object(s) detected in restricted zone!"
        message.attach(MIMEText(body, "plain"))
        message.attach(MIMEImage(img_data, name=os.path.basename(image_path)))

        try:
            self.smtp_server.send_message(message)
            self.email_sent = True
            print("📧 Alert email sent!")
        except Exception as e:
            print(f"❌ Failed to send email: {e}")

    def detect(self):
        # STEP 1: Show first frame to draw polygon
        ret, first_frame = self.cap.read()
        if not ret:
            print("❌ Could not read from video.")
            return

        print("🖱️ Draw polygon with LEFT click, RIGHT click to reset. Press ENTER to start detection.")

        while True:
            display_frame = first_frame.copy()
            if len(self.pts) > 1:
                cv2.polylines(display_frame, [np.array(self.pts)], isClosed=False, color=(0, 255, 255), thickness=2)

            for pt in self.pts:
                cv2.circle(display_frame, pt, 5, (255, 0, 0), -1)

            cv2.imshow("Video", display_frame)
            key = cv2.waitKey(1)
            if key == 13:  # ENTER key
                if len(self.pts) >= 4:
                    print("✅ Polygon defined. Starting detection...")
                    break
                else:
                    print("⚠️ At least 4 points needed.")

        # Restart capture to avoid skipping frames
        self.cap = cv2.VideoCapture(self.video_path)

        # STEP 2: Start detection loop
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_detected = frame.copy()
            frame = self.preprocess(frame)
            results = self.model(frame)

            for result in results:
                for box in result.boxes.data:
                    x1, y1, x2, y2, conf, cls = map(int, box.tolist()[:6])
                    name = result.names[int(cls)]

                    if name in self.target_classes:
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        if len(self.pts) >= 4:
                            polygon_np = np.array(self.pts)

                            if self.inside_polygon((center_x, center_y), polygon_np) and name == 'person':
                                if self.count < self.number_of_photos:
                                    save_path = os.path.join(self.results_folder, f"snapshot{self.count}.jpg")
                                    cv2.imwrite(save_path, frame_detected)
                                    self.count += 1

                                    if self.count == 1:
                                        self.send_email_alert(save_path, records=self.count)

                                if not pygame.mixer.music.get_busy():
                                    pygame.mixer.music.play()

                                cv2.putText(frame, "Target", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                cv2.putText(frame, "Person Detected", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # Draw polygon zone
            if len(self.pts) >= 4:
                cv2.polylines(frame, [np.array(self.pts)], isClosed=True, color=(0, 0, 255), thickness=2)

            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
