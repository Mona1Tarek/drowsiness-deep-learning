import cv2
import numpy as np
import pygame
import threading
import tensorflow as tf
from mainMediapipe_samples import FaceMeshDetector

# Initialize shared camera capture object
shared_cap = cv2.VideoCapture(0)
shared_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
shared_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Patch FaceMeshDetector to use the shared capture object
original_init = FaceMeshDetector.__init__
def patched_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    if hasattr(self, 'cap'):
        try:
            self.cap.release()
        except Exception:
            pass
    self.cap = shared_cap
FaceMeshDetector.__init__ = patched_init

class DrowsinessDetector:
    def __init__(self, model_path):
        self.face_mesh_detector = FaceMeshDetector()
        self.count = 0
        self.alarm_threshold = 30  # Number of frames before triggering alarm
        self.alarm_playing = False
        self.cap = shared_cap

        # Initialize alarm sound
        pygame.mixer.init()
        self.alarm_sound = pygame.mixer.Sound("alarm2.mp3")

        # Load TensorFlow Lite model
        self.model = tf.lite.Interpreter(model_path=model_path)
        self.model.allocate_tensors()

        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    def extract_eye_region(self, image, eye_coords):
        if not eye_coords or len(eye_coords) < 4:
            return None
        
        xs = [x for x, y in eye_coords]
        ys = [y for x, y in eye_coords]
        x_min, x_max = max(0, min(xs)), min(image.shape[1], max(xs))
        y_min, y_max = max(0, min(ys)), min(image.shape[0], max(ys))

        if x_min >= x_max or y_min >= y_max:
            return None

        return cv2.cvtColor(image[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)

    def preprocess_eye(self, eye_image):
        if eye_image is None or eye_image.size == 0:
            return None

        eye_image_resized = cv2.resize(eye_image, (224, 224))
        eye_image_normalized = eye_image_resized.astype(np.float32) / 255.0

        # Convert grayscale (1 channel) to 3-channel format
        eye_image_colored = np.stack([eye_image_normalized] * 3, axis=-1)

        return np.expand_dims(eye_image_colored, axis=0)  # Shape becomes (1, 224, 224, 3)


    def predict_eye_state(self, eye_image):
        if eye_image is None:
            return 1

        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()

        preprocessed_eye = self.preprocess_eye(eye_image)
        if preprocessed_eye is None:
            return 1

        self.model.set_tensor(input_details[0]['index'], preprocessed_eye)
        self.model.invoke()

        output_data = self.model.get_tensor(output_details[0]['index'])
        return output_data[0][0]

    def play_alarm(self):
        if not self.alarm_playing:
            self.alarm_playing = True
            self.alarm_sound.play(loops=-1)

    def stop_alarm(self):
        if self.alarm_playing:
            self.alarm_playing = False
            self.alarm_sound.stop()

    def process_frame(self, image):
        results = self.face_mesh_detector.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = image.shape
                left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in self.LEFT_EYE_INDICES]
                right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in self.RIGHT_EYE_INDICES]

                for (x, y) in left_eye + right_eye:
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

                left_eye_image = self.extract_eye_region(image, left_eye)
                right_eye_image = self.extract_eye_region(image, right_eye)

                left_eye_state = self.predict_eye_state(left_eye_image)
                right_eye_state = self.predict_eye_state(right_eye_image)

                if left_eye_state < 0.5 and right_eye_state < 0.5:
                    self.count += 1
                else:
                    self.count = max(0, self.count - 1)
                    self.stop_alarm()

                if self.count >= self.alarm_threshold:
                    cv2.putText(image, "DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    threading.Thread(target=self.play_alarm, daemon=True).start()
        else:
            self.count = 0
            self.stop_alarm()
        
        return image

    def run(self):
        while True:
            success, image = self.cap.read()
            if not success:
                continue

            image = cv2.flip(image, 1)
            processed_image = self.process_frame(image)

            cv2.imshow("Drowsiness Detection", processed_image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.stop_alarm()

if __name__ == "__main__":
    detector = DrowsinessDetector(model_path="optimized_eye_detection_model.tflite")
    detector.run()
