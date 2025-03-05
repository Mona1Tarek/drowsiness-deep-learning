import cv2
import numpy as np
import pygame
import threading
import tensorflow as tf
from mainMediapipe_samples import FaceMeshDetector

# Initialize shared camera capture object
shared_cap = cv2.VideoCapture(0)

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
        self.alarm_threshold = 20  # Number of frames before triggering alarm
        self.alarm_playing = False

        self.cap = shared_cap

        # Initialize alarm sound
        pygame.mixer.init()
        self.alarm_sound = pygame.mixer.Sound("alarm2.mp3")

        # Load TensorFlow Lite model
        self.model = tf.lite.Interpreter(model_path=model_path)
        self.model.allocate_tensors()

        # Corrected eye landmark indices for refined face mesh
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # Left eye indices for refined mesh
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # Right eye indices for refined mesh

    def extract_eye_region(self, image, eye_coords):
        """Safely extracts an eye region from the image."""
        if not eye_coords or len(eye_coords) < 4:
            return None  # Invalid eye coordinates

        # Get bounding box from eye coordinates
        xs = [x for x, y in eye_coords]
        ys = [y for x, y in eye_coords]
        x_min, x_max = max(0, min(xs)), min(image.shape[1], max(xs))
        y_min, y_max = max(0, min(ys)), min(image.shape[0], max(ys))

        if x_min >= x_max or y_min >= y_max:
            return None  # Invalid region

        return image[y_min:y_max, x_min:x_max]

    def preprocess_eye(self, eye_image):
        """Prepares the eye image for model input."""
        if eye_image is None or eye_image.size == 0:
            print("Warning: Empty eye image, skipping frame")
            return None

        eye_image_resized = cv2.resize(eye_image, (224, 224))  # Resize to model's expected input size
        eye_image_normalized = eye_image_resized / 255.0  # Normalize pixel values
        return np.expand_dims(eye_image_normalized.astype(np.float32), axis=0)  # Convert to FLOAT32 and add batch dimension


    def predict_eye_state(self, eye_image):
        """Runs inference on an eye image and predicts its state."""
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()

        preprocessed_eye = self.preprocess_eye(eye_image)
        if preprocessed_eye is None:
            return 1  # Assume eye is open if processing fails

        self.model.set_tensor(input_details[0]['index'], preprocessed_eye)
        self.model.invoke()

        output_data = self.model.get_tensor(output_details[0]['index'])
        return output_data[0][0]  # 1 = Open, 0 = Closed

    def play_alarm(self):
        """Plays an alarm in a separate thread."""
        if not self.alarm_playing:
            self.alarm_playing = True
            self.alarm_sound.play(loops=-1)  # Loop indefinitely

    def stop_alarm(self):
        """Stops the alarm if it's playing."""
        if self.alarm_playing:
            self.alarm_playing = False
            self.alarm_sound.stop()

    def get_eye_keypoints(self, face_landmarks, image_shape):
        """Extracts eye keypoints using the correct indices for refined face mesh."""
        h, w, _ = image_shape
        eye_keypoints = {
            "left_eye": [],
            "right_eye": []
        }

        # Extract left eye keypoints
        for idx in self.LEFT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            eye_keypoints["left_eye"].append((int(landmark.x * w), int(landmark.y * h)))

        # Extract right eye keypoints
        for idx in self.RIGHT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            eye_keypoints["right_eye"].append((int(landmark.x * w), int(landmark.y * h)))

        return eye_keypoints

    def process_frame(self, image):
        """Processes a single frame to detect drowsiness."""
        results = self.face_mesh_detector.face_mesh.process(image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Use the correct eye keypoints for refined face mesh
                eye_keypoints = self.get_eye_keypoints(face_landmarks, image.shape)

                # Check if eye keypoints are valid
                if not eye_keypoints["left_eye"] or not eye_keypoints["right_eye"]:
                    print("Warning: Eye keypoints not found, skipping frame")
                    return image

                # Extract eye regions
                left_eye_image = self.extract_eye_region(image, eye_keypoints["left_eye"])
                right_eye_image = self.extract_eye_region(image, eye_keypoints["right_eye"])

                # Skip processing if the extracted eye images are invalid
                if left_eye_image is None or right_eye_image is None:
                    print("Warning: Invalid eye region, skipping frame")
                    return image

                # Predict eye states
                left_eye_state = self.predict_eye_state(left_eye_image)
                right_eye_state = self.predict_eye_state(right_eye_image)

                # Check if both eyes are closed
                if left_eye_state < 0.5 and right_eye_state < 0.5:  # Using 0.5 as threshold
                    self.count += 1
                else:
                    self.count = max(0, self.count - 1)
                    self.stop_alarm()

                # Draw the eye keypoints on the image for visualization
                for (x, y) in eye_keypoints["left_eye"] + eye_keypoints["right_eye"]:
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

                # Trigger alarm if threshold is reached
                if self.count >= self.alarm_threshold:
                    cv2.putText(image, "DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    threading.Thread(target=self.play_alarm).start()
        else:
            print("Warning: No face detected, skipping frame")

        return image

    def run(self):
        """Main loop for capturing and processing video frames."""
        while True:
            success, image = self.cap.read()
            if not success:
                print("Warning: Failed to read from camera, retrying...")
                continue

            image = cv2.flip(image, 1)
            processed_image = self.process_frame(image)

            if processed_image is None:
                print("Skipping frame due to processing error")
                continue

            cv2.imshow("Drowsiness Detection", processed_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Exit loop when 'q' is pressed

        self.cap.release()
        cv2.destroyAllWindows()
        self.stop_alarm()

if __name__ == "__main__":
    detector = DrowsinessDetector(model_path="optimized_eye_detection_model.tflite")
    detector.run()