import cv2
import mediapipe as mp
import math
from typing import Tuple, Union, Dict, List


"""
Fn:
    Detects faces and their bounding boxes
    Extracts facial landmarks, particularly eyes and mouth keypoints
    Draws the face mesh and landmarks on the frame
    Displays the processed frame in real-time
"""


class FaceMeshDetector:
    def __init__(self, max_faces=1, min_detection_conf=0.5, min_tracking_conf=0.5):
        self.mp_drawing = mp.solutions.drawing_utils    #helps in drawing landmarks on the frame 
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh      #loads the facemesh model for detecting the facial landmarks
        self.mp_face_detection = mp.solutions.face_detection    #loads face detection for the bounding box 

        #Initializing face mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )

        #Initializing face detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_conf)
        self.cap = cv2.VideoCapture(0)

    #Converting normalized facial landmark coordinates into pixel coordinates x&y
    def _normalized_to_pixel_coordinates(self, normalized_x: float, normalized_y: float, image_width: int, image_height: int) -> Union[None, Tuple[int, int]]:
        def is_valid_normalized_value(value: float) -> bool:    #checks if the value is between 0 and 1
            return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
            return None
        
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px

    #Drawing bounding boxes around detected faces in the image 
    def visualize(self, image, detection_result):
        annotated_image = image.copy()
        height, width, _ = image.shape
        bbox_data = []

        for detection in detection_result.detections:        #draws bounding boxes around each detected face
            bbox = detection.location_data.relative_bounding_box
            start_point = int(bbox.xmin * width), int(bbox.ymin * height)
            end_point = int((bbox.xmin + bbox.width) * width), int((bbox.ymin + bbox.height) * height)
            cv2.rectangle(annotated_image, start_point, end_point, (0, 255, 0), 3)

            bbox_data.append({    #stores the bounding box data
                "start_point": start_point,
                "end_point": end_point,
                "width": bbox.width * width,
                "height": bbox.height * height
            })
        return annotated_image, bbox_data


    #Extracting eye & mouth keypoints
    def get_eye_mouth_keypoints(self, face_landmarks, image_shape) -> Dict[str, List[Tuple[int, int]]]:
        eye_mouth_keypoints = {
            "left_eye": [],
            "right_eye": [],
            "mouth": []
        }
        h, w, _ = image_shape

        LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  
        RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  
        MOUTH_INDICES = [61, 291, 39, 181, 17, 405]

        #stores keypoints
        for idx in LEFT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            eye_mouth_keypoints["left_eye"].append((int(landmark.x * w), int(landmark.y * h)))

        for idx in RIGHT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            eye_mouth_keypoints["right_eye"].append((int(landmark.x * w), int(landmark.y * h)))

        for idx in MOUTH_INDICES:
            landmark = face_landmarks.landmark[idx]
            eye_mouth_keypoints["mouth"].append((int(landmark.x * w), int(landmark.y * h)))

        return eye_mouth_keypoints


    #Processing the frame
    def process_frame(self, image):
        detection_results = self.face_detection.process(image)  #detects the faces 
        bbox_data = []
        eye_mouth_keypoints = {}

        if detection_results.detections:    #calling the visualize function 
            image_with_bbox, bbox_data = self.visualize(image, detection_results)
        else:
            image_with_bbox = image

        results = self.face_mesh.process(image)

        #draws facial landmarks & mesh connections
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=image_with_bbox,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                self.mp_drawing.draw_landmarks(
                    image=image_with_bbox,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )

                eye_mouth_keypoints = self.get_eye_mouth_keypoints(face_landmarks, image.shape)

        return image_with_bbox, bbox_data, eye_mouth_keypoints  #calling the get_eye_mouth_keypoints function


    def run(self):
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                break

            image_with_bbox, bbox_data, eye_mouth_keypoints = self.process_frame(image)
            cv2.imshow("Face Mesh Detection", cv2.flip(image_with_bbox, 1))

            print("Bounding Box Data:", bbox_data)
            print("Eye and Mouth Keypoints:", eye_mouth_keypoints)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = FaceMeshDetector()
    detector.run()