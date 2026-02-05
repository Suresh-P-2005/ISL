# MediaPipe Hand Detection Handler

import cv2
import mediapipe as mp
import numpy as np
from config import MEDIAPIPE_CONFIG

class MediaPipeHandler:
    """Handle MediaPipe hand detection and landmark extraction"""
    
    def __init__(self):
        """Initialize MediaPipe Hands solution"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize Hands detector
        self.hands = self.mp_hands.Hands(
            static_image_mode=MEDIAPIPE_CONFIG['static_image_mode'],
            max_num_hands=MEDIAPIPE_CONFIG['max_num_hands'],
            min_detection_confidence=MEDIAPIPE_CONFIG['min_detection_confidence'],
            min_tracking_confidence=MEDIAPIPE_CONFIG['min_tracking_confidence']
        )
        
        self.landmarks_list = []
        
    def extract_landmarks(self, image):
        """
        Extract hand landmarks from image
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            landmarks_array: Numpy array of normalized landmarks (63 features)
            hand_detected: Boolean indicating if hand was detected
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.hands.process(image_rgb)
        
        landmarks_array = np.zeros(63)  # 21 landmarks * 3 coordinates
        hand_detected = False
        
        if results.multi_hand_landmarks:
            # Get first hand (can be extended for two-hand gestures)
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_detected = True
            
            # Extract x, y, z coordinates
            for idx, landmark in enumerate(hand_landmarks.landmark):
                landmarks_array[idx * 3] = landmark.x
                landmarks_array[idx * 3 + 1] = landmark.y
                landmarks_array[idx * 3 + 2] = landmark.z
        
        return landmarks_array, hand_detected, results
    
    def extract_landmarks_sequence(self, video_path, sequence_length=30):
        """
        Extract landmark sequence from video file
        
        Args:
            video_path: Path to video file
            sequence_length: Number of frames to extract
            
        Returns:
            sequence: Numpy array of shape (sequence_length, 63)
        """
        cap = cv2.VideoCapture(video_path)
        sequence = []
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // sequence_length)
        
        while cap.isOpened() and len(sequence) < sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                landmarks, detected, _ = self.extract_landmarks(frame)
                if detected:
                    sequence.append(landmarks)
            
            frame_count += 1
        
        cap.release()
        
        # Pad or truncate to exact sequence length
        sequence = np.array(sequence)
        if len(sequence) < sequence_length:
            # Pad with zeros
            padding = np.zeros((sequence_length - len(sequence), 63))
            sequence = np.vstack([sequence, padding])
        elif len(sequence) > sequence_length:
            # Truncate
            sequence = sequence[:sequence_length]
        
        return sequence
    
    def draw_landmarks(self, image, results):
        """
        Draw hand landmarks on image
        
        Args:
            image: BGR image
            results: MediaPipe results object
            
        Returns:
            annotated_image: Image with drawn landmarks
        """
        annotated_image = image.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return annotated_image
    
    def calculate_hand_features(self, landmarks_array):
        """
        Calculate additional hand features from landmarks
        
        Features:
        - Palm center
        - Finger tip distances
        - Hand orientation
        - Hand size
        
        Args:
            landmarks_array: Array of 63 landmark coordinates
            
        Returns:
            features: Dictionary of calculated features
        """
        landmarks = landmarks_array.reshape(21, 3)
        
        # Palm center (average of wrist and middle finger base)
        wrist = landmarks[0]
        middle_base = landmarks[9]
        palm_center = (wrist + middle_base) / 2
        
        # Finger tips indices
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        
        # Calculate distances from palm center to each finger tip
        tip_distances = []
        for tip_idx in finger_tips:
            distance = np.linalg.norm(landmarks[tip_idx] - palm_center)
            tip_distances.append(distance)
        
        # Hand size (distance from wrist to middle finger tip)
        hand_size = np.linalg.norm(landmarks[12] - landmarks[0])
        
        # Hand orientation (angle of hand)
        wrist_to_middle = landmarks[9] - landmarks[0]
        hand_angle = np.arctan2(wrist_to_middle[1], wrist_to_middle[0])
        
        features = {
            'palm_center': palm_center,
            'tip_distances': tip_distances,
            'hand_size': hand_size,
            'hand_angle': hand_angle
        }
        
        return features
    
    def normalize_landmarks(self, landmarks_array):
        """
        Normalize landmarks relative to wrist position and hand size
        
        Args:
            landmarks_array: Raw landmark coordinates
            
        Returns:
            normalized_landmarks: Normalized coordinates
        """
        landmarks = landmarks_array.reshape(21, 3)
        
        # Get wrist position
        wrist = landmarks[0].copy()
        
        # Translate to wrist origin
        landmarks = landmarks - wrist
        
        # Calculate hand size (max distance from wrist)
        distances = np.linalg.norm(landmarks, axis=1)
        hand_size = np.max(distances)
        
        if hand_size > 0:
            # Scale by hand size
            landmarks = landmarks / hand_size
        
        return landmarks.flatten()
    
    def close(self):
        """Release MediaPipe resources"""
        self.hands.close()

# Utility function for real-time detection
def process_frame_for_display(frame, mediapipe_handler):
    """
    Process frame for display with landmarks
    
    Args:
        frame: Input frame
        mediapipe_handler: MediaPipeHandler instance
        
    Returns:
        processed_frame: Frame with landmarks drawn
        landmarks: Extracted landmarks
        detected: Boolean for detection status
    """
    landmarks, detected, results = mediapipe_handler.extract_landmarks(frame)
    processed_frame = mediapipe_handler.draw_landmarks(frame, results)
    
    return processed_frame, landmarks, detected

if __name__ == "__main__":
    # Test MediaPipe handler
    print("Testing MediaPipe Handler...")
    handler = MediaPipeHandler()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        landmarks, detected, results = handler.extract_landmarks(frame)
        annotated_frame = handler.draw_landmarks(frame, results)
        
        if detected:
            cv2.putText(annotated_frame, "Hand Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('MediaPipe Hand Detection', annotated_frame)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    handler.close()
