import numpy as np

def normalise_landmarks(flat):
    """
    Normalizes hand landmarks array (126 elements: 21 (x,y,z) points x 2 hands).
    Centers wrist landmark at origin and scales max distance to 1.
    """
    arr = np.array(flat, dtype=float)
    result = arr.copy()
    for offset in [0, 63]:
        chunk = arr[offset:offset+63].reshape(21, 3)
        if np.sum(np.abs(chunk)) < 1e-6:
            continue
        wrist = chunk[0].copy()
        chunk -= wrist
        scale = np.max(np.linalg.norm(chunk, axis=1))
        if scale > 1e-6:
            chunk /= scale
        result[offset:offset+63] = chunk.flatten()
    return result

def extract_mediapipe_landmarks(results):
    """
    Extracts raw 126-length landmark vector from MediaPipe multi_hand_landmarks.
    Left hand -> indices 0..62; Right hand -> indices 63..125
    """
    flat = [0.0] * 126
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_handedness in enumerate(results.multi_handedness):
            label = hand_handedness.classification[0].label  # 'Left' or 'Right'
            offset = 63 if label == 'Right' else 0
            hand_landmarks = results.multi_hand_landmarks[idx]
            for j, lm in enumerate(hand_landmarks.landmark):
                flat[offset + j * 3] = lm.x
                flat[offset + j * 3 + 1] = lm.y
                flat[offset + j * 3 + 2] = lm.z
    return flat
