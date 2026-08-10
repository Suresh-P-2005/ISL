import numpy as np
from src.ml.data.extractors import normalise_landmarks

def test_normalise_landmarks_zero_vector():
    flat = [0.0] * 126
    norm = normalise_landmarks(flat)
    assert len(norm) == 126
    assert np.allclose(norm, 0.0)

def test_normalise_landmarks_shape_preservation():
    flat = [float(i) for i in range(126)]
    norm = normalise_landmarks(flat)
    assert len(norm) == 126
    assert isinstance(norm, np.ndarray)
