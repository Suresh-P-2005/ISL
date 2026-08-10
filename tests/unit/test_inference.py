import pytest
import numpy as np
from src.backend.services.inference_service import InferenceService

def test_inference_service_zero_landmarks():
    config = {"HAND_REQUIREMENTS": {"A": 1}, "N_FEAT": 126}
    service = InferenceService(models_dir="models", config=config)
    
    # Zero landmark array
    zero_landmarks = [0.0] * 126
    res = service.predict_static(zero_landmarks, mode="alphabet")
    
    assert res["label"] == "---"
    assert res["confidence"] == 0.0
    assert res["engine"] == "none"

def test_inference_service_sequence_empty():
    config = {"KEYFRAMES": 30, "N_FEAT": 126}
    service = InferenceService(models_dir="models", config=config)
    
    res = service.predict_sequence(frames=[], num_hands=1)
    assert res["label"] == "---"
