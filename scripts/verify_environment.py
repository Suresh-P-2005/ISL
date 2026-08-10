import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from config.testing import TestingConfig
from src.backend import create_app

def run_tests():
    print("=" * 60)
    print("  RUNNING FASTAPI SYSTEM INTEGRATION & VERIFICATION CHECKS")
    print("=" * 60)

    app = create_app(TestingConfig())
    client = TestClient(app)

    # 1. Health Status Test
    res = client.get('/status')
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert data['status'] == 'online', f"Expected status 'online', got {data.get('status')}"
    print("  [PASS] GET /status - FastAPI Service Online")

    # 2. Hand Requirements Test
    res = client.get('/hand_requirements')
    assert res.status_code == 200
    assert res.json().get("A") == 2
    print("  [PASS] GET /hand_requirements")

    # 3. Pydantic Validation Test (Empty body should yield 422)
    res = client.post('/predict', json={})
    assert res.status_code == 422
    print("  [PASS] POST /predict (422 Pydantic Validation Error on empty payload)")

    # 4. Static Prediction Test (Valid landmarks)
    fake_landmarks = [0.1] * 126
    res = client.post('/predict', json={'landmarks': fake_landmarks, 'mode': 'alphabet'})
    assert res.status_code == 200
    pdata = res.json()
    assert 'label' in pdata and 'confidence' in pdata
    print(f"  [PASS] POST /predict (Result label: {pdata['label']}, engine: {pdata['engine']})")

    # 5. Translation Test
    res = client.post('/translate', json={'word': 'Hello', 'lang': 'hi-IN'})
    assert res.status_code == 200
    tdata = res.json()
    assert tdata['original'] == 'Hello'
    print(f"  [PASS] POST /translate (Original: 'Hello', Translated: '{tdata['translated']}')")

    # 6. Sentence Construction Test
    res = client.post('/make_sentence', json={'words': 'Hello Thank You', 'lang': 'en-US'})
    assert res.status_code == 200
    sdata = res.json()
    assert 'english_sentence' in sdata
    print(f"  [PASS] POST /make_sentence (Sentence: '{sdata['english_sentence']}')")

    # 7. OpenAPI Docs Endpoint Test
    res = client.get('/docs')
    assert res.status_code == 200
    print("  [PASS] GET /docs - OpenAPI Swagger Documentation Ready")

    print("\n  ALL FASTAPI VERIFICATION CHECKS PASSED SUCCESSFULLY!\n")

if __name__ == '__main__':
    run_tests()
