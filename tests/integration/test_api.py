def test_health_status_endpoint(client):
    response = client.get('/status')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'online'
    assert 'hand_requirements' in data

def test_hand_requirements_endpoint(client):
    response = client.get('/hand_requirements')
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert data.get('A') == 2

def test_predict_endpoint_empty_body(client):
    response = client.post('/predict', json={})
    assert response.status_code == 422  # FastAPI Pydantic validation error

def test_predict_endpoint_valid_landmarks(client):
    fake_landmarks = [0.1] * 126
    response = client.post('/predict', json={'landmarks': fake_landmarks, 'mode': 'alphabet'})
    assert response.status_code == 200
    data = response.json()
    assert 'label' in data
    assert 'confidence' in data

def test_translate_endpoint(client):
    response = client.post('/translate', json={'word': 'Hello', 'lang': 'hi-IN'})
    assert response.status_code == 200
    data = response.json()
    assert data['original'] == 'Hello'
    assert 'translated' in data
