import pytest
import numpy as np
from app.schemas import DetectRequest
from app.geometry import get_vertices, sample_polyline_points, bilinear_interpolate
from app.utils import normalize_features

def test_detect_request_validation():
    # Valid
    req = DetectRequest(image_path="/some/path.jpg", coords=[1,2,3,4,5,6])
    assert len(req.coords) == 6
    
    # Invalid length
    with pytest.raises(ValueError):
        DetectRequest(image_path="/some/path.jpg", coords=[1,2,3])
        
    # Empty coords
    with pytest.raises(ValueError):
        DetectRequest(image_path="/some/path.jpg", coords=[])

def test_geometry_vertices():
    P = (0, 0)
    X = (10, 0)
    Y = (0, 20)
    
    vertices = get_vertices(P, X, Y)
    assert len(vertices) == 6
    assert vertices[0] == (0, 0)
    assert vertices[1] == (0, 20)
    assert vertices[2] == (10, 20)
    assert vertices[3] == (10, 0)
    assert vertices[4] == (5, 0)
    assert vertices[5] == (5, 20)

def test_polyline_sampling():
    P = (0, 0)
    X = (10, 0)
    Y = (0, 20)
    vertices = get_vertices(P, X, Y)
    
    samples = sample_polyline_points(vertices, num_samples=80)
    assert len(samples) == 80
    
    # Start and end should match exactly
    assert samples[0] == vertices[0]
    assert samples[-1] == vertices[-1]

def test_bilinear_interpolate():
    img = np.array([
        [10, 20],
        [30, 40]
    ], dtype=np.uint8)
    
    # Exact corners
    assert bilinear_interpolate(img, 0, 0) == 10
    assert bilinear_interpolate(img, 1, 0) == 20
    assert bilinear_interpolate(img, 0, 1) == 30
    assert bilinear_interpolate(img, 1, 1) == 40
    
    # Center
    assert bilinear_interpolate(img, 0.5, 0.5) == 25.0
    
    # Out of bounds should clip securely
    assert bilinear_interpolate(img, -1, -5) == 10
    assert bilinear_interpolate(img, 5, 5) == 40

def test_normalize_features():
    features = np.array([10, 50, 110], dtype=float)
    normalized = normalize_features(features)
    assert normalized.shape == (3,)
    assert normalized[0] == 0.0
    assert normalized[2] == 1.0
    assert normalized[1] == 0.4
    
    # Test identical array
    flat = np.array([10, 10, 10], dtype=float)
    normalized_flat = normalize_features(flat)
    assert normalized_flat[0] == 0.0
