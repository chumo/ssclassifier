import math
import numpy as np
from typing import List, Tuple

def get_vertices(P: Tuple[float, float], X: Tuple[float, float], Y: Tuple[float, float]) -> List[Tuple[float, float]]:
    """
    Given bottom-left point P, and vectors X and Y, derive the 6 vertices
    of the sampling polyline spanning the 7-segment character.
    Path: A -> B -> C -> D -> E -> F
    """
    px, py = P
    xx, xy = X
    yx, yy = Y
    
    A = (px, py)
    B = (px + yx, py + yy)
    C = (px + yx + xx, py + yy + xy)
    D = (px + xx, py + xy)
    E = (px + 0.5 * xx, py + 0.5 * xy)
    F = (px + 0.5 * xx + yx, py + 0.5 * xy + yy)
    
    return [A, B, C, D, E, F]

def sample_polyline_points(vertices: List[Tuple[float, float]], num_samples: int = 80) -> List[Tuple[float, float]]:
    """
    Sample exactly num_samples points evenly spaced along the full polyline arc length.
    """
    segments = []
    total_length = 0.0
    for i in range(len(vertices) - 1):
        v1 = vertices[i]
        v2 = vertices[i+1]
        dist = math.hypot(v2[0] - v1[0], v2[1] - v1[1])
        segments.append(dist)
        total_length += dist

    if total_length == 0.0:
        return [vertices[0]] * num_samples
        
    sampled_points = []
    for i in range(num_samples):
        target_dist = (i / max(1, (num_samples - 1))) * total_length
        if i == 0:
            sampled_points.append(vertices[0])
            continue
        if i == num_samples - 1:
            sampled_points.append(vertices[-1])
            continue
            
        current_dist = 0.0
        for j in range(len(segments)):
            if current_dist + segments[j] >= target_dist - 1e-9:
                overshoot = target_dist - current_dist
                fraction = overshoot / segments[j] if segments[j] > 0 else 0
                
                v1 = vertices[j]
                v2 = vertices[j+1]
                px = v1[0] + (v2[0] - v1[0]) * fraction
                py = v1[1] + (v2[1] - v1[1]) * fraction
                sampled_points.append((px, py))
                break
            current_dist += segments[j]
            
    return sampled_points

def bilinear_interpolate(img: np.ndarray, x: float, y: float) -> float:
    """
    Sample pixel intensity securely from 2D grayscale image using bilinear interpolation.
    Bounds are checked explicitly.
    """
    h, w = img.shape
    x_clip = max(0.0, min(float(x), w - 1.0))
    y_clip = max(0.0, min(float(y), h - 1.0))
    
    x0 = int(math.floor(x_clip))
    x1 = min(x0 + 1, w - 1)
    y0 = int(math.floor(y_clip))
    y1 = min(y0 + 1, h - 1)
    
    wx = x_clip - x0
    wy = y_clip - y0
    
    val00 = img[y0, x0]
    val10 = img[y0, x1]
    val01 = img[y1, x0]
    val11 = img[y1, x1]
    
    valY0 = val00 * (1.0 - wx) + val10 * wx
    valY1 = val01 * (1.0 - wx) + val11 * wx
    val = valY0 * (1.0 - wy) + valY1 * wy
    
    return float(val)
