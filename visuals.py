"""
visuals.py - 2D and 3D Art Generation with Distinct Style Differentiation

This module provides style-based image transformation for the ArtGen system.
Each of the 5 styles (Dreamy, Retro, Cyber, Millennial, Nature) produces
visually distinct output through different color palettes and processing techniques.
"""
import math
import tempfile
import random
from typing import Dict, Tuple, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance


# ============================================================
# 0) Helper Functions
# ============================================================
def _ensure_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 RGB format"""
    if rgb is None:
        return None
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def _rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to BGR for OpenCV"""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB"""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _vignette_mask(h: int, w: int, strength: float = 0.35) -> np.ndarray:
    """Create vignette mask for edge darkening"""
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    mask = 1.0 - strength * np.clip(rr, 0, 1)
    return mask.astype(np.float32)


# ============================================================
# 1) Style Palettes - Distinct Colors Per Style
# ============================================================
STYLE_PALETTES = {
    # Dreamy: Soft pastels, lavender, pink, baby blue
    "Dreamy": {
        "Happy": ((255, 182, 193), (230, 190, 255)),      # Pink to lavender
        "Sad": ((147, 180, 220), (100, 130, 180)),        # Soft blue
        "Angry": ((255, 150, 170), (180, 100, 130)),      # Dusty rose
        "Surprise": ((255, 220, 180), (200, 170, 255)),   # Peach to lilac
        "Neutral": ((220, 210, 230), (180, 175, 200)),    # Soft gray-lavender
        "overlay_alpha": 0.25,
        "saturation_boost": 0.95,
        "contrast": 0.9,
    },
    # Retro: Warm oranges, browns, yellows, vintage film look
    "Retro": {
        "Happy": ((255, 200, 100), (200, 130, 50)),       # Golden yellow to brown
        "Sad": ((120, 100, 80), (70, 60, 50)),            # Sepia brown
        "Angry": ((200, 80, 50), (100, 40, 30)),          # Burnt orange
        "Surprise": ((255, 180, 80), (180, 100, 60)),     # Warm amber
        "Neutral": ((180, 160, 130), (120, 110, 90)),     # Vintage beige
        "overlay_alpha": 0.30,
        "saturation_boost": 0.85,
        "contrast": 1.15,
    },
    # Cyber: Neon cyan, magenta, electric blue, high contrast
    "Cyber": {
        "Happy": ((0, 255, 200), (255, 0, 150)),          # Cyan to magenta
        "Sad": ((50, 80, 180), (20, 0, 100)),             # Deep electric blue
        "Angry": ((255, 0, 80), (150, 0, 50)),            # Neon red/magenta
        "Surprise": ((0, 255, 255), (200, 0, 255)),       # Cyan to purple
        "Neutral": ((100, 150, 180), (50, 80, 120)),      # Steel blue
        "overlay_alpha": 0.35,
        "saturation_boost": 1.4,
        "contrast": 1.3,
    },
    # Millennial: Muted pastels, dusty pink, sage green, minimalist
    "Millennial": {
        "Happy": ((255, 200, 180), (230, 180, 170)),      # Dusty pink/peach
        "Sad": ((180, 190, 180), (150, 160, 155)),        # Sage gray
        "Angry": ((200, 150, 140), (160, 120, 115)),      # Muted terracotta
        "Surprise": ((220, 200, 180), (190, 180, 170)),   # Warm beige
        "Neutral": ((200, 195, 190), (170, 168, 165)),    # Warm gray
        "overlay_alpha": 0.20,
        "saturation_boost": 0.75,
        "contrast": 0.95,
    },
    # Nature: Greens, earth tones, forest colors
    "Nature": {
        "Happy": ((180, 220, 100), (100, 160, 60)),       # Bright green to forest
        "Sad": ((100, 130, 120), (60, 90, 80)),           # Muted teal/moss
        "Angry": ((160, 100, 60), (100, 60, 40)),         # Earth brown/rust
        "Surprise": ((200, 230, 150), (120, 180, 100)),   # Spring green
        "Neutral": ((150, 160, 130), (110, 120, 100)),    # Olive/sage
        "overlay_alpha": 0.28,
        "saturation_boost": 1.1,
        "contrast": 1.05,
    },
}


def get_style_palette(context_label: str, tag: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Get distinct color palette for context + style combination"""
    tag = tag or "Dreamy"
    context_label = context_label or "Neutral"
    palette = STYLE_PALETTES.get(tag, STYLE_PALETTES["Dreamy"])
    return palette.get(context_label, palette["Neutral"])


def get_style_params(tag: str) -> Dict:
    """Get style-specific rendering parameters"""
    tag = tag or "Dreamy"
    palette = STYLE_PALETTES.get(tag, STYLE_PALETTES["Dreamy"])
    return {
        "overlay_alpha": palette.get("overlay_alpha", 0.25),
        "saturation_boost": palette.get("saturation_boost", 1.0),
        "contrast": palette.get("contrast", 1.0),
    }


def _gradient_overlay(size: Tuple[int, int], c1: Tuple[int, int, int], c2: Tuple[int, int, int], 
                      direction: str = "vertical") -> Image.Image:
    """Create gradient overlay with different directions per style"""
    w, h = size
    bg = Image.new("RGB", (w, h), c1)
    d = ImageDraw.Draw(bg)
    
    if direction == "diagonal":
        for i in range(max(w, h) * 2):
            t = i / max(1, max(w, h) * 2 - 1)
            r = int(c1[0] * (1 - t) + c2[0] * t)
            g = int(c1[1] * (1 - t) + c2[1] * t)
            b = int(c1[2] * (1 - t) + c2[2] * t)
            d.line([(i, 0), (0, i)], fill=(r, g, b), width=2)
    elif direction == "radial":
        cx, cy = w // 2, h // 2
        max_dist = math.sqrt(cx**2 + cy**2)
        for y in range(h):
            for x in range(w):
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                t = min(1.0, dist / max_dist)
                r = int(c1[0] * (1 - t) + c2[0] * t)
                g = int(c1[1] * (1 - t) + c2[1] * t)
                b = int(c1[2] * (1 - t) + c2[2] * t)
                bg.putpixel((x, y), (r, g, b))
    else:  # vertical
        for y in range(h):
            t = y / max(1, (h - 1))
            r = int(c1[0] * (1 - t) + c2[0] * t)
            g = int(c1[1] * (1 - t) + c2[1] * t)
            b = int(c1[2] * (1 - t) + c2[2] * t)
            d.line([(0, y), (w, y)], fill=(r, g, b))
    return bg


# ============================================================
# 2) Style Filters - Each Style Has Unique Processing
# ============================================================
def _posterize_kmeans(bgr: np.ndarray, k: int = 8) -> np.ndarray:
    """K-means color quantization for posterize effect"""
    Z = bgr.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    out = centers[labels.flatten()].reshape(bgr.shape)
    return out


def _style_dreamy(bgr: np.ndarray, intensity: float) -> np.ndarray:
    """
    Dreamy Style: Soft blur, pastel wash, ethereal glow
    - Heavy watercolor effect
    - Strong gaussian blur for dreamy glow
    - Pastel pink/lavender color shift
    - Low saturation for soft pastels
    - Bright, lifted shadows
    """
    # Strong watercolor effect
    styled = cv2.stylization(bgr, sigma_s=120, sigma_r=0.35)
    
    # Heavy dreamy blur/glow
    blur = cv2.GaussianBlur(styled, (0, 0), 12 + 10 * intensity)
    styled = cv2.addWeighted(styled, 0.5, blur, 0.5, 0)
    
    # Push towards pastel pink/lavender tones
    hsv = cv2.cvtColor(styled, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    hue = (hue + 15) % 180  # Shift towards pink
    hsv[:, :, 0] = hue.astype(np.uint8)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.65, 0, 255).astype(np.uint8)  # Heavy desaturation
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2 + 35, 0, 255).astype(np.uint8)  # Brighter
    styled = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Add soft pink tint overlay
    pink_tint = np.full_like(styled, (220, 180, 255), dtype=np.uint8)  # Light pink in BGR
    styled = cv2.addWeighted(styled, 0.85, pink_tint, 0.15, 0)
    
    return styled


def _style_retro(bgr: np.ndarray, intensity: float) -> np.ndarray:
    """
    Retro Style: Film grain, sepia tint, high contrast, posterize
    - Heavy posterization for vintage look
    - Strong cartoon edges
    - Heavy sepia tone
    - Warm orange/brown tint
    - Heavy film grain
    - High contrast with vignette
    """
    # Heavy posterize for vintage look
    styled = _posterize_kmeans(bgr, k=5)
    
    # Strong cartoon edges
    gray = cv2.cvtColor(styled, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 3)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    styled = cv2.bitwise_and(styled, edges)
    
    # Strong sepia tone
    sepia_kernel = np.array([
        [0.189, 0.469, 0.089],
        [0.349, 0.686, 0.168],
        [0.472, 0.869, 0.239]
    ])
    styled = cv2.transform(styled, sepia_kernel)
    styled = np.clip(styled, 0, 255).astype(np.uint8)
    
    # Add warm orange/brown tint
    warm_tint = np.full_like(styled, (60, 120, 200), dtype=np.uint8)  # Warm orange in BGR
    styled = cv2.addWeighted(styled, 0.75, warm_tint, 0.25, 0)
    
    # Heavy film grain
    noise = np.random.normal(0, 18 + 12 * intensity, styled.shape).astype(np.float32)
    styled = np.clip(styled.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # High contrast
    styled = cv2.convertScaleAbs(styled, alpha=1.35, beta=-20)
    
    # Add vignette darkening for vintage look
    h, w = styled.shape[:2]
    vign = _vignette_mask(h, w, strength=0.4)
    styled = (styled.astype(np.float32) * vign[..., None]).clip(0, 255).astype(np.uint8)
    
    return styled


def _style_cyber(bgr: np.ndarray, intensity: float) -> np.ndarray:
    """
    Cyber Style: Neon glow, chromatic aberration, high saturation, glitch
    - Anime-style posterization
    - Neon cyan/magenta edge glow
    - Strong chromatic aberration
    - Heavy saturation boost
    - Scanlines for digital effect
    - Glitch blocks
    """
    # Anime-style posterization
    flat = _posterize_kmeans(bgr, k=10)
    flat = cv2.bilateralFilter(flat, 9, 35, 35)
    
    # Strong edges (anime look)
    gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 100)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    
    # Neon cyan/magenta edge glow
    edges_blur = cv2.GaussianBlur(edges, (9, 9), 4)
    edges_color = np.zeros_like(flat)
    edges_color[:, :, 0] = np.clip(edges_blur * 1.5, 0, 255).astype(np.uint8)  # Strong blue/cyan
    edges_color[:, :, 1] = np.clip(edges_blur * 0.8, 0, 255).astype(np.uint8)  # Some green for cyan
    edges_color[:, :, 2] = np.clip(edges_blur * 1.2, 0, 255).astype(np.uint8)  # Magenta component
    
    styled = cv2.add(flat, edges_color)
    
    # Strong chromatic aberration
    b, g, r = cv2.split(styled)
    shift = int(4 + 5 * intensity)
    r = np.roll(r, shift, axis=1)
    b = np.roll(b, -shift, axis=1)
    styled = cv2.merge([b, g, r])
    
    # Add neon cyan/magenta tint
    neon_tint = np.zeros_like(styled)
    neon_tint[:, :, 0] = 180  # Cyan blue
    neon_tint[:, :, 1] = 50   # Some green
    neon_tint[:, :, 2] = 150  # Magenta
    styled = cv2.addWeighted(styled, 0.7, neon_tint, 0.3, 0)
    
    # Heavy saturation boost
    hsv = cv2.cvtColor(styled, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.8, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255).astype(np.uint8)
    styled = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Add scanlines for digital effect
    for i in range(0, styled.shape[0], 3):
        styled[i:i+1, :, :] = (styled[i:i+1, :, :] * 0.6).astype(np.uint8)
    
    # Glitch blocks
    h, w = styled.shape[:2]
    num_glitches = int(3 + 5 * intensity)
    for _ in range(num_glitches):
        gy = random.randint(0, h - 10)
        gh = random.randint(3, 15)
        shift_amt = random.randint(-20, 20)
        if gy + gh < h:
            styled[gy:gy+gh, :, :] = np.roll(styled[gy:gy+gh, :, :], shift_amt, axis=1)
    
    return styled


def _style_millennial(bgr: np.ndarray, intensity: float) -> np.ndarray:
    """
    Millennial Style: Muted, desaturated, soft pencil sketch overlay, minimal
    - Pencil sketch effect
    - Very heavy desaturation (almost grayscale with hint of color)
    - Muted dusty rose/sage tint
    - Soft bilateral smoothing
    - Strong fade/lifted blacks
    - Subtle grain for film look
    """
    # Pencil sketch effect
    gray, color = cv2.pencilSketch(bgr, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    
    # Blend with original but heavily muted
    styled = cv2.addWeighted(bgr, 0.4, color, 0.6, 0)
    
    # Very heavily desaturate
    hsv = cv2.cvtColor(styled, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.3, 0, 255).astype(np.uint8)
    styled = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Add muted dusty rose/sage tint
    muted_tint = np.full_like(styled, (175, 185, 200), dtype=np.uint8)  # Dusty rose/sage in BGR
    styled = cv2.addWeighted(styled, 0.75, muted_tint, 0.25, 0)
    
    # Soft bilateral for smooth skin look
    styled = cv2.bilateralFilter(styled, 15, 60, 60)
    
    # Strong fade/lift blacks (faded look)
    styled = cv2.convertScaleAbs(styled, alpha=0.8, beta=50)
    
    # Add subtle grain for film look
    noise = np.random.normal(0, 5, styled.shape).astype(np.float32)
    styled = np.clip(styled.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return styled


def _style_nature(bgr: np.ndarray, intensity: float) -> np.ndarray:
    """
    Nature Style: Oil painting, earthy tones, organic textures
    - Oil painting effect
    - Hue shift towards greens/earth tones
    - Saturation boost for rich colors
    - Canvas texture overlay
    """
    # Oil painting effect
    try:
        styled = cv2.xphoto.oilPainting(bgr, 10, 1)
    except Exception:
        styled = cv2.stylization(bgr, sigma_s=80, sigma_r=0.5)
    
    # Push towards greens/earth tones
    hsv = cv2.cvtColor(styled, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    # Compress hue range towards green/yellow
    hue = np.where(hue < 90, hue * 0.8 + 20, hue)  # Push towards green
    hsv[:, :, 0] = np.clip(hue, 0, 179).astype(np.uint8)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255).astype(np.uint8)
    styled = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Add earth tone tint
    earth_tint = np.full_like(styled, (80, 120, 100), dtype=np.uint8)  # Earth green/brown in BGR
    styled = cv2.addWeighted(styled, 0.85, earth_tint, 0.15, 0)
    
    # Add subtle canvas texture
    h, w = styled.shape[:2]
    texture = np.random.normal(128, 8, (h, w)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (5, 5), 2)
    texture = ((texture - 128) * 0.1).astype(np.float32)
    for i in range(3):
        styled[:, :, i] = np.clip(styled[:, :, i].astype(np.float32) + texture, 0, 255).astype(np.uint8)
    
    return styled


def apply_style_filter(bgr: np.ndarray, tag: str, intensity: float) -> np.ndarray:
    """
    Apply style-specific filter - each produces DISTINCT look.
    
    Args:
        bgr: Input image in BGR format
        tag: Style tag ("Dreamy", "Retro", "Cyber", "Millennial", "Nature")
        intensity: Effect intensity 0-1
    
    Returns:
        Styled image in BGR format
    """
    tag = tag or "Dreamy"
    intensity = float(np.clip(intensity, 0.0, 1.0))
    
    if tag == "Dreamy":
        return _style_dreamy(bgr, intensity)
    elif tag == "Retro":
        return _style_retro(bgr, intensity)
    elif tag == "Cyber":
        return _style_cyber(bgr, intensity)
    elif tag == "Millennial":
        return _style_millennial(bgr, intensity)
    elif tag == "Nature":
        return _style_nature(bgr, intensity)
    else:
        return cv2.stylization(bgr, sigma_s=60, sigma_r=0.4)


# ============================================================
# 3) 2D Art Generation
# ============================================================
def make_2d_art_from_face(
    face_rgb: np.ndarray,
    context_label: str,
    intensity: float,
    tag: str,
    strength: float = 0.85
) -> np.ndarray:
    """
    Generate 2D art with distinct style differentiation.
    
    Args:
        face_rgb: Face/image region as RGB numpy array
        context_label: Contextual label for palette selection
        intensity: Effect intensity 0-1
        tag: Style tag ("Dreamy", "Retro", "Cyber", "Millennial", "Nature")
        strength: Blend strength 0-1 (higher = more stylized)
    
    Returns:
        Styled image as RGB numpy array
    """
    tag = tag or "Dreamy"
    strength = float(np.clip(strength, 0.0, 1.0))
    intensity = float(np.clip(intensity, 0.0, 1.0))

    face_rgb = _ensure_uint8_rgb(face_rgb)

    # Upscale for quality
    im = Image.fromarray(face_rgb).convert("RGB")
    target_w = 1024
    scale = target_w / max(1, im.size[0])
    im = im.resize((target_w, int(im.size[1] * scale)), Image.LANCZOS)

    rgb = np.array(im)
    bgr = _rgb_to_bgr(rgb)

    # Apply style-specific filter
    styled = apply_style_filter(bgr, tag, intensity)

    # Get style-specific palette and parameters
    c1, c2 = get_style_palette(context_label, tag)
    params = get_style_params(tag)
    
    # Different gradient directions per style
    gradient_directions = {
        "Dreamy": "radial",
        "Retro": "diagonal",
        "Cyber": "vertical",
        "Millennial": "vertical",
        "Nature": "diagonal",
    }
    grad_dir = gradient_directions.get(tag, "vertical")
    
    # Create gradient overlay
    overlay = _gradient_overlay((styled.shape[1], styled.shape[0]), c1, c2, grad_dir)
    overlay_bgr = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)
    
    # Apply overlay with style-specific alpha
    alpha = params["overlay_alpha"] * (0.8 + 0.4 * intensity)
    styled = cv2.addWeighted(styled, 1.0 - alpha, overlay_bgr, alpha, 0)

    # Blend styled with original based on strength
    out = cv2.addWeighted(styled, strength, bgr, 1.0 - strength, 0)

    # Apply style-specific saturation and contrast
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    sat_boost = params["saturation_boost"]
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_boost, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Contrast adjustment
    contrast = params["contrast"]
    out = cv2.convertScaleAbs(out, alpha=contrast, beta=0)

    # Style-specific vignette
    h, w = out.shape[:2]
    vignette_strengths = {
        "Dreamy": 0.15,
        "Retro": 0.35,
        "Cyber": 0.25,
        "Millennial": 0.10,
        "Nature": 0.20,
    }
    vig_strength = vignette_strengths.get(tag, 0.20)
    vign = _vignette_mask(h, w, strength=vig_strength)
    out = (out.astype(np.float32) * vign[..., None]).clip(0, 255).astype(np.uint8)

    return _bgr_to_rgb(out)


def make_2d_art_from_image(
    rgb: np.ndarray,
    context_label: str,
    intensity: float,
    tag: str,
    strength: float = 0.85,
) -> np.ndarray:
    """Same as make_2d_art_from_face, works on any image"""
    rgb = _ensure_uint8_rgb(rgb)
    return make_2d_art_from_face(
        rgb,
        context_label=context_label,
        intensity=float(intensity),
        tag=tag,
        strength=float(strength),
    )


# ============================================================
# 4) MiDaS Depth Estimation
# ============================================================
_MIDAS_MODEL = None
_MIDAS_TRANSFORM = None


def _load_midas():
    """Load MiDaS depth estimation model"""
    global _MIDAS_MODEL, _MIDAS_TRANSFORM
    if _MIDAS_MODEL is not None:
        return _MIDAS_MODEL, _MIDAS_TRANSFORM

    try:
        import torch
        _MIDAS_MODEL = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
        _MIDAS_MODEL.eval()
        _MIDAS_MODEL.to("cpu")

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        _MIDAS_TRANSFORM = transforms.small_transform
        return _MIDAS_MODEL, _MIDAS_TRANSFORM
    except Exception:
        _MIDAS_MODEL = None
        _MIDAS_TRANSFORM = None
        return None, None


def _depth_midas(rgb_uint8: np.ndarray) -> Optional[np.ndarray]:
    """Returns depth map in [0..1] float32, same HxW as rgb."""
    model, transform = _load_midas()
    if model is None:
        return None

    import torch

    rgb = rgb_uint8
    input_batch = transform(rgb).to("cpu")

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy().astype(np.float32)
    depth = depth - depth.min()
    depth = depth / (depth.max() + 1e-6)
    return depth


def _fallback_depth(face_rgb: np.ndarray, landmarks_xy: Dict[int, Tuple[float, float]]) -> np.ndarray:
    """Fallback depth if MiDaS not available: radial depth from center/landmarks"""
    h, w = face_rgb.shape[:2]
    
    # Use nose tip as center if available
    if landmarks_xy and 1 in landmarks_xy:
        cx, cy = landmarks_xy[1]
    else:
        cx, cy = w / 2, h / 2
    
    # Create radial depth map
    y = np.linspace(0, h - 1, h)
    x = np.linspace(0, w - 1, w)
    xx, yy = np.meshgrid(x, y)
    
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    max_dist = np.sqrt((w/2)**2 + (h/2)**2)
    
    depth = 1.0 - (dist / max_dist)
    depth = np.clip(depth, 0.0, 1.0).astype(np.float32)
    
    return depth


def _fallback_depth_simple(rgb: np.ndarray) -> np.ndarray:
    """Simple radial fallback depth for images without landmarks"""
    h, w = rgb.shape[:2]
    cx, cy = w / 2, h / 2
    
    y = np.linspace(0, h - 1, h)
    x = np.linspace(0, w - 1, w)
    xx, yy = np.meshgrid(x, y)
    
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    max_dist = np.sqrt((w/2)**2 + (h/2)**2)
    
    depth = 1.0 - (dist / max_dist)
    depth = np.clip(depth, 0.0, 1.0).astype(np.float32)
    
    return depth


# ============================================================
# 5) 3D Parallax Animation
# ============================================================
def _generate_parallax_frames(styled_bgr: np.ndarray, depth: np.ndarray, 
                              num_frames: int = 30, max_shift: int = 15) -> list:
    """Generate parallax animation frames"""
    h, w = styled_bgr.shape[:2]
    frames = []
    
    for i in range(num_frames):
        # Oscillate left-right
        t = i / num_frames
        angle = t * 2 * np.pi
        shift_x = int(max_shift * np.sin(angle))
        shift_y = int(max_shift * 0.3 * np.cos(angle))  # Smaller vertical movement
        
        # Create displacement map based on depth
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        
        for y in range(h):
            for x in range(w):
                d = depth[y, x]
                map_x[y, x] = x + shift_x * d
                map_y[y, x] = y + shift_y * d
        
        # Apply remapping
        frame = cv2.remap(styled_bgr, map_x, map_y, cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_REFLECT)
        
        # Convert to RGB for GIF
        frame_rgb = _bgr_to_rgb(frame)
        frames.append(frame_rgb)
    
    return frames


def _save_gif(frames: list, fps: int = 15) -> str:
    """Save frames as animated GIF"""
    path = tempfile.mktemp(suffix=".gif")
    imageio.mimsave(path, frames, fps=fps, loop=0)
    return path


def make_3d_parallax_gif_from_face(
    face_rgb: np.ndarray,
    landmarks_xy: Dict[int, Tuple[float, float]],
    intensity: float,
    tag: str,
    num_frames: int = 30,
    max_shift: int = 15
) -> str:
    """
    Generate 3D parallax animation GIF from face image.
    
    Args:
        face_rgb: Face region as RGB numpy array
        landmarks_xy: Facial landmarks for depth fallback
        intensity: Effect intensity 0-1
        tag: Style tag
        num_frames: Number of animation frames
        max_shift: Maximum parallax shift in pixels
    
    Returns:
        Path to generated GIF file
    """
    face_rgb = _ensure_uint8_rgb(face_rgb)
    
    # Get depth map
    depth = _depth_midas(face_rgb)
    if depth is None:
        depth = _fallback_depth(face_rgb, landmarks_xy)
    
    # Apply style filter
    bgr = _rgb_to_bgr(face_rgb)
    styled = apply_style_filter(bgr, tag, intensity)
    
    # Apply style overlay
    context_label = "Neutral"  # Default for 3D
    c1, c2 = get_style_palette(context_label, tag)
    params = get_style_params(tag)
    
    gradient_directions = {
        "Dreamy": "radial",
        "Retro": "diagonal",
        "Cyber": "vertical",
        "Millennial": "vertical",
        "Nature": "diagonal",
    }
    grad_dir = gradient_directions.get(tag, "vertical")
    
    overlay = _gradient_overlay((styled.shape[1], styled.shape[0]), c1, c2, grad_dir)
    overlay_bgr = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)
    
    alpha = params["overlay_alpha"] * 0.7
    styled = cv2.addWeighted(styled, 1.0 - alpha, overlay_bgr, alpha, 0)
    
    # Generate frames
    frames = _generate_parallax_frames(styled, depth, num_frames, max_shift)
    
    # Save GIF
    return _save_gif(frames)


def make_3d_parallax_gif_from_image(
    rgb: np.ndarray,
    intensity: float,
    tag: str,
    num_frames: int = 30,
    max_shift: int = 15
) -> str:
    """Same as make_3d_parallax_gif_from_face, works on any image"""
    rgb = _ensure_uint8_rgb(rgb)
    
    # Get depth map
    depth = _depth_midas(rgb)
    if depth is None:
        depth = _fallback_depth_simple(rgb)
    
    # Apply style filter
    bgr = _rgb_to_bgr(rgb)
    styled = apply_style_filter(bgr, tag, intensity)
    
    # Apply style overlay
    context_label = "Neutral"
    c1, c2 = get_style_palette(context_label, tag)
    params = get_style_params(tag)
    
    gradient_directions = {
        "Dreamy": "radial",
        "Retro": "diagonal",
        "Cyber": "vertical",
        "Millennial": "vertical",
        "Nature": "diagonal",
    }
    grad_dir = gradient_directions.get(tag, "vertical")
    
    overlay = _gradient_overlay((styled.shape[1], styled.shape[0]), c1, c2, grad_dir)
    overlay_bgr = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)
    
    alpha = params["overlay_alpha"] * 0.7
    styled = cv2.addWeighted(styled, 1.0 - alpha, overlay_bgr, alpha, 0)
    
    # Generate frames
    frames = _generate_parallax_frames(styled, depth, num_frames, max_shift)
    
    # Save GIF
    return _save_gif(frames)
