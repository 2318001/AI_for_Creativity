"""
server.py - FaceTales FastAPI Server
"""
import os
import io
import uuid
import random
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from emotion_model import analyze_face
from visuals import (
    make_2d_art_from_face,
    make_3d_parallax_gif_from_face,
    make_2d_art_from_image,
    make_3d_parallax_gif_from_image,
)

BASE_DIR = Path(__file__).parent.absolute()
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

AUDIO_DIR_CANDIDATES = [
    BASE_DIR / "assets" / "audio",
    BASE_DIR / "assets" / "music",
    BASE_DIR / "assets music",
]
SUPPORTED_AUDIO_EXT = (".wav", ".mp3", ".ogg", ".flac", ".m4a")
STYLE_TAGS = ["Dreamy", "Retro", "Cyber", "Millennial", "Nature"]

# Poetry Banks
POET_LINES = {
    "Dreamy": {
        "Happy": ["Soft light dances through pastel clouds.", "Your smile floats on cotton candy skies.", "Joy whispers in shades of lavender."],
        "Sad": ["Muted tears fall like watercolor rain.", "The world blurs at the edges, soft and slow.", "Quiet blue wraps you in its hush."],
        "Angry": ["Even fire can burn in soft pink hues.", "Thunder rumbles behind dreamy veils.", "Fury wrapped in gossamer threads."],
        "Surprise": ["Eyes open wide to wonder's soft call.", "A gasp catches in clouds of light.", "Wonder blooms in pastel explosions."],
        "Neutral": ["Still moments drift like feathers falling.", "Balance rests in lavender shadows.", "The calm between breaths, soft as silk."],
    },
    "Retro": {
        "Happy": ["Golden laughter echoes through vinyl grooves.", "Sunshine yellow, like grandmother's kitchen.", "Warm amber memories dance in sepia light."],
        "Sad": ["Old film grain captures tears in monochrome.", "Faded photographs hold forgotten sorrows.", "Melancholy plays on a scratched record."],
        "Angry": ["Burnt orange fury crackles like old radio static.", "Rage develops slowly, like darkroom prints.", "Fire trapped in amber, waiting to release."],
        "Surprise": ["A flash bulb pops—frozen in sepia.", "Time stops like a skipping record.", "Wonder captured on Polaroid film."],
        "Neutral": ["Steady as a grandfather clock's tick.", "Balance found in weathered wooden floors.", "Stillness wrapped in warm brown tones."],
    },
    "Cyber": {
        "Happy": ["Neon joy pulses through digital veins.", "Happiness glitches in cyan and magenta.", "Binary bliss cascades in pixel rain."],
        "Sad": ["Blue screens of sorrow flicker endlessly.", "Data streams carry encrypted tears.", "Digital ghosts haunt neon corridors."],
        "Angry": ["Red alerts flash—system overload.", "Fury compiles into viral executables.", "Error codes cascade in crimson waves."],
        "Surprise": ["System interrupt—unexpected input detected.", "Glitch in the matrix reveals new dimensions.", "Quantum surprise collapses probability waves."],
        "Neutral": ["Idle state—awaiting next command.", "Standby mode hums with electric patience.", "Neutral zone between signal and noise."],
    },
    "Millennial": {
        "Happy": ["Self-care Sunday wrapped in dusty rose.", "Intentional joy, mindfully cultivated.", "Authentic happiness, no filter needed."],
        "Sad": ["Aesthetic melancholy in muted sage.", "Sad but make it minimalist.", "Low-key sorrow, high-key valid."],
        "Angry": ["Sustainable fury, ethically sourced.", "Mindful rage against the machine.", "No toxic vibes allowed in this space."],
        "Surprise": ["Plot twist—unexpected character development.", "Main character energy activated.", "Growth opportunity disguised as chaos."],
        "Neutral": ["Neutral palette, neutral vibes.", "Unbothered, moisturized, in my lane.", "Balance is the new hustle."],
    },
    "Nature": {
        "Happy": ["Sunlight filters through emerald canopy.", "Joy grows wild like meadow flowers.", "Happiness roots deep in rich earth."],
        "Sad": ["Rain washes grief into moss and stone.", "Tears join the river's ancient flow.", "Mist rises from pools of quiet grief."],
        "Angry": ["Thunder rolls across mountain peaks.", "Storm fury bends the tallest trees.", "Lightning cracks—nature's primal rage."],
        "Surprise": ["A deer appears—time holds its breath.", "Sudden bloom in unexpected places.", "The forest whispers secret revelations."],
        "Neutral": ["Still pond reflects the patient sky.", "Balance found in ancient forest rhythms.", "Earth breathes slow and steady."],
    },
}


def make_small_poem(context_label: str, tag: str) -> str:
    context_label = context_label or "Neutral"
    tag = tag or "Dreamy"
    style_poems = POET_LINES.get(tag, POET_LINES["Dreamy"])
    lines = style_poems.get(context_label, style_poems["Neutral"])
    a = random.choice(lines)
    b = random.choice(lines)
    while b == a and len(lines) > 1:
        b = random.choice(lines)
    suffix = {"Cyber": " // neon.exe", "Retro": " // vintage.film", "Nature": " // earth.whisper", "Millennial": " // minimal.vibes", "Dreamy": " // pastel.drift"}.get(tag, "")
    return f"{a}\n{b}{suffix}"


def list_audio_files() -> List[str]:
    files = []
    for d in AUDIO_DIR_CANDIDATES:
        if d.is_dir():
            for f in d.iterdir():
                if f.suffix.lower() in SUPPORTED_AUDIO_EXT:
                    files.append(str(f))
    files.sort()
    return files


def get_audio_filenames() -> List[str]:
    return [os.path.basename(f) for f in list_audio_files()]


def find_audio_by_name(name: str) -> Optional[str]:
    for f in list_audio_files():
        if os.path.basename(f) == name:
            return f
    return None


def auto_pick_song(context_label: str, tag: str, audio_files: List[str]) -> Optional[str]:
    if not audio_files:
        return None
    context_key = (context_label or "").lower()
    tag_key = (tag or "").lower()
    matches = [p for p in audio_files if context_key in os.path.basename(p).lower() and tag_key in os.path.basename(p).lower()]
    if matches:
        return random.choice(matches)
    matches = [p for p in audio_files if tag_key in os.path.basename(p).lower()]
    if matches:
        return random.choice(matches)
    matches = [p for p in audio_files if context_key in os.path.basename(p).lower()]
    if matches:
        return random.choice(matches)
    return random.choice(audio_files)


def ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def save_image(img: np.ndarray, suffix: str = ".png") -> str:
    filename = f"{uuid.uuid4().hex}{suffix}"
    filepath = OUTPUT_DIR / filename
    Image.fromarray(img).save(filepath)
    return f"/output/{filename}"


def save_gif(gif_path: str) -> Optional[str]:
    if not gif_path or not os.path.exists(gif_path):
        return None
    filename = f"{uuid.uuid4().hex}.gif"
    dest = OUTPUT_DIR / filename
    shutil.copy(gif_path, dest)
    return f"/output/{filename}"


# FastAPI App
app = FastAPI(title="FaceTales API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/audio-list")
async def get_audio_list():
    return JSONResponse({"songs": get_audio_filenames()})


@app.post("/api/process")
async def process_image_api(
    image: UploadFile = File(...),
    mode: str = Form("2D"),
    style: str = Form("Dreamy"),
    strength: float = Form(0.90),
    music_mode: str = Form("Auto"),
    manual_song: str = Form("(none)"),
):
    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        rgb = np.array(pil_img)
        rgb = ensure_uint8_rgb(rgb)
        
        if style not in STYLE_TAGS:
            style = "Dreamy"
        strength = max(0.0, min(1.0, float(strength)))
        audio_files = list_audio_files()
        
        face_result = analyze_face(rgb)
        art2d_url, art3d_url = None, None
        
        if face_result is not None:
            context_label = face_result.context_label
            intensity = float(face_result.intensity)
            face_rgb = face_result.face_rgb
            landmarks = face_result.landmarks_xy
            
            art2d = make_2d_art_from_face(face_rgb, context_label, intensity, style, strength=strength)
            if art2d is not None:
                art2d_url = save_image(art2d)
            
            if mode == "3D":
                art3d_path = make_3d_parallax_gif_from_face(face_rgb, landmarks, intensity, style)
                if art3d_path:
                    art3d_url = save_gif(art3d_path)
        else:
            context_label = "Neutral"
            intensity = 0.65
            
            art2d = make_2d_art_from_image(rgb, context_label, intensity, style, strength=strength)
            if art2d is not None:
                art2d_url = save_image(art2d)
            
            if mode == "3D":
                art3d_path = make_3d_parallax_gif_from_image(rgb, intensity, style)
                if art3d_path:
                    art3d_url = save_gif(art3d_path)
        
        # Audio
        audio_url = None
        if music_mode == "Auto":
            chosen_audio = auto_pick_song(context_label, style, audio_files)
        elif music_mode == "Manual" and manual_song and manual_song != "(none)":
            chosen_audio = find_audio_by_name(manual_song)
        else:
            chosen_audio = None
        
        if chosen_audio and os.path.isfile(chosen_audio):
            ext = os.path.splitext(chosen_audio)[1]
            audio_filename = f"{uuid.uuid4().hex}{ext}"
            shutil.copy(chosen_audio, OUTPUT_DIR / audio_filename)
            audio_url = f"/output/{audio_filename}"
        
        poem = make_small_poem(context_label, style)
        
        return JSONResponse({
            "art2d_url": art2d_url,
            "art3d_url": art3d_url,
            "audio_url": audio_url,
            "poem": poem,
            "context": context_label,
            "style": style,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/styles")
async def get_styles():
    descriptions = {
        "Dreamy": "Soft pastels, ethereal glow, watercolor effects",
        "Retro": "Vintage film look, sepia tones, grain texture",
        "Cyber": "Neon glow, chromatic aberration, glitch effects",
        "Millennial": "Muted pastels, minimalist, desaturated",
        "Nature": "Oil painting effect, earth tones, canvas texture",
    }
    return JSONResponse({"styles": STYLE_TAGS, "descriptions": descriptions})


# Static files
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"message": "FaceTales API is running. Place your index.html in the static folder."})


@app.on_event("startup")
async def startup_event():
    print("FaceTales server started at http://127.0.0.1:7860")
    for f in OUTPUT_DIR.iterdir():
        try:
            f.unlink()
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860)
