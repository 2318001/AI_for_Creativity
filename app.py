"""
app.py
Multi-Modal Art Generator: 2D/3D Art with Style-Based Poetry and Music

This application generates stylized artwork based on USER-SELECTED artistic styles, The optional face detection serves only for image cropping
and contextual enhancement.
"""
import os
import random
from typing import List, Optional

import gradio as gr
import numpy as np

from emotion_model import analyze_face
from visuals import (
    make_2d_art_from_face,
    make_3d_parallax_gif_from_face,
    make_2d_art_from_image,
    make_3d_parallax_gif_from_image,
)

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR_CANDIDATES = [
    os.path.join(BASE_DIR, "assets", "audio"),
    os.path.join(BASE_DIR, "assets", "music"),
    os.path.join(BASE_DIR, "assets music"),
    os.path.join(BASE_DIR, "assets music", "audio"),
    os.path.join(BASE_DIR, "assets music", "music"),
    "assets/audio",
    "assets/music",
    "assets music",
]
SUPPORTED_AUDIO_EXT = (".wav", ".mp3", ".ogg", ".flac", ".m4a")

# Available artistic styles
STYLE_TAGS = ["Dreamy", "Retro", "Cyber", "Millennial", "Nature"]

# ----------------------------------------------------
# Custom CSS
# ----------------------------------------------------
CUSTOM_CSS = """
/*footer/branding */
footer, #footer, .gradio-container footer {display:none !important;}
a[href*="gradio"] {display:none !important;}
.show-api, .api-docs, .built-with {display:none !important;}

.block.svelte-12cmxck {
    position: relative;
    margin: 0;
    box-shadow: rgb(11 23 197 / 63%) 0px 1px 1px 10px;
    border-width: var(--block-border-width);
    border-color: #1453d0;
    border-radius: var(--block-radius);
    background: #ebe5e5;
    width: 100%;
    line-height: var(--line-sm);
}

/* Compact layout */
.gradio-container {max-width: 1200px !important; margin: 0 auto !important;}
.block {padding: 8px !important;}
.wrap {gap: 10px !important;}
label, .label {font-size: 13px !important;}
.prose h1, .prose h2 {font-size: 20px !important; margin: 0 0 8px 0 !important;}
.prose p {font-size: 13px !important; margin: 0 !important; color: #666 !important;}
button {padding: 10px 12px !important; font-size: 13px !important;}
input, select, textarea {font-size: 13px !important;}

/* Style preview cards */
.style-info {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 10px;
}
"""

# ----------------------------------------------------
# Poetry Banks - Style-Specific Poetry
# ----------------------------------------------------
POET_LINES = {
    "Dreamy": {
        "Happy": [
            "Soft light dances through pastel clouds.",
            "Your smile floats on cotton candy skies.",
            "Joy whispers in shades of lavender.",
            "A gentle warmth wraps around the moment.",
        ],
        "Sad": [
            "Muted tears fall like watercolor rain.",
            "The world blurs at the edges, soft and slow.",
            "Even sorrow wears a gentle gown.",
            "Quiet blue wraps you in its hush.",
        ],
        "Angry": [
            "Even fire can burn in soft pink hues.",
            "Thunder rumbles behind dreamy veils.",
            "Fury wrapped in gossamer threads.",
            "Heat rises, filtered through morning mist.",
        ],
        "Surprise": [
            "Eyes open wide to wonder's soft call.",
            "A gasp catches in clouds of light.",
            "The unexpected arrives on butterfly wings.",
            "Wonder blooms in pastel explosions.",
        ],
        "Neutral": [
            "Still moments drift like feathers falling.",
            "Balance rests in lavender shadows.",
            "The calm between breaths, soft as silk.",
            "Quietude paints itself in gentle strokes.",
        ],
    },
    "Retro": {
        "Happy": [
            "Golden laughter echoes through vinyl grooves.",
            "Sunshine yellow, like grandmother's kitchen.",
            "Joy tastes like butterscotch and old photographs.",
            "Warm amber memories dance in sepia light.",
        ],
        "Sad": [
            "Old film grain captures tears in monochrome.",
            "Faded photographs hold forgotten sorrows.",
            "Melancholy plays on a scratched record.",
            "Brown paper bags of yesterday's grief.",
        ],
        "Angry": [
            "Burnt orange fury crackles like old radio static.",
            "Rage develops slowly, like darkroom prints.",
            "Heat rises through vintage radiators.",
            "Fire trapped in amber, waiting to release.",
        ],
        "Surprise": [
            "A flash bulb pops—frozen in sepia.",
            "Time stops like a skipping record.",
            "Wonder captured on Polaroid film.",
            "The unexpected preserved in vintage frames.",
        ],
        "Neutral": [
            "Steady as a grandfather clock's tick.",
            "Balance found in weathered wooden floors.",
            "Calm like Sunday afternoons in autumn.",
            "Stillness wrapped in warm brown tones.",
        ],
    },
    "Cyber": {
        "Happy": [
            "Neon joy pulses through digital veins.",
            "Happiness glitches in cyan and magenta.",
            "Electric smiles light up the grid.",
            "Binary bliss cascades in pixel rain.",
        ],
        "Sad": [
            "Blue screens of sorrow flicker endlessly.",
            "Data streams carry encrypted tears.",
            "Melancholy runs in corrupted code.",
            "Digital ghosts haunt neon corridors.",
        ],
        "Angry": [
            "Red alerts flash—system overload.",
            "Fury compiles into viral executables.",
            "Rage burns through fiber optic cables.",
            "Error codes cascade in crimson waves.",
        ],
        "Surprise": [
            "System interrupt—unexpected input detected.",
            "Glitch in the matrix reveals new dimensions.",
            "Wonder protocols activated unexpectedly.",
            "Quantum surprise collapses probability waves.",
        ],
        "Neutral": [
            "Idle state—awaiting next command.",
            "Standby mode hums with electric patience.",
            "Balanced voltage maintains the system.",
            "Neutral zone between signal and noise.",
        ],
    },
    "Millennial": {
        "Happy": [
            "Self-care Sunday wrapped in dusty rose.",
            "Intentional joy, mindfully cultivated.",
            "Authentic happiness, no filter needed.",
            "Gratitude journaled in minimalist spaces.",
        ],
        "Sad": [
            "Aesthetic melancholy in muted sage.",
            "Sad but make it minimalist.",
            "Curated grief on neutral backgrounds.",
            "Low-key sorrow, high-key valid.",
        ],
        "Angry": [
            "Sustainable fury, ethically sourced.",
            "Mindful rage against the machine.",
            "Boundaries established, energy protected.",
            "No toxic vibes allowed in this space.",
        ],
        "Surprise": [
            "Plot twist—unexpected character development.",
            "Main character energy activated.",
            "Universe said surprise, and here we are.",
            "Growth opportunity disguised as chaos.",
        ],
        "Neutral": [
            "Neutral palette, neutral vibes.",
            "Unbothered, moisturized, in my lane.",
            "Just existing, and that's enough.",
            "Balance is the new hustle.",
        ],
    },
    "Nature": {
        "Happy": [
            "Sunlight filters through emerald canopy.",
            "Joy grows wild like meadow flowers.",
            "Laughter ripples across forest streams.",
            "Happiness roots deep in rich earth.",
        ],
        "Sad": [
            "Rain washes grief into moss and stone.",
            "Tears join the river's ancient flow.",
            "Autumn leaves carry sorrow downstream.",
            "Mist rises from pools of quiet grief.",
        ],
        "Angry": [
            "Thunder rolls across mountain peaks.",
            "Storm fury bends the tallest trees.",
            "Lightning cracks—nature's primal rage.",
            "Volcanic heat builds beneath calm soil.",
        ],
        "Surprise": [
            "A deer appears—time holds its breath.",
            "Sudden bloom in unexpected places.",
            "Nature reveals her hidden wonders.",
            "The forest whispers secret revelations.",
        ],
        "Neutral": [
            "Still pond reflects the patient sky.",
            "Balance found in ancient forest rhythms.",
            "Roots and branches know their place.",
            "Earth breathes slow and steady.",
        ],
    },
}


def make_small_poem(context_label: str, tag: str) -> str:
    """Generate style-specific poem based on context label and selected style"""
    context_label = context_label or "Neutral"
    tag = tag or "Dreamy"
    
    style_poems = POET_LINES.get(tag, POET_LINES["Dreamy"])
    lines = style_poems.get(context_label, style_poems["Neutral"])
    
    a = random.choice(lines)
    b = random.choice(lines)
    while b == a and len(lines) > 1:
        b = random.choice(lines)

    # Style-specific signature
    suffix = {
        "Cyber": " // neon.exe",
        "Retro": " // vintage.film",
        "Nature": " // earth.whisper",
        "Millennial": " // minimal.vibes",
        "Dreamy": " // pastel.drift",
    }.get(tag, "")

    return f"{a}\n{b}{suffix}"


# ----------------------------------------------------
# Audio Helpers
# ----------------------------------------------------
def _existing_audio_dirs() -> List[str]:
    return [d for d in AUDIO_DIR_CANDIDATES if os.path.isdir(d)]


def list_audio_files() -> List[str]:
    files: List[str] = []
    for d in _existing_audio_dirs():
        for f in os.listdir(d):
            if f.lower().endswith(SUPPORTED_AUDIO_EXT):
                files.append(os.path.join(d, f))
    files.sort()
    return files


def auto_pick_song(context_label: str, tag: str, audio_files: List[str]) -> Optional[str]:
    """Pick song matching context and/or style"""
    if not audio_files:
        return None
    
    context_key = (context_label or "").lower()
    tag_key = (tag or "").lower()
    
    # Priority 1: Match both context and style
    matches = [p for p in audio_files 
               if context_key in os.path.basename(p).lower() 
               and tag_key in os.path.basename(p).lower()]
    if matches:
        return random.choice(matches)
    
    # Priority 2: Match style
    matches = [p for p in audio_files if tag_key in os.path.basename(p).lower()]
    if matches:
        return random.choice(matches)
    
    # Priority 3: Match context
    matches = [p for p in audio_files if context_key in os.path.basename(p).lower()]
    if matches:
        return random.choice(matches)
    
    # Fallback: Random
    return random.choice(audio_files)


# ----------------------------------------------------
# Core Processing Function
# ----------------------------------------------------
def process_image(
    image: np.ndarray,
    visual_mode: str,
    style_tag: str,
    art_strength: float,
    music_mode: str,
    manual_song: str,
):
    """
    Main processing function.
    
    The USER'S STYLE SELECTION (style_tag) is the primary driver of art generation.
       Returns: (2d_art, 3d_gif_path, audio_path, poem, status_message)
    """
    audio_files = list_audio_files()

    if image is None:
        return (None, None, None, "", "Please upload an image or use webcam.")

    # Ensure uint8 RGB
    rgb = image
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # Try face detection for cropping (optional enhancement)
    face_result = analyze_face(rgb)
    
    if face_result is not None:
        context_label = face_result.context_label
        intensity = float(face_result.intensity)
        face_rgb = face_result.face_rgb
        landmarks = face_result.landmarks_xy
        status = f"Face detected | Style: {style_tag} | Processing..."
        
        # Generate 2D art from face region
        art2d = make_2d_art_from_face(
            face_rgb, context_label, intensity, style_tag, strength=float(art_strength)
        )
        
        # Generate 3D if requested
        art3d_path = None
        if visual_mode == "3D":
            art3d_path = make_3d_parallax_gif_from_face(
                face_rgb, landmarks, intensity, style_tag
            )
    else:
        # No face detected - use full image with default context
        context_label = "Neutral"
        intensity = 0.65
        status = f"Processing full image | Style: {style_tag}"
        
        # Generate 2D art from full image
        art2d = make_2d_art_from_image(
            rgb, context_label, intensity, style_tag, strength=float(art_strength)
        )
        
        # Generate 3D if requested
        art3d_path = None
        if visual_mode == "3D":
            art3d_path = make_3d_parallax_gif_from_image(
                rgb, intensity, style_tag
            )

    # Audio selection based on style
    if music_mode == "Off":
        chosen_audio = None
    elif music_mode == "Auto":
        chosen_audio = auto_pick_song(context_label, style_tag, audio_files)
    else:  # Manual
        chosen_audio = manual_song if manual_song and manual_song != "(none)" else None

    # Verify audio file exists
    if chosen_audio and not os.path.isfile(chosen_audio):
        chosen_audio = None

    # Generate style-matched poetry
    poem = make_small_poem(context_label, style_tag)

    return (art2d, art3d_path, chosen_audio, poem, status)


# ----------------------------------------------------
# Style Description Helper
# ----------------------------------------------------
def get_style_description(style: str) -> str:
    """Return description of what each style produces"""
    descriptions = {
        "Dreamy": "Soft pastels, ethereal glow, watercolor effects, lavender and pink tones",
        "Retro": "Vintage film look, sepia tones, grain texture, warm amber colors, posterized",
        "Cyber": "Neon glow, chromatic aberration, high saturation, cyan/magenta, scanlines",
        "Millennial": "Muted pastels, desaturated, minimalist pencil sketch overlay, dusty rose/sage",
        "Nature": "Oil painting effect, earth tones, greens and browns, canvas texture",
    }
    return descriptions.get(style, "Artistic transformation")


# ----------------------------------------------------
# Gradio UI
# ----------------------------------------------------
def build_ui():
    audio_files = list_audio_files()
    manual_choices = ["(none)"] + audio_files

    with gr.Blocks(title="ArtGen - Multi-Modal Art Generator", css=CUSTOM_CSS) as demo:
        gr.Markdown("""
        ## ArtGen - Style-Based Multi-Modal Art Generator
        
        **Select an artistic style** to generate coordinated 2D/3D art, poetry, and music.
        Each style produces **distinctly different** visual and textual output.
        
        > **Note:** This is NOT an emotion detection system. Your selected style drives all generation.
        """)

        with gr.Row(equal_height=True):
            # Left Column - Input Controls
            with gr.Column(scale=1, min_width=450):
                img_in = gr.Image(
                    sources=["webcam", "upload"],
                    type="numpy",
                    label="Input Image",
                    height=280,
                )

                with gr.Row():
                    visual_mode = gr.Radio(
                        ["2D", "3D"], 
                        value="2D", 
                        label="Output Mode",
                        info="2D: Static art | 3D: Animated parallax GIF"
                    )
                    style_tag = gr.Dropdown(
                        STYLE_TAGS, 
                        value="Dreamy", 
                        label="Art Style (Primary)",
                        info="This selection drives all art generation"
                    )

                style_info = gr.Textbox(
                    label="Style Preview",
                    value=get_style_description("Dreamy"),
                    interactive=False,
                    lines=2,
                )

                art_strength = gr.Slider(
                    0.0, 1.0, 
                    value=0.90, 
                    step=0.01, 
                    label="Art Strength",
                    info="Higher = more stylized, Lower = more photorealistic"
                )

                with gr.Row():
                    music_mode = gr.Radio(
                        ["Off", "Auto", "Manual"], 
                        value="Auto", 
                        label="Music Mode"
                    )
                    manual_song = gr.Dropdown(
                        manual_choices, 
                        value="(none)", 
                        label="Select Song"
                    )

            # Right Column - Output
            with gr.Column(scale=1, min_width=450):
                with gr.Row():
                    out_2d = gr.Image(
                        type="numpy", 
                        label="2D Art Output", 
                        height=280
                    )
                    out_3d = gr.Image(
                        type="filepath", 
                        label="3D Parallax (GIF)", 
                        height=280
                    )

                out_audio = gr.Audio(
                    label="Background Music", 
                    type="filepath"
                )

                out_poem = gr.Textbox(
                    label="Generated Poetry",
                    lines=3,
                    max_lines=4,
                    interactive=False,
                    placeholder="Style-matched poetry will appear here...",
                )

                status = gr.Textbox(
                    label="Status",
                    lines=1,
                    interactive=False,
                )

        # Event handlers
        inputs = [img_in, visual_mode, style_tag, art_strength, music_mode, manual_song]
        outputs = [out_2d, out_3d, out_audio, out_poem, status]

        # Update style description when style changes
        style_tag.change(
            fn=lambda s: get_style_description(s),
            inputs=[style_tag],
            outputs=[style_info]
        )

        # Process on any input change
        img_in.change(process_image, inputs, outputs)
        visual_mode.change(process_image, inputs, outputs)
        style_tag.change(process_image, inputs, outputs)
        art_strength.change(process_image, inputs, outputs)
        music_mode.change(process_image, inputs, outputs)
        manual_song.change(process_image, inputs, outputs)

        # Style comparison info
        gr.Markdown("""
        ### Style Comparison:
        | Style | Colors | Effect | Mood |
        |-------|--------|--------|------|
        | **Dreamy** | Pastels, lavender, pink | Soft blur, watercolor | Ethereal, romantic |
        | **Retro** | Sepia, amber, brown | Film grain, posterize | Nostalgic, vintage |
        | **Cyber** | Neon cyan, magenta | Glitch, chromatic shift | Futuristic, electric |
        | **Millennial** | Muted sage, dusty rose | Sketch, desaturated | Minimal, aesthetic |
        | **Nature** | Greens, earth tones | Oil painting, texture | Organic, grounded |
        
        ---
        
        **How it works:**
        1. Upload an image or use webcam
        2. **Select your preferred artistic style** (this is the key input!)
        3. Adjust art strength and music options
        4. Receive coordinated 2D/3D art, poetry, and music
        """)

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.queue().launch()
