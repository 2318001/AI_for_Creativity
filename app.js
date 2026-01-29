const startCamBtn = document.getElementById("startCamBtn");
const captureBtn = document.getElementById("captureBtn");
const generateBtn = document.getElementById("generateBtn");

const fileInput = document.getElementById("fileInput");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const previewImg = document.getElementById("previewImg");

const styleTag = document.getElementById("styleTag");
const strength = document.getElementById("strength");
const strengthVal = document.getElementById("strengthVal");

const song = document.getElementById("song");

const art2d = document.getElementById("art2d");
const art3d = document.getElementById("art3d");
const audio = document.getElementById("audio");
const poem = document.getElementById("poem");

const statusEl = document.getElementById("status");
const loading = document.getElementById("loading");

// hidden inputs for backend values
const modeHidden = document.getElementById("mode");         // "2D" | "3D"
const musicHidden = document.getElementById("musicMode");   // "Off" | "Auto" | "Manual"

let stream = null;
let currentFile = null;

// ---------- helpers ----------
function setStatus(t) { statusEl.textContent = t; }
function setLoading(on) { loading.classList.toggle("hidden", !on); }
function enableGenerate(on) { generateBtn.disabled = !on; }

function setActiveButton(groupSelector, valueAttr, value) {
  document.querySelectorAll(groupSelector).forEach(btn => {
    btn.classList.toggle("active", btn.getAttribute(valueAttr) === value);
  });
}

function canAutoGenerate() {
  return !!currentFile && !generateBtn.disabled;
}

// debounce for slider auto-generate
let debounceTimer = null;
function debounceGenerate(ms = 300) {
  if (!canAutoGenerate()) return;
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    generate();
  }, ms);
}

// ---------- strength ----------
strength.addEventListener("input", () => {
  strengthVal.textContent = Number(strength.value).toFixed(2);
  // ✅ auto-generate when moving slider
  debounceGenerate(250);
});

// ---------- segmented buttons: mode ----------
document.querySelectorAll(".segBtn[data-mode]").forEach(btn => {
  btn.addEventListener("click", () => {
    const v = btn.dataset.mode; // "2D" or "3D"
    modeHidden.value = v;
    setActiveButton(".segBtn[data-mode]", "data-mode", v);

    // if switching to 2D, clear 3D output image
    if (v === "2D") art3d.removeAttribute("src");

    // optional: auto-generate on mode change
    debounceGenerate(150);
  });
});

// ---------- segmented buttons: music ----------
document.querySelectorAll(".segBtn[data-music]").forEach(btn => {
  btn.addEventListener("click", () => {
    const v = btn.dataset.music; // "Off" | "Auto" | "Manual"
    musicHidden.value = v;
    setActiveButton(".segBtn[data-music]", "data-music", v);

    // enable song dropdown only for manual
    song.disabled = (v !== "Manual");

    // optional: auto-generate on music change
    debounceGenerate(150);
  });
});

// ---------- load songs ----------
async function loadSongs() {
  try {
    const r = await fetch("/api/audio-list");
    const data = await r.json();
    song.innerHTML = `<option value="(none)">(none)</option>`;
    (data.songs || []).forEach(name => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      song.appendChild(opt);
    });
  } catch (e) {
    console.warn("audio list failed", e);
  }
}
loadSongs();

// if manual song changes, auto-generate
song.addEventListener("change", () => {
  if (musicHidden.value === "Manual") debounceGenerate(150);
});

// ---------- webcam ----------
startCamBtn.addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    video.classList.remove("hidden");
    previewImg.classList.add("hidden");
    captureBtn.disabled = false;
    setStatus("Webcam ready");
  } catch (e) {
    console.error(e);
    setStatus("Webcam blocked / unavailable");
  }
});

captureBtn.addEventListener("click", async () => {
  if (!video.videoWidth) return;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0);

  const blob = await new Promise(res => canvas.toBlob(res, "image/jpeg", 0.95));
  currentFile = new File([blob], "webcam.jpg", { type: "image/jpeg" });

  previewImg.src = URL.createObjectURL(blob);
  previewImg.classList.remove("hidden");
  video.classList.add("hidden");

  enableGenerate(true);
  generateBtn.disabled = false;
  setStatus("Captured (ready)");

  // auto-generate after capture
  debounceGenerate(50);
});

// ---------- upload ----------
fileInput.addEventListener("change", () => {
  const f = fileInput.files?.[0];
  if (!f) return;

  currentFile = f;
  previewImg.src = URL.createObjectURL(f);
  previewImg.classList.remove("hidden");
  video.classList.add("hidden");

  enableGenerate(true);
  setStatus("Image loaded (ready)");

  // auto-generate after upload
  debounceGenerate(50);
});

// ---------- generate ----------
async function generate() {
  if (!currentFile) return;

  setLoading(true);
  setStatus("Generating…");

  try {
    const fd = new FormData();
    fd.append("image", currentFile);
    fd.append("mode", modeHidden.value);                 // "2D" or "3D"
    fd.append("style", styleTag.value);
    fd.append("strength", String(strength.value));
    fd.append("music_mode", musicHidden.value);          // Off/Auto/Manual
    fd.append("manual_song", song.value || "(none)");

    const r = await fetch("/api/process", { method: "POST", body: fd });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();

    const t = `?t=${Date.now()}`;

    if (data.art2d_url) art2d.src = data.art2d_url + t;
    else art2d.removeAttribute("src");

    if (data.art3d_url && modeHidden.value === "3D") art3d.src = data.art3d_url + t;
    else art3d.removeAttribute("src");

    if (data.audio_url) {
      audio.src = data.audio_url + t;
      audio.load();
    } else {
      audio.removeAttribute("src");
      audio.load();
    }

    poem.textContent = data.poem || "—";
    setStatus("Done");
  } catch (e) {
    console.error(e);
    setStatus("Error (check server console)");
  } finally {
    setLoading(false);
  }



  // ---------- Fullscreen viewer ----------
const viewer = document.getElementById("viewer");
const viewerImg = document.getElementById("viewerImg");
const viewerClose = document.getElementById("viewerClose");
const viewerBtnClose = document.getElementById("viewerBtnClose");
const viewerDownload = document.getElementById("viewerDownload");

function openViewer(src){
  if(!src) return;
  viewerImg.src = src;
  viewerDownload.href = src;
  viewer.classList.remove("hidden");
}

function closeViewer(){
  viewer.classList.add("hidden");
  viewerImg.removeAttribute("src");
}

viewerClose.addEventListener("click", closeViewer);
viewerBtnClose.addEventListener("click", closeViewer);

document.addEventListener("keydown", (e) => {
  if(e.key === "Escape") closeViewer();
});

// Click output images -> fullscreen
art2d.addEventListener("click", () => openViewer(art2d.src));
art3d.addEventListener("click", () => openViewer(art3d.src));

}


generateBtn.addEventListener("click", generate);

// ---------- style change auto-generate ----------
styleTag.addEventListener("change", () => debounceGenerate(150));
