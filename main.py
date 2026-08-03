"""
AI Prompt Extractor - Backend
------------------------------
Alur kerja:
1. User kirim link (YouTube/TikTok/IG/dll) ATAU upload file gambar/video.
2. Kalau link -> download pakai yt-dlp.
3. Kalau video -> ekstrak beberapa frame pakai ffmpeg.
4. Kirim frame ke Claude Vision -> hasilkan prompt teks deskriptif.
5. Kembalikan hasil prompt ke frontend.
6. Hapus file sementara setelah selesai (privasi + hemat storage).

Cara jalankan:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY="sk-ant-xxxx"
    uvicorn main:app --reload --port 8000
"""

import os
import uuid
import base64
import shutil
import subprocess
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import anthropic

load_dotenv()

APP_DIR = Path(__file__).parent
TEMP_DIR = APP_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    print("[WARNING] ANTHROPIC_API_KEY belum diset. Set di file .env")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

app = FastAPI(title="AI Prompt Extractor")

# Izinkan frontend (ganti origin sesuai domain kamu nanti saat deploy)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ganti ke domain spesifik saat production
    allow_methods=["*"],
    allow_headers=["*"],
)

STYLE_PRESETS = {
    "midjourney": "Format hasil sebagai prompt gaya Midjourney: deskriptif, padat, diakhiri parameter seperti --ar 16:9 --style raw.",
    "stable_diffusion": "Format hasil sebagai prompt gaya Stable Diffusion: kata kunci dipisah koma, urutan subjek, gaya, pencahayaan, kualitas.",
    "cinematic": "Format hasil sebagai deskripsi prompt sinematik naratif, cocok untuk text-to-video (Runway/Pika/Sora), sertakan camera movement dan mood.",
    "general": "Format hasil sebagai deskripsi prompt umum yang detail: subjek, gaya visual, pencahayaan, komposisi, warna, mood.",
}


def cleanup(path: Path):
    """Hapus file/folder sementara."""
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()
    except Exception as e:
        print(f"[cleanup] gagal hapus {path}: {e}")


def download_from_url(url: str, job_dir: Path) -> Path:
    """Download video/gambar dari link sosial media pakai yt-dlp."""
    output_template = str(job_dir / "source.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]/best",
        "-o", output_template,
        "--no-playlist",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise HTTPException(
            status_code=400,
            detail=f"Gagal mengunduh dari link. Pastikan link valid & publik. Detail: {result.stderr[-300:]}",
        )

    files = list(job_dir.glob("source.*"))
    if not files:
        raise HTTPException(status_code=400, detail="Tidak ada file yang berhasil diunduh.")
    return files[0]


def extract_frames(video_path: Path, job_dir: Path, num_frames: int = 4) -> List[Path]:
    """Ekstrak beberapa frame representatif dari video pakai ffmpeg."""
    frames_dir = job_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # Ambil durasi video dulu
    probe_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True)
    try:
        duration = float(probe.stdout.strip())
    except ValueError:
        duration = 5.0  # fallback

    frame_paths = []
    for i in range(num_frames):
        timestamp = max(0.1, (duration / (num_frames + 1)) * (i + 1))
        out_path = frames_dir / f"frame_{i}.jpg"
        cmd = [
            "ffmpeg", "-y", "-ss", str(timestamp), "-i", str(video_path),
            "-frames:v", "1", "-q:v", "2", str(out_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        if out_path.exists():
            frame_paths.append(out_path)

    return frame_paths


def image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_prompt_from_images(image_paths: List[Path], style: str) -> str:
    """Kirim gambar ke Claude Vision, minta deskripsi jadi prompt AI-generator."""
    style_instruction = STYLE_PRESETS.get(style, STYLE_PRESETS["general"])

    content_blocks = []
    for p in image_paths:
        content_blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_to_base64(p),
            },
        })

    instruction = (
        "Kamu adalah ahli prompt engineering untuk AI image/video generator. "
        "Analisis gambar berikut (bisa lebih dari satu, mewakili frame-frame dari satu video). "
        "Buat SATU prompt teks yang menggambarkan: subjek utama, gaya visual, komposisi, "
        "pencahayaan, warna, mood/atmosfer, dan gerakan jika terlihat dari perbedaan antar frame. "
        f"{style_instruction} "
        "Jawab hanya dengan prompt akhirnya saja, tanpa penjelasan tambahan, tanpa tanda kutip."
    )
    content_blocks.append({"type": "text", "text": instruction})

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": content_blocks}],
    )

    return "".join(block.text for block in message.content if hasattr(block, "text")).strip()


@app.get("/")
def health_check():
    return {"status": "ok", "message": "AI Prompt Extractor backend berjalan."}


@app.post("/api/extract-from-url")
async def extract_from_url(url: str = Form(...), style: str = Form("general")):
    """Endpoint utama: terima link sosmed, kembalikan prompt teks."""
    job_id = str(uuid.uuid4())
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        source_path = download_from_url(url, job_dir)

        if source_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            frames = [source_path]
        else:
            frames = extract_frames(source_path, job_dir, num_frames=4)
            if not frames:
                raise HTTPException(status_code=400, detail="Gagal mengekstrak frame dari video.")

        prompt_result = generate_prompt_from_images(frames, style)

        return JSONResponse({"success": True, "prompt": prompt_result, "frames_used": len(frames)})

    finally:
        cleanup(job_dir)


@app.post("/api/extract-from-upload")
async def extract_from_upload(file: UploadFile = File(...), style: str = Form("general")):
    """Endpoint alternatif: user upload file langsung (lebih aman secara legal)."""
    job_id = str(uuid.uuid4())
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        ext = Path(file.filename).suffix or ".mp4"
        source_path = job_dir / f"source{ext}"
        with open(source_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if ext.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            frames = [source_path]
        else:
            frames = extract_frames(source_path, job_dir, num_frames=4)
            if not frames:
                raise HTTPException(status_code=400, detail="Gagal mengekstrak frame dari video.")

        prompt_result = generate_prompt_from_images(frames, style)

        return JSONResponse({"success": True, "prompt": prompt_result, "frames_used": len(frames)})

    finally:
        cleanup(job_dir)
