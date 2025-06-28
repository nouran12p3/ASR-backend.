from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import torch
import librosa
import numpy as np
import io
import os
import logging
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define frontend directories
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
style_dir = os.path.join(frontend_dir, "style")
img_dir = os.path.join(frontend_dir, "img")

# Serve static assets
app.mount("/style", StaticFiles(directory=style_dir), name="style")
app.mount("/img", StaticFiles(directory=img_dir), name="img")
app.mount("/scripts", StaticFiles(directory=frontend_dir), name="scripts")  # serves script.js

# HTML routes
@app.get("/")
async def serve_home():
    return FileResponse(os.path.join(frontend_dir, "home.html"))

@app.get("/speech_recognition.html")
async def serve_speech_recognition():
    return FileResponse(os.path.join(frontend_dir, "speech_recognition.html"))

@app.get("/contact_us.html")
async def serve_contact_us():
    return FileResponse(os.path.join(frontend_dir, "contact_us.html"))

@app.get("/script.js")
async def serve_script():
    script_path = os.path.join(frontend_dir, "script.js")
    if not os.path.exists(script_path):
        raise HTTPException(status_code=404, detail="script.js not found")
    return FileResponse(script_path, media_type="application/javascript")

# Load Whisper model
try:
    logger.info("Loading Whisper model...")
    processor = WhisperProcessor.from_pretrained("Marwan-Kasem/whisper-medium-hi32")
    model = WhisperForConditionalGeneration.from_pretrained("Marwan-Kasem/whisper-medium-hi32")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.generation_config.forced_decoder_ids = None
    model.generation_config.max_length = 225
    logger.info(f"Model moved to device: {device}")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    raise HTTPException(status_code=500, detail="Model loading failed")

# Audio upload endpoint
@app.post("/upload")
async def upload_audio(audio: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {audio.filename}")
        if audio.size and audio.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Max 10MB.")

        audio_data = await audio.read()
        if not audio_data:
            raise HTTPException(status_code=400, detail="Empty audio file received")

        audio_stream = io.BytesIO(audio_data)
        audio_array, sampling_rate = librosa.load(audio_stream, sr=16000)
        audio_array = librosa.util.normalize(audio_array)

        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(input_features, attention_mask=attention_mask)
            transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)

        transcription = transcription.strip() or "Text not available"
        logger.info(f"Transcription: {transcription}")
        return {"filename": audio.filename, "transcription": transcription}

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup():
    logger.info("Server running at http://127.0.0.1:8000")


# to run locally
# C:\Users\Smart\Desktop\asr_project_website\backend
# uvicorn main:app --reload