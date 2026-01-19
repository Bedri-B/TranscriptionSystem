import os
import json
import logging
import torch
import ollama
from faster_whisper import WhisperModel
from moviepy import VideoFileClip

# --- Configuration ---
VIDEO_FOLDER = "videos"
INTERMEDIATE_FILE = "transcriptions_fast.json"
FINAL_OUTPUT_FILE = "gym_exercises_final.json"
OLLAMA_MODEL = "mistral:latest"
# 'distil-large-v3' is incredibly fast and accurate, or use 'large-v3' / 'medium'
WHISPER_MODEL_SIZE = "distil-large-v3"

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("process_fast.log"), logging.StreamHandler()],
)


def check_gpu():
    """Checks for NVIDIA GPU and sets compute precision."""
    if torch.cuda.is_available():
        logging.info(f"✅ CUDA Detected: {torch.cuda.get_device_name(0)}")
        return "cuda", "float16"  # Use float16 for speed on GPU
    else:
        logging.warning("⚠️ CUDA not found. Running on CPU (slow).")
        return "cpu", "int8"


def extract_and_transcribe(device, compute_type):
    """Phase 1: High-speed transcription using Faster-Whisper."""
    if not os.path.exists(VIDEO_FOLDER):
        logging.error(f"Folder '{VIDEO_FOLDER}' not found.")
        return []

    logging.info(f"--- PHASE 1: Loading Faster-Whisper ({WHISPER_MODEL_SIZE}) ---")

    try:
        # Load model using CTranslate2 backend
        model = WhisperModel(
            WHISPER_MODEL_SIZE, device=device, compute_type=compute_type
        )
    except Exception as e:
        logging.critical(f"Failed to load model: {e}")
        return []

    results = []
    files = [
        f
        for f in os.listdir(VIDEO_FOLDER)
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
    ]

    for filename in files:
        video_path = os.path.join(VIDEO_FOLDER, filename)
        temp_audio = "temp_fast.wav"

        try:
            logging.info(f"Processing: {filename}")

            # 1. Fast Audio Extraction
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(temp_audio, verbose=False, logger=None)
            clip.close()

            # 2. Fast Transcription
            # beam_size=1 is faster; increase to 5 for slightly better accuracy on difficult audio
            segments, info = model.transcribe(temp_audio, beam_size=1, language="en")

            # Combine segments into one string
            full_text = " ".join([segment.text for segment in segments]).strip()

            if full_text:
                results.append({"file": filename, "transcription": full_text})
                logging.info(
                    f"  -> Success ({len(full_text)} chars). detected lang: {info.language}"
                )
            else:
                logging.warning(f"  -> No speech detected.")

        except Exception as e:
            logging.error(f"Error on {filename}: {e}")

        finally:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)

    # Save Phase 1 results
    with open(INTERMEDIATE_FILE, "w") as f:
        json.dump(results, f, indent=4)

    # Cleanup VRAM
    del model
    torch.cuda.empty_cache()
    logging.info("🧹 GPU Memory Cleared.")

    return results


def analyze_with_ollama(transcription_data):
    """Phase 2: Ollama Analysis (Same as before)"""
    logging.info("--- PHASE 2: Analyzing with Ollama ---")
    final_data = []

    for item in transcription_data:
        filename = item["file"]
        text = item["transcription"]

        prompt = f"""
        Analyze this gym video transcription. Return ONLY the name of the exercise performed.
        If it is unclear, return "Unknown".
        
        Transcription: "{text}"
        """

        try:
            logging.info(f"Analyzing {filename}...")
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            item["exercise_type"] = response["message"]["content"].strip()
            final_data.append(item)
            logging.info(f"  -> Identified: {item['exercise_type']}")

        except Exception as e:
            logging.error(f"Ollama error: {e}")

    return final_data


def main():
    device, compute_type = check_gpu()

    # Resumability check
    if os.path.exists(INTERMEDIATE_FILE):
        choice = input(
            f"Found '{INTERMEDIATE_FILE}'. Use existing transcriptions? (y/n): "
        ).lower()
        if choice == "y":
            with open(INTERMEDIATE_FILE, "r") as f:
                transcriptions = json.load(f)
        else:
            transcriptions = extract_and_transcribe(device, compute_type)
    else:
        transcriptions = extract_and_transcribe(device, compute_type)

    if transcriptions:
        final_results = analyze_with_ollama(transcriptions)
        with open(FINAL_OUTPUT_FILE, "w") as f:
            json.dump(final_results, f, indent=4)
        logging.info(f"✅ Job Complete. See {FINAL_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
