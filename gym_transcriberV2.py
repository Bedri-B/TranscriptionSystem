import os
import json
import logging
import torch
import ollama
from openai import OpenAI
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()

# FIXED IMPORT FOR MOVIEPY 2.0+
from moviepy import VideoFileClip

# --- Configuration ---
VIDEO_FOLDER = "videos"
INTERMEDIATE_FILE = "transcriptions_fast.json"
FINAL_OUTPUT_FILE = "gym_exercises_final.json"
OLLAMA_MODEL = "gemma2:latest"
OPENAI_MODEL = "gpt-4.1-mini"
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
        return "cuda", "float16"
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
            # Note: In MoviePy 2.0, logger=None is the default behavior if not specified
            try:
                clip = VideoFileClip(video_path)
                clip.audio.write_audiofile(temp_audio, logger=None)
                clip.close()
            except Exception as e:
                # Fallback for some MoviePy versions that might complain about logger
                logging.warning(
                    f"Standard export failed ({e}), retrying without logger arg..."
                )
                clip = VideoFileClip(video_path)
                clip.audio.write_audiofile(temp_audio)
                clip.close()

            # 2. Fast Transcription
            segments, info = model.transcribe(temp_audio, beam_size=1, language="en")

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
                try:
                    os.remove(temp_audio)
                except PermissionError:
                    pass  # File sometimes held by process on Windows

    # Save Phase 1 results
    with open(INTERMEDIATE_FILE, "w") as f:
        json.dump(results, f, indent=4)

    # Cleanup VRAM
    del model
    torch.cuda.empty_cache()
    logging.info("🧹 GPU Memory Cleared.")

    return results


def analyze_with_ollama(transcription_data):
    """Phase 2: Ollama Analysis"""
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


def analyze_with_openai(transcription_data):
    """Phase 2: OpenAI Analysis"""
    logging.info("--- PHASE 2: Analyzing with OpenAI ---")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("CRITICAL: OPENAI_API_KEY not found in environment variables.")
        return []

    client = OpenAI(api_key=api_key)
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
            logging.info(f"Analyzing {filename} with OpenAI...")
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            item["exercise_type"] = response.choices[0].message.content.strip()
            final_data.append(item)
            logging.info(f"  -> Identified: {item['exercise_type']}")

        except Exception as e:
            logging.error(f"OpenAI error: {e}")

    return final_data


def main():
    device, compute_type = check_gpu()

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
        print("\nChoose analysis backend:")
        print("1. Ollama (Local)")
        print("2. OpenAI (Cloud)")
        choice = input("Enter choice (1/2): ").strip()

        if choice == "2":
            final_results = analyze_with_openai(transcriptions)
        else:
            final_results = analyze_with_ollama(transcriptions)

        with open(FINAL_OUTPUT_FILE, "w") as f:
            json.dump(final_results, f, indent=4)
        logging.info(f"✅ Job Complete. See {FINAL_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
