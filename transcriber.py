import os
import json
import logging

try:
    import torch
except ImportError:
    torch = None

try:
    import ollama
except ImportError:
    ollama = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# FIXED IMPORT FOR MOVIEPY 2.0+
from moviepy import VideoFileClip


# --- Load Configuration ---
def load_config():
    """Load configuration from config.json file."""
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        logging.warning(f"Config file '{config_path}' not found. Using default values.")
        return {
            "paths": {
                "video_folder": "videos",
                "intermediate_file": "transcriptions_fast.json",
                "final_output_file": "gym_exercises_final.json",
                "log_file": "process_fast.log",
            },
            "models": {
                "ollama_model": "gemma2:latest",
                "openai_model": "gpt-4.1-mini",
                "whisper_model_size": "distil-large-v3",
            },
            "whisper": {
                "beam_size": 1,
                "language": "en",
                "supported_video_formats": [".mp4", ".mov", ".avi", ".mkv"],
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(levelname)s - %(message)s",
            },
            "prompts": {
                "exercise_analysis": 'Analyze this gym video transcription. Return ONLY the name of the exercise performed.\nIf it is unclear, return "Unknown".\n\nTranscription: "{text}"'
            },
        }


# Load configuration
CONFIG = load_config()

# --- Configuration ---
VIDEO_FOLDER = CONFIG["paths"]["video_folder"]
RESULTS_FOLDER = CONFIG["paths"].get("results_folder", "results")
INTERMEDIATE_FILE = os.path.join(RESULTS_FOLDER, CONFIG["paths"]["intermediate_file"])
FINAL_OUTPUT_FILE = os.path.join(RESULTS_FOLDER, CONFIG["paths"]["final_output_file"])
LOG_FILE = CONFIG["paths"][
    "log_file"
]  # Keep logs in root or move them too? Let's move them to results as well if desired, but user said "results" usually for data. Let's stick to JSONs in results for now as per "results folder" usually implies output data. But to keep it clean, let's put logs in results too or just root. I'll keep logs in root unless specified, but wait, if it's "results", maybe all outputs?
# Let's check the previous values. intermediate and final were just filenames.
# I will make intermediate and final files go into RESULTS_FOLDER. log file I will leave as is unless I decide to move it.
# Actually, having logs in results is good practice. I will move logs to results folder too if it makes sense.
# But config has "log_file": "process_fast.log". I'll keep it simple: intermediate and final files in results.

# Ensure results folder exists
os.makedirs(RESULTS_FOLDER, exist_ok=True)

OLLAMA_MODEL = CONFIG["models"]["ollama_model"]
OPENAI_MODEL = CONFIG["models"]["openai_model"]
WHISPER_MODEL_SIZE = CONFIG["models"]["whisper_model_size"]
BEAM_SIZE = CONFIG["whisper"]["beam_size"]
LANGUAGE = CONFIG["whisper"]["language"]
SUPPORTED_FORMATS = tuple(CONFIG["whisper"]["supported_video_formats"])
EXERCISE_ANALYSIS_PROMPT = CONFIG["prompts"]["exercise_analysis"]

# --- Setup Logging ---
logging.basicConfig(
    level=getattr(logging, CONFIG["logging"]["level"]),
    format=CONFIG["logging"]["format"],
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)


def check_gpu():
    """Checks for NVIDIA GPU and sets compute precision."""
    if torch and torch.cuda.is_available():
        logging.info(f"CUDA Detected: {torch.cuda.get_device_name(0)}")
        return "cuda", "float16"
    else:
        logging.warning(
            "CUDA not found or Torch not installed. Running on CPU (slow) or using API."
        )
        return "cpu", "int8"


def extract_and_transcribe(device, compute_type):
    """Phase 1: High-speed transcription using Faster-Whisper."""
    if not os.path.exists(VIDEO_FOLDER):
        logging.error(f"Folder '{VIDEO_FOLDER}' not found.")
        return []

    if WhisperModel is None:
        logging.critical(
            "Faster-Whisper is not installed. Use 'openai' backend or install 'faster-whisper'."
        )
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
        f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith(SUPPORTED_FORMATS)
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
            segments, info = model.transcribe(
                temp_audio, beam_size=BEAM_SIZE, language=LANGUAGE
            )

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
    if torch:
        torch.cuda.empty_cache()
    logging.info("GPU Memory Cleared.")

    return results


def transcribe_with_openai_api():
    """Phase 1: Transcription using OpenAI Whisper API."""
    if not os.path.exists(VIDEO_FOLDER):
        logging.error(f"Folder '{VIDEO_FOLDER}' not found.")
        return []

    logging.info("--- PHASE 1: Transcribing with OpenAI API ---")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("CRITICAL: OPENAI_API_KEY not found in environment variables.")
        return []

    client = OpenAI(api_key=api_key)
    results = []

    files = [
        f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith(SUPPORTED_FORMATS)
    ]

    for filename in files:
        video_path = os.path.join(VIDEO_FOLDER, filename)
        temp_audio = "temp_api.mp3"  # OpenAI supports mp3, lighter upload

        try:
            logging.info(f"Processing: {filename}")

            # 1. Fast Audio Extraction
            try:
                clip = VideoFileClip(video_path)
                # Lower bitrate for faster upload, mono is fine for speech
                clip.audio.write_audiofile(
                    temp_audio, logger=None, bitrate="64k", nbytes=2, codec="libmp3lame"
                )
                clip.close()
            except Exception as e:
                logging.warning(
                    f"Standard export failed ({e}), retrying without logger arg..."
                )
                clip = VideoFileClip(video_path)
                clip.audio.write_audiofile(
                    temp_audio, bitrate="64k", nbytes=2, codec="libmp3lame"
                )
                clip.close()

            # 2. API Transcription
            with open(temp_audio, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1", file=audio_file, language=LANGUAGE
                )

            full_text = transcript.text.strip()

            if full_text:
                results.append({"file": filename, "transcription": full_text})
                logging.info(f"  -> Success ({len(full_text)} chars).")
            else:
                logging.warning(f"  -> No speech detected.")

        except Exception as e:
            logging.error(f"Error on {filename}: {e}")

        finally:
            if os.path.exists(temp_audio):
                try:
                    os.remove(temp_audio)
                except PermissionError:
                    pass

    # Save Phase 1 results
    with open(INTERMEDIATE_FILE, "w") as f:
        json.dump(results, f, indent=4)

    return results


def analyze_with_ollama(transcription_data):
    """Phase 2: Ollama Analysis"""
    logging.info("--- PHASE 2: Analyzing with Ollama ---")

    if ollama is None:
        logging.critical(
            "Ollama library is not installed. Install it or use OpenAI backend."
        )
        return []

    final_data = []

    for item in transcription_data:
        filename = item["file"]
        text = item["transcription"]

        prompt = EXERCISE_ANALYSIS_PROMPT.format(text=text)

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

        prompt = EXERCISE_ANALYSIS_PROMPT.format(text=text)

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

    TRANSCRIPTION_BACKEND = CONFIG["whisper"].get("transcription_backend", "local")

    if os.path.exists(INTERMEDIATE_FILE):
        choice = input(
            f"Found '{INTERMEDIATE_FILE}'. Use existing transcriptions? (y/n): "
        ).lower()
        if choice == "y":
            with open(INTERMEDIATE_FILE, "r") as f:
                transcriptions = json.load(f)
        else:
            if TRANSCRIPTION_BACKEND == "openai":
                transcriptions = transcribe_with_openai_api()
            else:
                transcriptions = extract_and_transcribe(device, compute_type)
    else:
        if TRANSCRIPTION_BACKEND == "openai":
            transcriptions = transcribe_with_openai_api()
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
        logging.info(f"Job Complete. See {FINAL_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
