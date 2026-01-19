# Gym Exercise Transcription System

A powerful video transcription and exercise classification system that uses AI to automatically identify gym exercises from video files.

## Features

- **Dual Transcription Backend**:
  - **Local (Hybrid)**: Free, unlimited transcription using Faster-Whisper (GPU recommended).
  - **Cloud (OpenAI API)**: Lightweight, no GPU required, pay-per-minute (requires OpenAI key).
- **AI-Powered Exercise Classification**: Supports both local (Ollama) and cloud-based (OpenAI) analysis.
- **GPU Acceleration**: Automatically detects and utilizes NVIDIA CUDA for faster local processing.
- **Flexible Configuration**: JSON-based config for all settings.

## Choose Your Setup

This project supports two installation modes depending on your hardware and preferences.

### Option A: Hybrid / Local Setup (Recommended for GPU users)

**Best for:** Free usage, sensitive data, offline capability.  
**Requirements:** NVIDIA GPU (Recommended), Python 3.8+  
**Pros:** Free to run. **Cons:** Requires installing PyTorch & heavier dependencies.

1. **Create Virtual Environment**:

   ```bash
   python -m venv .venv_local
   # Windows:
   .venv_local\Scripts\activate
   # Mac/Linux:
   source .venv_local/bin/activate
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements\requirements.txt
   ```

3. **Install FFmpeg**: (See Installation Section below)

### Option B: Full Cloud / API Setup (Lightweight)

**Best for:** Laptops without GPU, quick setup, cloud preference.  
**Requirements:** OpenAI API Key.  
**Pros:** Easy install, works on any machine. **Cons:** Costs money ($0.006/min for Whisper).

1. **Create Virtual Environment**:

   ```bash
   python -m venv .venv_api
   # Windows:
   .venv_api\Scripts\activate
   # Mac/Linux:
   source .venv_api/bin/activate
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements\requirements_api.txt
   ```

3. **Install FFmpeg**: (See Installation Section below)

---

## Installation & Setup

1. **Clone the repository**
2. **Install FFmpeg**:
   - Windows: [Download FFmpeg](https://ffmpeg.org/download.html) and add to PATH.
   - Mac: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg`

3. **Environment Variables**:
   Create a `.env` file in the project root:

   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

4. **Configuration (`config.json`)**:
   Set your preferred transcription method:
   - For **Local Whisper**: Set `"transcription_backend": "local"`
   - For **OpenAI API**: Set `"transcription_backend": "openai"`

   ```json
   {
     "whisper": {
       "transcription_backend": "openai"
     }
   }
   ```

## Usage

1. **Add Videos**: Place video files in the `videos` folder.
2. **Run the Script**:
   ```bash
   python transcriber.py
   ```
3. **Follow Prompts**: The script will verify your GPU (if local) and ask which AI model to use for the final exercise analysis (Ollama or OpenAI).

## Troubleshooting

- **"Module not found"**: Ensure you activated the correct `.venv` and installed the matching requirements file.
- **"CUDA not found"**: Only matters for 'local' mode. If you lack a GPU, correct this by installing the CPU version of torch or switching to 'openai' backend.
