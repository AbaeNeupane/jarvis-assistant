# Jarvis - Offline Personal Assistant

This is an open-source, fully offline personal assistant inspired by Jarvis. It uses a wake word to activate, transcribes speech to text, processes it with a local LLM, and speaks the response.

## Technology Stack

- **Wake Word:** OpenWakeWord
- **Speech-to-Text (STT):** Whisper.cpp
- **LLM Server:** Ollama
- **LLM Model:** Mistral (7B) new Phi3:mini
- **Text-to-Speech (TTS):** Piper TTS
- **UI Framework:** Flask

## Setup

1.  **Create the File Structure:**
    - Create all the folders and empty files as laid out in the project structure.

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install/Compile External Tools:**
    - **Ollama:** Download and install from [https://ollama.com/](https://ollama.com/). Then run `ollama pull mistral` or `ollama pull phi3:mini`.
    - **Whisper.cpp:** Compile from source from its official GitHub repository.
    - **Piper:** Download the executable for your OS from the Piper TTS GitHub releases page.

4.  **Download Audio Models:**
    - Download the required models for OpenWakeWord, Whisper.cpp, and Piper TTS.
    - Place them in the corresponding subdirectories inside `/audio_models`.
    - **Important:** Update the file paths at the top of `stt_module.py` and `tts_module.py` to point to your compiled executables and downloaded models.

5.  **Run the application:**
    ```bash
    python  main.py
    ```
    Then, open your web browser to `http://127.0.0.1:5000`.
