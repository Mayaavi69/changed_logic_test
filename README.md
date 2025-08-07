# Live Theatre Dubbing System

This project implements a real-time live dubbing system for theatre performances. It detects Hindi/Sanskrit speech from a live microphone input, transcribes it, matches the first few tokens with a predefined `script_cues.json` file, and plays the corresponding pre-recorded English audio file.

## Features

- **Real-time Speech-to-Text (STT):** Utilizes `faster-whisper` for efficient transcription of live speech.
- **Fuzzy Text Matching:** Uses `rapidfuzz` library for intelligent matching of detected speech against predefined cues with configurable threshold scoring.
- **Smart Audio Playback:** Plays English audio files with playback suspension to avoid interference during audio output.
- **Manual Overrides:** Hotkeys for playing next, previous, or repeating the last played audio cue.
- **Logging:** Provides clear terminal output for detected speech and matched cues with fuzzy match scores.
- **Offline Capability:** Designed to work entirely offline, suitable for theatre environments.

## Prerequisites

- Python 3.8+
- `pip` (Python package installer)

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    # If this were a git repo, you'd clone it here.
    # For this task, assume you have the files in your working directory.
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `faster-whisper` will download the specified Whisper model (`base` by default) on its first run. The `rapidfuzz` library is required for fuzzy text matching.*

3.  **Prepare audio files:**
    Create an `audio` directory in the project root and place your pre-recorded English `.wav` files inside it. Ensure the filenames match those referenced in `script_cues.json` (e.g., `1.wav`, `2.wav`).

    ```
    your_project_root/
    ├── app.py
    ├── script_cues.json
    ├── requirements.txt
    └── audio/
        ├── 1.wav
        ├── 2.wav
        └── ...
    ```

4.  **Configure `script_cues.json`:**
    Edit the `script_cues.json` file to define your cues and their corresponding English audio files.

    Example `script_cues.json`:
    ```json
    [
      {
        "id": 1,
        "hi_text": "Shri Ram Bolo",
        "en_audio": "1.wav"
      },
      {
        "id": 2,
        "hi_text": "Hanuman Ji Aayenge",
        "en_audio": "2.wav"
      }
    ]
    ```
    - `id`: A unique identifier for the cue.
    - `hi_text`: The text that will be matched against detected speech using fuzzy matching.
    - `en_audio`: The filename of the corresponding English audio file (should be in the `audio/` directory).

## Usage

Run the application from your terminal:

```bash
python app.py
```

The system will start listening to your microphone.

### Hotkeys

-   **N**: Play the next audio cue in `script_cues.json`.
-   **P**: Play the previous audio cue in `script_cues.json`.
-   **R**: Repeat the last played audio cue.
-   **Esc**: Exit the application.

## Logging

The terminal will display real-time logs:

```
[20:38:57] Detected: 'Okay. Okay.'
[20:38:57] Fuzzy match 'Hanuman Ji Aayenge' (score 41.666666666666664)
[20:39:15] Match! Cue 1 → 1.wav
```

## How Fuzzy Matching Works

The system uses the `rapidfuzz` library to intelligently match detected speech against predefined cues:

- **Partial Ratio Scoring:** Uses fuzzy string matching that can handle variations in pronunciation, minor speech recognition errors, and partial matches.
- **Configurable Threshold:** Only matches with scores above `MATCH_THRESHOLD_SCORE` (default: 70) are accepted.
- **Best Match Selection:** Always selects the highest-scoring match from all available cues.
- **Language Flexibility:** Works with any language as the matching is done on transcribed text, not audio directly.

This approach is more robust than exact token matching and handles real-world speech recognition challenges better.

## Customization

-   **Whisper Model Size:** In `app.py`, change `WHISPER_MODEL_SIZE` to `small`, `medium`, or `large` for different accuracy/performance trade-offs.
-   **Language:** The `LANGUAGE` setting in `app.py` is set to `"en"` for English transcription, which works well for detecting various languages.
-   **Fuzzy Match Threshold:** Adjust `MATCH_THRESHOLD_SCORE` in `app.py` to control how closely speech must match cues (0-100, where 100 is exact match).
-   **Silence Detection:** Adjust `SILENCE_THRESHOLD` and `SILENCE_DURATION` in `app.py` to fine-tune when an utterance is considered complete.
-   **Match Cooldown:** Modify `MATCH_COOLDOWN` in `app.py` to control how long the system ignores new matches after a playback.
-   **Playback Suspension:** The system automatically suspends speech detection during audio playback to prevent interference.

## Potential Enhancements (Bonus)

-   **Dockerfile:** For containerized deployment with ALSA/PulseAudio support.
-   **Keyword Spotting:** Integrate `openWakeWord` or a TF-Lite KWS model for more robust initial token detection.
-   **GUI:** A simple graphical user interface for easier control and monitoring.
-   **Error Handling:** More robust error handling for audio devices and file operations.