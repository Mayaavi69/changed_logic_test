import json
import os
import time
import threading
import queue
import logging
import asyncio
import numpy as np
from transcriber import Transcriber
from playsound import playsound
from pynput import keyboard
from rapidfuzz import process, fuzz

# --- Configuration ---
AUDIO_DIR = "audio"
SCRIPT_CUES_FILE = "script_cues.json"
SAMPLE_RATE = 16000  # Whisper model expects 16kHz
CHUNK_SIZE = 1024    # Audio buffer size
WHISPER_MODEL_SIZE = "base"  # or "small", "medium", "large"
LANGUAGE = "en"  # English for transcription
SILENCE_THRESHOLD = 0.01  # Adjust as needed
SILENCE_DURATION = 1.0  # Seconds of silence to consider end of utterance
MATCH_COOLDOWN = 5  # Seconds to ignore new matches after a playback
MATCH_THRESHOLD_SCORE = 50  # Fuzzy match threshold (0-100)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s.%(msecs)03d] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# --- Global Variables ---
audio_queue = queue.Queue() # This queue is now primarily for internal Transcriber use, but kept for compatibility if needed
transcription_queue = queue.Queue() # This queue will receive output from Transcriber
playback_queue = queue.Queue()
script_cues = []
cue_texts = [] # This will now store English texts for matching
last_played_cue_id = None
last_match_time = 0
current_cue_index = -1  # For manual override
keyboard_listener = None
is_playing = False  # Flag to suspend listening during playback
transcription_active = False # Flag to control transcription
current_playing_audio_file = None # To track the audio file currently being played
transcriber_instance = None # Global instance of Transcriber

# --- Load Script Cues ---
def load_script_cues():
    global script_cues, cue_texts
    try:
        with open(SCRIPT_CUES_FILE, 'r', encoding='utf-8') as f:
            script_cues = json.load(f)
        # Use 'en_text' for fuzzy matching as transcription is in English
        cue_texts = [cue['en_text'] for cue in script_cues]
        log.info(f"Loaded {len(script_cues)} script cues from {SCRIPT_CUES_FILE}")
    except FileNotFoundError:
        log.error(f"Error: {SCRIPT_CUES_FILE} not found.")
        exit(1)
    except json.JSONDecodeError:
        log.error(f"Error: Could not decode JSON from {SCRIPT_CUES_FILE}.")
        exit(1)

# --- Transcription and Audio Handling with Transcriber Class ---
async def run_transcriber_listen():
    global transcriber_instance, transcription_active
    try:
        transcriber_instance = Transcriber()
        log.info("Transcriber instance created.")
        while True:
            if transcription_active and not transcriber_instance._is_streaming:
                transcriber_instance.start_stream()
                log.info("Transcriber audio stream started.")
            elif not transcription_active and transcriber_instance._is_streaming:
                transcriber_instance.stop_stream()
                log.info("Transcriber audio stream stopped.")

            if transcription_active:
                async for message in transcriber_instance.listen():
                    if message["type"] == "transcript":
                        transcription_queue.put(message["text"])
                    elif message["type"] == "error":
                        log.error(f"Transcriber error: {message['message']}")
                    elif message["type"] == "status":
                        log.info(f"Transcriber status: {message['message']}")
            else:
                await asyncio.sleep(0.1) # Wait if transcription is not active
    except Exception as e:
        log.error(f"Error in run_transcriber_listen: {e}")
    finally:
        if transcriber_instance and transcriber_instance._is_streaming:
            transcriber_instance.stop_stream()
            log.info("Transcriber audio stream stopped.")

# --- Playback Thread ---
def audio_playback():
    global is_playing, current_playing_audio_file
    while True:
        try:
            audio_file = playback_queue.get()
            if audio_file:
                is_playing = True
                current_playing_audio_file = audio_file
                log.info(f"Playing '{audio_file}'...")
                
                # Emit event to web server that cue has started playing
                playsound(audio_file, block=True)
                is_playing = False
                current_playing_audio_file = None
        except Exception as e:
            log.error(f"Audio playback error: {e}")

# --- Main Logic Thread ---
def main_logic():
    global last_match_time, last_played_cue_id, current_cue_index

    while True:
        if is_playing:
            time.sleep(0.1)
            continue
        try:
            transcribed_text = transcription_queue.get(timeout=1)
            log.info(f"Detected: '{transcribed_text}'")

            if time.time() - last_match_time < MATCH_COOLDOWN:
                continue

            match, score, idx = process.extractOne(
                transcribed_text, cue_texts, scorer=fuzz.partial_ratio
            )
            log.info(f"Fuzzy match '{match}' (score {score})")

            if score >= MATCH_THRESHOLD_SCORE:
                cue = script_cues[idx]
                current_cue_index = idx
                log.info(f"Match! Cue {cue['id']} → {cue['en_audio']}")
                playback_queue.put(os.path.join(AUDIO_DIR, cue['en_audio'].split('/')[-1]))
                last_match_time = time.time()
                last_played_cue_id = cue['id']
                
        except queue.Empty:
            continue
        except Exception as e:
            log.error(f"Main logic error: {e}")

# --- Manual Override Hotkeys ---
def on_press(key):
    global current_cue_index, last_match_time, last_played_cue_id
    try:
        if key == keyboard.Key.esc:
            return False
        if hasattr(key, 'char'):
            c = key.char.lower()
            if c in ('n', 'p', 'r'):
                if c == 'n' and current_cue_index < len(script_cues) - 1:
                    current_cue_index += 1
                elif c == 'p' and current_cue_index > 0:
                    current_cue_index -= 1
                elif c == 'r' and last_played_cue_id is not None:
                    current_cue_index = next((i for i, cue in enumerate(script_cues) if cue['id'] == last_played_cue_id), current_cue_index)
                cue = script_cues[current_cue_index]
                log.info(f"Manual '{c}' → Cue {cue['id']} playing {cue['en_audio']}")
                playback_queue.put(os.path.join(AUDIO_DIR, cue['en_audio'].split('/')[-1]))
                last_played_cue_id = cue['id']
                last_match_time = time.time()
                
    except Exception:
        pass

def start_keyboard_listener():
    global keyboard_listener
    keyboard_listener = keyboard.Listener(on_press=on_press)
    keyboard_listener.start()
    log.info("Keyboard listener started. Press N/P/R or Esc.")

# --- Main Execution ---
if __name__ == "__main__":
    print("app.py: Starting application...")
    load_script_cues()

    # Start the transcriber in a separate thread with its own asyncio loop
    def start_transcriber_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_transcriber_listen())

    threading.Thread(target=start_transcriber_thread, daemon=True).start()
    threading.Thread(target=audio_playback, daemon=True).start()
    threading.Thread(target=main_logic, daemon=True).start()

    start_keyboard_listener()

    print("app.py: All threads started. Application running.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Application interrupted by user.")
        print("app.py: Application interrupted by user.")
    finally:
        if keyboard_listener:
            keyboard_listener.stop()
        log.info("Application shutting down.")
        print("app.py: Application shutting down.")
