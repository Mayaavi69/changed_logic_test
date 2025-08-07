#!/usr/bin/env python3
"""
Transcriber Module - Clean speech-to-text streaming interface
Refactored from existing AI-Kilo codebase for real-time WebSocket streaming
"""

import asyncio
import json
import time
import logging
import numpy as np
import whisper
import pyaudio
from typing import Optional, AsyncGenerator
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class Transcriber:
    """
    Clean, maintainable speech-to-text transcriber with streaming capabilities
    """
    
    def __init__(self, microphone_index: Optional[int] = None):
        """
        Initialize the transcriber with audio configuration
        
        Args:
            microphone_index: Specific microphone to use (None for default)
        """
        # Audio configuration
        self.microphone_index = microphone_index
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.channels = 1
        self.audio_format = pyaudio.paFloat32
        
        # Silence detection parameters
        self.silence_threshold = 0.02
        self.silence_duration = 1.5  # Seconds
        self.min_audio_length = 0.5  # Minimum seconds for transcription
        
        # Internal state
        self._audio = None
        self._stream = None
        self._whisper_model = None
        self._is_streaming = False
        self._audio_queue = queue.Queue()
        self._stop_event = threading.Event()
        
        # Initialize components
        self._initialize_whisper()
        self._initialize_audio()
    
    def _initialize_whisper(self):
        """Initialize Whisper model"""
        try:
            logger.info("Loading Whisper model (base)...")
            self._whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def _initialize_audio(self):
        """Initialize PyAudio"""
        try:
            self._audio = pyaudio.PyAudio()
            
            # List available devices for debugging
            if logger.level <= logging.DEBUG:
                self._log_audio_devices()
                
        except Exception as e:
            logger.error(f"Failed to initialize audio: {e}")
            raise
    
    def _log_audio_devices(self):
        """Log available audio devices for debugging"""
        logger.debug("Available audio devices:")
        for i in range(self._audio.get_device_count()):
            info = self._audio.get_device_info_by_index(i)
            logger.debug(f"  {i}: {info['name']} (inputs: {info['maxInputChannels']})")
    
    def start_stream(self):
        """Start the audio stream"""
        if self._is_streaming:
            logger.warning("Stream is already running")
            return
        
        try:
            # Open audio stream
            self._stream = self._audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.microphone_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=None
            )
            
            self._is_streaming = True
            self._stop_event.clear()
            
            # Start audio capture thread
            self._audio_thread = threading.Thread(target=self._audio_capture_loop, daemon=True)
            self._audio_thread.start()
            
            logger.info("Audio stream started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            raise
    
    def stop_stream(self):
        """Gracefully stop the audio stream"""
        if not self._is_streaming:
            return
        
        logger.info("Stopping audio stream...")
        
        # Signal stop
        self._stop_event.set()
        self._is_streaming = False
        
        # Close stream
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                logger.error(f"Error closing audio stream: {e}")
            finally:
                self._stream = None
        
        # Wait for thread to finish
        if hasattr(self, '_audio_thread'):
            self._audio_thread.join(timeout=2.0)
        
        logger.info("Audio stream stopped")
    
    def _audio_capture_loop(self):
        """Audio capture loop running in separate thread"""
        audio_buffer = []
        silence_start = None
        
        while not self._stop_event.is_set() and self._is_streaming:
            try:
                # Read audio data
                data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                
                # Calculate volume level
                volume = np.sqrt(np.mean(audio_chunk**2))
                
                if volume > self.silence_threshold:
                    # Speech detected
                    silence_start = None
                    audio_buffer.extend(audio_chunk)
                else:
                    # Silence detected
                    if len(audio_buffer) > 0:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start >= self.silence_duration:
                            # End of utterance detected
                            if len(audio_buffer) >= self.sample_rate * self.min_audio_length:
                                # Queue audio for transcription
                                audio_array = np.array(audio_buffer, dtype=np.float32)
                                self._audio_queue.put(audio_array)
                            
                            # Reset buffer
                            audio_buffer = []
                            silence_start = None
                
                # Prevent buffer from growing too large
                if len(audio_buffer) > self.sample_rate * 30:  # 30 seconds max
                    logger.warning("Audio buffer too large, clearing...")
                    audio_buffer = []
                    silence_start = None
                
            except Exception as e:
                if self._is_streaming:  # Only log if we're supposed to be streaming
                    logger.error(f"Audio capture error: {e}")
                break
    
    async def listen(self) -> AsyncGenerator[dict, None]:
        """
        Async generator yielding transcribed text chunks
        
        Yields:
            dict: JSON message with transcription results or errors
        """
        if not self._is_streaming:
            yield {
                "type": "error",
                "message": "Audio stream not started. Call start_stream() first.",
                "timestamp": time.time()
            }
            return
        
        logger.info("Starting transcription listener...")
        
        # Initial connection message
        yield {
            "type": "status",
            "message": "Connected - listening for speech...",
            "timestamp": time.time()
        }
        
        while self._is_streaming:
            try:
                # Check for new audio to transcribe
                if not self._audio_queue.empty():
                    audio_data = self._audio_queue.get_nowait()
                    
                    # Transcribe audio
                    transcript = await self._transcribe_audio(audio_data)
                    
                    if transcript and transcript.strip():
                        yield {
                            "type": "transcript",
                            "text": transcript.strip(),
                            "timestamp": time.time()
                        }
                
                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Transcription error: {e}")
                yield {
                    "type": "error",
                    "message": f"Transcription error: {str(e)}",
                    "timestamp": time.time()
                }
                
                # Brief pause before continuing
                await asyncio.sleep(1.0)
        
        logger.info("Transcription listener stopped")
        yield {
            "type": "status",
            "message": "Transcription stopped",
            "timestamp": time.time()
        }
    
    async def _transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """
        Transcribe audio data using Whisper
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Transcribed text or None if transcription fails
        """
        try:
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._whisper_model.transcribe(
                    audio_data,
                    language="en",  # Can be made configurable
                    fp16=False
                )
            )
            
            return result["text"]
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return None
    
    def __enter__(self):
        """Context manager entry"""
        self.start_stream()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_stream()
        
        # Cleanup
        if self._audio:
            self._audio.terminate()
    
    def __del__(self):
        """Destructor - ensure cleanup"""
        try:
            self.stop_stream()
            if self._audio:
                self._audio.terminate()
        except:
            pass  # Ignore cleanup errors in destructor


# Example usage and testing
if __name__ == "__main__":
    async def test_transcriber():
        """Test function for the transcriber"""
        transcriber = Transcriber()
        
        try:
            transcriber.start_stream()
            
            print("🎤 Listening for speech... (speak now)")
            print("Press Ctrl+C to stop\n")
            
            async for message in transcriber.listen():
                if message["type"] == "transcript":
                    print(f"📝 [{time.strftime('%H:%M:%S')}] {message['text']}")
                elif message["type"] == "error":
                    print(f"❌ Error: {message['message']}")
                elif message["type"] == "status":
                    print(f"ℹ️  {message['message']}")
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
        finally:
            transcriber.stop_stream()
    
    # Run test
    asyncio.run(test_transcriber())
