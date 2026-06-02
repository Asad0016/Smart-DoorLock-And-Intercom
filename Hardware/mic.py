import pyaudio
import numpy as np
import wave
import time
import datetime
import os
import threading
from logs.logger import getLogger
import pyttsx3

class INMP441MicRecorder:
    def __init__(self, 
                 chunk=1024,
                 rate=48000,
                 channels=1,
                 device_index=0,
                 output_folder='./data/',
                 volume_gain=10.0):
        self.chunk = chunk
        self.rate = rate
        self.channels = channels
        self.device_index = device_index
        self.output_folder = output_folder
        self.volume_gain = volume_gain
        
        self.format = pyaudio.paInt16
        self.buffer_format = np.int16
        
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        
        self.logger = getLogger('INMP441_Mic')
        self.logger.info("INMP441 Mic Recorder initialized")
        
        os.makedirs(self.output_folder, exist_ok=True)

        # TTS
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 1.0)

    def speak(self, text):
        self.logger.info(f"Speaking: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def play_beep(self):
        self.speak("Beep")

    def _start_stream(self):
        self.stream = self.p.open(
            format=self.format,
            rate=self.rate,
            channels=self.channels,
            input_device_index=self.device_index,
            input=True,
            frames_per_buffer=self.chunk
        )
        self.stream.stop_stream()

    def start_recording(self):
        """Non-blocking start – background thread mein recording chalegi"""
        if self.is_recording:
            self.logger.warning("Mic already recording")
            return

        if self.stream is None:
            self._start_stream()

        self.frames = []
        self.is_recording = True
        
        self.stream.start_stream()
        self.stream.read(self.chunk, exception_on_overflow=False)  # Flush
        self.logger.info("Mic recording started (background)")

        def _record_loop():
            while self.is_recording:
                try:
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=self.buffer_format)
                    adjusted = np.int16(audio_data * self.volume_gain)
                    self.frames.append(adjusted.tobytes())
                except Exception as e:
                    if self.is_recording:
                        self.logger.error(f"Read error: {e}")
        
        threading.Thread(target=_record_loop, daemon=True).start()

    def stop_recording(self, custom_filename=None):
        """Stop background recording and save file"""
        if not self.is_recording:
            self.logger.warning("No active recording")
            return None

        self.is_recording = False
        time.sleep(0.3)

        self.stream.stop_stream()
        self.logger.info("Mic recording stopped")

        t_0 = datetime.datetime.now()
        if custom_filename:
            filename = custom_filename
        else:
            filename = datetime.datetime.strftime(t_0, '%Y_%m_%d_%H_%M_%S_mic') + '.wav'
        
        filepath = os.path.join(self.output_folder, filename)
        
        try:
            wf = wave.open(filepath, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            self.logger.info(f"Audio saved: {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Save failed: {e}")
            return None

    def cleanup(self):
        if self.stream:
            try:
                self.stream.close()
            except:
                pass
        self.p.terminate()
        self.tts_engine.stop()
        self.logger.info("Mic recorder cleaned up")

# if __name__ == "__main__":
#     mic = INMP441MicRecorder(
#         chunk=1024,
#         rate=48000,
#         channels=1,
#         device_index=0,
#         volume_gain=10.0
#     )

#     print("Press Enter to start 10 second recording...")
#     input()
    
#     mic.start_recording()
#     time.sleep(10)
#     filepath = mic.stop_recording()
    
#     if filepath:
#         print(f"Recording saved: {filepath}")
    
#     mic.cleanup()