import os
import time
import datetime
import subprocess
import signal
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt
import pyttsx3
from logs.logger import getLogger

class INMP441MicRecorder:
    def __init__(self, 
                 rate=48000, 
                 channels=2,
                 device_string="hw:1,0",
                 output_folder='/home/doorlock/DoorLock/data',
                 volume_gain=8.0):
        
        self.rate = rate
        self.channels = channels
        self.device_string = device_string
        self.output_folder = output_folder
        self.volume_gain = volume_gain
        
        self.logger = getLogger('INMP441_Mic')
        self.is_recording = False
        self.process = None
        self.filepath = None
        
        os.makedirs(self.output_folder, exist_ok=True)

        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 1.0)
        except Exception as e:
            self.logger.error(f"TTS Engine Initialization failed: {e}")
            self.tts_engine = None

    def speak(self, text):
        if self.tts_engine:
            self.logger.info(f"Speaking: {text}")
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                self.logger.error(f"TTS Speech execution interrupted: {e}")

    def play_beep(self):
        self.speak("Beep")

    # ── YOUR ORIGINAL FILTER — kept exactly ──────────────────

    def _butter_highpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def _apply_noise_filter(self, data):
        # 1. Remove DC Offset
        centered_data = data - np.mean(data)
        # 2. High-Pass Filter above 150Hz
        b, a = self._butter_highpass(cutoff=150.0, fs=self.rate, order=4)
        from scipy.signal import lfilter
        filtered_audio = lfilter(b, a, centered_data)
        # 3. Noise Gate
        threshold = 30.0
        filtered_audio[np.abs(filtered_audio) < threshold] = 0.0
        return filtered_audio

    # ── NEW: added on top of your existing filter ─────────────

    def _bandpass_voice(self, data: np.ndarray) -> np.ndarray:
        """
        Keeps only 100Hz–4000Hz — the exact human voice band.
        Runs AFTER your existing highpass so we get double filtering.
        Uses SOS (second order sections) — stable at 48kHz unlike ba form.
        """
        sos = sosfilt(
            butter(4, [100.0, 4000.0], btype='bandpass', fs=self.rate, output='sos'),
            data
        )
        return sos

    def _noise_gate_rms(self, data: np.ndarray) -> np.ndarray:
        """
        Zeros out 20ms blocks that are below RMS energy threshold.
        Removes hiss/hum between spoken words without touching speech.
        Threshold 0.01 — lower if voice is being cut, raise if hiss remains.
        """
        block  = int(0.02 * self.rate)   # 20ms
        thresh = 0.01
        out    = data.copy().astype(np.float32)
        for i in range(0, len(out) - block, block):
            rms = np.sqrt(np.mean(out[i:i+block] ** 2))
            if rms < thresh:
                out[i:i+block] = 0.0
        return out

    # ── RECORDING — YOUR ORIGINAL CODE unchanged ─────────────

    def start_recording(self):
        if self.is_recording:
            self.logger.warning("Mic driver instance is already recording actively.")
            return

        t_0 = datetime.datetime.now()
        filename = datetime.datetime.strftime(t_0, '%Y_%m_%d_%H_%M_%S_mic') + '.wav'
        self.filepath = os.path.join(self.output_folder, filename)

        self.cmd = [
            'arecord',
            '-D', self.device_string,
            '-c', str(self.channels),
            '-r', str(self.rate),
            '-f', 'S32_LE',
            '-t', 'wav',
            self.filepath
        ]

        try:
            self.is_recording = True
            self.logger.info(f"Spawning filtered ALSA pipeline on device: {self.device_string}")
            self.process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except Exception as e:
            self.logger.error(f"Failed to spawn native system recording process: {e}")
            self.is_recording = False

    def stop_recording(self, custom_filename=None):
        if not self.is_recording or not self.process:
            self.logger.warning("Execution halt requested but no active tracking process found.")
            return None

        self.is_recording = False
        self.logger.info("Terminating ALSA capture pipe and engaging DSP filtration...")
        
        try:
            self.process.send_signal(signal.SIGINT)
            self.process.wait(timeout=2)
            time.sleep(0.2)
            
            if os.path.exists(self.filepath):
                fs, stereo_data = wavfile.read(self.filepath)
                
                if len(stereo_data.shape) > 1:
                    # ── YOUR ORIGINAL processing ──────────────
                    left_channel  = stereo_data[:, 0]
                    parsed_audio  = left_channel >> 14       # 32-bit left-justified → 18-bit
                    clean_audio   = self._apply_noise_filter(parsed_audio)  # your original filter
                    boosted_audio = clean_audio * self.volume_gain
                    audio_matrix  = np.clip(boosted_audio, -32768, 32767).astype(np.float32)

                    # ── NEW: normalize to [-1, 1] then apply extra filters ──
                    peak = np.max(np.abs(audio_matrix))
                    if peak > 0:
                        normalized = audio_matrix / peak     # float [-1, 1]
                    else:
                        normalized = audio_matrix

                    # Extra bandpass — keeps only voice frequencies
                    voice_only = self._bandpass_voice(normalized)

                    # RMS noise gate — zeros hiss between words
                    gated = self._noise_gate_rms(voice_only)

                    # Final int16 output
                    final_audio = np.clip(gated * 32767, -32768, 32767).astype(np.int16)

                else:
                    # Mono fallback — your original path
                    boosted = stereo_data * self.volume_gain
                    final_audio = np.clip(boosted, -32768, 32767).astype(np.int16)

                if custom_filename:
                    # Support both full paths and plain filenames
                    if os.path.isabs(custom_filename):
                        new_filepath = custom_filename
                    else:
                        new_filepath = os.path.join(self.output_folder, custom_filename)
                    wavfile.write(new_filepath, self.rate, final_audio)
                    if os.path.exists(self.filepath) and new_filepath != self.filepath:
                        os.remove(self.filepath)
                    self.filepath = new_filepath
                else:
                    wavfile.write(self.filepath, self.rate, final_audio)

                self.logger.info(f"Filtered audio saved: {self.filepath}")
                return self.filepath
            else:
                self.logger.error("ALSA pipeline closed but output file not found.")
                return None

        except Exception as e:
            self.logger.error(f"DSP processing pipeline crashed: {e}")
            return None

    def cleanup(self):
        if self.process:
            try:
                self.process.kill()
            except Exception:
                pass
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except Exception:
                pass
        self.logger.info("Hardware mic management layer and filters safely closed.")


if __name__ == "__main__":
    mic = INMP441MicRecorder(
        rate=48000,
        channels=2,
        device_string="hw:1,0",
        volume_gain=15.0
    )

    print("Press Enter to start 10 second filtered test recording...")
    input()
    
    mic.start_recording()
    time.sleep(10)
    
    filepath = mic.stop_recording()
    if filepath:
        print(f"Success! Noise-filtered file generated at path: {filepath}")
        
    mic.cleanup()