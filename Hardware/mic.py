# import pyaudio
# import numpy as np
# import wave
# import time
# import datetime
# import os
# import threading
# from scipy.signal import butter, lfilter

# from logs.logger import getLogger


# class INMP441MicRecorder:
#     def __init__(self,
#                  chunk=4096,
#                  rate=48000,
#                  channels=2,
#                  device_string="googlevoicehat",
#                  output_folder="./data/",
#                  volume_gain=60.0,  # Keeping your working gain configuration
#                  highpass_cutoff_hz=100): # Raised slightly to cut more low-end hum

#         self.chunk           = chunk
#         self.rate            = rate
#         self.channels        = channels
#         self.device_string   = device_string
#         self.output_folder   = output_folder
#         self.volume_gain     = volume_gain

#         self.format        = pyaudio.paInt32
#         self.buffer_format = np.int32

#         self.p              = pyaudio.PyAudio()
#         self.stream         = None
#         self.frames         = []
#         self.is_recording   = False
#         self._record_thread = None
#         self._chunk_count   = 0

#         self.logger = getLogger("INMP441_Mic")

#         self.device_index = self._find_device_index()
#         if self.device_index is None:
#             self.logger.error(f'No device matching "{device_string}" found.')
#         else:
#             info = self.p.get_device_info_by_index(self.device_index)
#             self.logger.info(
#                 f"Device [{self.device_index}]: {info['name']} | "
#                 f"max_inputs={info['maxInputChannels']} | "
#                 f"default_rate={int(info['defaultSampleRate'])}"
#             )

#         # High-pass filter setup to clean DC offset and room rumble
#         nyq = rate / 2.0
#         b, a = butter(4, highpass_cutoff_hz / nyq, btype='high')
#         self._hp_b = b
#         self._hp_a = a

#         os.makedirs(output_folder, exist_ok=True)

#         # TTS setup
#         self.tts_engine = None
#         try:
#             import pyttsx3
#             self.tts_engine = pyttsx3.init()
#             self.tts_engine.setProperty('rate', 150)
#             self.tts_engine.setProperty('volume', 1.0)
#         except Exception as e:
#             self.logger.warning(f"TTS unavailable: {e}")

#     def _find_device_index(self):
#         count = self.p.get_device_count()
#         self.logger.info(f"Scanning {count} PyAudio devices…")
#         for i in range(count):
#             info = self.p.get_device_info_by_index(i)
#             if (info["maxInputChannels"] > 0
#                     and self.device_string.lower() in info["name"].lower()):
#                 self.logger.info(f'Matched device [{i}]: {info["name"]}')
#                 return i
#         self.logger.warning(f'"{self.device_string}" not found. Available inputs:')
#         for i in range(count):
#             info = self.p.get_device_info_by_index(i)
#             if info["maxInputChannels"] > 0:
#                 self.logger.warning(f'  [{i}] {info["name"]}')
#         return None

#     def speak(self, text: str):
#         if self.tts_engine:
#             try:
#                 self.tts_engine.say(text)
#                 self.tts_engine.runAndWait()
#             except Exception as e:
#                 self.logger.error(f"TTS error: {e}")

#     def _open_stream(self):
#         formats_to_try = [
#             (pyaudio.paInt32, np.int32,  "paInt32"),
#             (pyaudio.paInt16, np.int16,  "paInt16"),
#         ]

#         for pa_fmt, np_fmt, name in formats_to_try:
#             try:
#                 s = self.p.open(
#                     format             = pa_fmt,
#                     rate               = self.rate,
#                     channels           = self.channels,
#                     input_device_index = self.device_index,
#                     input              = True,
#                     frames_per_buffer  = self.chunk,
#                 )
#                 self.logger.info(f"Stream opened with format {name}")
#                 self.active_fmt_name = name
#                 return s
#             except Exception as e:
#                 self.logger.warning(f"Format {name} failed: {e}")

#         raise RuntimeError("No working audio format found for this device.")

#     def _start_stream(self):
#         if self.device_index is None:
#             raise RuntimeError("No valid audio device.")
#         self.stream = self._open_stream()
#         self.stream.stop_stream()
#         self.logger.info("Stream ready (stopped).")

#     def start_recording(self):
#         if self.is_recording:
#             self.logger.warning("Already recording.")
#             return

#         if self.stream is None:
#             self._start_stream()

#         self.frames       = []
#         self.is_recording = True
#         self._chunk_count = 0

#         self.stream.start_stream()

#         # Flush hardware buffer
#         self.logger.info("Flushing hardware buffer…")
#         for _ in range(4):
#             self.stream.read(self.chunk, exception_on_overflow=False)

#         self.logger.info(
#             f"Recording started | fmt={self.active_fmt_name} | "
#             f"rate={self.rate} | channels={self.channels} | "
#             f"device={self.device_index} | gain={self.volume_gain}"
#         )

#         self._record_thread = threading.Thread(
#             target=self._record_loop,
#             daemon=True,
#             name="MicRecordLoop",
#         )
#         self._record_thread.start()

#     def _record_loop(self):
#         from scipy.signal import lfilter
        
#         # --- DSP State Variables ---
#         dc_estimate = 0.0
#         alpha_dc = 0.999    # Time constant for DC tracking
#         smoothed_val = 0.0
#         alpha_smooth = 0.4  # Low-pass smoothing factor for high-frequency hiss
        
#         # AGC (Automatic Gain Control) parameters
#         current_volume_envelope = 0.01
#         target_level = 0.5   # Target peak amplitude (out of 1.0)
#         max_agc_gain = 120.0 # Maximum allowed dynamic amplification
#         min_agc_gain = 1.0
#         agc_gain = self.volume_gain # Start with your default gain (60.0)

#         while self.is_recording:
#             try:
#                 raw = self.stream.read(self.chunk, exception_on_overflow=False)
#                 self._chunk_count += 1

#                 # 1. Decode Raw Data
#                 pcm32 = np.frombuffer(raw, dtype=np.int32).copy()
#                 right = pcm32[1::2] # Target the channel where L/R pin dictates

#                 # 2. Extract 24-bit MSB Core
#                 shifted = (right >> 8).astype(np.int32)
#                 signal = shifted.astype(np.float64) / 8388608.0

#                 # 3. DSP STEP 1: Hardware DC Offset Removal (Leaky Integrator)
#                 # This centers the audio wave precisely at 0.0 before applying any gain
#                 output_signal = np.zeros_like(signal)
#                 for i in range(len(signal)):
#                     dc_estimate = (alpha_dc * dc_estimate) + ((1 - alpha_dc) * signal[i])
#                     output_signal[i] = signal[i] - dc_estimate
                
#                 # 4. DSP STEP 2: Scipy High-Pass Filter (Phase-2 Rumble Cut)
#                 output_signal = lfilter(self._hp_b, self._hp_a, output_signal)

#                 # 5. DSP STEP 3: High-Frequency Hiss / Clock Jitter Reduction
#                 # Exponential Moving Average acting as a low-pass filter to deaden harsh static
#                 for i in range(len(output_signal)):
#                     smoothed_val = (alpha_smooth * smoothed_val) + ((1 - alpha_smooth) * output_signal[i])
#                     output_signal[i] = smoothed_val

#                 # 6. DSP STEP 4: Adaptive Automatic Gain Control (AGC)
#                 # Calculates the envelope of this block to dynamically fix low volume
#                 block_peak = np.max(np.abs(output_signal))
#                 if block_peak > 0.001:
#                     # Smooth the volume envelope tracking over chunks
#                     current_volume_envelope = (0.8 * current_volume_envelope) + (0.2 * block_peak)
                    
#                     # Calculate ideal gain required to bring the peak to the target level
#                     ideal_gain = target_level / current_volume_envelope
                    
#                     # Clamp the gain to safe operational boundaries to prevent runaway feedback
#                     agc_gain = np.clip(ideal_gain, min_agc_gain, max_agc_gain)

#                 # Apply the dynamically adjusted adaptive gain
#                 output_signal *= agc_gain

#                 # 7. DSP STEP 5: Soft Peak Limiter
#                 # Prevents any leftover transient peaks from generating harsh square-wave distortion
#                 output_signal = np.clip(output_signal, -1.0, 1.0)

#                 # Diagnostic logging to monitor dynamic adjustments
#                 if self._chunk_count % 10 == 0:
#                     self.logger.info(
#                         f"[DSP Chunk {self._chunk_count}] "
#                         f"Tracked Peak={block_peak:.4f} | "
#                         f"Dynamic AGC Gain Applied={agc_gain:.1f} | "
#                         f"Output Signal Range=[{output_signal.min():.2f}, {output_signal.max():.2f}]"
#                     )

#                 # Convert to clean 16-bit PCM streaming frames
#                 pcm16 = (output_signal * 32767).astype(np.int16)
#                 self.frames.append(pcm16.tobytes())

#             except Exception as e:
#                 if self.is_recording:
#                     self.logger.error(f"DSP Record loop error: {e}", exc_info=True)

#     def stop_recording(self, custom_filename: str | None = None) -> str | None:
#         if not self.is_recording:
#             self.logger.warning("Not recording.")
#             return None

#         self.is_recording = False

#         if self._record_thread:
#             self._record_thread.join(timeout=3.0)

#         self.stream.stop_stream()

#         total_chunks = self._chunk_count
#         total_seconds = (total_chunks * self.chunk) / self.rate
#         self.logger.info(
#             f"Recording stopped | chunks={total_chunks} | "
#             f"approx_duration={total_seconds:.1f}s | "
#             f"frames_stored={len(self.frames)}"
#         )

#         if not self.frames:
#             self.logger.error("ZERO frames captured — nothing to save.")
#             return None

#         all_audio = np.frombuffer(b"".join(self.frames), dtype=np.int16)
#         max_val = np.max(np.abs(all_audio))
#         energy = np.mean(np.abs(all_audio))
        
#         self.logger.info(
#             f"Final audio stats: mean_abs={energy:.1f} max={max_val} "
#             f"({'HAS SIGNAL' if max_val > 200 else 'SILENT OR UNREADABLE'})"
#         )

#         if custom_filename:
#             filepath = custom_filename
#         else:
#             ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_mic")
#             filepath = os.path.join(self.output_folder, ts + ".wav")

#         try:
#             with wave.open(filepath, "wb") as wf:
#                 wf.setnchannels(1)      # Output dynamic single-track mono track
#                 wf.setsampwidth(2)      # Standard 16-bit sample width (2 bytes)
#                 wf.setframerate(self.rate) # Write out at proper hardware clock speed (48000 Hz)
#                 wf.writeframes(b"".join(self.frames))
#             self.logger.info(f"Saved: {filepath}")
#             return filepath
#         except Exception as e:
#             self.logger.error(f"WAV write failed: {e}")
#             return None

#     def cleanup(self):
#         self.is_recording = False
#         if self.stream:
#             try:
#                 self.stream.close()
#             except Exception:
#                 pass
#         self.p.terminate()
#         self.logger.info("Mic cleaned up.")


# if __name__ == "__main__":
#     # Initialize with your verified working gain configurations
#     mic = INMP441MicRecorder(
#         chunk              = 4096,
#         rate               = 48000,
#         channels           = 2,
#         device_string      = "googlevoicehat",
#         volume_gain        = 100.0,
#         highpass_cutoff_hz = 100,
#     )

#     print("\nSpeak loudly after pressing Enter. Recording 10 seconds…")
#     input()

#     mic.start_recording()
#     time.sleep(10)

#     path = mic.stop_recording()
#     if path:
#         print(f"\n[SUCCESS] Saved to: {path}")
#     else:
#         print("\n[FAILED] No file saved.")

#     mic.cleanup()
import pyaudio
import numpy as np
import wave
import time
import datetime
import os
import threading
from scipy.signal import butter, lfilter

from logs.logger import getLogger


class INMP441MicRecorder:
    def __init__(self,
                 chunk=4096,
                 rate=48000,
                 channels=2,
                 device_string="googlevoicehat",
                 output_folder="./data/",
                 volume_gain=120.0,  # Boosted default baseline gain stage
                 highpass_cutoff_hz=100): 

        self.chunk           = chunk
        self.rate            = rate
        self.channels        = channels
        self.device_string   = device_string
        self.output_folder   = output_folder
        self.volume_gain     = volume_gain

        self.format        = pyaudio.paInt32
        self.buffer_format = np.int32

        self.p              = pyaudio.PyAudio()
        self.stream         = None
        self.frames         = []
        self.is_recording   = False
        self._record_thread = None
        self._chunk_count   = 0

        self.logger = getLogger("INMP441_Mic")

        self.device_index = self._find_device_index()
        if self.device_index is None:
            self.logger.error(f'No device matching "{device_string}" found.')
        else:
            info = self.p.get_device_info_by_index(self.device_index)
            self.logger.info(
                f"Device [{self.device_index}]: {info['name']} | "
                f"max_inputs={info['maxInputChannels']} | "
                f"default_rate={int(info['defaultSampleRate'])}"
            )

        # High-pass filter setup to clean DC offset and room rumble
        nyq = rate / 2.0
        b, a = butter(4, highpass_cutoff_hz / nyq, btype='high')
        self._hp_b = b
        self._hp_a = a

        os.makedirs(output_folder, exist_ok=True)

        # TTS setup
        self.tts_engine = None
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 1.0)
        except Exception as e:
            self.tts_engine = None

    def _find_device_index(self):
        count = self.p.get_device_count()
        self.logger.info(f"Scanning {count} PyAudio devices...")
        for i in range(count):
            info = self.p.get_device_info_by_index(i)
            if (info["maxInputChannels"] > 0 and self.device_string.lower() in info["name"].lower()):
                self.logger.info(f'Matched device [{i}]: {info["name"]}')
                return i
        return 2

    def speak(self, text: str):
        if self.tts_engine:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                self.logger.error(f"TTS error: {e}")

    def _open_stream(self):
        try:
            s = self.p.open(
                format             = self.format,
                rate               = self.rate,
                channels           = self.channels,
                input_device_index = self.device_index,
                input              = True,
                frames_per_buffer  = self.chunk,
            )
            self.logger.info("Stream opened with 32-bit container depth.")
            self.active_fmt_name = "paInt32"
            return s
        except Exception as e:
            raise RuntimeError(f"Audio buffer allocation failed: {e}")

    def _start_stream(self):
        if self.device_index is None:
            raise RuntimeError("No valid audio device.")
        self.stream = self._open_stream()
        self.stream.stop_stream()
        self.logger.info("Stream ready (stopped).")

    def start_recording(self):
        if self.is_recording:
            self.logger.warning("Already recording.")
            return

        if self.stream is None:
            self._start_stream()

        self.frames       = []
        self.is_recording = True
        self._chunk_count = 0

        self.stream.start_stream()

        # Flush hardware buffer
        self.logger.info("Flushing hardware buffer...")
        for _ in range(4):
            self.stream.read(self.chunk, exception_on_overflow=False)

        self._record_thread = threading.Thread(
            target=self._record_loop,
            daemon=True,
            name="MicRecordLoop",
        )
        self._record_thread.start()

    def _record_loop(self):
        # --- DSP State Variables ---
        dc_estimate = 0.0
        alpha_dc = 0.999

        smoothed_val = 0.0
        alpha_smooth = 0.25

        # --- Spectral noise reduction state ---
        # This is the actual noise-REMOVAL stage (the gate/AGC below only turn
        # things up or down in time — they can't pull noise out from underneath
        # your voice). We continuously learn what the room's background noise
        # looks like in the frequency domain, then subtract that exact profile
        # out of every single frame, including while you're talking.
        noise_mag_profile    = None    # learned noise spectrum (built up over time)
        noise_profile_alpha  = 0.95    # how slowly the learned noise profile updates
        over_subtraction     = 1.6     # how aggressively to subtract the noise profile
        spectral_floor_ratio = 0.06    # residual floor -> prevents "musical noise" artifacts
        raw_noise_floor      = 0.0008  # time-domain RMS estimate of the quiet background
        calibration_chunks   = 5       # first ~0.4s assumed to be background noise only
        chunk_index          = 0

        # --- AGC (envelope) state: drives gain UP when you are actually speaking ---
        current_volume_envelope = 0.01
        target_level   = 1.0     # aim closer to full scale before the soft limiter
        max_agc_gain    = 400.0
        min_agc_gain    = 1.0
        agc_gain        = self.volume_gain

        # Asymmetric envelope tracking: rise FAST on loud peaks (prevents overshoot),
        # fall SLOWLY as sound fades. This is what stops the "gain spike" that used
        # to blast noise right at the moment you finish speaking and trail off.
        envelope_attack  = 0.5
        envelope_release = 0.04

        # Gain rate-limit: agc_gain is only allowed to change by a small percentage
        # per chunk. Combined with the asymmetric envelope above, this removes the
        # end-of-recording noise burst entirely (no sudden gain jumps are possible).
        max_gain_step_pct = 0.06

        # --- Noise-gate state: second-stage safety net for whatever residual
        # noise the spectral stage above didn't catch, during pure silence ---
        gate_gain              = 1.0
        noise_floor            = 0.0008
        gate_threshold_factor  = 3.0
        gate_attack            = 0.6
        gate_release           = 0.05
        silence_attenuation    = 0.06

        # --- Loudness on playback ---
        makeup_gain = 2.2  # extra loudness applied after gating, before the soft limiter

        while self.is_recording:
            try:
                raw = self.stream.read(self.chunk, exception_on_overflow=False)
                self._chunk_count += 1
                chunk_index += 1

                # 1. Decode Raw Bit Data Streams
                pcm32 = np.frombuffer(raw, dtype=np.int32).copy()

                # Try 1::2 (Right Channel), switch to 0::2 if wire layout dictates Left alignment
                right_channel = pcm32[1::2]

                # 2. Baseline map over 32-bit boundary scale
                signal = right_channel.astype(np.float64) / 2147483648.0

                # Toned-down fixed pre-gain; AGC + makeup gain handle final loudness now.
                signal = signal * 18.0

                # 3. SPECTRAL NOISE REDUCTION (the actual noise removal step)
                # Transform to the frequency domain, learn/subtract the noise profile,
                # then transform back. This pulls hiss/hum out even while you talk,
                # which time-domain gating alone can never do.
                spectrum  = np.fft.rfft(signal)
                magnitude = np.abs(spectrum)
                phase     = np.angle(spectrum)

                if noise_mag_profile is None:
                    noise_mag_profile = magnitude.copy()

                raw_rms = np.sqrt(np.mean(signal ** 2))
                is_quiet_frame = (chunk_index <= calibration_chunks) or (raw_rms < raw_noise_floor * 2.5)

                if is_quiet_frame:
                    raw_noise_floor = max((0.95 * raw_noise_floor) + (0.05 * raw_rms), 1e-5)
                    noise_mag_profile = (
                        (noise_profile_alpha * noise_mag_profile)
                        + ((1 - noise_profile_alpha) * magnitude)
                    )

                clean_magnitude = magnitude - (over_subtraction * noise_mag_profile)
                clean_magnitude = np.maximum(clean_magnitude, spectral_floor_ratio * magnitude)

                clean_spectrum = clean_magnitude * np.exp(1j * phase)
                signal = np.fft.irfft(clean_spectrum, n=len(signal))

                # 4. Hardware DC Offset Removal
                output_signal = np.zeros_like(signal)
                for i in range(len(signal)):
                    dc_estimate = (alpha_dc * dc_estimate) + ((1 - alpha_dc) * signal[i])
                    output_signal[i] = signal[i] - dc_estimate

                # 5. Filter Rumble
                output_signal = lfilter(self._hp_b, self._hp_a, output_signal)

                # 6. Light residual smoothing
                for i in range(len(output_signal)):
                    smoothed_val = (alpha_smooth * smoothed_val) + ((1 - alpha_smooth) * output_signal[i])
                    output_signal[i] = smoothed_val

                block_peak = np.max(np.abs(output_signal))

                # 7a. Track an adaptive noise floor, but only update it while it's clearly quiet
                if block_peak < noise_floor * 4.0:
                    noise_floor = (0.98 * noise_floor) + (0.02 * block_peak)
                noise_floor = max(noise_floor, 1e-5)

                voice_present = block_peak > max(noise_floor * gate_threshold_factor, 0.0025)

                # 7b. Envelope tracking: fast rise on loud peaks, slow fall as sound fades.
                if block_peak > current_volume_envelope:
                    current_volume_envelope = (
                        (1 - envelope_attack) * current_volume_envelope + envelope_attack * block_peak
                    )
                else:
                    current_volume_envelope = (
                        (1 - envelope_release) * current_volume_envelope + envelope_release * block_peak
                    )

                desired_gain = np.clip(
                    target_level / max(current_volume_envelope, 1e-6),
                    min_agc_gain, max_agc_gain
                )

                # Rate-limit how fast agc_gain itself can move. No sudden gain jumps.
                max_step = agc_gain * max_gain_step_pct
                if desired_gain > agc_gain:
                    agc_gain = agc_gain + min(desired_gain - agc_gain, max_step)
                else:
                    agc_gain = agc_gain - min(agc_gain - desired_gain, max_step)

                if voice_present:
                    gate_gain = (1 - gate_attack) * gate_gain + gate_attack * 1.0
                else:
                    # Gate closes smoothly (no click) and suppresses the noise floor
                    # instead of amplifying it.
                    gate_gain = (1 - gate_release) * gate_gain + gate_release * silence_attenuation

                output_signal = output_signal * agc_gain * gate_gain * makeup_gain

                # 8. Soft-knee limiter (tanh) instead of a hard clip.
                output_signal = np.tanh(output_signal)

                # Convert float arrays back cleanly to standard 16-bit WAV PCM
                pcm16 = (output_signal * 32767).astype(np.int16)
                self.frames.append(pcm16.tobytes())

            except Exception as e:
                if self.is_recording:
                    self.logger.error(f"DSP Record loop error: {e}")

    def stop_recording(self, custom_filename: str | None = None) -> str | None:
        if not self.is_recording:
            return None

        self.is_recording = False
        if self._record_thread:
            self._record_thread.join(timeout=3.0)

        self.stream.stop_stream()

        if not self.frames:
            self.logger.error("Empty frames array stack.")
            return None

        if custom_filename:
            filepath = custom_filename
        else:
            ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_boosted")
            filepath = os.path.join(self.output_folder, ts + ".wav")

        try:
            # Apply a short fade-out over the final ~120ms of audio. Even with the
            # AGC fix above, this guarantees there's never an abrupt cut or pop at
            # the very end of the recording.
            all_bytes = b"".join(self.frames)
            audio = np.frombuffer(all_bytes, dtype=np.int16).copy()

            fade_ms = 120
            fade_samples = int(self.rate * fade_ms / 1000.0)
            if 0 < fade_samples < len(audio):
                ramp = np.linspace(1.0, 0.0, fade_samples)
                tail = audio[-fade_samples:].astype(np.float64) * ramp
                audio[-fade_samples:] = tail.astype(np.int16)

            with wave.open(filepath, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.rate)
                wf.writeframes(audio.tobytes())
            self.logger.info(f"Boosted audio track stored safely at: {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"WAV pipeline write failure: {e}")
            return None

    def cleanup(self):
        self.is_recording = False
        if self.stream:
            try: self.stream.close()
            except Exception: pass
        self.p.terminate()


if __name__ == "__main__":
    mic = INMP441MicRecorder(
        chunk=4096,
        rate=48000,
        channels=2,
        device_string="googlevoicehat",
        volume_gain=150.0, # Starting baseline processing amplification level
    )

    print("\n[READY] Speak closely and loudly... Press ENTER to start recording 10s.")
    input()

    mic.start_recording()
    time.sleep(10)

    path = mic.stop_recording()
    if path:
        print(f"\n[SUCCESS] Track successfully saved to: {path}")
    else:
        print("\n[FAILED] Audio generation failure.")

    mic.cleanup()