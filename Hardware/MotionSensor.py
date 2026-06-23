# MotionSensor.py
"""
PIR Sensor with software debouncing and explicit processing sync.
Includes a standalone heavy processing simulator test at the end.
"""

import time
import sys
from gpiozero import MotionSensor
from logging import getLogger, StreamHandler, Formatter

# ─────────────────────── DEBOUNCE CONFIG ──────────────────────
DEBOUNCE_MIN_DURATION = 2.0   # seconds
MOTION_COOLDOWN = 10.0        # seconds
MOTION_EXTEND_WINDOW = 5.0    # seconds
# ──────────────────────────────────────────────────────────────

class PIRSENSOR:
    def __init__(self, pir_pin: int = 23):
        self.logger = getLogger("PIR_SENSOR")

        if not self.logger.handlers:
            handler = StreamHandler(sys.stdout)
            formatter = Formatter(
                "%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel("INFO")

        self.pir = MotionSensor(pir_pin)

        # ── Internal debounce state ────────────────────────────
        self._raw_motion        = False   
        self._confirmed_motion  = False   
        self._motion_start_time = None    
        self._motion_end_time   = None    
        self._last_confirm_end  = None    

        self.pir.when_motion    = self._on_raw_motion
        self.pir.when_no_motion = self._on_raw_no_motion

        self.logger.info(
            f"PIR Sensor initialised on GPIO {pir_pin}  |  "
            f"debounce={DEBOUNCE_MIN_DURATION}s  "
            f"cooldown={MOTION_COOLDOWN}s  "
            f"extend={MOTION_EXTEND_WINDOW}s"
        )
        time.sleep(2)  # hardware warm-up

    def _on_raw_motion(self):
        now = time.time()
        if (
            self._last_confirm_end is not None
            and (now - self._last_confirm_end) < MOTION_COOLDOWN
        ):
            return

        if not self._raw_motion:
            self._raw_motion        = True
            self._motion_start_time = now

    def _on_raw_no_motion(self):
        if self._raw_motion:
            self._raw_motion        = False
            self._motion_start_time = None

    def reset_cooldown(self):
        """
        CRITICAL FIX: Call this immediately AFTER your face recognition/heavy processing finishes.
        It forces the 10-second cooldown to start from THIS EXACT MOMENT.
        """
        now = time.time()
        self._confirmed_motion = False
        self._raw_motion = False
        self._motion_start_time = None
        self._motion_end_time = None
        self._last_confirm_end = now
        self.logger.info(f"Processing finished. Cooldown FORCE-STARTED for next {MOTION_COOLDOWN}s.")

    def is_motion_active(self) -> bool:
        now = time.time()

        # In cooldown? Lock out any active states.
        if (
            self._last_confirm_end is not None
            and (now - self._last_confirm_end) < MOTION_COOLDOWN
        ):
            return False

        # Check if raw motion sustained long enough
        if (
            self._raw_motion
            and self._motion_start_time is not None
            and not self._confirmed_motion
        ):
            held_for = now - self._motion_start_time
            if held_for >= DEBOUNCE_MIN_DURATION:
                self._confirmed_motion = True
                self._motion_end_time  = None
                self.logger.info(f"Motion CONFIRMED (raw HIGH held for {held_for:.1f}s)")

        if self._confirmed_motion and self._raw_motion:
            self._motion_end_time = None   

        if self._confirmed_motion and not self._raw_motion:
            if self._motion_end_time is None:
                self._motion_end_time = now
            elif (now - self._motion_end_time) >= MOTION_EXTEND_WINDOW:
                self._confirmed_motion = False
                self._last_confirm_end = now
                self._motion_end_time  = None
                self.logger.info(f"Motion ENDED naturally. Cooldown active for {MOTION_COOLDOWN}s.")

        return self._confirmed_motion

    def cleanup(self):
        self.pir.close()
        self.logger.info("PIR Sensor cleaned up.")


# ══════════════════════════════════════════════════════════════
#  UPDATED STANDALONE TEST (SIMULATING HEAVY WORKFLOW)
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main_logger = getLogger("MAIN")
    if not main_logger.handlers:
        handler = StreamHandler(sys.stdout)
        formatter = Formatter(
            "%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s"
        )
        handler.setFormatter(formatter)
        main_logger.addHandler(handler)
        main_logger.setLevel("INFO")

    main_logger.info("PIR Sync-Lock test engine started...")
    
    pir = PIRSENSOR()
    last_state = False

    try:
        while True:
            current = pir.is_motion_active()
            
            if current != last_state:
                if current:
                    main_logger.info(">>> CONFIRMED MOTION ACTIVE <<<")
                    
                    # ─── SIMULATING YOUR DOOR OPERATIONAL/FACE REC WORKLOAD ───
                    main_logger.info("[WORKLOAD] Simulating Face Recognition / Door Lock Operations (Takes 5s)...")
                    for i in range(5, 0, -1):
                        main_logger.info(f"[WORKLOAD] Processing... {i}s remaining")
                        time.sleep(1.0)
                    
                    main_logger.info("[WORKLOAD] Operations finished successfully!")
                    # ──────────────────────────────────────────────────────────
                    
                    # Force reset the cooldown right here so old background state is wiped
                    pir.reset_cooldown()
                    
                    # Reset our tracking state to match the fresh cooldown setup
                    last_state = False
                else:
                    main_logger.info(">>> Motion cleared naturally <<<")
                    last_state = current
                    
            time.sleep(0.1)

    except KeyboardInterrupt:
        main_logger.info("Test stopped by user.")
    finally:
        pir.cleanup()