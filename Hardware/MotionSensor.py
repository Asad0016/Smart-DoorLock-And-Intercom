# MotionSensor.py
"""
PIR Sensor with software debouncing.

False positive problem:
  Raw PIR signals flicker — motion detected for 2-3 seconds, then gone,
  then again immediately.  This is caused by:
    - Sensitivity pot set too high on the HC-SR501 module
    - Time-delay pot set too low (single-trigger mode cycling rapidly)
    - IR interference (sunlight, heater, camera heat)

Software fix — two-stage filter:
  1. DEBOUNCE_MIN_DURATION  : motion is only "real" if the PIR stays HIGH
                              for at least this many seconds continuously.
                              Eliminates 1-3 second spikes entirely.
  2. MOTION_COOLDOWN        : after motion ends, ignore the next N seconds
                              of new triggers.  Prevents rapid re-trigger
                              from the same physical event.
  3. MOTION_EXTEND_WINDOW   : keeps the motion flag alive for this many
                              seconds after the PIR goes LOW.  Prevents
                              the camera from sleeping mid-recognition
                              due to a momentary PIR dropout.

Tune these three constants to match your environment.
"""

import time
import sys
from gpiozero import MotionSensor
from logging import getLogger, StreamHandler, Formatter


# ─────────────────────── DEBOUNCE CONFIG ──────────────────────
# Minimum continuous seconds the PIR must stay HIGH to count as real motion.
# Your false positives last ~2-3s → set this above that.
DEBOUNCE_MIN_DURATION = 2.0   # seconds

# After confirmed motion ends, ignore PIR for this many seconds.
# Prevents the same physical event triggering multiple recording cycles.
MOTION_COOLDOWN = 10.0        # seconds

# Keep the motion flag alive for this long after PIR goes LOW.
# Bridges momentary dropouts while someone is still in the frame.
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
        self._raw_motion        = False   # what PIR hardware currently says
        self._confirmed_motion  = False   # debounced "real" motion flag
        self._motion_start_time = None    # when raw HIGH began (for min-duration check)
        self._motion_end_time   = None    # when confirmed motion ended (for extend window)
        self._last_confirm_end  = None    # when last confirmed motion ended (for cooldown)

        self.pir.when_motion    = self._on_raw_motion
        self.pir.when_no_motion = self._on_raw_no_motion

        self.logger.info(
            f"PIR Sensor initialised on GPIO {pir_pin}  |  "
            f"debounce={DEBOUNCE_MIN_DURATION}s  "
            f"cooldown={MOTION_COOLDOWN}s  "
            f"extend={MOTION_EXTEND_WINDOW}s"
        )
        time.sleep(2)   # hardware warm-up

    # ── Raw hardware callbacks (called by gpiozero thread) ────

    def _on_raw_motion(self):
        now = time.time()

        # In cooldown? Ignore completely.
        if (
            self._last_confirm_end is not None
            and (now - self._last_confirm_end) < MOTION_COOLDOWN
        ):
            remaining = MOTION_COOLDOWN - (now - self._last_confirm_end)
            self.logger.debug(
                f"PIR raw HIGH during cooldown — ignored "
                f"({remaining:.1f}s remaining in cooldown)"
            )
            return

        if not self._raw_motion:
            self._raw_motion        = True
            self._motion_start_time = now
            self.logger.debug("PIR raw HIGH — debounce timer started …")

    def _on_raw_no_motion(self):
        if self._raw_motion:
            self._raw_motion        = False
            self._motion_start_time = None
            self.logger.debug("PIR raw LOW — debounce timer cleared.")

    # ── Public API ────────────────────────────────────────────

    def is_motion_active(self) -> bool:
        """
        Returns True only when CONFIRMED (debounced) motion is present.
        Call this in a polling loop (e.g. every 100 ms from your sensor thread).

        State transitions:
          raw HIGH ≥ DEBOUNCE_MIN_DURATION  →  confirmed = True
          raw LOW  (after confirmed)         →  start extend window
          extend window expired              →  confirmed = False + start cooldown
          cooldown active                    →  new raw HIGH is ignored
        """
        now = time.time()

        # ── Check if raw motion has been sustained long enough ──
        if (
            self._raw_motion
            and self._motion_start_time is not None
            and not self._confirmed_motion
        ):
            held_for = now - self._motion_start_time
            if held_for >= DEBOUNCE_MIN_DURATION:
                self._confirmed_motion = True
                self._motion_end_time  = None
                self.logger.info(
                    f"Motion CONFIRMED (raw HIGH held for {held_for:.1f}s)"
                )

        # ── If confirmed, reset end-time while raw is still HIGH ──
        if self._confirmed_motion and self._raw_motion:
            self._motion_end_time = None   # still moving, don't start extend

        # ── If confirmed but raw just went LOW, start extend window ──
        if self._confirmed_motion and not self._raw_motion:
            if self._motion_end_time is None:
                self._motion_end_time = now
                self.logger.debug(
                    f"PIR went LOW — extend window open for {MOTION_EXTEND_WINDOW}s …"
                )
            elif (now - self._motion_end_time) >= MOTION_EXTEND_WINDOW:
                # Extend window expired → confirmed motion ends
                self._confirmed_motion = False
                self._last_confirm_end = now
                self._motion_end_time  = None
                self.logger.info(
                    f"Motion ENDED (extend window expired). "
                    f"Cooldown active for {MOTION_COOLDOWN}s."
                )

        return self._confirmed_motion

    def cleanup(self):
        self.pir.close()
        self.logger.info("PIR Sensor cleaned up.")


# ══════════════════════════════════════════════════════════════
#  STANDALONE TEST
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

    main_logger.info("PIR Sensor debounce test started …")
    main_logger.info(
        f"Config: debounce={DEBOUNCE_MIN_DURATION}s  "
        f"cooldown={MOTION_COOLDOWN}s  "
        f"extend={MOTION_EXTEND_WINDOW}s"
    )

    pir = PIRSENSOR()
    last_state = False

    try:
        while True:
            current = pir.is_motion_active()
            if current != last_state:
                if current:
                    main_logger.info(">>> CONFIRMED MOTION ACTIVE <<<")
                else:
                    main_logger.info(">>> Motion cleared <<<")
                last_state = current
            time.sleep(0.1)

    except KeyboardInterrupt:
        main_logger.info("Test stopped by user.")
    finally:
        pir.cleanup()