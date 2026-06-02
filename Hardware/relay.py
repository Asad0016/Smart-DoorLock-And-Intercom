"""
LockDriver.py - Class-level hardware controller for the physical door lock actuator.
Handles GPIO initialization, latch triggering, and cleanup context.
"""

import RPi.GPIO as GPIO
from time import sleep
from logs.logger import getLogger

logger = getLogger("LockDriver")

class MagneticLock:
    def __init__(self, pin: int = 18):
        """
        Initializes the Lock Driver configuration.
        Default pin is set to GPIO 18 (BCM notation).
        """
        self.lock_pin = pin
        self.logger = logger
        self._setup_gpio()

    def _setup_gpio(self):
        """
        Internal helper to initialize GPIO blocks safely.
        """
        try:
            GPIO.setwarnings(False)
            # Ensure safe configuration checking before setting mode
            if GPIO.getmode() is None:
                GPIO.setmode(GPIO.BCM)
                
            GPIO.setup(self.lock_pin, GPIO.OUT)
            
            # Keep the door LOCKED by default on startup
            # Active Low Relay Context: High (1) keeps the relay off/locked
            GPIO.output(self.lock_pin, GPIO.HIGH)
            self.logger.info(f"Hardware initialization complete. Pin {self.lock_pin} configured as LOCK_OUTPUT.")
        except Exception as e:
            self.logger.error(f"Failed to initialize GPIO pin {self.lock_pin}: {e}")

    def unlock(self, hold_time: float = 3.0) -> bool:
        """
        Fires the relay state to break/complete the circuit, releasing the lock actuator.
        Automatically relocks after the hold_time expires.
        """
        try:
            self.logger.info(f"🔑 Hardware Action: Releasing door latch actuator...")
            
            # Active Low Relay: Low (0) turns the relay ON (Unlocks the door)
            GPIO.output(self.lock_pin, GPIO.LOW)
            
            # Keep the door open for the user to pass through
            sleep(hold_time)
            
            # Relock automatically
            GPIO.output(self.lock_pin, GPIO.HIGH)
            self.logger.info("🔒 Hardware Action: Latch engaged. Door Relocked safely.")
            return True
            
        except Exception as e:
            self.logger.error(f"Hardware breakdown during lock actuation layer: {e}")
            # Safety fallback: attempt to relock in case of code mid-crash
            GPIO.output(self.lock_pin, GPIO.HIGH)
            return False

    def cleanup(self):
        """
        Explicitly releases the GPIO pin allocations back to the kernel.
        """
        try:
            GPIO.cleanup(self.lock_pin)
            self.logger.info(f"GPIO Pin {self.lock_pin} released safely via teardown routine.")
        except Exception as e:
            self.logger.error(f"Error executing GPIO pin cleanup: {e}")

    # Context Manager support for 'with' blocks inside main.py
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()