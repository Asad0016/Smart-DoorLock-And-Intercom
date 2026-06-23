"""
LockDriver.py - Class-level hardware controller for the physical door lock actuator.
Production baseline for Raspberry Pi 5. Handles precise state synchronization.
"""

from gpiozero import OutputDevice
from time import sleep
from logs.logger import getLogger

logger = getLogger("LockDriver")

class MagneticLock:
    def __init__(self, pin: int = 17):
        self.lock_pin = pin
        self.logger = logger
        self.device = None
        self._setup_gpio()

    def _setup_gpio(self):
        try:
            # Active-Low Relay ke liye direct standard architecture initialize karein
            self.device = OutputDevice(self.lock_pin, active_high=False, initial_value=False)
            self.logger.info(f"🔒 [SAFE INIT] GPIO Pin {self.lock_pin} baseline established successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize GPIO: {e}")

    def unlock(self, hold_time: float = 10.0) -> bool:
        return self.unlock_and_release(hold_time=hold_time)

    def unlock_and_release(self, hold_time: float = 10.0) -> bool:
        if not self.device:
            self._setup_gpio()
            
        try:
            # 1. Door Unlocks on Authorized Input
            self.logger.info("🔑 [AUTHORIZED TRIGGER] Match Found. Opening lock... Relay ON")
            self.device.on()  
            
            # 2. Strict Hold Time
            self.logger.info(f"⌛ Keeping door unlocked for exactly {hold_time} seconds...")
            sleep(hold_time)
            
            # 3. Forcing Absolute Lock Release
            self.logger.info("🔒 [HOLD TIME EXPIRED] Cutting off pulse... Relay OFF")
            self.device.off()   
            
            # Note: close() nahi call karenge taake state floating na ho aur logic lock rahe
            self.logger.info("🔒 System returned to standing lock position.")
            return True
        except Exception as e:
            self.logger.error(f"Error during actuation layer: {e}")
            if self.device:
                self.device.off()
            return False

    def cleanup(self):
        try:
            if self.device:
                self.device.off()
                self.device.close()
        except:
            pass

# =========================================================================
# PRODUCTION STANDBY APPLICATION TEST BENCH
# =========================================================================
if __name__ == "__main__":
    import sys
    print("==========================================================")
    print("         🔒 SYSTEM DRIVER BASELINE INTERACTION            ")
    print("==========================================================")
    
    lock = MagneticLock(pin=17)
    print("\n--- System Standing By. Use Main App or press Enter to test ---")
    
    try:
        while True:
            input("\n[STANDBY] Press ENTER to trigger 10-second authorized access...")
            lock.unlock_and_release(hold_time=10.0)
    except KeyboardInterrupt:
        print("\nExiting cleanly...")
    finally:
        lock.cleanup()