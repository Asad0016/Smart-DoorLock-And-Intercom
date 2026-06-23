# rfid.py
import time
from logs.logger import getLogger
from mfrc522 import SimpleMFRC522

class RFIDManager:
    def __init__(self):
        self.reader = None
        self.logger = getLogger('RFIDManager')
        
        # authorized cards
        self.authorized_cards = {
            "223208080116" : "Admin",
            "631362104739" : "Admin",
        }
        self.last_scanned_uid = None
        self.last_scaned_label = None
        self.logger.info("RFIDManager initialized")
        self.logger.info(f"Authorized cards: {self.authorized_cards}")
        
        self.initialize_reader()
    # ==================== ADD RUNTIME CARD ====================
    def add_authorized_card(self, card_uid: str, label: str = "Unknown"):
        """Dynamically adds a new RFID card UID to the memory cache"""
        card_uid_str = str(card_uid).strip()
        if card_uid_str:
            self.authorized_cards[card_uid_str] = label
            self.logger.info(f"[HARDWARE MAP] Added Card: {card_uid_str} for {label}")
            self.logger.info(f"Current Authorized Cards: {self.authorized_cards}")
            return True
        return False
    def initialize_reader(self):
        """Initialize the MFRC522 reader"""
        try:
            self.reader = SimpleMFRC522()
            self.logger.info("MFRC522 reader successfully initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize MFRC522: {e}")
            self.logger.error("Check SPI enabled and wiring (RST pin optional)")
            self.reader = None
            return False

    def is_authorized_card(self):
        if self.reader is None:
            return False
        try:
            tag_id, _ = self.reader.read_no_block()
            if tag_id:
                tag_id_str = str(tag_id)
                self.logger.info(f"RFID Card Detected: {tag_id_str}")
                if tag_id_str in self.authorized_cards:
                    self.last_scanned_uid = tag_id_str
                    self.last_scanned_label = self.authorized_cards[tag_id_str]
                    self.logger.info(f"AUTHORIZED CARD ({self.last_scanned_label}) – Access Granted!")
                    return True
                self.logger.info("Unauthorized card – Access Denied")
            return False
        except Exception as e:
            self.logger.debug(f"RFID read error (normal in loop): {e}")
            return False

    def read_card_blocking(self):
        """
        Blocking read – use if you want to wait for card
        Returns (tag_id, text) or (None, None)
        """
        if self.reader is None:
            return None, None

        try:
            self.logger.info("Waiting for card (blocking)...")
            tag_id, text = self.reader.read()
            tag_id_str = str(tag_id)
            self.logger.info(f"Card read: ID = {tag_id_str}")
            return tag_id_str, text.strip() if text else ""
        except Exception as e:
            self.logger.error(f"Blocking read error: {e}")
            return None, None

    def stop(self):
        """Cleanup (not strictly needed in SimpleMFRC522, but good practice)"""
        self.logger.info("RFIDManager stopped")

# ================== TEST ==================
if __name__ == "__main__":
    rfid = RFIDManager()
    
    if not rfid.reader:
        print("RFID initialization failed. Exiting.")
    else:
        try:
            while True:
                if rfid.is_authorized_card():
                    print(">>> AUTHORIZED CARD DETECTED! <<<")
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping RFID Manager...")
        finally:
            rfid.stop()