import os
os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"   # Pi 5 GPIO backend

from devicemanager.DeviceManager import DeviceManager

def main():
    dm = DeviceManager()
    if dm.boot():
        dm.run_loop()

if __name__ == "__main__":
    main()