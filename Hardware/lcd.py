import time
import socket
from datetime import datetime

from luma.core.interface.serial import spi
from luma.lcd.device import ili9341
from PIL import Image, ImageDraw, ImageFont
import RPi.GPIO as GPIO


class TouchKeypad:
    # ==================== THEME ====================
    COLOR_BG        = (16, 18, 26)
    COLOR_PANEL     = (24, 26, 38)
    COLOR_HEADER    = (10, 12, 20)
    COLOR_BTN       = (34, 37, 52)
    COLOR_BTN_EDGE  = (54, 58, 78)
    COLOR_ACCENT    = (88, 166, 255)
    COLOR_SUCCESS   = (46, 204, 113)
    COLOR_DANGER    = (231, 76, 60)
    COLOR_AMBER     = (240, 165, 60)
    COLOR_VIOLET    = (170, 130, 255)
    COLOR_TEXT      = (235, 237, 245)
    COLOR_TEXT_DIM  = (120, 126, 145)
    COLOR_DOT_EMPTY = (55, 58, 76)
    COLOR_WIFI_OFF  = (90, 94, 110)

    MAX_CONNECT_ATTEMPTS = 10
    NETWORK_RECHECK_SECS = 20

    # ── Doorbell anti-spam tuning ──
    BELL_NOTIFY_COOLDOWN_SECS = 20    # min gap between two accepted notifications
    BELL_SPAM_THRESHOLD       = 3     # presses inside the cooldown window before lockout
    BELL_LOCKOUT_SECS         = 200   # how long the button stays disabled after spam

    def __init__(self):
        try:
            GPIO.setmode(GPIO.BCM)
        except Exception:
            pass

        # ==================== HARDWARE INIT ====================
        self.lcd_serial = spi(
            port=0, device=1, gpio_DC=24, gpio_RST=12, clock_frequency=8000000
        )
        self.display = ili9341(self.lcd_serial, width=320, height=240, rotate=1, bgr=True)

        self.W = self.display.size[0]
        self.H = self.display.size[1]
        print(f"[SYSTEM] Engine Initialized at Core: {self.W}x{self.H}")

        # ==================== FONTS ====================
        self.font_title = self._load_font(16)
        self.font_big = self._load_font(28)
        self.font_subtitle = self._load_font(12)
        self.font_btn = self._load_font(20)
        self.font_status = self._load_font(14)
        self.font_clock = self._load_font(13)

        # ==================== TOUCH PINS ====================
        self.T_CLK = 5; self.T_CS = 6; self.T_DIN = 13; self.T_DO = 16; self.T_IRQ = 26

        GPIO.setup(self.T_CLK, GPIO.OUT)
        GPIO.setup(self.T_CS, GPIO.OUT)
        GPIO.setup(self.T_DIN, GPIO.OUT)
        GPIO.setup(self.T_DO, GPIO.IN)
        GPIO.setup(self.T_IRQ, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.output(self.T_CS, GPIO.HIGH)

        # Calibration
        self.X_RAW_TOP = 150
        self.X_RAW_BOTTOM = 1800
        self.Y_RAW_LEFT = 1815
        self.Y_RAW_RIGHT = 272

        # State
        self.correct_password = "1234"
        self.current_input = ""
        self.mode = "HOME"                 # ← boots into the new home screen
        self.header_title = "DOOR LOCK"
        self.status_text = ""
        self.status_color = self.COLOR_ACCENT
        self.show_icon = None
        self.online = False

        # AlertManager is injected later (it doesn't exist yet when LCD boots)
        self.alert_mgr = None

        # Doorbell anti-spam state
        self._last_bell_time = None
        self._bell_spam_count = 0
        self._bell_disabled_until = None

        self.buttons = {}
        self._setup_keypad_geometry()
        self._setup_home_geometry()

        # Boot
        self._wait_for_internet()
        self._show_splash_screen()
        time.sleep(1.5)
        self._draw_home_ui()
        self._last_refresh = time.time()

    def _load_font(self, size):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
        return ImageFont.load_default()

    def _center_text_x(self, draw, text, font, area_w, area_x=0):
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        return area_x + (area_w - text_w) // 2

    # ==================== ALERT MANAGER INJECTION ====================
    def set_alert_manager(self, alert_mgr):
        """
        Called by DeviceManager right after AlertManager is constructed.
        LCD boots before AlertManager exists, so this can't be a constructor arg.
        """
        self.alert_mgr = alert_mgr

    # ==================== NETWORK ====================
    def _has_internet(self, timeout=1.5):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect(("8.8.8.8", 53))
            s.close()
            return True
        except OSError:
            return False

    def _wifi_quality_percent(self):
        try:
            with open("/proc/net/wireless") as f:
                lines = f.readlines()
            if len(lines) < 3: return None
            parts = lines[2].split()
            quality = float(parts[2].strip('.'))
            return max(0, min(100, int(quality / 70 * 100)))
        except Exception:
            return None

    def _wifi_bars(self):
        if not self.online: return 0
        pct = self._wifi_quality_percent()
        return 4 if pct is None else max(1, min(4, 1 + int(pct / 25)))

    def _wait_for_internet(self):
        attempt = 0
        while attempt < self.MAX_CONNECT_ATTEMPTS:
            self._show_connecting_screen(attempt + 1)
            if self._has_internet():
                self.online = True
                return
            attempt += 1
            time.sleep(1.2)
        self.online = False

    def _maybe_recheck_network(self):
        self.online = self._has_internet(timeout=1.0)

    # ==================== TOUCH ====================
    def _bitbang_touch_tx_rx(self, cmd):
        GPIO.output(self.T_CS, GPIO.LOW)
        for i in range(8):
            GPIO.output(self.T_DIN, (cmd >> (7 - i)) & 1)
            GPIO.output(self.T_CLK, GPIO.HIGH)
            GPIO.output(self.T_CLK, GPIO.LOW)

        result = 0
        for i in range(12):
            GPIO.output(self.T_CLK, GPIO.HIGH)
            result = (result << 1) | GPIO.input(self.T_DO)
            GPIO.output(self.T_CLK, GPIO.LOW)

        GPIO.output(self.T_CS, GPIO.HIGH)
        return result

    def get_touch_coordinates(self):
        if GPIO.input(self.T_IRQ) == GPIO.LOW:
            raw_x = self._bitbang_touch_tx_rx(0x90)
            raw_y = self._bitbang_touch_tx_rx(0xD0)
            if 50 < raw_x < 2200 and 100 < raw_y < 2200:
                val_x = int((self.Y_RAW_LEFT - raw_y) * self.W / (self.Y_RAW_LEFT - self.Y_RAW_RIGHT))
                val_y = int((raw_x - self.X_RAW_TOP) * self.H / (self.X_RAW_BOTTOM - self.X_RAW_TOP))
                final_x = max(0, min(val_x, self.W - 1))
                final_y = max(0, min(val_y, self.H - 1))
                return final_x, final_y
        return None

    # ==================== LAYOUT ====================
    def _setup_keypad_geometry(self):
        matrix = [['1','2','3'],['4','5','6'],['7','8','9'],['P','0','C']]
        btn_w, btn_h = 64, 42
        gap_x, gap_y = 12, 12
        grid_w = 3 * btn_w + 2 * gap_x
        grid_h = 4 * btn_h + 3 * gap_y
        start_x = (self.W - grid_w) // 2
        start_y = 64
        self.grid_bounds = (start_x, start_y, start_x + grid_w, start_y + grid_h)

        for r, row in enumerate(matrix):
            for c, char in enumerate(row):
                x1 = start_x + c * (btn_w + gap_x)
                y1 = start_y + r * (btn_h + gap_y)
                self.buttons[char] = (x1, y1, x1 + btn_w, y1 + btn_h)

        self.status_panel_top = start_y + grid_h + 14

    def _setup_home_geometry(self):
        """Two big buttons stacked under the header: DOOR BELL and ENTER PIN."""
        btn_w, btn_h = 220, 70
        gap = 20
        start_x = (self.W - btn_w) // 2
        bell_y = 64
        pin_y  = bell_y + btn_h + gap

        self.home_buttons = {
            "BELL": (start_x, bell_y, start_x + btn_w, bell_y + btn_h),
            "PIN":  (start_x, pin_y,  start_x + btn_w, pin_y + btn_h),
        }
        self.home_status_y = pin_y + btn_h + 22

    # ==================== DRAWING HELPERS ====================
    def _rounded(self, draw, bounds, radius, **kwargs):
        try:
            draw.rounded_rectangle(bounds, radius=radius, **kwargs)
        except AttributeError:
            draw.rectangle(bounds, **kwargs)

    def _draw_wifi_icon(self, draw, cx, cy, connected, bars):
        dot_r = 2
        draw.ellipse((cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r),
                     fill=self.COLOR_ACCENT if bars >= 1 else self.COLOR_WIFI_OFF)

        radii = (6, 10, 14)
        for i, r in enumerate(radii):
            tier = i + 2
            color = self.COLOR_ACCENT if (connected and bars >= tier) else self.COLOR_WIFI_OFF
            draw.arc((cx - r, cy - r, cx + r, cy + r), start=210, end=330,
                     fill=color, width=2)

        if not connected:
            draw.line((cx - 16, cy + 10, cx + 16, cy - 14), fill=self.COLOR_DANGER, width=2)

    def _draw_header(self, draw):
        """Time top-left, title centered, wifi icon top-right — shared by HOME and keypad pages."""
        draw.rectangle((0, 0, self.W, 48), fill=self.COLOR_HEADER)
        header_center_y = 24

        time_str = datetime.now().strftime("%H:%M")
        draw.text((12, header_center_y - 7), time_str, font=self.font_clock, fill=self.COLOR_TEXT_DIM)

        tx = self._center_text_x(draw, self.header_title, self.font_title, self.W)
        draw.text((tx, header_center_y - 9), self.header_title, font=self.font_title, fill=self.COLOR_TEXT)

        self._draw_wifi_icon(draw, self.W - 32, header_center_y + 2, self.online, self._wifi_bars())

    # ==================== BEAUTIFUL SCREENS ====================
    def _show_connecting_screen(self, attempt):
        img = Image.new("RGB", (self.W, self.H), self.COLOR_BG)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, self.W, 130), fill=self.COLOR_HEADER)
        cx = self.W // 2
        draw.rounded_rectangle((cx - 18, 55, cx + 18, 85), radius=6, outline=self.COLOR_ACCENT, width=3)
        draw.arc((cx - 14, 30, cx + 14, 70), start=180, end=360, fill=self.COLOR_ACCENT, width=3)

        title = "SMART DOOR LOCK"
        tx = self._center_text_x(draw, title, self.font_title, self.W)
        draw.text((tx, 95), title, font=self.font_title, fill=self.COLOR_TEXT)

        msg = "Connecting to Internet..."
        mx = self._center_text_x(draw, msg, self.font_subtitle, self.W)
        draw.text((mx, 190), msg, font=self.font_subtitle, fill=self.COLOR_ACCENT)

        dots = "." * (1 + attempt % 3)
        sub = f"Attempt {attempt}/{self.MAX_CONNECT_ATTEMPTS}{dots}"
        sx = self._center_text_x(draw, sub, self.font_subtitle, self.W)
        draw.text((sx, 215), sub, font=self.font_subtitle, fill=self.COLOR_TEXT_DIM)

        self.display.display(img)

    def _show_splash_screen(self):
        img = Image.new("RGB", (self.W, self.H), self.COLOR_BG)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, self.W, 130), fill=self.COLOR_HEADER)
        cx = self.W // 2
        draw.rounded_rectangle((cx - 18, 55, cx + 18, 85), radius=6, outline=self.COLOR_ACCENT, width=3)
        draw.arc((cx - 14, 30, cx + 14, 70), start=180, end=360, fill=self.COLOR_ACCENT, width=3)

        title = "SMART DOOR LOCK"
        sub = "INTERCOM SYSTEM"
        tx = self._center_text_x(draw, title, self.font_title, self.W)
        sx = self._center_text_x(draw, sub, self.font_subtitle, self.W)
        draw.text((tx, 95), title, font=self.font_title, fill=self.COLOR_TEXT)
        draw.text((sx, 118), sub, font=self.font_subtitle, fill=self.COLOR_TEXT_DIM)

        status_msg = "Online" if self.online else "Offline Mode"
        status_color = self.COLOR_SUCCESS if self.online else self.COLOR_DANGER
        bx = self._center_text_x(draw, status_msg, self.font_subtitle, self.W)
        draw.text((bx, 200), status_msg, font=self.font_subtitle, fill=status_color)

        self.display.display(img)

    def _show_success_screen(self):
        img = Image.new("RGB", (self.W, self.H), self.COLOR_BG)
        draw = ImageDraw.Draw(img)

        draw.rectangle((0, 0, self.W, 4), fill=self.COLOR_SUCCESS)

        cx, cy = self.W // 2, self.H // 2 - 20
        ring_r = 38
        draw.ellipse(
            (cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r),
            outline=self.COLOR_SUCCESS, width=2
        )
        inner_r = ring_r - 4
        draw.ellipse(
            (cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r),
            fill=(46, 204, 113, 0)
        )
        draw.line((cx - 16, cy + 1, cx - 4, cy + 14), fill=self.COLOR_SUCCESS, width=3)
        draw.line((cx - 4, cy + 14, cx + 18, cy - 14), fill=self.COLOR_SUCCESS, width=3)

        title = "ACCESS GRANTED"
        tx = self._center_text_x(draw, title, self.font_status, self.W)
        draw.text((tx, cy + ring_r + 14), title, font=self.font_status, fill=self.COLOR_SUCCESS)

        sub = "Door unlocked  ·  Welcome home"
        sx = self._center_text_x(draw, sub, self.font_subtitle, self.W)
        draw.text((sx, cy + ring_r + 36), sub, font=self.font_subtitle, fill=self.COLOR_TEXT_DIM)

        self.display.display(img)

    def _show_failure_screen(self):
        img = Image.new("RGB", (self.W, self.H), self.COLOR_BG)
        draw = ImageDraw.Draw(img)

        draw.rectangle((0, 0, self.W, 4), fill=self.COLOR_DANGER)

        cx, cy = self.W // 2, self.H // 2 - 20
        ring_r = 38
        draw.ellipse(
            (cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r),
            outline=self.COLOR_DANGER, width=2
        )
        offset = 14
        draw.line((cx - offset, cy - offset, cx + offset, cy + offset), fill=self.COLOR_DANGER, width=3)
        draw.line((cx + offset, cy - offset, cx - offset, cy + offset), fill=self.COLOR_DANGER, width=3)

        title = "ACCESS DENIED"
        tx = self._center_text_x(draw, title, self.font_status, self.W)
        draw.text((tx, cy + ring_r + 14), title, font=self.font_status, fill=self.COLOR_DANGER)

        sub = "Incorrect PIN  ·  Please try again"
        sx = self._center_text_x(draw, sub, self.font_subtitle, self.W)
        draw.text((sx, cy + ring_r + 36), sub, font=self.font_subtitle, fill=self.COLOR_TEXT_DIM)

        self.display.display(img)

    # ==================== NEW CUSTOM MESSAGE FUNCTION ====================
    def show_custom_message(self, title: str, subtitle: str = "", color=None, duration: float = 3.0):
        """Beautiful full-screen custom message"""
        if color is None:
            color = self.COLOR_ACCENT

        img = Image.new("RGB", (self.W, self.H), self.COLOR_BG)
        draw = ImageDraw.Draw(img)

        draw.rectangle((0, 0, self.W, 6), fill=color)

        title_y = 80
        tx = self._center_text_x(draw, title, self.font_big, self.W)
        draw.text((tx, title_y), title, font=self.font_big, fill=color)

        if subtitle:
            sx = self._center_text_x(draw, subtitle, self.font_subtitle, self.W)
            draw.text((sx, title_y + 55), subtitle, font=self.font_subtitle, fill=self.COLOR_TEXT_DIM)

        self.display.display(img)
        time.sleep(duration)

        # Return to whichever screen the user was actually on
        if self.mode == "HOME":
            self._draw_home_ui()
        else:
            self._draw_keypad_ui()

    # ==================== HOME PAGE ====================
    def _draw_home_ui(self):
        img = Image.new("RGB", (self.W, self.H), self.COLOR_BG)
        draw = ImageDraw.Draw(img)
        self._draw_header(draw)

        locked_out = self._bell_disabled_until is not None and time.time() < self._bell_disabled_until

        # DOOR BELL button
        bx1, by1, bx2, by2 = self.home_buttons["BELL"]
        bell_edge = self.COLOR_TEXT_DIM if locked_out else self.COLOR_AMBER
        self._rounded(draw, (bx1, by1, bx2, by2), radius=14, fill=self.COLOR_BG, outline=bell_edge, width=2)
        bell_label = "BELL LOCKED" if locked_out else "DOOR BELL"
        bbox = draw.textbbox((0, 0), bell_label, font=self.font_btn)
        lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            ((bx1 + bx2) // 2 - lw // 2, (by1 + by2) // 2 - lh // 2 - bbox[1]),
            bell_label, font=self.font_btn, fill=bell_edge
        )

        # ENTER PIN button
        px1, py1, px2, py2 = self.home_buttons["PIN"]
        self._rounded(draw, (px1, py1, px2, py2), radius=14, fill=self.COLOR_BG, outline=self.COLOR_ACCENT, width=2)
        pin_label = "ENTER PIN"
        bbox = draw.textbbox((0, 0), pin_label, font=self.font_btn)
        lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            ((px1 + px2) // 2 - lw // 2, (py1 + py2) // 2 - lh // 2 - bbox[1]),
            pin_label, font=self.font_btn, fill=self.COLOR_ACCENT
        )

        if self.status_text:
            sx = self._center_text_x(draw, self.status_text, self.font_status, self.W)
            draw.text((sx, self.home_status_y), self.status_text, font=self.font_status, fill=self.status_color)

        self.display.display(img)

    def _handle_home_touch(self, x, y):
        bx1, by1, bx2, by2 = self.home_buttons["BELL"]
        px1, py1, px2, py2 = self.home_buttons["PIN"]

        if bx1 <= x <= bx2 and by1 <= y <= by2:
            self._handle_doorbell_press()
            return

        if px1 <= x <= px2 and py1 <= y <= py2:
            self.mode = "LOGIN"
            self.current_input = ""
            self.header_title = "DOOR LOCK"
            self.status_text = "ENTER PIN"
            self.status_color = self.COLOR_ACCENT
            self.show_icon = None
            self._draw_keypad_ui()
            return

        self._draw_home_ui()

    # ==================== DOORBELL LOGIC ====================
    def _handle_doorbell_press(self):
        now = time.time()

        # 1) Currently locked out from a previous spam burst?
        if self._bell_disabled_until is not None:
            if now < self._bell_disabled_until:
                remaining = int(self._bell_disabled_until - now)
                self._flash_home_status(f"BELL LOCKED ({remaining}s)", self.COLOR_DANGER)
                return
            # lockout window passed — clear it and start fresh
            self._bell_disabled_until = None
            self._bell_spam_count = 0

        # 2) Still inside the cooldown window since the last accepted ring?
        if self._last_bell_time is not None and (now - self._last_bell_time) < self.BELL_NOTIFY_COOLDOWN_SECS:
            self._bell_spam_count += 1

            if self._bell_spam_count >= self.BELL_SPAM_THRESHOLD:
                self._bell_disabled_until = now + self.BELL_LOCKOUT_SECS
                self._flash_home_status("TOO MANY PRESSES - LOCKED", self.COLOR_DANGER, hold_seconds=3.0)
                if self.alert_mgr:
                    self.alert_mgr.trigger_cloud_alert(
                        "doorbell_spam",
                        "Doorbell disabled for 200s after repeated rapid presses."
                    )
            else:
                wait_left = int(self.BELL_NOTIFY_COOLDOWN_SECS - (now - self._last_bell_time))
                self._flash_home_status(f"PLEASE WAIT ({wait_left}s)", self.COLOR_AMBER)
            return

        # 3) Accepted ring — notify the cloud
        self._last_bell_time = now
        self._bell_spam_count = 1

        if self.alert_mgr:
            self.alert_mgr.trigger_cloud_alert(
                "doorbell_ring",
                "Someone is ringing the doorbell!"
            )

        self._flash_home_status("BELL RUNG - NOTIFIED", self.COLOR_SUCCESS)

    def _flash_home_status(self, message, color, hold_seconds=2.0):
        self.status_text = message
        self.status_color = color
        self._draw_home_ui()
        time.sleep(hold_seconds)
        self.status_text = ""
        self._draw_home_ui()

    # ==================== KEYPAD UI (unchanged) ====================
    def _draw_keypad_ui(self):
        img = Image.new("RGB", (self.W, self.H), self.COLOR_BG)
        draw = ImageDraw.Draw(img)
        self._draw_header(draw)

        for char, bounds in self.buttons.items():
            if char == 'C':
                fill, edge, txt_color, width = self.COLOR_BG, self.COLOR_AMBER, self.COLOR_AMBER, 2
            elif char == 'P':
                if self.mode == "LOGIN":
                    fill, edge, txt_color, width = self.COLOR_BG, self.COLOR_VIOLET, self.COLOR_VIOLET, 2
                else:
                    fill, edge, txt_color, width = self.COLOR_BG, self.COLOR_DANGER, self.COLOR_DANGER, 2
            elif char == '0':
                fill, edge, txt_color, width = self.COLOR_BTN, self.COLOR_ACCENT, self.COLOR_TEXT, 1
            else:
                fill, edge, txt_color, width = self.COLOR_BTN, self.COLOR_BTN_EDGE, self.COLOR_TEXT, 1

            self._rounded(draw, bounds, radius=10, fill=fill, outline=edge, width=width)

            label = "DEL" if char == 'C' else "BACK" if char == 'P' and self.mode != "LOGIN" else "PIN" if char == 'P' else char
            bbox = draw.textbbox((0, 0), label, font=self.font_btn)
            lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            cx = (bounds[0] + bounds[2]) // 2 - lw // 2
            cy = (bounds[1] + bounds[3]) // 2 - lh // 2 - bbox[1]
            draw.text((cx, cy), label, font=self.font_btn, fill=txt_color)

        divider_y = self.status_panel_top - 8
        draw.line((16, divider_y, self.W - 16, divider_y), fill=self.COLOR_BTN_EDGE, width=1)

        dots_y = self.status_panel_top + 8
        if self.show_icon:
            self._draw_result_icon(draw, self.show_icon, self.W // 2, dots_y)
        else:
            dot_r = 6
            dot_gap = 22
            total_dots_w = dot_gap * 3
            dots_start_x = (self.W - total_dots_w) // 2
            for i in range(4):
                cx = dots_start_x + i * dot_gap
                filled = i < len(self.current_input)
                color = self.status_color if filled else self.COLOR_DOT_EMPTY
                if filled:
                    draw.ellipse((cx - dot_r, dots_y - dot_r, cx + dot_r, dots_y + dot_r), fill=color)
                else:
                    draw.ellipse((cx - dot_r, dots_y - dot_r, cx + dot_r, dots_y + dot_r), outline=color, width=2)

        status_y = dots_y + 18
        sx = self._center_text_x(draw, self.status_text, self.font_status, self.W)
        draw.text((sx, status_y), self.status_text, font=self.font_status, fill=self.status_color)

        self.display.display(img)

    def _draw_result_icon(self, draw, icon, cx, cy, size=15):
        if icon == "check":
            color = self.COLOR_SUCCESS
            draw.line((cx - size, cy, cx - size*0.25, cy + size*0.75), fill=color, width=4)
            draw.line((cx - size*0.25, cy + size*0.75, cx + size, cy - size*0.7), fill=color, width=4)
        elif icon == "cross":
            color = self.COLOR_DANGER
            draw.line((cx - size*0.7, cy - size*0.7, cx + size*0.7, cy + size*0.7), fill=color, width=4)
            draw.line((cx - size*0.7, cy + size*0.7, cx + size*0.7, cy - size*0.7), fill=color, width=4)

    # ==================== RESULT & STATE ====================
    def show_access_result(self, granted: bool):
        if granted:
            self._show_success_screen()
        else:
            self._show_failure_screen()
        time.sleep(2.5)
        self._reset_to_login()
        self._draw_home_ui()

    def _reset_to_login(self):
        """After any keypad flow finishes, go back to the HOME screen."""
        self.mode = "HOME"
        self.current_input = ""
        self.header_title = "DOOR LOCK"
        self.status_text = ""
        self.status_color = self.COLOR_ACCENT
        self.show_icon = None

    def _show_inline_message(self, message, color, icon, hold_seconds=2.0):
        self.status_text = message
        self.status_color = color
        self.show_icon = icon
        self._draw_keypad_ui()
        time.sleep(hold_seconds)

    def set_pin(self, new_pin: str) -> bool:
        """Updates the master PIN from local keypad or external webhooks cleanly"""
        if len(new_pin) == 4 and new_pin.isdigit():
            self.correct_password = new_pin
            print(f"[HARDWARE] Master PIN successfully updated to: {self.correct_password}")

            self.status_text = f"PIN UPDATED: {new_pin}"
            self.status_color = self.COLOR_SUCCESS
            self.show_icon = "check"
            self._draw_keypad_ui()

            self._reset_to_login()
            self._draw_home_ui()
            return True

        return False

    # ==================== INPUT ====================
    def authenticate_user(self):
        pos = self.get_touch_coordinates()
        if not pos: return None

        x, y = pos

        if self.mode == "HOME":
            self._handle_home_touch(x, y)
            time.sleep(0.35)
            return None

        pressed = None
        for char, bounds in self.buttons.items():
            if bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]:
                pressed = char
                break
        if pressed is None: return None

        result = None

        if pressed == 'C':
            self.current_input = ""
        elif pressed == 'P':
            if self.mode == "LOGIN":
                self.mode = "VERIFY_OLD"
                self.current_input = ""
                self.header_title = "CHANGE PIN"
                self.status_text = "ENTER CURRENT PIN"
                self.status_color = self.COLOR_ACCENT
                self.show_icon = None
            else:
                self._reset_to_login()
        elif pressed.isdigit() and len(self.current_input) < 4:
            self.current_input += pressed
            if len(self.current_input) == 4:
                entered = self.current_input
                if self.mode == "LOGIN":
                    if entered == self.correct_password:
                        self.show_access_result(True)
                        result = True
                    else:
                        self.show_access_result(False)
                        result = False
                    return result
                elif self.mode == "VERIFY_OLD":
                    if entered == self.correct_password:
                        self.mode = "ENTER_NEW"
                        self.current_input = ""
                        self.status_text = "ENTER NEW PIN"
                        self.status_color = self.COLOR_ACCENT
                        self.show_icon = None
                    else:
                        self._show_inline_message("WRONG CURRENT PIN", self.COLOR_DANGER, "cross")
                        self._reset_to_login()
                elif self.mode == "ENTER_NEW":
                    self.set_pin(entered)
                    return None

        if self.mode == "HOME":
            self._draw_home_ui()
        else:
            self._draw_keypad_ui()
        time.sleep(0.35)
        return result

    def tick(self):
        now = time.time()
        idle = self.mode == "HOME" or (self.mode == "LOGIN" and not self.current_input)
        if now - self._last_refresh >= self.NETWORK_RECHECK_SECS and idle:
            self._maybe_recheck_network()
            if self.mode == "HOME":
                self._draw_home_ui()
            else:
                self._draw_keypad_ui()
            self._last_refresh = now


if __name__ == "__main__":
    try:
        print("Launching Door Lock Keypad...")
        keypad = TouchKeypad()
        while True:
            keypad.authenticate_user()
            keypad.tick()
            time.sleep(0.01)
    except KeyboardInterrupt:
        GPIO.cleanup()