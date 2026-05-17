import  os
import  sys
import  logging
from    logging.handlers import TimedRotatingFileHandler

class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG':    '\033[94m',  # Blue
        'INFO':     '\033[92m',  # Green
        'WARNING':  '\033[93m',  # Yellow
        'ERROR':    '\033[91m',  # Red
        'CRITICAL': '\033[95m',  # Magenta
        'RESET':    '\033[0m',   # Reset color
    }

    def format(self, record):
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        record.levelname = f"{level_color}{record.levelname}{reset}"
        return super().format(record)

def getLogger(name: str, log_level=logging.INFO):
    logger = logging.getLogger(name)

    if not logger.handlers:
        color_formatter = ColorFormatter(
            '%(asctime)s | %(name)-15s | %(levelname)-16s | %(threadName)-32s  | %(message)s',
            datefmt='%H:%M:%S'
        )

        plain_formatter = logging.Formatter(
            '%(asctime)s | %(name)-15s | %(levelname)-16s | %(threadName)-32s | %(message)s',
            datefmt='%H:%M:%S'
        )
  
        stream_handler = logging.StreamHandler(sys.stderr)              # Console handler
        stream_handler.setFormatter(color_formatter)
        logger.addHandler(stream_handler)

        log_dir = 'logs'                                                # File handler (daily rotation, keep last 7 days)
        os.makedirs(log_dir, exist_ok=True)                             # Create logs/ directory if it doesn't exist
        log_file = os.path.join(log_dir, f"{name}.log")

        file_handler = TimedRotatingFileHandler(
            filename=log_file,
            when='midnight',                                            # Rotate at midnight
            interval=1,                                                 # Every day
            backupCount=7,                                              # Keep last 7 days
            encoding='utf-8',
            utc=True                                                    # Use UTC time for consistency
        )
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)
        
        logger.setLevel(log_level)                                      # Set global log level

        logger.propagate = False                                        # Prevent duplicate logs if root logger is also used

    return logger