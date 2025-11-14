import time
import re
import json
import io
import base64
import os
import sys
import socket
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional, Tuple

from PIL import Image
import cv2
import numpy as np
import mss
import ollama
from flask import Flask, jsonify, render_template_string, request, render_template
import tkinter as tk
from tkinter import ttk, colorchooser
import keyboard  # hotkey support
import tkinter.messagebox as messagebox
import subprocess
import shutil
import webbrowser
import logging
import logging.handlers

# Configure logging to both console and file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# File handler with rotation (keeps last 5 files, max 10MB each)
file_handler = logging.handlers.RotatingFileHandler(
    'scanning_tool.log', 
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def show_installation_message(system_name: str) -> None:
    """Present final installation instructions, using a GUI prompt on Windows."""
    message = (
        f"Ollama installation initiated for {system_name.title()}.\n\n"
        "After installation completes:\n"
        "1. Restart this program\n"
        "2. The first scan will download the AI model automatically\n\n"
        "Visit https://ollama.com/ for troubleshooting."
    )

    if system_name == "windows":
        temp_root = None
        try:
            temp_root = tk.Tk()
            temp_root.withdraw()
            messagebox.showinfo("Ollama Installation", message, parent=temp_root)
        except Exception as exc:
            logger.debug(f"Unable to show Windows message box: {exc}")
            logger.info(message)
        else:
            logger.info(message)
        finally:
            if temp_root is not None:
                temp_root.destroy()
    else:
        logger.info(message)

def ensure_ollama_installed():
    """
    Check if Ollama is installed on the system (cross-platform).
    If not found, offer OS-specific installation options.
    Works on Windows and Linux.
    """
    if not shutil.which("ollama"):
        import platform
        system = platform.system().lower()
        
        logger.info("Ollama not found on your system.")
        logger.info("Ollama is required for AI-powered code recognition.")
        logger.info("")

        if system == "windows":
            # Windows - offer automatic download and install
            logger.info("=== Windows Installation Options ===")
            logger.info("1. Automatic download and install (Recommended)")
            logger.info("2. Manual download from website")
            logger.info("")

            download_url = "https://ollama.com/download/OllamaSetup.exe"
            logger.info("Opening the Ollama download link in your default browser...")
            logger.info(f"Download URL: {download_url}")
            try:
                opened = webbrowser.open(download_url)
                if opened:
                    logger.info("Browser opened successfully. Follow the prompts to install Ollama.")
                else:
                    logger.warning("The browser did not report success. Please open the link manually if nothing happens.")
            except Exception as e:
                logger.error(f"Unable to open browser automatically: {e}")
                logger.info("Please open the link manually to download Ollama.")

        elif system == "linux":
            # Linux - detect distribution and offer package manager commands
            logger.info("=== Linux Installation Options ===")

            # Try to detect Linux distribution
            distro_info = ""
            package_cmd = ""
            
            try:
                with open("/etc/os-release", "r") as f:
                    os_release = f.read().lower()
                    
                if "debian" in os_release or "ubuntu" in os_release or "mint" in os_release:
                    distro_info = "Debian/Ubuntu/Mint"
                    package_cmd = "curl -fsSL https://ollama.com/install.sh | sh"
                elif "arch" in os_release or "manjaro" in os_release:
                    distro_info = "Arch/Manjaro"
                    package_cmd = "sudo pacman -S ollama"
                elif "fedora" in os_release or "rhel" in os_release or "centos" in os_release:
                    distro_info = "RedHat/Fedora/CentOS"
                    package_cmd = "curl -fsSL https://ollama.com/install.sh | sh"
                elif "gentoo" in os_release or "funtoo" in os_release:
                    distro_info = "Gentoo/Funtoo"
                    package_cmd = "sudo emerge --ask ollama"
                elif "suse" in os_release or "opensuse" in os_release:
                    distro_info = "SUSE/openSUSE"
                    package_cmd = "sudo zypper install ollama"
                else:
                    distro_info = "Unknown Linux"
                    package_cmd = "curl -fsSL https://ollama.com/install.sh | sh"
            except:
                distro_info = "Unknown Linux"
                package_cmd = "curl -fsSL https://ollama.com/install.sh | sh"
            
            logger.info(f"Detected: {distro_info}")
            logger.info(f"Recommended command: {package_cmd}")
            logger.info("")
            logger.info("1. Run the recommended installation command")
            logger.info("2. Manual installation from website")
            logger.info("")

            choice = input("Would you like to run the installation command? (y/n): ").lower().strip()
            
            if choice in ['y', 'yes', '1', '']:
                logger.info(f"Running: {package_cmd}")
                logger.info("Please enter your password if prompted...")
                try:
                    result = subprocess.run(package_cmd, shell=True, check=False)
                    if result.returncode == 0:
                        logger.info("Ollama installation completed!")
                        logger.info("Please restart this program to continue.")
                    else:
                        logger.warning("Installation failed or was cancelled.")
                        logger.info("You can try installing manually from https://ollama.com/")
                except Exception as e:
                    logger.error(f"Error running installation command: {e}")
                    logger.info("Please visit https://ollama.com/ for manual installation.")
            else:
                logger.info("Opening Ollama website for manual installation...")
                webbrowser.open("https://ollama.com/")
        
        else:
            # Unsupported OS
            logger.info("=== Unsupported Operating System ===")
            logger.info("This tool currently supports Windows and Linux only.")
            logger.info("Please install Ollama manually from: https://ollama.com/")
            webbrowser.open("https://ollama.com/")
        
        # Show final message
        show_installation_message(system)

        input("\nPress ENTER after installing Ollama to close this program...")
        sys.exit(0)

    else:
        try:
            version = subprocess.check_output(["ollama", "--version"], text=True).strip()
            logger.info(f"Ollama found: {version}")
        except Exception as e:
            logger.error(f"Error checking Ollama: {e}")
            sys.exit("Please install Ollama and rerun this program.")

def ensure_model_installed(model="qwen2.5vl:3b"):
    """Ensure the Ollama model is pulled locally."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if model not in result.stdout:
            logger.info(f"Model {model} not found. Pulling now...")
            subprocess.run(["ollama", "pull", model], check=True)
            logger.info(f"Model {model} installed successfully.")
        else:
            logger.info(f"Model {model} already installed.")
    except Exception as e:
        logger.error(f"Error ensuring model: {e}")
        sys.exit("Failed to ensure Ollama model.")



# ---------- CONFIG ----------
CONFIG_FILE = "config.json"

CAP_REGION = {"left": 1260, "top": 310, "width": 160, "height": 30}
ANCHOR_REGION = {"left": 1100, "top": 240, "width": 320, "height": 140}
ANCHOR_OFFSET = {"x": 36, "y": 56}
ANCHOR_THRESHOLD = 0.82
AUTO_ALIGN_ENABLED = True
ANCHOR_TEMPLATE_DIR = "assets/anchor_templates"
ALIGNMENT_POLL_INTERVAL_MS = 500
CONTINUOUS_CAPTURE_INTERVAL = 2.0
INFO_OVERLAY_OFFSET = {"x": 0, "y": 0}
label_color = "yellow"
MIN_CONFIDENCE = 0.65
DEBUG_SHOW_OVERLAY = True
OLLAMA_MODEL = "qwen2.5vl:3b"   # vision model

# Regex for codes
CODE_RE = re.compile(
    r"(?:[A-Za-z]?-?\d[\d,\.]{1,10}|\d{2,10})",
    re.IGNORECASE
)

last_result = {"code": None, "code_raw": None, "info": None,
               "confidence": 0.0, "raw_text": ""}
last_alignment_info = {
    "enabled": AUTO_ALIGN_ENABLED,
    "matched": False,
    "template": None,
    "score": 0.0,
    "match_left": None,
    "match_top": None,
    "capture_left": None,
    "capture_top": None,
}


GUI_CONTROL_STATE = {
    "capture": {"left": None, "top": None, "width": None, "height": None},
    "anchor": {"left": None, "top": None, "width": None, "height": None, "offset_x": None, "offset_y": None},
    "overlay": {"offset_x": None, "offset_y": None},
    "syncing": {"capture": False, "anchor": False, "overlay": False},
}


def register_capture_sliders(left: tk.Scale, top: tk.Scale, width: tk.Scale, height: tk.Scale) -> None:
    GUI_CONTROL_STATE["capture"].update({
        "left": left,
        "top": top,
        "width": width,
        "height": height,
    })


def register_anchor_sliders(
    left: tk.Scale,
    top: tk.Scale,
    width: tk.Scale,
    height: tk.Scale,
    offset_x: tk.Scale,
    offset_y: tk.Scale,
) -> None:
    GUI_CONTROL_STATE["anchor"].update({
        "left": left,
        "top": top,
        "width": width,
        "height": height,
        "offset_x": offset_x,
        "offset_y": offset_y,
    })


def register_overlay_sliders(offset_x: tk.Scale, offset_y: tk.Scale) -> None:
    GUI_CONTROL_STATE["overlay"].update({"offset_x": offset_x, "offset_y": offset_y})


def sync_capture_sliders() -> None:
    state = GUI_CONTROL_STATE
    widgets = state["capture"]
    widget = widgets["left"]
    if not widget:
        return
    if state["syncing"]["capture"]:
        return

    def _apply() -> None:
        if state["syncing"]["capture"]:
            return
        state["syncing"]["capture"] = True
        try:
            try:
                widgets["left"].set(int(CAP_REGION["left"]))
                widgets["top"].set(int(CAP_REGION["top"]))
                widgets["width"].set(int(CAP_REGION["width"]))
                widgets["height"].set(int(CAP_REGION["height"]))
            except tk.TclError:
                pass
        finally:
            state["syncing"]["capture"] = False

    try:
        widget.after(0, _apply)
    except tk.TclError:
        pass


def sync_anchor_sliders() -> None:
    state = GUI_CONTROL_STATE
    widgets = state["anchor"]
    widget = widgets["left"]
    if not widget:
        return
    if state["syncing"]["anchor"]:
        return

    def _apply() -> None:
        if state["syncing"]["anchor"]:
            return
        state["syncing"]["anchor"] = True
        try:
            try:
                widgets["left"].set(int(ANCHOR_REGION["left"]))
                widgets["top"].set(int(ANCHOR_REGION["top"]))
                widgets["width"].set(int(ANCHOR_REGION["width"]))
                widgets["height"].set(int(ANCHOR_REGION["height"]))
                widgets["offset_x"].set(int(ANCHOR_OFFSET["x"]))
                widgets["offset_y"].set(int(ANCHOR_OFFSET["y"]))
            except tk.TclError:
                pass
        finally:
            state["syncing"]["anchor"] = False


def sync_overlay_sliders() -> None:
    state = GUI_CONTROL_STATE
    widgets = state["overlay"]
    widget = widgets["offset_x"]
    if not widget:
        return
    if state["syncing"]["overlay"]:
        return

    def _apply() -> None:
        if state["syncing"]["overlay"]:
            return
        state["syncing"]["overlay"] = True
        try:
            try:
                widgets["offset_x"].set(int(INFO_OVERLAY_OFFSET["x"]))
                widgets["offset_y"].set(int(INFO_OVERLAY_OFFSET["y"]))
            except tk.TclError:
                pass
        finally:
            state["syncing"]["overlay"] = False

    try:
        widget.after(0, _apply)
    except tk.TclError:
        pass


# ---------- Config Handling ----------
def ensure_anchor_directory(path: str) -> None:
    """Ensure the directory for anchor templates exists."""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(f"Unable to ensure anchor template directory {path}: {exc}")


class AnchorRegionTracker:
    """Manage template loading and anchor matching for auto alignment."""

    def __init__(self, template_dir: str, threshold: float = 0.82) -> None:
        self.template_dir = template_dir
        self.threshold = threshold
        self.templates: List[Tuple[str, np.ndarray]] = []
        self.last_loaded_count = 0
        self.load_templates()

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold

    def set_directory(self, template_dir: str) -> int:
        self.template_dir = template_dir
        return self.load_templates()

    def load_templates(self) -> int:
        ensure_anchor_directory(self.template_dir)
        loaded: List[Tuple[str, np.ndarray]] = []
        directory = Path(self.template_dir)
        if not directory.exists():
            logger.debug(f"Anchor template directory does not exist: {directory}")
            self.templates = []
            self.last_loaded_count = 0
            return 0

        supported_ext = {".png", ".jpg", ".jpeg", ".bmp"}
        for path in sorted(directory.glob("**/*")):
            if path.suffix.lower() not in supported_ext or not path.is_file():
                continue
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.warning(f"Failed to load anchor template: {path}")
                continue
            loaded.append((path.name, image))

        self.templates = loaded
        self.last_loaded_count = len(loaded)
        if self.last_loaded_count == 0:
            logger.warning(
                "No anchor templates were loaded. Head sway compensation will remain disabled until templates are added."
            )
        else:
            logger.info(f"Loaded {self.last_loaded_count} anchor templates from {directory}")
        return self.last_loaded_count

    def locate_anchor(self, region: Dict[str, int]) -> Optional[Dict[str, float]]:
        if not self.templates:
            return None

        monitor = {
            "left": int(region["left"]),
            "top": int(region["top"]),
            "width": int(region["width"]),
            "height": int(region["height"]),
        }

        with mss.mss() as sct:
            try:
                screenshot = sct.grab(monitor)
            except Exception as exc:
                logger.error(f"Anchor capture failed: {exc}")
                return None

        anchor_image = np.array(screenshot)
        if anchor_image.ndim == 3 and anchor_image.shape[2] == 4:
            anchor_gray = cv2.cvtColor(anchor_image, cv2.COLOR_BGRA2GRAY)
        else:
            anchor_gray = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2GRAY)

        best_score = -1.0
        best_loc: Optional[Tuple[int, int]] = None
        best_template: Optional[Tuple[str, np.ndarray]] = None

        for template_name, template_img in self.templates:
            if anchor_gray.shape[0] < template_img.shape[0] or anchor_gray.shape[1] < template_img.shape[1]:
                continue
            res = cv2.matchTemplate(anchor_gray, template_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = float(max_val)
                best_loc = max_loc
                best_template = (template_name, template_img)

        if best_loc is None or best_template is None:
            return None

        if best_score < self.threshold:
            logger.debug(
                f"Anchor match below threshold ({best_score:.3f} < {self.threshold:.3f}) using template {best_template[0]}"
            )
            return None

        match_left = monitor["left"] + best_loc[0]
        match_top = monitor["top"] + best_loc[1]
        return {
            "match_left": float(match_left),
            "match_top": float(match_top),
            "score": best_score,
            "template": best_template[0],
            "template_width": float(best_template[1].shape[1]),
            "template_height": float(best_template[1].shape[0]),
        }


anchor_tracker: Optional[AnchorRegionTracker] = None


def load_config():
    global CAP_REGION, label_color, AUTO_ALIGN_ENABLED, ANCHOR_REGION, ANCHOR_OFFSET, ANCHOR_THRESHOLD, ANCHOR_TEMPLATE_DIR
    global ALIGNMENT_POLL_INTERVAL_MS, CONTINUOUS_CAPTURE_INTERVAL, INFO_OVERLAY_OFFSET
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                CAP_REGION = data.get("CAP_REGION", CAP_REGION)
                label_color = data.get("label_color", label_color)
                AUTO_ALIGN_ENABLED = data.get("AUTO_ALIGN_ENABLED", AUTO_ALIGN_ENABLED)
                ANCHOR_REGION = data.get("ANCHOR_REGION", ANCHOR_REGION)
                ANCHOR_OFFSET = data.get("ANCHOR_OFFSET", ANCHOR_OFFSET)
                ANCHOR_THRESHOLD = data.get("ANCHOR_THRESHOLD", ANCHOR_THRESHOLD)
                ANCHOR_TEMPLATE_DIR = data.get("ANCHOR_TEMPLATE_DIR", ANCHOR_TEMPLATE_DIR)
                ALIGNMENT_POLL_INTERVAL_MS = data.get("ALIGNMENT_POLL_INTERVAL_MS", ALIGNMENT_POLL_INTERVAL_MS)
                CONTINUOUS_CAPTURE_INTERVAL = data.get("CONTINUOUS_CAPTURE_INTERVAL", CONTINUOUS_CAPTURE_INTERVAL)
                INFO_OVERLAY_OFFSET = data.get("INFO_OVERLAY_OFFSET", INFO_OVERLAY_OFFSET)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Config file invalid or empty, resetting: {e}")
            save_config()
    else:
        save_config()

    ensure_anchor_directory(ANCHOR_TEMPLATE_DIR)
    last_alignment_info["enabled"] = AUTO_ALIGN_ENABLED



def save_config():
    global CAP_REGION, label_color, AUTO_ALIGN_ENABLED, ANCHOR_REGION, ANCHOR_OFFSET, ANCHOR_THRESHOLD, ANCHOR_TEMPLATE_DIR
    global ALIGNMENT_POLL_INTERVAL_MS, CONTINUOUS_CAPTURE_INTERVAL, INFO_OVERLAY_OFFSET
    data = {
        "CAP_REGION": CAP_REGION,
        "label_color": label_color,
        "AUTO_ALIGN_ENABLED": AUTO_ALIGN_ENABLED,
        "ANCHOR_REGION": ANCHOR_REGION,
        "ANCHOR_OFFSET": ANCHOR_OFFSET,
        "ANCHOR_THRESHOLD": ANCHOR_THRESHOLD,
        "ANCHOR_TEMPLATE_DIR": ANCHOR_TEMPLATE_DIR,
        "ALIGNMENT_POLL_INTERVAL_MS": ALIGNMENT_POLL_INTERVAL_MS,
        "CONTINUOUS_CAPTURE_INTERVAL": CONTINUOUS_CAPTURE_INTERVAL,
        "INFO_OVERLAY_OFFSET": INFO_OVERLAY_OFFSET,
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=4)
    logger.info("Config saved.")


# ---------- Load Rock Types JSON ----------

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller bundle."""
    if hasattr(sys, "_MEIPASS"):  # running inside PyInstaller bundle
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

rock_file = resource_path("RockTypes_2025-09-16.json")
with open(rock_file, "r") as f:
    ROCK_DATA = json.load(f)


# ---------- Multiplier Codes ----------
MULTIPLIER_CODES = {
    1700: {"key": "CTYPE", "display_name": "C-Type", "rarity": "common", "category": "Rock Deposits"},
    1900: {"key": "ETYPE", "display_name": "E-Type", "rarity": "common", "category": "Rock Deposits"},
    1660: {"key": "ITYPE", "display_name": "I-Type", "rarity": "common", "category": "Rock Deposits"},
    1850: {"key": "MTYPE", "display_name": "M-Type", "rarity": "common", "category": "Rock Deposits"},
    1750: {"key": "PTYPE", "display_name": "P-Type", "rarity": "common", "category": "Rock Deposits"},
    1870: {"key": "QTYPE", "display_name": "Q-Type", "rarity": "common", "category": "Rock Deposits"},
    1720: {"key": "STYPE", "display_name": "S-Type", "rarity": "common", "category": "Rock Deposits"},
    1800: {"key": "ATACAMITE", "display_name": "Atacamite", "rarity": "common", "category": "Rock Deposits"},
    1770: {"key": "FELSIC", "display_name": "Felsic", "rarity": "common", "category": "Rock Deposits"},
    1840: {"key": "GNEISS", "display_name": "Gneiss", "rarity": "common", "category": "Rock Deposits"},
    1920: {"key": "GRANITE", "display_name": "Granite", "rarity": "common", "category": "Rock Deposits"},
    1950: {"key": "IGNEOUS", "display_name": "Igneous", "rarity": "common", "category": "Rock Deposits"},
    1790: {"key": "OBSIDIAN", "display_name": "Obsidian", "rarity": "common", "category": "Rock Deposits"},
    1820: {"key": "QUARTZITE", "display_name": "Quartzite", "rarity": "common", "category": "Rock Deposits"},
    1730: {"key": "SHALE", "display_name": "Shale", "rarity": "common", "category": "Rock Deposits"},
    620: {"key": "GEMS", "display_name": "Gems", "rarity": "common", "category": "Gems"},
    2000: {"key": "SALVAGE", "display_name": "Metal Pannals", "rarity": "common", "category": "Savlage"},
}

# ---------- Ore Value Tiers ----------
ORE_TIERS = {
    "HIGHEST": {"ores": ["QUANTANIUM", "STILERON", "RICCITE"], "color": "#E88AFF"},
    "HIGH": {"ores": ["TARANITE", "BEXALITE", "GOLD"], "color": "#63E64C"},
    "MEDIUM": {"ores": ["LARANITE", "BORASE", "BERYL", "AGRICIUM", "HEPHAESTANITE"], "color": "#E6E14C"},
    "LOW": {"ores": ["TUNGSTEN", "TITANIUM", "SILICON", "IRON", "QUARTZ", "CORUNDUM", "COPPER", "TIN", "ALUMINUM", "ICE"], "color": "#E69E4C"},
}
ORE_VALUE_MAP = {}
for tier, data in ORE_TIERS.items():
    for ore in data["ores"]:
        ORE_VALUE_MAP[ore.upper()] = {"tier": tier, "color": data["color"]}


# ---------- Build Deposit Tables ----------
def build_deposit_tables(rock_data):
    deposit_tables = {}
    for deposit_name, details in rock_data.items():
        ores = details.get("ores", {})
        table = []
        for ore_name, ore_data in ores.items():
            name_up = ore_name.upper()
            value_info = ORE_VALUE_MAP.get(name_up, {"tier": "OTHER", "color": "#888"})
            table.append({
                "name": ore_name.title(),
                "prob": f"{ore_data.get('prob', 0)*100:.0f}%",
                "min": f"{ore_data.get('minPct', 0)*100:.0f}%",
                "max": f"{ore_data.get('maxPct', 0)*100:.0f}%",
                "med": f"{ore_data.get('medPct', 0)*100:.0f}%",
                "tier": value_info["tier"],
                "color": value_info["color"]
            })
        tier_order = ["HIGHEST", "HIGH", "MEDIUM", "LOW", "OTHER"]
        table.sort(key=lambda x: tier_order.index(x["tier"]))
        deposit_tables[deposit_name.upper()] = table
    return deposit_tables

DEPOSIT_TABLES = {
    "STANTON": build_deposit_tables(ROCK_DATA.get("STANTON", {})),
    "PYRO": build_deposit_tables(ROCK_DATA.get("PYRO", {}))
}

# ---------- OCR with Ollama ----------
def ocr_with_ollama(pil_img: Image.Image, model=OLLAMA_MODEL) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    try:
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": "Extract the numeric code shown in this image. Only return the code, no extra words.",
                "images": [img_bytes],
            }],
        )
        return response["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Ollama OCR error: {e}")
        return ""


def extract_code_from_text(raw_text: str):
    if not raw_text:
        return None, None
    matches = CODE_RE.findall(raw_text)
    if not matches:
        return None, raw_text
    raw = matches[0].upper()
    if any(ch.isdigit() for ch in raw):
        m = re.match(r"([A-Za-z]?-?)([\d,\.]+)", raw)
        if m:
            prefix, digits = m.groups()
            digits = digits.replace(",", "").replace(".", "")
            candidate = prefix + digits
        else:
            candidate = raw.replace(",", "").replace(".", "")
    else:
        candidate = raw
    return candidate, raw


# ---------- Deposit Lookup ----------
def lookup_deposit(code: str):
    if not code:
        return None
    try:
        m = re.search(r"(\d+)$", code)
        if not m:
            return None
        num_code = int(m.group(1))
        for base_code, info in MULTIPLIER_CODES.items():
            if num_code % base_code == 0:
                deposits = num_code // base_code
                return {
                    "name": info["display_name"],
                    "key": info["key"],
                    "rarity": info["rarity"],
                    "base_code": base_code,
                    "deposits": deposits,
                    "category": info.get("category", "Ore")
                }
    except ValueError:
        pass
    return None


# ---------- Capture / Overlay ----------
continuous_mode = False
show_border = True
border_canvas = None

capture_overlay_root = None
capture_overlay_canvas = None
capture_rect_id = None

info_overlay_root = None
info_overlay_canvas = None
info_text_id = None
info_overlay_geometry = {"screen_width": None, "screen_height": None, "width": 0, "height": 0}
overlay_text = ""
last_overlay_time = 0

anchor_overlay_root = None
anchor_overlay_canvas = None
anchor_rect_id = None
anchor_overlay_visible = True


def perform_auto_alignment() -> bool:
    """Attempt to adjust the capture region based on anchor template matches."""
    global CAP_REGION, last_alignment_info

    last_alignment_info["enabled"] = AUTO_ALIGN_ENABLED

    if not AUTO_ALIGN_ENABLED:
        return False

    if anchor_tracker is None:
        logger.debug("Anchor tracker not initialised; skipping auto alignment.")
        return False

    anchor_tracker.set_threshold(float(ANCHOR_THRESHOLD))
    detection = anchor_tracker.locate_anchor(ANCHOR_REGION)

    if not detection:
        last_alignment_info.update({
            "matched": False,
            "template": None,
            "score": 0.0,
            "match_left": None,
            "match_top": None,
            "capture_left": None,
            "capture_top": None,
        })
        return False

    template_w = detection.get("template_width", float(CAP_REGION["width"]))
    template_h = detection.get("template_height", float(CAP_REGION["height"]))
    base_left = detection["match_left"] + (template_w / 2.0) - (CAP_REGION["width"] / 2.0)
    base_top = detection["match_top"] + (template_h / 2.0) - (CAP_REGION["height"] / 2.0)

    new_left = int(round(base_left + ANCHOR_OFFSET.get("x", 0)))
    new_top = int(round(base_top + ANCHOR_OFFSET.get("y", 0)))

    CAP_REGION["left"] = max(0, new_left)
    CAP_REGION["top"] = max(0, new_top)

    last_alignment_info.update({
        "matched": True,
        "template": detection["template"],
        "score": float(detection["score"]),
        "match_left": detection["match_left"],
        "match_top": detection["match_top"],
        "capture_left": CAP_REGION["left"],
        "capture_top": CAP_REGION["top"],
    })

    sync_capture_sliders()

    if capture_overlay_root:
        try:
            capture_overlay_root.after(0, update_capture_overlay_region)
        except (RuntimeError, tk.TclError):
            update_capture_overlay_region()

    logger.debug(
        "Auto alignment applied using %s (score %.3f) => CAP_REGION left/top updated to (%d, %d)",
        detection["template"],
        detection["score"],
        CAP_REGION["left"],
        CAP_REGION["top"],
    )
    return True


def toggle_border():
    """Toggle visibility of the debug red border."""
    global show_border, border_canvas
    show_border = not show_border
    if border_canvas:
        border_canvas.itemconfig("border", state="normal" if show_border else "hidden")


def update_overlay_label(info, *, code: Optional[str] = None, raw_text: Optional[str] = None) -> None:
    """Update the floating label with the latest scan result."""
    global overlay_text, info_overlay_canvas, info_text_id, last_overlay_time

    message = ""
    if info:
        message = f"{info['name']} x{info['deposits']}" if "deposits" in info else info["name"]

    overlay_text = message
    if message:
        last_overlay_time = time.time()
    else:
        last_overlay_time = 0
    if info_overlay_canvas and info_text_id:
        info_overlay_canvas.itemconfig(info_text_id, text=overlay_text, fill=label_color)


def compute_info_overlay_geometry(screen_width: int, screen_height: int) -> Tuple[int, int, int, int]:
    overlay_width = max(400, min(800, screen_width - 40))
    overlay_height = 120
    base_left = max(0, (screen_width - overlay_width) // 2)
    base_top = max(0, int(screen_height * 0.35) - overlay_height // 2)

    offset_x = int(INFO_OVERLAY_OFFSET.get("x", 0))
    offset_y = int(INFO_OVERLAY_OFFSET.get("y", 0))

    max_left = max(0, screen_width - overlay_width)
    max_top = max(0, screen_height - overlay_height)

    left = min(max(0, base_left + offset_x), max_left)
    top = min(max(0, base_top + offset_y), max_top)
    return overlay_width, overlay_height, left, top


def reposition_info_overlay() -> None:
    global info_overlay_canvas, info_text_id
    if not info_overlay_root or not info_overlay_canvas or not info_text_id:
        return
    if not info_overlay_root.winfo_exists():
        return

    try:
        screen_width = info_overlay_root.winfo_screenwidth()
        screen_height = info_overlay_root.winfo_screenheight()
    except tk.TclError:
        geo = info_overlay_geometry
        screen_width = geo.get("screen_width") or 1920
        screen_height = geo.get("screen_height") or 1080

    overlay_width, overlay_height, left, top = compute_info_overlay_geometry(screen_width, screen_height)

    info_overlay_root.geometry(f"{overlay_width}x{overlay_height}+{left}+{top}")
    info_overlay_canvas.config(width=overlay_width, height=overlay_height)
    info_overlay_canvas.coords(info_text_id, overlay_width // 2, overlay_height // 2)
    info_overlay_canvas.itemconfig(info_text_id, width=overlay_width - 60)

    info_overlay_geometry.update(
        {
            "screen_width": screen_width,
            "screen_height": screen_height,
            "width": overlay_width,
            "height": overlay_height,
        }
    )


def start_label_timeout(window: Optional[tk.Toplevel]) -> None:
    """Background loop to clear overlay label if no update for 10s."""
    global info_overlay_canvas, info_text_id, last_overlay_time

    if info_overlay_canvas and info_text_id:
        if last_overlay_time and (time.time() - last_overlay_time > 10):
            info_overlay_canvas.itemconfig(info_text_id, text="")
            last_overlay_time = 0

    if window and window.winfo_exists():
        window.after(500, lambda: start_label_timeout(window))



def enforce_topmost(window: tk.Toplevel, interval_ms: int = 1500) -> None:
    """Continuously lift the overlay window so it stays above focused apps."""
    if window is None:
        return
    if not window.winfo_exists():
        return
    try:
        window.attributes("-topmost", True)
        window.lift()
    except tk.TclError:
        return
    window.after(interval_ms, lambda: enforce_topmost(window, interval_ms))


def create_overlay_window(width: int, height: int, left: int, top: int) -> tk.Toplevel:
    """Create a transparent always-on-top overlay window."""
    window = tk.Toplevel()
    window.attributes("-transparentcolor", "black")
    window.attributes("-topmost", True)
    window.overrideredirect(True)
    window.configure(bg="black")
    window.geometry(f"{width}x{height}+{left}+{top}")
    enforce_topmost(window)
    return window



# ---------- GUI + Overlay ----------
def choose_label_color():
    global label_color, info_overlay_canvas, info_text_id
    color = colorchooser.askcolor(title="Choose Label Color")[1]
    if color:
        label_color = color
        if info_overlay_canvas and info_text_id:
            info_overlay_canvas.itemconfig(info_text_id, fill=label_color)


def show_capture_overlay():
    global border_canvas, capture_overlay_canvas, capture_overlay_root, capture_rect_id

    if capture_overlay_root and capture_overlay_root.winfo_exists():
        try:
            capture_overlay_root.destroy()
        except tk.TclError:
            pass
        capture_overlay_canvas = None
        capture_rect_id = None
        border_canvas = None

    cap_w, cap_h = int(CAP_REGION['width']), int(CAP_REGION['height'])
    padding_x, padding_y = 100, 40

    overlay_width = cap_w + padding_x
    overlay_height = cap_h + padding_y
    left = int(CAP_REGION['left']) - (padding_x // 2)
    top = int(CAP_REGION['top']) - padding_y

    capture_overlay_root = create_overlay_window(overlay_width, overlay_height, left, top)

    capture_overlay_canvas = tk.Canvas(
        capture_overlay_root,
        width=overlay_width,
        height=overlay_height,
        bg="black",
        highlightthickness=0,
    )
    capture_overlay_canvas.pack()
    border_canvas = capture_overlay_canvas

    capture_rect_id = capture_overlay_canvas.create_rectangle(
        padding_x // 2,
        padding_y,
        padding_x // 2 + cap_w,
        padding_y + cap_h,
        outline="red",
        width=3,
        tags=("border",),
    )


def show_info_overlay(screen_width: int, screen_height: int) -> None:
    """Display a floating status overlay near the top-center of the screen."""
    global info_overlay_root, info_overlay_canvas, info_text_id

    if info_overlay_root and info_overlay_root.winfo_exists():
        try:
            info_overlay_root.destroy()
        except tk.TclError:
            pass
        info_overlay_canvas = None
        info_text_id = None

    overlay_width, overlay_height, left, top = compute_info_overlay_geometry(screen_width, screen_height)

    info_overlay_root = create_overlay_window(overlay_width, overlay_height, left, top)

    info_overlay_canvas = tk.Canvas(
        info_overlay_root,
        width=overlay_width,
        height=overlay_height,
        bg="black",
        highlightthickness=0,
    )
    info_overlay_canvas.pack()

    info_text_id = info_overlay_canvas.create_text(
        overlay_width // 2,
        overlay_height // 2,
        text=overlay_text,
        fill=label_color,
        font=("Arial", 18, "bold"),
        width=overlay_width - 60,
        justify="center",
    )

    info_overlay_geometry.update(
        {
            "screen_width": screen_width,
            "screen_height": screen_height,
            "width": overlay_width,
            "height": overlay_height,
        }
    )

    start_label_timeout(info_overlay_root)


def update_capture_overlay_region():
    global capture_overlay_canvas, capture_rect_id, capture_overlay_root
    if not capture_overlay_canvas or not capture_rect_id or not capture_overlay_root:
        return
    cap_w, cap_h = int(CAP_REGION['width']), int(CAP_REGION['height'])
    padding_x, padding_y = 100, 40
    overlay_width = cap_w + padding_x
    overlay_height = cap_h + padding_y
    left = int(CAP_REGION['left']) - (padding_x // 2)
    top = int(CAP_REGION['top']) - padding_y

    capture_overlay_canvas.config(width=overlay_width, height=overlay_height)

    capture_overlay_canvas.coords(
        capture_rect_id,
        padding_x // 2,
        padding_y,
        padding_x // 2 + cap_w,
        padding_y + cap_h,
    )
    capture_overlay_root.geometry(f"{overlay_width}x{overlay_height}+{left}+{top}")
    try:
        capture_overlay_root.lift()
    except tk.TclError:
        pass


def show_anchor_overlay():
    global anchor_overlay_root, anchor_overlay_canvas, anchor_rect_id, anchor_overlay_visible

    if not anchor_overlay_visible:
        return

    if anchor_overlay_root and anchor_overlay_root.winfo_exists():
        try:
            anchor_overlay_root.destroy()
        except tk.TclError:
            pass
        anchor_overlay_canvas = None
        anchor_rect_id = None

    pad = 40
    width = int(ANCHOR_REGION['width']) + pad
    height = int(ANCHOR_REGION['height']) + pad
    left = int(ANCHOR_REGION['left']) - (pad // 2)
    top = int(ANCHOR_REGION['top']) - (pad // 2)

    anchor_overlay_root = create_overlay_window(width, height, left, top)

    anchor_overlay_canvas = tk.Canvas(
        anchor_overlay_root,
        width=width,
        height=height,
        bg="black",
        highlightthickness=0,
    )
    anchor_overlay_canvas.pack()

    anchor_rect_id = anchor_overlay_canvas.create_rectangle(
        pad // 2,
        pad // 2,
        pad // 2 + int(ANCHOR_REGION['width']),
        pad // 2 + int(ANCHOR_REGION['height']),
        outline="#00d4ff",
        width=2,
    )

    anchor_overlay_canvas.create_text(
        width // 2,
        5,
        text="ANCHOR REGION",
        fill="#00d4ff",
        font=("Arial", 12, "bold"),
        anchor="n",
    )


def update_anchor_overlay_region():
    global anchor_overlay_root, anchor_overlay_canvas, anchor_rect_id
    if (
        not anchor_overlay_visible
        or not anchor_overlay_root
        or not anchor_overlay_canvas
        or not anchor_rect_id
    ):
        return

    pad = 40
    width = int(ANCHOR_REGION['width']) + pad
    height = int(ANCHOR_REGION['height']) + pad
    left = int(ANCHOR_REGION['left']) - (pad // 2)
    top = int(ANCHOR_REGION['top']) - (pad // 2)

    anchor_overlay_canvas.config(width=width, height=height)

    anchor_overlay_canvas.coords(
        anchor_rect_id,
        pad // 2,
        pad // 2,
        pad // 2 + int(ANCHOR_REGION['width']),
        pad // 2 + int(ANCHOR_REGION['height']),
    )
    anchor_overlay_root.geometry(f"{width}x{height}+{left}+{top}")
    try:
        anchor_overlay_root.lift()
    except tk.TclError:
        pass


def hide_anchor_overlay():
    global anchor_overlay_root, anchor_overlay_canvas, anchor_rect_id

    if anchor_overlay_root and anchor_overlay_root.winfo_exists():
        try:
            anchor_overlay_root.destroy()
        except tk.TclError:
            pass

    anchor_overlay_root = None
    anchor_overlay_canvas = None
    anchor_rect_id = None


def show_overlay(screen_width: Optional[int] = None, screen_height: Optional[int] = None) -> None:
    show_anchor_overlay()
    show_capture_overlay()

    if screen_width is None or screen_height is None:
        try:
            if capture_overlay_root and capture_overlay_root.winfo_exists():
                screen_width = capture_overlay_root.winfo_screenwidth()
                screen_height = capture_overlay_root.winfo_screenheight()
        except tk.TclError:
            screen_width = screen_width or 1920
            screen_height = screen_height or 1080

    if screen_width is not None and screen_height is not None:
        show_info_overlay(screen_width, screen_height)


def update_overlay_region():
    update_anchor_overlay_region()
    update_capture_overlay_region()


def launch_gui():
    def update_region_from_sliders(*args):
        if GUI_CONTROL_STATE["syncing"]["capture"]:
            return
        CAP_REGION["left"] = int(slider_left.get())
        CAP_REGION["top"] = int(slider_top.get())
        CAP_REGION["width"] = int(slider_width.get())
        CAP_REGION["height"] = int(slider_height.get())
        status_var.set(f"CAP_REGION updated: {CAP_REGION}")
        update_capture_overlay_region()

    def update_anchor_region_from_sliders(*args):
        if GUI_CONTROL_STATE["syncing"]["anchor"]:
            return
        ANCHOR_REGION["left"] = int(anchor_left.get())
        ANCHOR_REGION["top"] = int(anchor_top.get())
        ANCHOR_REGION["width"] = int(anchor_width.get())
        ANCHOR_REGION["height"] = int(anchor_height.get())
        set_anchor_status(f"Anchor region updated: {ANCHOR_REGION}", hold=2.0)
        if AUTO_ALIGN_ENABLED:
            perform_auto_alignment()
        update_anchor_overlay_region()

    def update_anchor_offset_from_sliders(*args):
        if GUI_CONTROL_STATE["syncing"]["anchor"]:
            return
        ANCHOR_OFFSET["x"] = int(anchor_offset_x.get())
        ANCHOR_OFFSET["y"] = int(anchor_offset_y.get())
        set_anchor_status(f"Anchor offset updated: {ANCHOR_OFFSET}", hold=2.0)
        if AUTO_ALIGN_ENABLED:
            perform_auto_alignment()

    def update_info_overlay_from_sliders(*args):
        if GUI_CONTROL_STATE["syncing"].get("overlay"):
            return
        INFO_OVERLAY_OFFSET["x"] = int(info_offset_x.get())
        INFO_OVERLAY_OFFSET["y"] = int(info_offset_y.get())
        status_var.set(
            f"Display offset updated: x={INFO_OVERLAY_OFFSET['x']}, y={INFO_OVERLAY_OFFSET['y']}"
        )
        reposition_info_overlay()

    def toggle_auto_align():
        global AUTO_ALIGN_ENABLED
        AUTO_ALIGN_ENABLED = auto_align_var.get()
        last_alignment_info["enabled"] = AUTO_ALIGN_ENABLED
        if AUTO_ALIGN_ENABLED:
            set_anchor_status("Head sway compensation enabled.")
            perform_auto_alignment()
        else:
            set_anchor_status("Head sway compensation disabled.")

    def reload_anchor_templates():
        global anchor_tracker
        ensure_anchor_directory(ANCHOR_TEMPLATE_DIR)
        if anchor_tracker is None:
            anchor_tracker = AnchorRegionTracker(ANCHOR_TEMPLATE_DIR, ANCHOR_THRESHOLD)
        count = anchor_tracker.set_directory(ANCHOR_TEMPLATE_DIR)
        set_anchor_status(f"Loaded {count} anchor template(s) from {ANCHOR_TEMPLATE_DIR}.")

    def manual_realign():
        success = perform_auto_alignment()
        if success:
            set_anchor_status(
                f"Anchor locked using {last_alignment_info['template']} (score {last_alignment_info['score']:.2f}).",
                hold=2.5,
            )
            status_var.set(f"Auto alignment adjusted CAP_REGION: {CAP_REGION}")
        else:
            set_anchor_status("Anchor match not found. Adjust search region or add templates.")

    def open_anchor_directory():
        path = os.path.abspath(ANCHOR_TEMPLATE_DIR)
        ensure_anchor_directory(path)
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform.startswith("darwin"):
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            set_anchor_status(f"Unable to open template folder: {exc}", hold=3.0)
        else:
            set_anchor_status(f"Opened template folder: {path}")

    def update_threshold(*_args):
        global ANCHOR_THRESHOLD
        try:
            value = float(threshold_var.get())
        except (tk.TclError, ValueError):
            return
        value = max(0.1, min(0.99, value))
        ANCHOR_THRESHOLD = value
        if anchor_tracker is not None:
            anchor_tracker.set_threshold(ANCHOR_THRESHOLD)
        set_anchor_status(f"Anchor detection threshold set to {ANCHOR_THRESHOLD:.2f}")

    def toggle_anchor_overlay_visibility():
        global anchor_overlay_visible
        anchor_overlay_visible = anchor_overlay_var.get()
        if anchor_overlay_visible:
            show_anchor_overlay()
            set_anchor_status("Anchor overlay shown.")
        else:
            hide_anchor_overlay()
            set_anchor_status("Anchor overlay hidden.")

    def update_alignment_interval(*_args):
        global ALIGNMENT_POLL_INTERVAL_MS
        try:
            value = int(alignment_interval_var.get())
        except (tk.TclError, ValueError):
            return
        value = max(100, min(5000, value))
        ALIGNMENT_POLL_INTERVAL_MS = value
        set_anchor_status(f"Alignment interval set to {ALIGNMENT_POLL_INTERVAL_MS} ms", hold=2.0)

    def update_capture_interval(*_args):
        global CONTINUOUS_CAPTURE_INTERVAL
        try:
            value = float(capture_interval_var.get())
        except (tk.TclError, ValueError):
            return
        value = max(0.2, min(30.0, value))
        CONTINUOUS_CAPTURE_INTERVAL = value
        status_var.set(f"Continuous capture interval set to {CONTINUOUS_CAPTURE_INTERVAL:.1f}s")

    def alignment_poll():
        now = time.time()
        message: Optional[str] = None

        if AUTO_ALIGN_ENABLED:
            if anchor_tracker is None or not getattr(anchor_tracker, "templates", None):
                message = "Add anchor templates to enable head sway compensation."
                last_alignment_info.update(
                    {
                        "matched": False,
                        "template": None,
                        "score": 0.0,
                        "match_left": None,
                        "match_top": None,
                        "capture_left": None,
                        "capture_top": None,
                    }
                )
            else:
                match_found = perform_auto_alignment()
                info = last_alignment_info
                if info.get("matched"):
                    message = (
                        f"Anchor locked using {info['template']} (score {info['score']:.2f})."
                    )
                    capture_msg = f"Auto alignment adjusted CAP_REGION: {CAP_REGION}"
                    if status_var.get() != capture_msg:
                        status_var.set(capture_msg)
                elif not match_found:
                    message = "Anchor match not found. Adjust search region or add templates."
        else:
            message = "Head sway compensation disabled."
            last_alignment_info.update(
                {
                    "matched": False,
                    "template": None,
                    "score": 0.0,
                    "match_left": None,
                    "match_top": None,
                    "capture_left": None,
                    "capture_top": None,
                }
            )

        if message and now >= anchor_status_hold["until"]:
            if message != alignment_status_cache.get("message") or anchor_status_var.get() != message:
                anchor_status_var.set(message)
                alignment_status_cache["message"] = message

        try:
            interval = max(100, int(ALIGNMENT_POLL_INTERVAL_MS))
            root.after(interval, alignment_poll)
        except tk.TclError:
            pass

    def on_close():
        global capture_overlay_root, capture_overlay_canvas, capture_rect_id
        global anchor_overlay_root, anchor_overlay_canvas, anchor_rect_id
        global info_overlay_root, info_overlay_canvas, info_text_id

        save_config()
        try:
            for window in (capture_overlay_root, anchor_overlay_root, info_overlay_root):
                if window and window.winfo_exists():
                    window.destroy()
        except Exception:
            pass

        capture_overlay_root = None
        capture_overlay_canvas = None
        capture_rect_id = None
        anchor_overlay_root = None
        anchor_overlay_canvas = None
        anchor_rect_id = None
        info_overlay_root = None
        info_overlay_canvas = None
        info_text_id = None

        root.destroy()

    root = tk.Tk()
    root.title("Star Citizen Scanner Control")
    root.protocol("WM_DELETE_WINDOW", on_close)

    status_var = tk.StringVar(value="Ready.")
    anchor_status_var = tk.StringVar(value="Head sway compensation ready.")

    alignment_status_cache = {"message": None}
    anchor_status_hold = {"until": 0.0}

    def set_anchor_status(message: str, hold: float = 1.5) -> None:
        anchor_status_var.set(message)
        anchor_status_hold["until"] = time.time() + hold
        alignment_status_cache["message"] = None

    frm_region = ttk.LabelFrame(root, text="Capture Region")
    frm_region.pack(fill="x", padx=5, pady=5)

    slider_left = tk.Scale(frm_region, from_=0, to=3000, orient="horizontal",
                           label="Left", command=update_region_from_sliders)
    slider_left.set(CAP_REGION["left"])
    slider_left.pack(fill="x")

    slider_top = tk.Scale(frm_region, from_=0, to=2000, orient="horizontal",
                          label="Top", command=update_region_from_sliders)
    slider_top.set(CAP_REGION["top"])
    slider_top.pack(fill="x")

    slider_width = tk.Scale(frm_region, from_=50, to=1000, orient="horizontal",
                            label="Width", command=update_region_from_sliders)
    slider_width.set(CAP_REGION["width"])
    slider_width.pack(fill="x")

    slider_height = tk.Scale(frm_region, from_=20, to=500, orient="horizontal",
                             label="Height", command=update_region_from_sliders)
    slider_height.set(CAP_REGION["height"])
    slider_height.pack(fill="x")

    register_capture_sliders(slider_left, slider_top, slider_width, slider_height)
    sync_capture_sliders()

    frm_anchor = ttk.LabelFrame(root, text="Head Sway Compensation")
    frm_anchor.pack(fill="x", padx=5, pady=5)

    auto_align_var = tk.BooleanVar(value=AUTO_ALIGN_ENABLED)
    chk_auto_align = tk.Checkbutton(frm_anchor, text="Enable auto alignment", variable=auto_align_var,
                                    command=toggle_auto_align)
    chk_auto_align.pack(anchor="w", padx=5, pady=(5, 0))

    anchor_overlay_var = tk.BooleanVar(value=anchor_overlay_visible)
    chk_anchor_overlay = tk.Checkbutton(
        frm_anchor,
        text="Show anchor overlay",
        variable=anchor_overlay_var,
        command=toggle_anchor_overlay_visibility,
    )
    chk_anchor_overlay.pack(anchor="w", padx=5, pady=(0, 5))

    interval_row = ttk.Frame(frm_anchor)
    interval_row.pack(fill="x", padx=5, pady=(0, 5))
    ttk.Label(interval_row, text="Alignment interval (ms)").pack(side="left")
    alignment_interval_var = tk.IntVar(value=int(ALIGNMENT_POLL_INTERVAL_MS))
    alignment_interval_spin = tk.Spinbox(
        interval_row,
        from_=100,
        to=5000,
        increment=50,
        textvariable=alignment_interval_var,
        width=6,
        command=update_alignment_interval,
    )
    alignment_interval_spin.pack(side="left", padx=5)
    alignment_interval_var.trace_add("write", update_alignment_interval)

    threshold_row = ttk.Frame(frm_anchor)
    threshold_row.pack(fill="x", padx=5, pady=5)
    ttk.Label(threshold_row, text="Detection threshold").pack(side="left")
    threshold_var = tk.DoubleVar(value=ANCHOR_THRESHOLD)
    threshold_spin = tk.Spinbox(threshold_row, from_=0.10, to=0.99, increment=0.01,
                                 textvariable=threshold_var, width=6, command=update_threshold)
    threshold_spin.pack(side="left", padx=5)
    threshold_var.trace_add("write", update_threshold)

    anchor_left = tk.Scale(frm_anchor, from_=0, to=3840, orient="horizontal",
                           label="Anchor Left", command=update_anchor_region_from_sliders)
    anchor_left.set(ANCHOR_REGION["left"])
    anchor_left.pack(fill="x")

    anchor_top = tk.Scale(frm_anchor, from_=0, to=2160, orient="horizontal",
                          label="Anchor Top", command=update_anchor_region_from_sliders)
    anchor_top.set(ANCHOR_REGION["top"])
    anchor_top.pack(fill="x")

    anchor_width = tk.Scale(frm_anchor, from_=50, to=1200, orient="horizontal",
                            label="Anchor Width", command=update_anchor_region_from_sliders)
    anchor_width.set(ANCHOR_REGION["width"])
    anchor_width.pack(fill="x")

    anchor_height = tk.Scale(frm_anchor, from_=50, to=800, orient="horizontal",
                             label="Anchor Height", command=update_anchor_region_from_sliders)
    anchor_height.set(ANCHOR_REGION["height"])
    anchor_height.pack(fill="x")

    anchor_offset_x = tk.Scale(frm_anchor, from_=-300, to=600, orient="horizontal",
                               label="Offset X", command=update_anchor_offset_from_sliders)
    anchor_offset_x.set(ANCHOR_OFFSET["x"])
    anchor_offset_x.pack(fill="x")

    anchor_offset_y = tk.Scale(frm_anchor, from_=-300, to=600, orient="horizontal",
                               label="Offset Y", command=update_anchor_offset_from_sliders)
    anchor_offset_y.set(ANCHOR_OFFSET["y"])
    anchor_offset_y.pack(fill="x")

    register_anchor_sliders(
        anchor_left,
        anchor_top,
        anchor_width,
        anchor_height,
        anchor_offset_x,
        anchor_offset_y,
    )
    sync_anchor_sliders()

    anchor_btn_row = ttk.Frame(frm_anchor)
    anchor_btn_row.pack(fill="x", padx=5, pady=5)
    ttk.Button(anchor_btn_row, text="Reload Templates", command=reload_anchor_templates).pack(side="left", padx=5)
    ttk.Button(anchor_btn_row, text="Realign Now", command=manual_realign).pack(side="left", padx=5)
    ttk.Button(anchor_btn_row, text="Open Template Folder", command=open_anchor_directory).pack(side="left", padx=5)

    frm_display = ttk.LabelFrame(root, text="Result Display")
    frm_display.pack(fill="x", padx=5, pady=5)

    info_offset_x = tk.Scale(
        frm_display,
        from_=-800,
        to=800,
        orient="horizontal",
        label="Display offset X",
        command=update_info_overlay_from_sliders,
    )
    info_offset_x.set(int(INFO_OVERLAY_OFFSET.get("x", 0)))
    info_offset_x.pack(fill="x")

    info_offset_y = tk.Scale(
        frm_display,
        from_=-600,
        to=600,
        orient="horizontal",
        label="Display offset Y",
        command=update_info_overlay_from_sliders,
    )
    info_offset_y.set(int(INFO_OVERLAY_OFFSET.get("y", 0)))
    info_offset_y.pack(fill="x")

    register_overlay_sliders(info_offset_x, info_offset_y)
    sync_overlay_sliders()

    frm_ctrl = ttk.LabelFrame(root, text="Controls")
    frm_ctrl.pack(fill="x", padx=5, pady=5)

    capture_interval_frame = ttk.Frame(frm_ctrl)
    capture_interval_frame.pack(fill="x", padx=5, pady=(5, 10))
    ttk.Label(capture_interval_frame, text="Continuous capture interval (s)").pack(side="left")
    capture_interval_var = tk.DoubleVar(value=float(CONTINUOUS_CAPTURE_INTERVAL))
    capture_interval_spin = tk.Spinbox(
        capture_interval_frame,
        from_=0.2,
        to=30.0,
        increment=0.1,
        textvariable=capture_interval_var,
        width=6,
        format="%.1f",
        command=update_capture_interval,
    )
    capture_interval_spin.pack(side="left", padx=5)
    capture_interval_var.trace_add("write", update_capture_interval)

    button_row = ttk.Frame(frm_ctrl)
    button_row.pack(fill="x", padx=5, pady=(0, 5))

    tk.Button(button_row, text="Single Scan", command=capture_once).pack(side="left", padx=5)
    tk.Button(button_row, text="Loop Toggle", command=toggle_continuous).pack(side="left", padx=5)
    tk.Button(button_row, text="Update Overlay", command=update_overlay_region).pack(side="left", padx=5)
    tk.Button(button_row, text="Set Label Color", command=choose_label_color).pack(side="left", padx=5)
    tk.Button(button_row, text="Save Config", command=save_config).pack(side="left", padx=5)
    tk.Button(button_row, text="Toggle Border", command=toggle_border).pack(side="left", padx=5)

    ttk.Label(root, textvariable=status_var, anchor="w", justify="left").pack(fill="x", padx=5, pady=(5, 0))
    ttk.Label(root, textvariable=anchor_status_var, anchor="w", justify="left", foreground="#8aa6ff").pack(
        fill="x", padx=5, pady=(0, 5)
    )

    root.update_idletasks()
    show_overlay(root.winfo_screenwidth(), root.winfo_screenheight())
    alignment_poll()
    root.mainloop()


# ---------- Scanning Functions ----------
def capture_once():
    """Capture one scan from CAP_REGION and update overlay."""
    global last_result
    auto_aligned = perform_auto_alignment()
    if AUTO_ALIGN_ENABLED:
        logger.debug("Auto alignment %s before capture.", "succeeded" if auto_aligned else "did not match")
    with mss.mss() as sct:
        monitor = {
            "left": CAP_REGION["left"],
            "top": CAP_REGION["top"],
            "width": CAP_REGION["width"],
            "height": CAP_REGION["height"],
        }
        img = sct.grab(monitor)
        pil_img = Image.frombytes("RGB", img.size, img.rgb)

    raw_text = ocr_with_ollama(pil_img)
    code, raw = extract_code_from_text(raw_text)
    info = lookup_deposit(code)

    last_result = {"code": code, "code_raw": raw, "info": info, "raw_text": raw_text}
    update_overlay_label(info, code=code, raw_text=raw or raw_text)
    logger.info(f"Scan result: {last_result}")


def toggle_continuous():
    """Toggle continuous scanning mode."""
    global continuous_mode
    continuous_mode = not continuous_mode
    logger.info(f"Continuous mode: {continuous_mode}")
    if continuous_mode:
        Thread(target=continuous_scan_loop, daemon=True).start()


def continuous_scan_loop():
    """Run scans repeatedly until continuous_mode is turned off."""
    while continuous_mode:
        capture_once()
        interval = max(0.1, float(CONTINUOUS_CAPTURE_INTERVAL))
        time.sleep(interval)



# ---------- Network helpers ----------
def get_local_ip() -> str:
    """Best-effort detection of the primary local network IP address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip_address = sock.getsockname()[0]
            if ip_address:
                return ip_address
    except Exception as exc:
        logger.debug(f"Unable to determine local IP automatically: {exc}")
    return "127.0.0.1"


# ---------- Flask / Hotkeys ----------
template_folder = resource_path("templates")
app = Flask(__name__, template_folder=template_folder)

@app.route("/")
def index():
    return render_template("overlay.html")



@app.route("/status")
def status():
    """Return the latest scan information for the overlay UI."""

    selected_region = request.args.get("region", "STANTON").upper()
    result = last_result or {}
    info = result.get("info") if isinstance(result, dict) else None

    table = None
    if info:
        deposit_key = (info.get("key") or info.get("name") or "").upper()
        region_tables = DEPOSIT_TABLES.get(selected_region, {})
        table = region_tables.get(deposit_key)

    response = {
        # Legacy keys kept for compatibility with any external tools.
        "region": CAP_REGION,
        "label_color": label_color,
        "last": last_result,
        "alignment": last_alignment_info,
        # Data consumed by the overlay web page.
        "selected_region": selected_region,
        "info": info,
        "code": result.get("code") if isinstance(result, dict) else None,
        "code_raw": result.get("code_raw") if isinstance(result, dict) else None,
        "confidence": float(result.get("confidence", 0.0)) if isinstance(result, dict) else 0.0,
        "raw_text": result.get("raw_text") if isinstance(result, dict) else None,
        "table": table,
    }

    return jsonify(response)


def hotkey_listener():
    """Set up hotkey listeners with cross-platform error handling."""
    try:
        keyboard.add_hotkey("7", capture_once)
        keyboard.add_hotkey("ctrl+7", toggle_continuous)
        keyboard.add_hotkey("8", toggle_border)
        logger.info("Hotkeys registered: '7' for single scan, 'Ctrl+7' for continuous toggle, '8' for border toggle")
        keyboard.wait()
    except Exception as e:
        logger.warning(f"Could not set up global hotkeys: {e}")
        logger.info("Note: Linux Support is being tested.")


# ---------- Main ----------
if __name__ == "__main__":
    # Ensure Ollama + model before starting
    ensure_ollama_installed()
    ensure_model_installed("qwen2.5vl:3b")

    load_config()
    anchor_tracker = AnchorRegionTracker(ANCHOR_TEMPLATE_DIR, ANCHOR_THRESHOLD)
    Thread(target=hotkey_listener, daemon=True).start()
    local_ip = get_local_ip()
    logger.info(
        "Starting overlay server: http://127.0.0.1:5000 (this device) | "
        f"http://{local_ip}:5000 (local network)"
    )
    Thread(target=lambda: app.run(host="0.0.0.0", port=5000, debug=False), daemon=True).start()
    launch_gui()
