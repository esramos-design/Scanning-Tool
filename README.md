# Star Citizen Deposit Scanner

🚀 **An easy-to-use tool that automatically reads and identifies Star Citizen mining deposit codes from your screen!**

[![GitHub release](https://img.shields.io/github/v/release/FrozenButton/Scanning-Tool)](https://github.com/FrozenButton/Scanning-Tool/releases)
[![GitHub stars](https://img.shields.io/github/stars/FrozenButton/Scanning-Tool)](https://github.com/FrozenButton/Scanning-Tool/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/FrozenButton/Scanning-Tool)](https://github.com/FrozenButton/Scanning-Tool/issues)

## 📥 Download

**Get the latest version from GitHub:**

### Option 1: Download ZIP (Easiest)
1. Click the **green "Code" button** at the top of this page
2. Click **"Download ZIP"**
3. **Extract the ZIP file** to any folder on your computer
4. Follow the instructions below to run it

### Option 2: Git Clone (For developers)
```bash
git clone https://github.com/FrozenButton/Scanning-Tool.git
cd Scanning-Tool
```

### Option 3: Direct Release Download
- Go to **[Releases](https://github.com/FrozenButton/Scanning-Tool/releases)** and download the latest version

---

## What does this do?

This tool helps Star Citizen miners by:
- 📸 **Taking screenshots** of deposit codes on your screen
- 🤖 **Reading the numbers** using AI
- 📊 **Telling you what type of deposit** it is and how valuable the materials are
- 📍 **Showing an overlay** on your screen with the deposit information
- 🎯 **Locking onto a HUD anchor** so the capture box follows head sway and ship movement

## 💻 System Requirements

**⚠️ This tool is designed for mid to high-end gaming PCs and is NOT intended for low-spec machines.**

Since this tool runs alongside Star Citizen, you'll need:

### Recommended Requirements
- **VRAM**: 8GB+ dedicated graphics memory (tool uses ~1.73GB for AI model)
- **RAM**: 32GB+ system memory (Star Citizen + this tool)
- **CPU**: Modern multi-core processor
- **OS**: Windows 10/11 or Linux (64-bit)

### Why these requirements?
- The AI model (**qwen2.5vl:3b**) requires **~1.73GB of VRAM** to run efficiently
- Star Citizen is already demanding on system resources
- Running both simultaneously requires adequate hardware

> 💡 **Performance Tip**: If you experience lag or performance issues, close the scanner when not actively mining, or consider upgrading your graphics card or RAM.

> ⚠️ **VRAM Note**: If your graphics card doesn't have enough VRAM, Ollama will automatically fall back to using your CPU for AI processing. This will work but will be significantly slower and may impact game performance more than GPU processing.

---

## 🎮 How to use it (Super Easy!)

### For Windows Users (Most People)

1. **Download this tool** from GitHub (see Download section above)
2. **Extract the ZIP file** to any folder on your computer  
3. **Double-click** the file called `launch_windows.bat`
4. **Wait** - The tool will automatically download everything it needs (this might take a few minutes the first time)
5. **That's it!** The scanner will open and be ready to use

> ⚠️ **First time setup takes 2-5 minutes** while it downloads Python and other components. After that, it starts instantly!

### For Linux Users

1. **Download this tool** from GitHub (see Download section above)
2. **Extract the files** to any folder on your computer
3. **Open a terminal** in that folder
4. **Type:** `./launch_linux.sh` and press Enter
5. **Follow any prompts** to install Python if needed
6. **That's it!** The scanner will open and be ready to use

## 🎯 What you need to install separately

### Ollama (Required for AI scanning)

**This tool needs "Ollama" to read the deposit codes with AI.**

**How to install Ollama:**
1. Go to **https://ollama.com/** in your web browser
2. Click the **download button for Windows** (or Linux if you use Linux)
3. **Run the installer** - just click "Next" through everything
4. **Restart your computer** when it's done

> 💡 **Don't worry!** If you forget to install Ollama, the scanner tool will remind you and even open the website for you.

## 🎮 How to use the scanner

### Setting up the scan area

1. **Start Star Citizen** and go to a mining area
2. **Open the scanner tool** (it will show a red capture box on your screen)
3. **Drag the sliders** in the scanner window to move the red box over where deposit codes appear
4. **Make the red box** just big enough to cover the deposit code numbers

### 🧭 Head sway compensation (anchor alignment)

The tool can now follow the in-game HUD even when head sway is enabled. It does this by
locking onto a **stable anchor icon** on your ship screen and then adjusting the red
capture box using an offset.

1. **Show the anchor overlay** – The cyan “Anchor Region” frame is always-on-top so you
   can position it over a reliable HUD element. Toggle its visibility with the
   **"Show anchor overlay"** checkbox if you need a clearer view.
2. **Position the anchor region** – Use the *Anchor Left/Top/Width/Height* sliders in the
   **Head Sway Compensation** panel to cover the icon or shape you want to track.
3. **Capture or add templates** – Click **Open Template Folder** to jump to
   `assets/anchor_templates/` and drop in one or more cropped screenshots (PNG/JPG/BMP).
   Each template should be a tight crop of the anchor icon without extra background. Use
   the Windows Snipping Tool or your favourite editor to grab these still frames from the
   game. You can organise them in subfolders—everything in this directory is loaded.
4. **Reload templates** – After adding files, click **Reload Templates**. The status bar
   will confirm how many templates were loaded. If none load, check the file extension and
   make sure the images are not empty.
5. **Calibrate offsets** – With the game open, click **Realign Now**. When the anchor is
   matched you’ll see a status message showing the template name and match score. Adjust
   the *Offset X* and *Offset Y* sliders until the red capture box snaps directly over the
   deposit code after a realign.
6. **Fine-tune the threshold** – If matching is inconsistent, tweak the detection
   threshold (default `0.82`). Lower values make matching easier but risk false positives;
   higher values require a closer match.

Tips:

- You can store multiple templates for different lighting conditions or ship displays.
- Auto alignment runs before every scan when **Enable auto alignment** is checked. Turn it
  off temporarily if you want to position the capture region manually.
- Both the capture and anchor overlays continuously lift themselves, so they stay visible
  even while Star Citizen is in focus or fullscreen.

### Scanning deposits

**Option 1 - Hotkeys (Easy):**
- Press **"7"** to scan once
- Press **"Ctrl+7"** to start auto-scanning every 2 seconds
- Press **"8"** to hide/show the red box

**Option 2 - Buttons (If hotkeys don't work):**
- Click **"Single Scan"** to scan once
- Click **"Loop Toggle"** to start/stop auto-scanning

### Reading the results

- The tool will show **deposit type and quantity** above the red box
- **Green text** = High-value materials (like Quantanium, Gold)
- **Yellow text** = Medium-value materials  
- **Orange text** = Lower-value materials

## 🛠️ If something goes wrong

### "Python not found" or similar errors
- **Solution:** Just run `launch_windows.bat` again - it will download Python automatically

### "Ollama not installed" message
- **Solution:** Go to https://ollama.com/ and download/install Ollama, then restart the scanner

### Hotkeys don't work
- **Solution:** Use the buttons in the scanner window instead - they do the same thing

### The red box doesn't appear
- **Solution:** Click "Update Overlay" button in the scanner window

### Can't see the deposit code numbers clearly
- **Solution:** Adjust your Star Citizen graphics settings to make text clearer, or make the red box bigger

## 🎯 Tips for best results

1. **Make sure Star Citizen text is clear** - adjust graphics settings if text looks blurry
2. **Position the red box precisely** over just the deposit code numbers
3. **Don't make the red box too big** - it works better when focused on just the code
4. **Wait for the code to fully appear** before scanning (don't scan while the code is still appearing)

## 🔧 For Advanced Users

If you want to set things up manually instead of using the automatic installer:

### Manual Installation
1. Install Python 3.8+ from python.org
2. Install Ollama from ollama.com  
3. Open terminal/command prompt in the tool folder
4. Run: `python -m venv venv`
5. Activate venv: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux)
6. Run: `pip install -r requirements.txt`
7. Run: `python scan_deposits.py`

## 🆘 Still need help?

1. **Make sure Ollama is installed** from https://ollama.com/
2. **Try restarting your computer** after installing Ollama
3. **Run the launch script again** - it will re-download anything that's missing
4. **Check that Star Citizen is running** and you're in a mining area where deposit codes appear

## 🐛 Found a bug or need help?

- **Report issues:** [GitHub Issues](https://github.com/FrozenButton/Scanning-Tool/issues)
- **Request features:** [GitHub Issues](https://github.com/FrozenButton/Scanning-Tool/issues)
- **Get updates:** [Watch this repository](https://github.com/FrozenButton/Scanning-Tool) for notifications

**When reporting issues, please include:**
- Your operating system (Windows 10, Windows 11, Linux, etc.)
- What you were trying to do
- Any error messages you saw
- Screenshots if helpful

## ⭐ Like this tool?

If this tool helps your mining operations, please:
- ⭐ **Star this repository** on GitHub
- 🍴 **Share it** with other Star Citizen miners
- 🐛 **Report bugs** to help make it better
- 🤝 **Contribute code** via pull requests

---

**Happy mining, citizen! 🪨⛏️**

*Repository: https://github.com/FrozenButton/Scanning-Tool* 
