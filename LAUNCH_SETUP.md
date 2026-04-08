# OCR Chassis Launch Setup

Two files have been created to help you launch the application:

## Files Created

1. **ocrChassi.sh** - Shell script that activates the Python environment and runs the app
2. **ocrChassi.desktop** - Desktop shortcut file for Linux

## Setup Instructions

### Option 1: Create Desktop Shortcut (Recommended)

```bash
cd /home/sense-bike/Área de trabalho/ocrChassi

# Make the shell script executable
chmod +x ocrChassi.sh

# Copy the desktop file to your Desktop
cp ocrChassi.desktop ~/Desktop/

# Make the desktop file executable
chmod +x ~/Desktop/ocrChassi.desktop
```

Then double-click the shortcut on your Desktop to launch the application.

### Option 2: Add to Applications Menu

```bash
cd /home/sense-bike/Área de trabalho/ocrChassi

# Make the shell script executable
chmod +x ocrChassi.sh

# Create applications folder if it doesn't exist
mkdir -p ~/.local/share/applications

# Copy the desktop file to applications
cp ocrChassi.desktop ~/.local/share/applications/

# Refresh the application menu
update-desktop-database ~/.local/share/applications/
```

Then search for "OCR Chassis" in your application menu.

### Option 3: Quick Launch from Terminal

```bash
cd /home/sense-bike/Área de trabalho/ocrChassi
./ocrChassi.sh
```

## Notes

- The application will run with the Python environment from `venv_linux`
- The working directory is automatically set to the application folder
- The terminal window will close automatically when the application exits (when using the desktop shortcut)
- If you encounter permission issues, ensure both files are executable: `chmod +x ocrChassi.sh ocrChassi.desktop`
