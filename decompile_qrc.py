import base64

# Copy and paste the base64 string from resources_rc.py
# (This is just an example; your string will be much longer)
base64_data = b"iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAARnQU1BAAC" \
              b"..." # The full string will be much longer

# Decode and save the data
with open("extracted_image.png", "wb") as f:
    f.write(base64.b64decode(base64_data))
