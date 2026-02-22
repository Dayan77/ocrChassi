import re
import os

files_to_process = [
    '/Users/dayansantos/Dev/Sense/ocrChassi/components/cardpanel.py',
    '/Users/dayansantos/Dev/Sense/ocrChassi/components/cardmodel.py'
]

for file_path in files_to_process:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove all setMaximumWidth and setFixedWidth constraints which cause button text to clip
    content = re.sub(r'\s+[\w\.]+\.setMaximumWidth\(\d+\)', '', content)
    content = re.sub(r'\s+[\w\.]+\.setFixedWidth\(\d+\)', '', content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

print("Removed explicit widths.")
