import re
import os

files_to_process = [
    '/Users/dayansantos/Dev/Sense/ocrChassi/components/cardpanel.py',
    '/Users/dayansantos/Dev/Sense/ocrChassi/components/cardmodel.py'
]

for file_path in files_to_process:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Strip the static QLineEdit, QSpinBox, and QPushButton setStyleSheets that force transparency and white bounds
    pattern_style = r'\s+[\w\.]+\.setStyleSheet\(\"\"\"\n\s+(?:QPushButton|QLineEdit|QSpinBox|QGroupBox) \{(?:.|\n)*?\"\"\"\)'
    content = re.sub(pattern_style, '', content)

    # 2. Let's fix specific fixed dimensions in cardpanel.py
    if 'cardpanel.py' in file_path:
        # CustomCardContent constraints
        content = re.sub(r'\s+self\.image_segmentation_box\.setMinimumHeight\(.*?\)', '', content)
        content = re.sub(r'\s+self\.image_segmentation_box\.setMaximumHeight\(.*?\)', '', content)
        content = re.sub(r'\s+self\.image_segmentation_box\.setMinimumWidth\(.*?\)', '', content)
        content = re.sub(r'\s+self\.image_segmentation_box\.setMaximumWidth\(.*?\)', '', content)
        
        # Replace the tabs creation with scroll areas
        tabs_orig = """        self.tab_segmentation_config = QWidget()
        self.tab_segmentation_config.setFixedHeight(480)
        self.tab_segmentation_config.setFixedWidth(800)
        self.tab_segmentation_results = QWidget()
        self.tab_segmentation_results.setFixedHeight(480)
        self.tab_segmentation_results.setFixedWidth(800)

        self.tabs.addTab(self.tab_segmentation_config, "Segmentação")
        self.tabs.addTab(self.tab_segmentation_results, "Caracteres")"""
        
        tabs_new = """        self.tab_segmentation_config = QWidget()
        self.tab_segmentation_results = QWidget()

        scroll_config = QScrollArea()
        scroll_config.setWidgetResizable(True)
        scroll_config.setFrameShape(QFrame.Shape.NoFrame)
        scroll_config.setWidget(self.tab_segmentation_config)

        scroll_results = QScrollArea()
        scroll_results.setWidgetResizable(True)
        scroll_results.setFrameShape(QFrame.Shape.NoFrame)
        scroll_results.setWidget(self.tab_segmentation_results)

        self.tabs.addTab(scroll_config, "Segmentação")
        self.tabs.addTab(scroll_results, "Caracteres")"""

        content = content.replace(tabs_orig, tabs_new)
        
        # Ensure QScrollArea is imported
        if 'QScrollArea' not in content:
            content = content.replace('QFrame,', 'QFrame, QScrollArea,')

    # 3. Fix specific fixed dimensions in cardmodel.py
    if 'cardmodel.py' in file_path:
        # Remove ModelView constraint 
        content = re.sub(r'\s+self\.model_train_box\.setMinimumHeight\(.*?\)', '', content)
        content = re.sub(r'\s+self\.model_train_box\.setMaximumHeight\(.*?\)', '', content)
        content = re.sub(r'\s+self\.model_train_box\.setMinimumWidth\(.*?\)', '', content)
        content = re.sub(r'\s+self\.model_train_box\.setMaximumWidth\(.*?\)', '', content)

        # Remove rigid tab sizes
        content = re.sub(r'\s+self\.tabs\.setFixedHeight\(height\)', '', content)
        content = re.sub(r'\s+self\.tabs\.setFixedWidth\(800\)', '', content)
        content = re.sub(r'\s+self\.tab_\w+\.setFixedHeight\(height\)', '', content)
        content = re.sub(r'\s+self\.tab_\w+\.setFixedWidth\(800\)', '', content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

print("Done replacing styles and fixed sizes.")
