import re

file_path = '/Users/dayansantos/Dev/Sense/ocrChassi/components/cardmodel.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

def wrap_in_scroll(view_name, layout_name):
    global content
    original = f"{layout_name}.addWidget(self.{view_name})"
    replacement = f"""scroll_{view_name} = QScrollArea()
        scroll_{view_name}.setWidgetResizable(True)
        scroll_{view_name}.setFrameShape(QFrame.Shape.NoFrame)
        scroll_{view_name}.setWidget(self.{view_name})
        {layout_name}.addWidget(scroll_{view_name})"""
    content = content.replace(original, replacement)

wrap_in_scroll('dataset_view', 'edit_tab_layout')
wrap_in_scroll('training_summary_view', 'train_tab_layout')
wrap_in_scroll('results_view', 'results_tab_layout')
wrap_in_scroll('easyocr_view', 'easyocr_tab_layout')
wrap_in_scroll('inference_view', 'inference_tab_layout')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Wrapped cardmodel views in QScrollAreas.")
