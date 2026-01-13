import os
import sys
import json
import random
import shutil
import yaml
import cv2

import components.pv_visionlib as vision_lib

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QIcon, QFont
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QScrollArea,
    QWidget,
    QVBoxLayout,
    QApplication,
    QPushButton,
    QMenu,
    QFileDialog,
    QLabel,
    QHBoxLayout,
    QLineEdit,
    QStyle,
    QMessageBox,
)


class ClickableLabel(QLabel):
    """A QLabel that emits a 'clicked' signal with its stored path when clicked."""
    clicked = Signal(str)

    # Signals for context menu actions
    deleteRequested = Signal(str)
    moveRequested = Signal(str, str) # image_path, target_class

    def __init__(self, image_path, class_list, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.class_list = class_list
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.image_path)

    def show_context_menu(self, pos):
        """Creates and shows a context menu for the image."""
        # This method will be implemented in DatasetView to have access to all class widgets
        pass

class DatasetView(QWidget):
    """
    A QWidget to display images from a folder, grouped by subfolder name in a tree view.
    Each subfolder is treated as a class, and its images are shown as children.
    """    

    imageClicked = Signal(str)
    datasetLoaded = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dataset Viewer")

        # --- State tracking for file operations ---
        self.pending_deletions = []
        self.pending_moves = [] # List of {'source': path, 'destination': path}
        self.class_widgets = {} # {class_name: {'header': QLabel, 'layout': QHBoxLayout}}

        self.layout = QVBoxLayout(self)

        # --- Toolbar for folder selection ---
        toolbar_layout = QHBoxLayout()
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setPlaceholderText("Select a dataset folder...")
        self.folder_path_edit.setReadOnly(True)

        browse_button = QPushButton()
        browse_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        browse_button.setIcon(browse_icon)
        browse_button.setToolTip("Browse for a dataset folder")
        browse_button.clicked.connect(self.browse_folder)

        self.apply_button = QPushButton("Apply Changes")
        self.apply_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        self.apply_button.setToolTip("Apply pending deletions and moves to the filesystem")
        self.apply_button.clicked.connect(self.apply_changes)
        self.apply_button.setEnabled(False)

        toolbar_layout.addWidget(QLabel("Dataset Folder:"))
        toolbar_layout.addWidget(self.folder_path_edit)
        toolbar_layout.addWidget(browse_button)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.apply_button)

        self.layout.addLayout(toolbar_layout)

        # --- Main Vertical Scroll Area for folder groups ---
        self.main_scroll_area = QScrollArea()
        self.main_scroll_area.setWidgetResizable(True)
        self.main_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.scroll_content_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content_widget)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.main_scroll_area.setWidget(self.scroll_content_widget)
        self.layout.addWidget(self.main_scroll_area)

    def _clear_layout(self, layout):
        """Removes all widgets from a layout."""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
    
    def browse_folder(self):
        """Opens a dialog to select the root folder of the dataset."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder_path:
            self.folder_path_edit.setText(folder_path)
            self.load_dataset(folder_path)
            self.pending_deletions.clear()
            self.pending_moves.clear()
            self.apply_button.setEnabled(False)

    def load_dataset(self, base_path):
        """
        Scans the given base_path for subdirectories and image files,
        then populates the tree view.
        """
        self._clear_layout(self.scroll_layout)
        self.class_widgets.clear()
        total_image_count = 0

        if not os.path.isdir(base_path):
            return

        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

        # Iterate through subdirectories (which represent classes)
        class_names = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

        for class_name in class_names:
            class_path = os.path.join(base_path, class_name)
            # We already checked for isdir

            image_count = 0
            image_widgets = []
            # Iterate through files in the subdirectory
            for filename in sorted(os.listdir(class_path)):
                if filename.lower().endswith(image_extensions):
                    image_path = os.path.join(class_path, filename)

                    # --- Create a widget for each image ---
                    image_widget = QWidget()
                    image_widget.setFixedWidth(120) # Give it a fixed width
                    
                    image_layout = QVBoxLayout(image_widget)
                    image_layout.setContentsMargins(2, 2, 2, 2)
                    image_layout.setSpacing(2)
                    
                    image_label = ClickableLabel(image_path, class_names)
                    image_label.setPixmap(QIcon(image_path).pixmap(QSize(100, 100)))
                    image_label.setScaledContents(True)
                    image_label.setFixedSize(100, 100)
                    image_label.clicked.connect(self.imageClicked.emit)
                    image_label.customContextMenuRequested.connect(
                        lambda pos, label=image_label: self.show_image_context_menu(pos, label)
                    )

                    name_label = QLabel(filename)
                    name_label.setFont(QFont("Arial", 7))
                    name_label.setWordWrap(True)
                    name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    
                    image_layout.addWidget(image_label)
                    image_layout.addWidget(name_label)
                    
                    image_widgets.append(image_widget)
                    image_count += 1
            
            if image_count > 0:
                # --- Create a header for the class ---                
                header_label = QLabel()
                self.scroll_layout.addWidget(header_label)

                # --- Create a horizontal scroll area for this class's images ---
                image_scroll_area = QScrollArea()
                image_scroll_area.setWidgetResizable(True)
                image_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                image_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
                image_scroll_area.setFixedHeight(150)

                image_container = QWidget()
                image_layout = QHBoxLayout(image_container)
                image_layout.setSpacing(5)
                image_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
                
                for widget in image_widgets:
                    image_layout.addWidget(widget)

                image_scroll_area.setWidget(image_container)
                self.scroll_layout.addWidget(image_scroll_area)

                # Store widgets for later access
                self.class_widgets[class_name] = {
                    'header': header_label,
                    'layout': image_layout,
                    'count': image_count
                }
                self._update_header_label(class_name)
                total_image_count += image_count
        
        self.datasetLoaded.emit({'classes': len(self.class_widgets), 'images': total_image_count})

    def _update_header_label(self, class_name):
        """Updates the text of a class header to show the image count."""
        if class_name in self.class_widgets:
            widget_dict = self.class_widgets[class_name]
            count = widget_dict['count']
            rich_text = (
                f"<span style='font-size: 14pt; border: 1px solid black; border-radius: 5px; padding: 2px;font-weight: bold; color: orange;'>{class_name}</span> "
                f"<span style='font-size: 10pt; color: #cccccc;'>({count} imagens)</span>"
            )
            widget_dict['header'].setText(rich_text)

    def show_image_context_menu(self, pos, label_widget):
        """Creates and shows a context menu for a ClickableLabel."""
        menu = QMenu()

        # --- Delete Action ---
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(lambda: self.delete_image(label_widget))
        menu.addAction(delete_action)

        # --- Move To Submenu ---
        move_menu = menu.addMenu("Move to")
        current_class = os.path.basename(os.path.dirname(label_widget.image_path))
        
        for class_name in sorted(self.class_widgets.keys()):
            if class_name != current_class:
                move_action = QAction(class_name, self)
                move_action.triggered.connect(
                    lambda checked=False, src=label_widget, dst_class=class_name: self.move_image(src, dst_class)
                )
                move_menu.addAction(move_action)

        menu.exec(label_widget.mapToGlobal(pos))

    def delete_image(self, label_widget):
        """Handles the UI logic for deleting an image."""
        image_path = label_widget.image_path
        image_widget = label_widget.parentWidget()
        
        # Find which class layout it belongs to and remove it
        for class_name, data in self.class_widgets.items():
            if image_widget.parent() == data['layout'].parentWidget():
                data['layout'].removeWidget(image_widget)
                image_widget.deleteLater()
                data['count'] -= 1
                self._update_header_label(class_name)
                break

        self.pending_deletions.append(image_path)
        self.apply_button.setEnabled(True)

    def move_image(self, label_widget, destination_class):
        """Handles the UI logic for moving an image to another class."""
        source_path = label_widget.image_path
        image_widget = label_widget.parentWidget()
        source_class = os.path.basename(os.path.dirname(source_path))
        filename = os.path.basename(source_path)

        # --- Remove from old layout ---
        source_data = self.class_widgets[source_class]
        source_data['layout'].removeWidget(image_widget)
        source_data['count'] -= 1
        self._update_header_label(source_class)

        # --- Add to new layout ---
        destination_data = self.class_widgets[destination_class]
        destination_data['layout'].addWidget(image_widget)
        destination_data['count'] += 1
        self._update_header_label(destination_class)

        # --- Update internal state for applying changes ---
        base_dir = os.path.dirname(os.path.dirname(source_path))
        destination_path = os.path.join(base_dir, destination_class, filename)
        
        self.pending_moves.append({'source': source_path, 'destination': destination_path})
        label_widget.image_path = destination_path # Update the path in the label
        self.apply_button.setEnabled(True)

    def apply_changes(self):
        """Applies pending file deletions and moves to the filesystem."""
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Confirm Changes")
        msg_box.setText("This will permanently modify your dataset on disk.")
        details = f"Pending Deletions: {len(self.pending_deletions)}\n"
        details += f"Pending Moves: {len(self.pending_moves)}"
        msg_box.setInformativeText(f"Are you sure you want to apply these changes?\n\n{details}")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Apply | QMessageBox.StandardButton.Cancel)
        msg_box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        
        ret = msg_box.exec()

        if ret == QMessageBox.StandardButton.Apply:
            # Perform deletions
            for file_path in self.pending_deletions:
                if os.path.exists(file_path):
                    os.remove(file_path)
            # Perform moves
            for move in self.pending_moves:
                os.rename(move['source'], move['destination'])
            
            # Clear pending changes and reload
            self.browse_folder() # This will clear pending lists and reload the view

    def prepare_yolo_dataset(self, source_path, destination_path, parent_widget=None):
        """
        Converts the annotated dataset (images + .json files) into YOLO format.
        """
        if not source_path or not os.path.isdir(source_path):
            QMessageBox.warning(parent_widget, "No Source Dataset", "Please set a valid 'Annotation Dataset' path in the 'Modelo' tab.")
            return False

        if not destination_path:
            QMessageBox.warning(parent_widget, "No Destination Path", "Please set a 'Detector Dataset (YOLO)' path in the 'Modelo' tab.")
            return False

        # --- Ask to clear destination folder if it exists and is not empty ---
        if os.path.exists(destination_path) and os.listdir(destination_path):
            reply = QMessageBox.question(
                parent_widget,
                "Clear Destination",
                f"The destination folder '{destination_path}' is not empty.\n\n"
                "Do you want to clear it before preparing the new dataset?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                shutil.rmtree(destination_path)
            else:
                QMessageBox.information(parent_widget, "Cancelled", "Dataset preparation cancelled.")
                return False

        # --- Create YOLO directory structure ---
        paths = {
            "train_images": os.path.join(destination_path, "images", "train"),
            "val_images": os.path.join(destination_path, "images", "val"),
            "train_labels": os.path.join(destination_path, "labels", "train"),
            "val_labels": os.path.join(destination_path, "labels", "val"),
        }
        for path in paths.values():
            os.makedirs(path, exist_ok=True)

        # --- Find all images with corresponding .json annotations ---
        image_files = []
        for root, _, files in os.walk(source_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    json_path = os.path.join(root, file + ".json")
                    if os.path.exists(json_path):
                        image_files.append(os.path.join(root, file))
        if not image_files:
            QMessageBox.information(parent_widget, "No Annotations", "No images with corresponding .json annotation files were found.")
            return False

        # --- Split data into training and validation sets ---
        random.seed(42)
        random.shuffle(image_files)
        split_index = int(len(image_files) * 0.8)
        train_files = image_files[:split_index]
        val_files = image_files[split_index:]

        

        self.process_files(train_files, paths["train_images"], paths["train_labels"])
        self.process_files(val_files, paths["val_images"], paths["val_labels"])

        # --- Create data.yaml file ---
        data_yaml_path = os.path.join(destination_path, "data.yaml")
        yaml_content = {
            'train': os.path.abspath(paths["train_images"]),
            'val': os.path.abspath(paths["val_images"]),
            'nc': 1,
            'names': ['character']
        }
        with open(data_yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        QMessageBox.information(parent_widget, "Success", f"YOLO dataset created successfully at:\n{destination_path}", QMessageBox.Ok)
        return True



    # --- Process files and create YOLO labels ---
    def process_files(self, file_list, image_dest, label_dest):
        for img_path in file_list:
            try:
                shutil.copy(img_path, image_dest)
                json_path = img_path + ".json"
                with open(json_path, 'r') as f:
                    annotations = json.load(f)

                img = cv2.imread(img_path)
                if img is None: continue
                img_height, img_width, _ = img.shape
                
                label_filename = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
                label_path = os.path.join(label_dest, label_filename)

                with open(label_path, 'w') as f_label:
                    for roi_id, data in annotations.items():
                        box = data['box']
                        x, y, w, h = box['x'], box['y'], box['w'], box['h']
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        width_norm = w / img_width
                        height_norm = h / img_height
                        f_label.write(f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
            except Exception as e:
                print(f"Error processing file {img_path}: {e}")

    def _calculate_iou(boxA, boxB):
        # From [x, y, w, h] to [x1, y1, x2, y2]
        boxA_coords = [boxA['x'], boxA['y'], boxA['x'] + boxA['w'], boxA['y'] + boxA['h']]
        inter_x1 = max(boxA_coords[0], boxB[0])
        inter_y1 = max(boxA_coords[1], boxB[1])
        inter_x2 = min(boxA_coords[2], boxB[2])
        inter_y2 = min(boxA_coords[3], boxB[3])
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        boxA_area = (boxA_coords[2] - boxA_coords[0]) * (boxA_coords[3] - boxA_coords[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter_area / float(boxA_area + boxB_area - inter_area) if float(boxA_area + boxB_area - inter_area) > 0 else 0

    def prepare_recognition_dataset(self, source_path, destination_path, detector_path, model_data, parent_widget=None):
        """
        Uses a trained YOLO detector to find characters in the source dataset,
        preprocesses them, and saves them to the recognition dataset folder, sorted by class.
        """
        if not YOLO:
            QMessageBox.critical(parent_widget, "Error", "'ultralytics' package not found. Please install it.")
            return False

        if not source_path or not os.path.isdir(source_path):
            QMessageBox.warning(parent_widget, "No Source Dataset", "Please set a valid 'Annotation Dataset' path.")
            return False

        if not destination_path:
            QMessageBox.warning(parent_widget, "No Destination Path", "Please set a 'Recognition Train Dataset' path.")
            return False

        if not detector_path or not os.path.exists(detector_path):
            QMessageBox.warning(parent_widget, "No Detector Model", "Please set a valid 'Detector Model' path.")
            return False

        # --- Ask to clear destination folder ---
        if os.path.exists(destination_path) and os.listdir(destination_path):
            reply = QMessageBox.question(
                parent_widget, "Clear Destination",
                f"The destination folder '{destination_path}' is not empty.\n\n"
                "This will delete all existing images in it. Are you sure you want to continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                shutil.rmtree(destination_path)
            else:
                QMessageBox.information(parent_widget, "Cancelled", "Data preparation cancelled.")
                return False

        os.makedirs(destination_path, exist_ok=True)

        # --- Load detector model ---
        try:
            detector_model = YOLO(detector_path)
        except Exception as e:
            QMessageBox.critical(parent_widget, "Error", f"Failed to load detector model: {e}")
            return False

        # --- Process each image in the source annotation folder ---
        visionlib = vision_lib.pvVisionLib()
        img_h, img_w = model_data.image_height, model_data.image_width
        count = 0

        for filename in os.listdir(source_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(source_path, filename)
            json_path = img_path + ".json"
            if not os.path.exists(json_path):
                continue

            with open(json_path, 'r') as f:
                annotations = json.load(f)

            source_image = cv2.imread(img_path)
            gray_img = visionlib.convert_to_gray(source_image)

            # Use detector to get bounding boxes
            detection_results = detector_model(source_image, verbose=False)
            boxes = detection_results[0].boxes.xyxy.cpu().numpy()
            
            # --- Match detected boxes with annotations using IoU ---
            annotation_list = list(annotations.values())

            for i, detected_box in enumerate(boxes):
                best_match = None
                highest_iou = 0.5  # Set a threshold to avoid bad matches

                for ann in annotation_list:
                    iou = self._calculate_iou(ann['box'], detected_box)
                    if iou > highest_iou:
                        highest_iou = iou
                        best_match = ann

                if best_match:
                    true_char = best_match['char']
                    if true_char == '?': continue # Skip unlabeled characters

                    x_min, y_min, x_max, y_max = map(int, detected_box)

                    # This is the exact same preprocessing as in InferenceView
                    char_img = gray_img[y_min:y_max, x_min:x_max]
                    h, w = char_img.shape
                    if h > w:
                        pad = (h - w) // 2
                        char_img = cv2.copyMakeBorder(char_img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0])
                    elif w > h:
                        pad = (w - h) // 2
                        char_img = cv2.copyMakeBorder(char_img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0])
                    
                    resized_char = cv2.resize(char_img, (img_w, img_h), interpolation=cv2.INTER_AREA)

                    class_folder = os.path.join(destination_path, true_char)
                    os.makedirs(class_folder, exist_ok=True)
                    save_path = os.path.join(class_folder, f"{os.path.splitext(filename)[0]}_char_{i}.png")
                    cv2.imwrite(save_path, resized_char)
                    count += 1

        QMessageBox.information(parent_widget, "Success", f"Recognition dataset prepared successfully.\n{count} character images were created.")
        return True

# if __name__ == '__main__':
#     # Example of how to use the widget
#     app = QApplication(sys.argv)
#     widget = DatasetView()
#     widget.resize(400, 600)
#     widget.show()
#     sys.exit(app.exec())