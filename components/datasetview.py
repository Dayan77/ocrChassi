import os
import sys

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


if __name__ == '__main__':
    # Example of how to use the widget
    app = QApplication(sys.argv)
    widget = DatasetView()
    widget.resize(400, 600)
    widget.show()
    sys.exit(app.exec())