from dataclasses import asdict
import datetime
import json
import os
import sys
import cv2
import qtawesome as qta
from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QLabel, QListWidget, QListWidgetItem, QPushButton, QMessageBox,
    QSizePolicy, QGraphicsDropShadowEffect, QFrame,
    QCheckBox, QComboBox, QDateEdit, QDateTimeEdit, QDial, QDoubleSpinBox,
    QFontComboBox, QLCDNumber, QLineEdit, QProgressBar, QRadioButton, QSlider,
    QSpinBox, QTimeEdit, QGroupBox, QTabWidget
)

from PySide6.QtGui import QColor, QIcon, QPixmap
from qt_material import apply_stylesheet

import pyqtgraph as pg

from superqt import QRangeSlider, QLabeledRangeSlider

import pv_visionlib

from components.customlistview import CustomListView



class CardPanel(QWidget):

    def __init__(self, parent=None, color_bg="#2c3e50"):
        super().__init__(parent)
        self.parent_wnd = parent
        # Set a fixed background color for the card
        self.setStyleSheet(f"background-color: {color_bg}; border-radius: 10px;")
        
        # Create a layout for the card's content
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(1)
        
        # Apply a drop shadow effect
        self.shadow_effect = QGraphicsDropShadowEffect(self)
        self.shadow_effect.setBlurRadius(20)
        self.shadow_effect.setXOffset(0)
        self.shadow_effect.setYOffset(5)
        self.shadow_effect.setColor(QColor(0, 0, 0, 100)) # Semi-transparent black
        self.setGraphicsEffect(self.shadow_effect)
        
    def set_content(self, widget):
        """Adds a widget to the card's layout."""
        self.layout.addWidget(widget)

# Custom widget to display inside the card
class CustomCardContent(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.parent_wnd = parent
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        self.title_label = QLabel("Card Title")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.title_label.setMinimumHeight(60)
        self.title_label.setMaximumHeight(60)
        #layout.addWidget(self.title_label)
        
        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        #layout.addWidget(line)
        
        
        self.image_segmentation_box = ImageSegmentationView(self.parent_wnd)
        #image_segmentation_box.setStyleSheet("font-size: 12px; background-color: #536276ff")
        self.image_segmentation_box.setMinimumHeight(350)
        self.image_segmentation_box.setMaximumHeight(550)
        self.image_segmentation_box.setMinimumWidth(500)
        self.image_segmentation_box.setMaximumWidth(800)
        layout.addWidget(self.image_segmentation_box)


        #test_button = QPushButton
        test_button = QPushButton()
        test_button.setMaximumHeight(35)
        test_button.setMaximumWidth(60)
        test_icon = QIcon(":icons/icons/grid.svg") 
        test_button.setIcon(test_icon)

        # Style the button to be partially transparent and smaller
        test_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)

        #layout.addWidget(test_button)
        test_button.clicked.connect(self.test_pytesseract)

    def test_pytesseract(self):
        visionLib = pv_visionlib.pvVisionLib()
        visionLib.collect_samples(self.parent_wnd.cameras[0].image_path, '../models/', 'ocr_samples')

        





    




class ImageSegmentationView(QFrame):
    threshold = 50
    diameter = 9
    sigma = 75
    space = 75
    min_a = 0
    max_a = 100
    min_w = 3
    max_w = 100
    min_h = 1
    max_h = 100
    def __init__(self, parent=None):
        super().__init__(parent)

        

        self.parent_wnd = parent

        self.tabs = QTabWidget(self)

        self.tab_segmentation_config = QWidget()
        self.tab_segmentation_config.setFixedHeight(480)
        self.tab_segmentation_config.setFixedWidth(800)
        self.tab_segmentation_results = QWidget()
        self.tab_segmentation_results.setFixedHeight(480)
        self.tab_segmentation_results.setFixedWidth(800)

        self.tabs.addTab(self.tab_segmentation_config, "Segmentação")
        self.tabs.addTab(self.tab_segmentation_results, "Caracteres")

        # grid_fix = QWidget(self.tab_segmentation_config)
        # grid_fix.setMaximumHeight(300)

        grid = QHBoxLayout(self.tab_segmentation_config)

        layout = QVBoxLayout()
        layout.setSpacing(1)
        layout.setContentsMargins(1,1,1,1)

        grid.addLayout(layout)
        

        camA_radio = QRadioButton("Camera A")
        self.camB_radio = QRadioButton("Camera B")
        cam_select = QHBoxLayout()
        
        cam_select.addWidget(camA_radio)
        cam_select.addWidget(self.camB_radio)

        # layout.addLayout(cam_select)
        
        self.image_filtered = QLabel()
        self.image_filtered.setMinimumHeight(200)
        self.image_filtered.setPixmap(QPixmap(":images/images/No_Image_Available.jpg"))
        self.image_filtered.setScaledContents(True)
        image_buttons = QHBoxLayout()
        image_buttons.addWidget(self.image_filtered)
        image_buttons.addLayout(image_buttons)

        buttons = QVBoxLayout()
        test_button = QPushButton("Segmentação")
        test_button.setMaximumHeight(35)
        test_button.setMinimumWidth(120)
        test_button.setMaximumWidth(120)
        test_icon = QIcon(":icons/icons/grid.svg") 
        test_button.setIcon(test_icon)

        # Style the button to be partially transparent and smaller
        test_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        test_button.clicked.connect(self.test_pytesseract)

        buttons.addWidget(test_button)
        image_buttons.addLayout(buttons)
        
        layout.addLayout(image_buttons) #image_filtered)



        filters_row = QVBoxLayout()
        threshold_col = QVBoxLayout()
        label_threshold = QLabel("Threshold")
        label_threshold.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_threshold_value = QLabel(str(self.threshold))
        self.label_threshold_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_threshold = QDial()
        image_threshold.setMaximumSize(60,60)
        image_threshold.valueChanged.connect(self.update_threshold)
        image_threshold.setValue(self.threshold)
        image_threshold.setRange(0, 255)
        image_threshold.setSingleStep(1)
        threshold_col.addWidget(label_threshold)
        threshold_col.addWidget(image_threshold)
        threshold_col.addWidget(self.label_threshold_value)

        diameter_col = QVBoxLayout()
        image_diameter = QDial()    
        image_diameter.setMaximumSize(60,60)
        image_diameter.setValue(self.diameter)
        label_diameter = QLabel("Pixel")
        label_diameter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_diameter_value = QLabel(str(self.diameter))
        self.label_diameter_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_diameter.valueChanged.connect(self.update_diameter)
        image_diameter.setRange(0, 255)
        image_diameter.setSingleStep(1)
        diameter_col.addWidget(label_diameter)
        diameter_col.addWidget(image_diameter)
        diameter_col.addWidget(self.label_diameter_value)
        

        sigma_col = QVBoxLayout()
        image_sigma = QDial()
        image_sigma.setMaximumSize(60,60)
        image_sigma.setValue(self.sigma)
        label_sigma = QLabel("Sigma")
        label_sigma.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_sigma_value = QLabel(str(self.sigma))
        self.label_sigma_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_sigma.valueChanged.connect(self.update_sigma)
        image_sigma.setRange(0, 255)
        image_sigma.setSingleStep(1)
        sigma_col.addWidget(label_sigma)
        sigma_col.addWidget(image_sigma)
        sigma_col.addWidget(self.label_sigma_value)

        space_col = QVBoxLayout()
        image_space = QDial()
        image_space.setMaximumSize(60,60)
        image_space.setValue(self.space)
        label_space = QLabel("Similar")
        label_space.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_space_value = QLabel(str(self.space))
        self.label_space_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_space.valueChanged.connect(self.update_space)
        image_space.setRange(0, 255)
        image_space.setSingleStep(1)
        space_col.addWidget(label_space)
        space_col.addWidget(image_space)
        space_col.addWidget(self.label_space_value)

        filters_row.addLayout(threshold_col)
        filters_row.addLayout(diameter_col)
        filters_row.addLayout(sigma_col)
        filters_row.addLayout(space_col)

        grid.addLayout(filters_row)

       #width_height_row = QHBoxLayout()
        gbWidth_height = QGroupBox("Comprimento / Altura")
        gbWidth_height.setStyleSheet("""
            QGroupBox {
                background-color: #0e6e4b;
                border: 2px solid #555555;
                border-radius: 8px;
                margin-top: 1ex; /* Leave space for the title */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* Position at the top center */
                padding: 0 8px;
                background-color: #555555;
                color: white;
                border-radius: 4px;
            }
        """)
        gbWidth_height.setMaximumHeight(130)
        gbWidth_height.setMinimumHeight(130)
        width_height_row = QHBoxLayout()
        gbWidth = QGroupBox("Filtro de comprimento")
        gbWidth.setMaximumHeight(80)
        width_constrains_row = QHBoxLayout()
        width_constrains = QLabeledRangeSlider(Qt.Orientation.Horizontal)
        width_constrains.setValue((self.min_w, self.max_w))
        width_constrains.setBarMovesAllHandles(False)
        width_constrains_row.addWidget(width_constrains)
        
        gbWidth.setLayout(width_constrains_row)
        gbWidth.setStyleSheet("""
            QGroupBox {
                background-color: #0e6e4b;
                border: 2px solid #555555;
                border-radius: 8px;
                margin-top: 1ex; /* Leave space for the title */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* Position at the top center */
                padding: 0 8px;
                background-color: #555555;
                color: white;
                border-radius: 4px;
            }
        """)
        width_height_row.addWidget(gbWidth)
        gbWidth_height.setLayout(width_height_row)

        layout.addWidget(gbWidth_height)

        gbHeight = QGroupBox("Filtro de altura")
        gbHeight.setMaximumHeight(80)
        height_constrains_row = QHBoxLayout()
        height_constrains = QLabeledRangeSlider(Qt.Orientation.Horizontal)
        height_constrains.setValue((self.min_h, self.max_h))
        height_constrains.setBarMovesAllHandles(False)
        height_constrains_row.addWidget(height_constrains)
        
        gbHeight.setLayout(height_constrains_row)
        gbHeight.setStyleSheet("""
            QGroupBox {
                background-color: #0e6e4b;
                border: 2px solid #555555;
                border-radius: 8px;
                margin-top: 1ex; /* Leave space for the title */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* Position at the top center */
                padding: 0 8px;
                background-color: #555555;
                color: white;
                border-radius: 4px;
            }
        """)


        width_height_row.addWidget(gbHeight)

        gbArea = QGroupBox("Filtro de area")
        gbArea.setMaximumHeight(120)
        area_constrains_row = QHBoxLayout()
        area_constrains = QLabeledRangeSlider(Qt.Orientation.Horizontal)
        area_constrains.setMinimumHeight(50)
        area_constrains.setValue((self.min_a, self.max_a))
        #area_constrains.setBarMovesAllHandles(False)
        area_constrains_row.addWidget(area_constrains)

        gbArea.setLayout(area_constrains_row)
        gbArea.setStyleSheet("""
            QGroupBox {
                background-color: #0e6e4b;
                border: 2px solid #555555;
                border-radius: 8px;
                margin-top: 1ex; /* Leave space for the title */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* Position at the top center */
                padding: 0 8px;
                background-color: #555555;
                color: white;
                border-radius: 4px;
            }
        """)
        layout.addWidget(gbArea)
        grid.addStretch()
        layout.addStretch()
        

        #image_threshold.valueChanged.connect

    def update_diameter(self, value):
        self.diameter = value
        self.label_diameter_value.setText(str(value))
    def update_threshold(self, value):
        self.threshold = value
        self.label_threshold_value.setText(str(value))
    def update_sigma(self, value):
        self.sigma = value
        self.label_sigma_value.setText(str(value))
    def update_space(self, value):
        self.space = value
        self.label_space_value.setText(str(value))
    def update_w(self, values):
        self.w_min = values[0]

    
    def test_pytesseract(self):
        if self.camB_radio.isChecked():
            cam_index = 1
        else:
            cam_index = 0

        visionLib = pv_visionlib.pvVisionLib()
        
        img, bin, boxes, samples, chars = visionLib.collect_samples(self.parent_wnd.cameras[cam_index].image_path, '../models/', 'ocr_samples', self.threshold, self.diameter, self.sigma, self.space, self.min_w, self.max_w, self.min_h, self.max_h, self.min_a, self.max_a)
        self.refresh_image(bin)

        cam_index = 1 if self.camB_radio.isChecked() == True else 0
        imv = self.parent_wnd.cameras[cam_index].draw_rois(img, chars)

        self.populate_results(chars, samples, cam_index)
        pass
        
    
    def populate_results(self, chars, samples, cam_index):
        self.card_results = QVBoxLayout(self.tab_segmentation_results)

        #Characters
        self.alphabet_chars = QLineEdit()
        self.alphabet_chars.setMaxLength(100)
        self.alphabet_chars.setPlaceholderText("Caracteres para treinamento")
        # Style the QLineEddit to be partially transparent and smaller
        self.alphabet_chars.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)

        #buttons
        results_actions_row = QHBoxLayout()
        
        save_samples_btn = QPushButton("Exportar")
        save_samples_btn.setMaximumHeight(35)
        save_samples_btn.setMaximumWidth(100)
        save_samples_icon = QIcon(":icons/icons/external-link.svg") 
        save_samples_btn.setIcon(save_samples_icon)

        # Style the button to be partially transparent and smaller
        save_samples_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)

        clear_samples_btn = QPushButton("Limpar")
        clear_samples_btn.text = "Limpar"
        clear_samples_btn.setMaximumHeight(35)
        clear_samples_btn.setMaximumWidth(100)
        clear_samples_icon = QIcon(":icons/icons/trash-2.svg") 
        clear_samples_btn.setIcon(clear_samples_icon)

        # Style the button to be partially transparent and smaller
        clear_samples_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)

        delete_samples_btn = QPushButton()
        delete_samples_btn.setText("selecionado")
        delete_samples_btn.setMaximumHeight(35)
        delete_samples_btn.setFixedWidth(100)
        delete_samples_icon = QIcon(":icons/icons/trash.svg") 
        delete_samples_btn.setIcon(delete_samples_icon)

        # Style the button to be partially transparent and smaller
        delete_samples_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)

        refresh_samples_btn = QPushButton("Atualizar")
        refresh_samples_btn.setMaximumHeight(35)
        refresh_samples_btn.setFixedWidth(100)
        refresh_samples_icon = QIcon(":icons/icons/refresh-cw.svg") 
        refresh_samples_btn.setIcon(refresh_samples_icon)

        # Style the button to be partially transparent and smaller
        refresh_samples_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)

        add_roi_btn = QPushButton("Adicionar")
        add_roi_btn.setMaximumHeight(35)
        add_roi_btn.setFixedWidth(100)
        add_roi_icon = QIcon(":icons/icons/plus-square.svg") 
        add_roi_btn.setIcon(add_roi_icon)

        # Style the button to be partially transparent and smaller
        add_roi_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)

        results_actions_row.addWidget(refresh_samples_btn)
        results_actions_row.addWidget(add_roi_btn)
        results_actions_row.addWidget(save_samples_btn)
        results_actions_row.addWidget(delete_samples_btn)
        results_actions_row.addWidget(clear_samples_btn)

        self.cam_index = cam_index

        add_roi_btn.clicked.connect(self.parent_wnd.cameras[cam_index].create_new_roi)
        delete_samples_btn.clicked.connect(self.parent_wnd.cameras[cam_index].delete_selected_roi)
        clear_samples_btn.clicked.connect(self.parent_wnd.cameras[cam_index].delete_all_rois)
        refresh_samples_btn.clicked.connect(self.refresh_rois_listview)
        save_samples_btn.clicked.connect(self.export_rois_list)

        #list view with images
        self.list_segmentation_results = CustomListView("Segmentação de caracteres")
        self.list_segmentation_results.setFixedHeight(630)
        self.list_segmentation_results.setFixedWidth(350)
        

        for index, item in samples.items():
            img_cv = self.updateCV_Image(item["image"])
            self.list_segmentation_results.add_list_item(img_cv, item["char"])

        #Export images path
        self.export_images_path = QLineEdit()
        self.export_images_path.setMaxLength(100)
        self.export_images_path.setPlaceholderText("Pasta para armazenamento de anotações de imagens")
        # Style the QLineEddit to be partially transparent and smaller
        self.export_images_path.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)

        export_slices_btn = QPushButton("Exportar Imagens Treinamento")
        export_slices_btn.setMaximumHeight(35)
        export_slices_btn.setFixedWidth(250)
        export_slices_icon = QIcon(":icons/icons/image.svg") 
        export_slices_btn.setIcon(export_slices_icon)
        export_slices_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 5px;
            }
        """)

        #add widgets
        self.card_results.addWidget(self.alphabet_chars)
        self.card_results.addLayout(results_actions_row)

        results_data_row = QHBoxLayout()
        export_path_layout = QVBoxLayout()
        export_path_layout.addWidget(self.export_images_path)
        export_path_layout.addWidget(export_slices_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        export_slices_btn.clicked.connect(self.on_export_slices_clicked)

        results_data_row.addLayout(export_path_layout)
        results_data_row.addWidget(self.list_segmentation_results)
        self.card_results.addLayout(results_data_row)
        

        self.card_results.addStretch()

    

    def refresh_rois_listview(self):
        self.list_segmentation_results.list_widget.clear()
        samples = self.parent_wnd.cameras[self.cam_index].refresh_rois(self.alphabet_chars.text())

        # for _roi in self.parent_wnd.cameras[self.cam_index].rois:
        #     self.list_segmentation_results.add_list_item(_roi['image'], _roi['text'])
        for index, item in samples.items():
            img_cv = self.updateCV_Image(item["image"])
            self.list_segmentation_results.add_list_item(img_cv, item["char"])

        self.alphabet_chars.setText(self.parent_wnd.cameras[self.cam_index].image_chars)


    def export_rois_list(self):
        try:
            ###Create json with image rois informations
            samples = self.parent_wnd.cameras[self.cam_index].refresh_rois_dict(self.alphabet_chars.text())

            # samples_dict = asdict(samples)

            samples_json = json.dumps(samples, indent=4)
            
            filename = self.parent_wnd.cameras[self.cam_index].image_path+".json"

            with open(filename, 'w') as json_file:
                    json.dump(samples, json_file, indent=4)

            QMessageBox.information(self, "Sucesso", "Anotações JSON exportadas com sucesso!")

            # # Also export the image slices
            # if self.export_slices_images():
            #     QMessageBox.information(self, "Sucesso", "Anotações e imagens exportadas com sucesso!")
            # else:
            #     QMessageBox.warning(self, "Falha", "Ocorreu um erro ao exportar as imagens.")
            return True
        except:
            return False


    def export_slices_images(self):
        try:
            ###Create json with image rois informations
            samples = self.parent_wnd.cameras[self.cam_index].refresh_rois_dict(self.alphabet_chars.text())

            ###Export image slice of roi for training 
            path = self.export_images_path.text()
            if not path or not os.path.isdir(os.path.dirname(path)):
                 QMessageBox.warning(self, "Caminho Inválido", "Por favor, insira um caminho válido para exportar as imagens.")
                 return False

            if not os.path.exists(path):
                os.makedirs(path)

            origin_image_filename = os.path.basename(self.parent_wnd.cameras[self.cam_index].image_path)


            for item in samples.items():
                char = item[1]['char']
                if char == '?': continue # Skip unknown characters

                x = item[1]['box']['x']
                y = item[1]['box']['y']
                w = item[1]['box']['w']
                h = item[1]['box']['h']

                char_img = self.parent_wnd.cameras[self.cam_index].actual_image[y:y+h, x:x+w] 
                

                path_write = os.path.join(path, char)
                if not os.path.exists(path_write):
                    os.makedirs(path_write)

                # Create a more unique filename
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
                filename_exp = f"{char}_{item[0]}_{timestamp}_{origin_image_filename}"
                cv2.imwrite(os.path.join(path_write, filename_exp), char_img)

            return True
        except Exception as e:
            print(f"Error exporting slices: {e}")
            return False

    def on_export_slices_clicked(self):
        """Slot for the new export slices button."""
        if self.export_slices_images():
            QMessageBox.information(self, "Sucesso", "Imagens exportadas para treinamento com sucesso!")
        else:
            QMessageBox.warning(self, "Falha", "Ocorreu um erro ao exportar as imagens para treinamento.")

    def updateCV_Image(self, image):
        lib = pv_visionlib.pvVisionLib()
        qt_image = lib.convert_qt_image(image)
        return qt_image
        

    def refresh_image(self, img):
        visionLib = pv_visionlib.pvVisionLib()
        #binary
        qt_bin_image = visionLib.convert_qt_image(img)
        scaled_bin_pixmap = qt_bin_image.scaled(self.image_filtered.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.image_filtered.setPixmap(scaled_bin_pixmap)


       