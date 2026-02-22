
import sys
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QThread, Signal, Slot

from views.cameraView import CameraView
from components.cardpanel import CardPanel, CustomCardContent
from components.cardmodel import ModelView, CardModel, TrainingCardContent
from components.modeljson import ModelJson

import config_ini

class ProgramView(QtWidgets.QWidget):
    modelDataChanged = Signal()
    cameras = []
    camera_width = 0
    camera_height = 0
    model_json = ModelJson()
    segmentation_view = None
    
    def __init__(self, width, height):
        super().__init__()
        
        # Ensure ProgramView itself has a layout so its children resize dynamically
        self.root_layout = QtWidgets.QVBoxLayout(self)
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        
        self.layout = QtWidgets.QWidget(self)
        self.root_layout.addWidget(self.layout)
        self.area_work = QtWidgets.QGridLayout(self.layout)
        #self.area_work.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        self.area_work.setSpacing(1)
        self.area_work.setAlignment(Qt.AlignmentFlag.AlignLeading)
        
        self.layout.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.resize_views(width, height)
        
        

       
        self.toolbox = QtWidgets.QToolBox()
        
        
        # --- Create and populate a CardPanel Image segmentation ---
        card = CardPanel(self)
        # Let PySide control size based on expanding rules Instead of fixed math
        card_content = CustomCardContent(self)
        card_content.title_label.setText("Segmentação de caracteres")
        card.set_content(card_content)
        self.segmentation_view = card_content.image_segmentation_box
        
        self.toolbox.addItem(card, "Segmentação de caracteres")
        
        # --- Create and populate a CardPanel Model Training ---
        card_model = CardModel(self)

        card_model_content =  TrainingCardContent(self)
        # Connect the jsonLoaded signal from the view to our slot for handling it
        card_model_content.model_train_box.jsonLoaded.connect(self.on_json_loaded)
        card_model_content.title_label.setText("Treinamento de modelos IA")
        card_model.set_content(card_model_content)
        
        self.toolbox.addItem(card_model, "Treinamento de modelos IA")

        #Object to model json
        self.model_json = ModelJson()

        self.area_work.addWidget(self.toolbox, 1, 0, 1, 2)
        
        # Force the camera row and the card row to share 50/50 vertical space
        self.area_work.setRowStretch(0, 1)
        self.area_work.setRowStretch(1, 1)
        
    def resize_views(self, width, height):
        # We handle dynamic resizing in layout now, so no math here.
        pass
        
    @Slot(str)
    def on_json_loaded(self, filename):
        """Handles loading the JSON file and updating UI components."""
        if self.model_json.load_from_file(filename):
            # Directly update the UI elements in ModelView since we have access to it.
            self.findChild(ModelView).update_all_views()
            self.modelDataChanged.emit()
        

    def create_views(self, qty):
        #layout_cams = QtWidgets.QVBoxLayout()
        for i in range(qty):
            # Let layout handle geometry
            self.cameras.append(CameraView( config_ini.cam_usb_index[i] ))
            self.cameras[-1].setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            self.area_work.addWidget(self.cameras[-1], 0, i)

       




    
