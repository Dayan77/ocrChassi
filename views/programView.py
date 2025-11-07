
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
    
    def __init__(self, width, height):
        super().__init__()
        self.layout = QtWidgets.QWidget(self)
        self.area_work = QtWidgets.QGridLayout(self.layout)
        #self.area_work.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        self.area_work.setSpacing(1)
        self.area_work.setAlignment(Qt.AlignmentFlag.AlignLeading)
        
        self.layout.setMinimumHeight(height+30)
        self.layout.setMaximumHeight(height+30)
        self.layout.setMinimumWidth(width)
        self.layout.setMaximumWidth(width)

        self.resize_views(width, height)
        
        

       
        self.layout_data = QtWidgets.QWidget()
        self.data_col = QtWidgets.QHBoxLayout(self.layout_data)
        
        
        
        # --- Create and populate a CardPanel Image segmentation ---
        card = CardPanel(self)
        card.setMaximumHeight((height/2)-10)
        card.setMinimumHeight((height/2)-10)
        card_content = CustomCardContent(self)
        card_content.title_label.setText("Segmentação de caracteres")
        card.set_content(card_content)
        
        self.data_col.addWidget(card)
        
        # --- Create and populate a CardPanel Model Training ---
        card_model = CardModel(self)
        card_model.setMaximumHeight((height/2)-10)
        card_model.setMinimumHeight((height/2)-10)
        card_model_content =  TrainingCardContent(self)
        # Connect the jsonLoaded signal from the view to our slot for handling it
        card_model_content.model_train_box.jsonLoaded.connect(self.on_json_loaded)
        card_model_content.title_label.setText("Treinamento de modelos IA")
        card_model.set_content(card_model_content)
        
        self.data_col.addWidget(card_model)

        #Object to model json
        self.model_json = ModelJson()

        
        self.area_work.addWidget(self.layout_data, 1, 0, 1, 2)
       
        
    def resize_views(self, width, height):
        self.camera_width = int((width/2)-60)
        self.camera_height = int(height/2)
        
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
            self.cameras.append(CameraView( config_ini.cam_usb_index[i] , self.camera_width, self.camera_height))
            self.area_work.addWidget(self.cameras[len(self.cameras) -1], 0, i)

       




    
