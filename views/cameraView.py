# import os
# import sys
# from pathlib import Path

# from PySide6.QtMultimedia import (QAudioInput, QCamera, QCameraDevice,
#                                   QImageCapture, QMediaCaptureSession,
#                                   QMediaDevices, QMediaMetaData,
#                                   QMediaRecorder)
# from PySide6.QtWidgets import QDialog, QMainWindow, QMessageBox, QWidget
# from PySide6.QtGui import QAction, QActionGroup, QIcon, QImage, QPixmap
# from PySide6.QtCore import QDateTime, QDir, QTimer, Qt, Slot, qWarning

# class cameraView(QWidget):
#     pass


import glob
import json
import os
from pathlib import Path
import sys
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QColor
from PySide6.QtGui import QIcon, QPixmap, QImage
from PySide6.QtMultimedia import QMediaDevices
import cv2
import numpy as np

import pv_visionlib
import pyqtgraph as pg


import icons_rc, images_rc, config_ini
color_bg="#b1b5b99f"

class CameraView(QtWidgets.QWidget):
    image_path = None
    actual_image = None
    normal_pen = pg.mkPen('g', width=2)  # Verde para normal
    selected_pen = pg.mkPen('m', width=2) # Amarelo para selecionado
    rois = []
    def __init__(self, index, width, height):
        super().__init__()

        self.images_folder = config_ini.cam_files_path
        self.image_index = -1

        self.setMinimumWidth(width)
        self.setMaximumWidth(width)
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)
        
        

        self.camera_tool = QtWidgets.QHBoxLayout(self)
        self.camera_tool.setAlignment(Qt.AlignmentFlag.AlignLeading)
        #self.image_view = QtWidgets.QStackedLayout(self, stackingMode=QtWidgets.QStackedLayout.StackingMode.StackAll) #QStackedLayout(self)
        self.lists = ["1", "2", "3"]
        self.availableCameras = []

        self.camera_usb_index = index
        

        cam_actions_widget = QtWidgets.QWidget(self)
        cam_actions_widget.setMaximumWidth(80)
        cam_actions = QtWidgets.QVBoxLayout(cam_actions_widget)
        cam_actions.setAlignment(Qt.AlignmentFlag.AlignTop)
        # 3. Create the overlay button
        self.live_button = QtWidgets.QPushButton("Live")
        self.live_button.setMaximumHeight(35)
        self.live_button.setMaximumWidth(60)
        

        self.picture_button = QtWidgets.QPushButton()
        self.picture_button.setMaximumHeight(35)
        self.picture_button.setMaximumWidth(60)

        #picture_icon = QIcon(":icons/icons/camera.svg") 
        self.picture_button.setIcon(self.tint_icon(":icons/icons/camera.svg", "green"))

        self.file_button = QtWidgets.QPushButton()
        self.file_button.setMaximumHeight(35)
        self.file_button.setMaximumWidth(60)
        file_icon = QIcon(":icons/icons/folder.svg") 
        self.file_button.setIcon(file_icon)

        self.previous_button = QtWidgets.QPushButton()
        self.previous_button.setMaximumHeight(35)
        self.previous_button.setMaximumWidth(60)
        previous_icon = QIcon(":icons/icons/arrow-left.svg") 
        self.previous_button.setIcon(previous_icon)

        self.next_button = QtWidgets.QPushButton()
        self.next_button.setMaximumHeight(35)
        self.next_button.setMaximumWidth(60)
        next_icon = QIcon(":icons/icons/arrow-right.svg") 
        self.next_button.setIcon(next_icon)

        # Style the button to be partially transparent and smaller
        self.live_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 50); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 5px;
                padding: 7px;
            }
        """)
        
        # Style the button to be partially transparent and smaller
        self.picture_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 50); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 10px;                          
                padding: 7px;
            }
        """)
        # Style the button to be partially transparent and smaller
        self.file_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 50); /* Semi-transparent black */
                color: yellow;
                border: 2px solid white;
                border-radius: 15px;
                padding: 7px;
            }
        """)

        # Style the button to be partially transparent and smaller
        self.previous_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 50); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 10px;                          
                padding: 7px;
            }
        """)

        # Style the button to be partially transparent and smaller
        self.next_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 50); /* Semi-transparent black */
                color: white;                       
                border: 2px solid white;
                border-radius: 10px;                          
                padding: 7px;
            }
        """)

        pg.setConfigOptions(imageAxisOrder='row-major')
    
        label_widget = QtWidgets.QWidget()
        label_widget.setMinimumWidth(int(width/2)+280)
        
        self.label = pg.ImageView(label_widget, roi=None, normRoi=None, )
        self.label.setImage(  self.load_image_path("images/No_Image_Available.jpg") )#setPixmap(QPixmap(":images/images/No_Image_Available.jpg"))    
        self.label.getHistogramWidget().hide()
        self.label.ui.roiBtn.hide()
        self.label.ui.menuBtn.hide()
        self.label.autoRange()
        self.label.getView().scene().sigMouseClicked.connect(self.on_plot_clicked)
        #label_widget.addWidget(self.label)
        
        cam_actions.addWidget(self.live_button)
        cam_actions.addWidget(self.picture_button)
        cam_actions.addWidget(self.file_button)  
        cam_actions.addWidget(self.previous_button)
        cam_actions.addWidget(self.next_button)
        self.camera_tool.addWidget(cam_actions_widget)
        self.camera_tool.addWidget(label_widget) #self.label)
        
        
        
         

        self.getAvailableCameras()
        self.width = width
        self.height = height
        
        self.live_button.clicked.connect(self.start_live)
        self.file_button.clicked.connect(self.open_image_dialog)
        self.previous_button.clicked.connect(self.previous_image)
        self.next_button.clicked.connect(self.next_image)
        self.isLive = False

    def previous_image(self):
        self.image_index = self.image_index-1

        self.images_list = self.refresh_images_list()

        if self.image_index < 0:
            self.image_index = len(self.images_list)-1 
        
        self.label.setImage( self.load_image_path(self.images_list[self.image_index]) )
        self.image_path = self.images_list[self.image_index]
        self.search_annotation_file(self.images_list[self.image_index])
        

    def next_image(self):
        self.image_index = self.image_index+1

        self.images_list = self.refresh_images_list()

        if self.image_index > len(self.images_list)-1:
            self.image_index = 0
        
        self.label.setImage( self.load_image_path(self.images_list[self.image_index]) )
        self.image_path = self.images_list[self.image_index]
        self.search_annotation_file(self.images_list[self.image_index])


    def refresh_images_list(self):
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
        image_files = []
        for filename in os.listdir(self.images_folder):
            if filename.lower().endswith(image_extensions):
                image_files.append(os.path.join(self.images_folder, filename))
        return image_files
    


    def search_annotation_file(self, image_file):
        image_annotations = list(Path(self.images_folder).glob(os.path.basename(image_file+".json")))

        if len(image_annotations) > 0:
            self.draw_rois_json( image_annotations[0].as_posix())
            return image_annotations[0].as_posix()
        else:
            self.delete_all_rois()
            return None
    
    def draw_rois_json(self, json_file):
        dict_from_file = None
        with open(json_file, 'r') as f:
            dict_from_file = json.load(f)

        if dict_from_file:
            self.draw_rois_dict( dict_from_file)



    def draw_rois_dict(self, rois):
        
        self.delete_all_rois()
        self.image_chars = ""
        
        for i in rois:
            x = rois[i]['box']['x']
            y = rois[i]['box']['y']
            w = rois[i]['box']['w']
            h = rois[i]['box']['h']
            roi = pg.ROI( 
                pos=[int(x), int(y)], 
                size=[int(w), int(h)], 
                pen=pg.mkPen('r', width=2),
                handlePen=pg.mkPen('w', width=1),
            )
            roi.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
            roi.addScaleHandle([1, 1], [0.5, 0.5])  # Bottom-right corner for resizing
            roi.addRotateHandle([0, 0], [0.5, 0.5])  # Top-left corner for rotation
            roi.addScaleHandle([.5, 1], [0.2,0.2])
            roi.setZValue(10)
            self.rois.append(roi)

            #add chars
            self.image_chars += (rois[i]['char'])


            # Add the ROI to the view
            self.label.getView().addItem(self.rois[len(self.rois)-1])

            # Connect the ROI signal to a slot
            self.rois[len(self.rois)-1].sigRegionChanged.connect(self.on_roi_changed)
            
            roi.sigClicked.connect(self.on_roi_selected)

        

            




    def open_image_dialog(self):
        """Opens a file dialog for the user to select an image file."""
        # The getOpenFileName static method returns a tuple: (file_path, filter_string)
        # We only need the file_path, so we discard the filter_string with a placeholder '_'.
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            "", # Default directory (empty string opens user's home directory)
            "Image Files (*.png *.jpg *.jpeg *.bmp)" # File filter
        )
        
        if file_path:
            self.label.setImage( self.load_image_path(file_path) )
            self.image_path = file_path
            self.images_folder = os.path.dirname(file_path)
            


    def load_qimage(self, file_path):
        """Loads the image from the given path and displays it in the QLabel."""
        pixmap = QPixmap(file_path)
        
        if not pixmap.isNull():
            self.label.setPixmap(pixmap)
            self.label.setScaledContents(True)
            self.label.setText("") # Clear the text message
        else:
            self.label.setPixmap(QPixmap(":images/images/No_Image_Available.jpg"))
            self.label.setScaledContents(True)
            self.label.setText("Failed to load image.")



    def load_image_path(self, file_path):
        """Loads an image from a file using OpenCV and displays it."""
        try:
            

            # Load image using OpenCV
            # OpenCV loads images as BGR, so we convert to RGB
            image = cv2.imread(file_path)
            if image is None:
                raise FileNotFoundError(f"Image not found at {file_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.actual_image = image
            return image
        

        except FileNotFoundError as e:
            print(e)
            # Create dummy image data if file is not found
            data = np.random.normal(size=(500, 500), loc=100, scale=50).astype(np.float32)
            self.label.setImage(data)
        
            return data



    def start_live(self):
        if not self.isLive:
            self.isLive = True
            self.th = Thread(self)
            self.th.index = self.camera_usb_index
            #self.th.finished.connect(self.close)
            self.th.updateFrame.connect(self.setImage)
            self.runWebCam(0, self.width, self.height)
            # Style the button to be partially transparent and smaller
            self.live_button.setStyleSheet("""
                QPushButton {
                    background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                    color: red;
                    border: 2px solid white;
                    border-radius: 10px;
                    padding: 5px;
                }
            """)
        else:
            self.th.status = False
            #self.th.terminate()
            self.th = None
            self.live_button.setStyleSheet("""
                QPushButton {
                    background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
                    color: white;
                    border: 2px solid white;
                    border-radius: 10px;
                    padding: 5px;
                }
            """)
            self.isLive = False
            

    @Slot(QImage)
    def runWebCam(self, idx, width, height):
        #print(f"Selected the variable {idx} from combo {combo.id_number}")
        self.th.height = height-2
        self.th.width = width-2
        self.th.start()

    @Slot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))
    
    def updateCV_Image(self, image):
        lib = pv_visionlib.pvVisionLib()
        qt_image = lib.convert_qt_image(image)
        self.label.setPixmap(qt_image)
    
    def getAvailableCameras(self):
        cameras = QMediaDevices.videoInputs()
        for cameraDevice in cameras:
            self.availableCameras.append(cameraDevice.description())

    def tint_icon(self, icon_path, tint_color):
        """
        Loads an icon from a resource and tints it with the specified color.
        """
        pixmap = QPixmap(icon_path)
        mask = pixmap.createMaskFromColor(Qt.transparent, Qt.MaskOutColor)
        pixmap.fill(QColor(tint_color))
        pixmap.setMask(mask)
        return QIcon(pixmap)
    
    
    #######ROI########
    def draw_rois(self, img, characters):
        
        self.delete_all_rois()
        self.actual_image = img
        # Create a rectangular ROI item
        # The first argument is the initial position [x, y], the second is the initial size [w, h].
        # The handlePen is for the resizing handles, and the pen is for the box border.
        for i in characters:
            (x, y, w, h) = enumerate(i)
            roi = pg.ROI( 
                pos=[int(x[1]), int(y[1])], 
                size=[int(w[1]), int(h[1])], 
                pen=pg.mkPen('r', width=2),
                handlePen=pg.mkPen('w', width=1),
            )
            roi.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
            roi.addScaleHandle([1, 1], [0.5, 0.5])  # Bottom-right corner for resizing
            roi.addRotateHandle([0, 0], [0.5, 0.5])  # Top-left corner for rotation
            roi.addScaleHandle([.5, 1], [0.2,0.2])
            roi.setZValue(10)
            self.rois.append(roi)

            # Add the ROI to the view
            self.label.getView().addItem(self.rois[len(self.rois)-1])

            # Connect the ROI signal to a slot
            self.rois[len(self.rois)-1].sigRegionChanged.connect(self.on_roi_changed)
            #self.rois[len(self.rois)-1].sigClicked.connect(lambda r=self.rois[len(self.rois)-1]: self.on_roi_selected(r))
            roi.sigClicked.connect(self.on_roi_selected)

            #self.on_roi_selected(self.rois[len(self.rois)-1], None)

    def refresh_rois(self, chars):
        if not self.actual_image.any():
            return
        
        if len(self.rois) <= 0:
            return
        
        
        samples = {}
        i = 0
        for _roi in self.rois:
            x = int(_roi.state['pos'].x())
            y = int(_roi.state['pos'].y())
            a = int(_roi.state['angle'])
            w = int(_roi.state['size'].x())
            h = int(_roi.state['size'].y())
            char_img = self.actual_image[y:y+h, x:x+w] 

            if len(chars) <= 0:
                char = "?"
            else:
                if i >= len(chars):
                    char = "?"
                else:
                    char = chars[i]
            
            samples.update({str(i):{ "char": char, "box":{"x":x,"y":y,"w":w,"h":h}, "image":char_img} })
            i += 1
        return samples
    
    def refresh_rois_dict(self, chars):
        
        if len(self.rois) <= 0:
            return
        
        
        samples = {}
        i = 0
        for _roi in self.rois:
            x = int(_roi.state['pos'].x())
            y = int(_roi.state['pos'].y())
            a = int(_roi.state['angle'])
            w = int(_roi.state['size'].x())
            h = int(_roi.state['size'].y())

            if len(chars) <= 0:
                char = "?"
            else:
                if i >= len(chars):
                    char = "?"
                else:
                    char = chars[i]
            
            samples.update({str(i):{ "char": char, "box":{"x":x,"y":y,"w":w,"h":h}} })
            i += 1
        return samples
    
    
    
    def create_new_roi(self):
        x = config_ini.default_roi_x
        y = config_ini.default_roi_y
        w = config_ini.default_roi_w
        h = config_ini.default_roi_h
        roi = pg.ROI( 
            pos=[int(x), int(y)], 
            size=[int(w), int(h)], 
            pen=pg.mkPen('r', width=2),
            handlePen=pg.mkPen('w', width=1),
        )
        roi.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        roi.addScaleHandle([1, 1], [0.5, 0.5])  # Bottom-right corner for resizing
        roi.addRotateHandle([0, 0], [0.5, 0.5])  # Top-left corner for rotation
        roi.addScaleHandle([.5, 1], [0.2,0.2])
        roi.setZValue(10)
        self.rois.append(roi)

        # Add the ROI to the view
        self.label.getView().addItem(self.rois[len(self.rois)-1])

        # Connect the ROI signal to a slot
        self.rois[len(self.rois)-1].sigRegionChanged.connect(self.on_roi_changed)
        #self.rois[len(self.rois)-1].sigClicked.connect(lambda r=self.rois[len(self.rois)-1]: self.on_roi_selected(r))
        roi.sigClicked.connect(self.on_roi_selected)


    def on_roi_selected(self, roi_clicked, ev):
        """ Este método é chamado quando um ROI é clicado. """
        print(f"ROI selecionado: {roi_clicked}")
        
        # 4. Atualiza a referência do ROI selecionado
        self.selected_roi = roi_clicked
        
        # 5. Atualiza a aparência de todos os ROIs
        for roi in self.rois:
            if roi == self.selected_roi:
                roi.setPen(self.selected_pen) # Define o selecionado como amarelo
            else:
                roi.setPen(self.normal_pen)   # Define os outros como verdes
        if ev:
            ev.accept()

    def on_plot_clicked(self, ev):
        if ev.accepted:
            return
        """ Este método é chamado quando o fundo do gráfico é clicado. """
        print("Fundo do gráfico clicado. Deselecionando todos.")
        
        # 6. Limpa a seleção e restaura a cor de todos
        self.selected_roi = None
        for roi in self.rois:
            roi.setPen(self.normal_pen)


    def delete_selected_roi(self):
        """ Este método deleta o ROI armazenado em 'self.selected_roi'. """
        
        # 7. Verifica se há um ROI selecionado
        if self.selected_roi:
            print(f"Removendo ROI selecionado: {self.selected_roi}")
            
            # Remove do gráfico
            self.label.getView().removeItem(self.selected_roi)
            
            # Remove da nossa lista de rastreamento
            self.rois.remove(self.selected_roi)
            
            # Limpa a variável de seleção
            self.selected_roi = None
            
            print(f"ROIs restantes: {len(self.rois)}")
        else:
            print("Nenhum ROI selecionado para remover.")

    

    def on_roi_changed(self, *args):
        """Slot to handle changes in the ROI's position or size."""
        pos = self.rois[0].pos()
        size = self.rois[0].size()
        # self.status_label.setText(
        #     f"ROI Info: Position=({pos.x():.2f}, {pos.y():.2f}), Size=({size.x():.2f}, {size.y():.2f})"
        # )


    def delete_all_rois(self):
        if not self.rois:
            print("Não há ROIs para remover.")
            return

        print(f"Removindo todos os {len(self.rois)} ROIs...")
        
        for roi in self.rois:
            self.label.getView().removeItem(roi)
            
        self.rois.clear()
        
        # 8. Limpa a seleção, pois todos foram deletados
        self.selected_roi = None 
        
        print("Todos os ROIs foram removidos.")


class Thread(QThread):
    updateFrame = Signal(QImage)
    width = 0
    height = 0
    index = -1
    

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.status = True
        self.cap = True
        

    def run(self):
        self.cap = cv2.VideoCapture(self.index)
        while self.status:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Convert the frame from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            h, w, ch = rgb_frame.shape
            img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
            scaled_img = img.scaled(self.width, self.height, Qt.KeepAspectRatio)#Qt.KeepAspectRatio
            # Emit signal
            self.updateFrame.emit(scaled_img)
        self.cap.release()
        self.cap = None
        #sys.exit(-1)



if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = CameraView()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec_())