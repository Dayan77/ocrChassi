# import os
# import sys
# from pathlib import Path

# from PySide6.QtMultimedia import (QAudioInput, QCamera, QCameraDevice,
#                                   QMediaCaptureSession, QMediaDevices, QMediaMetaData,
#                                   QImageCapture, QMediaCaptureSession,
#                                   QMediaDevices, QMediaMetaData,
#                                   QMediaRecorder)
# from PySide6.QtWidgets import QDialog, QMainWindow, QMessageBox, QWidget
# from PySide6.QtGui import QAction, QActionGroup, QIcon, QImage, QPixmap
# from PySide6.QtCore import QDateTime, QDir, QTimer, Qt, Slot, qWarning

# class cameraView(QWidget):
#     pass


from asyncio import sleep
import glob
import json
import time
import os
import traceback
from pathlib import Path
import sys
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt, QThread, Signal, Slot, QDateTime, QDir
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
    annotation_data = None          # loaded JSON dict for current image
    def __init__(self, index):
        super().__init__()

        self.ready = False
        self.navigation_in_progress = False  # Additional lock for rapid clicks
        self.images_folder = config_ini.cam_files_path
        self.image_index = -1
        self.filter_annotated = False
        # debugging toggle for ROI handling
        self.debug_roi = False

        # when an image is loaded we keep its annotation data here;
        # drawing of the ROIs is done later when the user presses
        # "Atualizar" (refresh) rather than automatically during
        # navigation.  This avoids crashes tied to changing images.
        self.annotation_data = None

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

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

        self.save_button = QtWidgets.QPushButton()
        self.save_button.setMaximumHeight(35)
        self.save_button.setMaximumWidth(60)
        save_icon = QIcon(":icons/icons/save.svg")
        self.save_button.setIcon(save_icon)

        self.filter_btn = QtWidgets.QPushButton()
        self.filter_btn.setMaximumHeight(35)
        self.filter_btn.setMaximumWidth(60)
        filter_icon = QIcon(":icons/icons/filter.svg")
        self.filter_btn.setIcon(filter_icon)
        self.filter_btn.setToolTip("Ocultar imagens com anotações (JSON)")

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

        # Apply overlay style class
        self.live_button.setProperty("class", "overlay_btn")
        self.live_button.setCheckable(True) # Make sure it's checkable for the red state
        
        self.picture_button.setProperty("class", "overlay_btn")
        self.save_button.setProperty("class", "overlay_btn")
        self.filter_btn.setProperty("class", "overlay_btn")
        self.file_button.setProperty("class", "overlay_btn")
        self.previous_button.setProperty("class", "overlay_btn")
        self.next_button.setProperty("class", "overlay_btn")

        pg.setConfigOptions(imageAxisOrder='row-major')
    
        label_widget = QtWidgets.QWidget()
        # Create a layout for the image view widget
        image_layout = QtWidgets.QVBoxLayout(label_widget)
        image_layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = pg.ImageView(roi=None, normRoi=None)
        self.label.setImage(  self.load_image_path("images/No_Image_Available.jpg") )#setPixmap(QPixmap(":images/images/No_Image_Available.jpg"))    
        self.label.getHistogramWidget().hide()
        self.label.ui.roiBtn.hide()
        self.label.ui.menuBtn.hide()
        self.label.autoRange()
        self.label.getView().scene().sigMouseClicked.connect(self.on_plot_clicked)
        image_layout.addWidget(self.label, 1)  # Stretch factor 1 for image
        
        # Add character display below the image (horizontal, compact)
        self.char_list = QtWidgets.QLineEdit()
        self.char_list.setReadOnly(True)
        self.char_list.setMaximumHeight(32)
        self.char_list.setStyleSheet("""
            QLineEdit {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 4px;
                font-weight: bold;
                font-size: 14px;
                letter-spacing: 2px;
            }
        """)
        self.char_list.setPlaceholderText("Characters will appear here...")
        image_layout.addWidget(self.char_list)  # Stretch factor 0 for display
        
        cam_actions.addWidget(self.live_button)
        cam_actions.addWidget(self.picture_button)
        cam_actions.addWidget(self.save_button)
        cam_actions.addWidget(self.filter_btn)
        cam_actions.addWidget(self.file_button)  
        cam_actions.addWidget(self.previous_button)
        cam_actions.addWidget(self.next_button)
        self.camera_tool.addWidget(cam_actions_widget, 0) # Stretch factor of 0
        self.camera_tool.addWidget(label_widget, 1) # Stretch factor of 1
        
        
        
         

        self.getAvailableCameras()
        
        self.live_button.clicked.connect(self.start_live)
        self.file_button.clicked.connect(self.open_image_dialog)
        self.previous_button.clicked.connect(self.previous_image)
        self.next_button.clicked.connect(self.next_image)
        self.picture_button.clicked.connect(self.take_picture)
        self.save_button.clicked.connect(self.save_picture)
        self.filter_btn.clicked.connect(self.toggle_filter)
        self.isLive = False
        self.th = None

    def _apply_camera_settings(self, cap, index):
        """Applies camera settings from config_ini to a VideoCapture object."""
        # Exposure
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, config_ini.cam_auto_exposure[index])
        if not config_ini.cam_auto_exposure[index]:
            cap.set(cv2.CAP_PROP_EXPOSURE, config_ini.cam_exposure[index])
        # White Balance
        cap.set(cv2.CAP_PROP_AUTO_WB, config_ini.cam_auto_wb[index])
        if not config_ini.cam_auto_wb[index]:
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, config_ini.cam_wb_temperature[index])
        # Focus
        cap.set(cv2.CAP_PROP_AUTOFOCUS, config_ini.cam_auto_focus[index])
        if not config_ini.cam_auto_focus[index]:
            cap.set(cv2.CAP_PROP_FOCUS, config_ini.cam_focus[index])

    def take_picture(self):
        """Captures the current frame and saves it to a file."""
        if self.isLive:
            # If live, just stop the feed. The last frame is already in actual_image.
            self.start_live() # Toggles it off
        else:
            # If not live, open camera, grab a single frame, and close
            self.capture_single_frame()

    def capture_single_frame(self):
        """Opens the camera, captures a single frame, and displays it."""
        # If not live, open camera, grab a single frame, and close
        cap = cv2.VideoCapture(self.camera_usb_index)
        if not cap.isOpened():
            QtWidgets.QMessageBox.warning(self, "Camera Error", "Could not open camera.")
            return
        
        self._apply_camera_settings(cap, self.camera_usb_index)
        
        # Allow the camera to warm up and adjust exposure
        for _ in range(10):
            cap.read()

        # Re-apply settings after warm-up to ensure they stick
        self._apply_camera_settings(cap, self.camera_usb_index)

        ret, frame = cap.read()
        cap.release()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            captured_frame = cv2.flip(rgb_frame, 1)
            self.setImage(captured_frame) # Update the view

    def save_picture(self):
        """Saves the currently displayed image to a file."""
        if self.actual_image is None:
            QtWidgets.QMessageBox.warning(self, "No Image", "There is no image to save.")
            return

        # Open a file dialog to save the image
        timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
        default_filename = os.path.join(config_ini.cam_files_path or QDir.homePath(), f"capture_{timestamp}.png")
        
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Image", default_filename, "PNG Image (*.png)")
        if file_path:
            # Convert the currently displayed image (which is RGB) to BGR for OpenCV
            bgr_image = cv2.cvtColor(self.actual_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, bgr_image)
            print(f"Image saved to: {file_path}")
            QtWidgets.QMessageBox.information(self, "Image Saved", f"Image successfully saved to:\n{file_path}")

    def previous_image(self):
        # Double-check the ready flag
        if not self.ready or self.navigation_in_progress:
            print("Navigation already in progress, ignoring click")
            return
        
        self.ready = False
        self.navigation_in_progress = True  # Lock to prevent rapid clicks
        self.disable_btns() # disable buttons to avoid multiple clicks
        
        try:
            # FIRST: Clean up old ROIs before loading new image
            if len(self.rois) > 0:
                self.delete_all_rois(self.rois)
            
            self.image_index = self.image_index - 1
            self.images_list = self.refresh_images_list()

            if not self.images_list:
                QtWidgets.QMessageBox.warning(self, "No Images", "No images found in the folder.")
                self.image_index = -1
                return

            if self.image_index < 0 or self.image_index > len(self.images_list) - 1:
                self.image_index = len(self.images_list) - 1
            
            print(f"Previous image index: {self.image_index} / {len(self.images_list)}")
            
            # SECOND: Load and display the image
            try:
                image_data = self.load_image_path(self.images_list[self.image_index])
                self.label.setImage(image_data)
            except Exception as load_error:
                print(f"Error loading image: {load_error}")
                QtWidgets.QMessageBox.warning(self, "Image Load Error", f"Failed to load image:\n{load_error}")
                self.image_path = None
                return
            
            self.image_path = self.images_list[self.image_index]
            print(f"Getting annotation for image index: {self.image_index} / {len(self.images_list)}")
            
            # THIRD: Load ROIs for the new image
            try:
                self.load_annotation_file(self.images_list[self.image_index])
                self.display_annotation()
            except Exception as anno_error:
                print(f"Error searching annotation: {anno_error}")
                # Don't fail if annotation search has issues
                
        except Exception as e:
            print(f"Error loading previous image: {e}")
            QtWidgets.QMessageBox.critical(self, "Navigation Error", f"Error navigating to previous image:\n{str(e)}")
        finally:
            # Safety: immediately enable UI to avoid blocking the app if the
            # singleShot callback doesn't run for any reason. Keep the delayed
            # _enable_navigation as a backup to preserve debounce behavior.
            try:
                self.enable_btns()
                self.ready = True
                self.navigation_in_progress = False
            except Exception as e:
                print(f"Error in finalizing previous_image navigation: {e}")

            QtCore.QTimer.singleShot(500, self._enable_navigation)

    
    def disable_btns(self):
        self.live_button.setEnabled(False)
        self.picture_button.setEnabled(False)
        self.file_button.setEnabled(False)
        self.previous_button.setEnabled(False)
        self.next_button.setEnabled(False)

    def enable_btns(self):
        self.live_button.setEnabled(True)
        self.picture_button.setEnabled(True)
        self.file_button.setEnabled(True)
        self.previous_button.setEnabled(True)
        self.next_button.setEnabled(True)

    def _enable_navigation(self):
        """Re-enable navigation after a delay to prevent rapid clicks."""
        # Removed isValid check; widget may still exist or not, just attempt safely
        try:
            self.enable_btns()
            self.ready = True
            self.navigation_in_progress = False
            print("Navigation re-enabled")
        except Exception as e:
            print(f"Error re-enabling navigation: {e}")
            # Force set flags regardless
            try:
                self.ready = True
                self.navigation_in_progress = False
            except:
                pass


    def next_image(self):
        # Double-check the ready flag
        if not self.ready or self.navigation_in_progress:
            print("Navigation already in progress, ignoring click")
            return
        
        self.ready = False
        self.navigation_in_progress = True  # Lock to prevent rapid clicks
        self.disable_btns() # disable buttons to avoid multiple clicks
        
        try:
            # FIRST: Clean up old ROIs before loading new image
            if len(self.rois) > 0:
                self.delete_all_rois(self.rois)
            
            self.image_index = self.image_index + 1
            self.images_list = self.refresh_images_list()

            if not self.images_list:
                QtWidgets.QMessageBox.warning(self, "No Images", "No images found in the folder.")
                self.image_index = -1
                return

            if self.image_index > len(self.images_list) - 1 or self.image_index < 0:
                self.image_index = 0
            
            print(f"Next image index: {self.image_index} / {len(self.images_list)}")
            
            # SECOND: Load and display the image
            try:
                image_data = self.load_image_path(self.images_list[self.image_index])
                self.label.setImage(image_data)
            except Exception as load_error:
                print(f"Error loading image: {load_error}")
                QtWidgets.QMessageBox.warning(self, "Image Load Error", f"Failed to load image:\n{load_error}")
                self.image_path = None
                return
            
            self.image_path = self.images_list[self.image_index]
            print(f"Getting annotation for image index: {self.image_index} / {len(self.images_list)}")
            
            # THIRD: Load ROIs for the new image
            try:
                self.load_annotation_file(self.images_list[self.image_index])
                self.display_annotation()
            except Exception as anno_error:
                print(f"Error searching annotation: {anno_error}")
                # Don't fail if annotation search has issues
                
        except Exception as e:
            print(f"Error loading next image: {e}")
            QtWidgets.QMessageBox.critical(self, "Navigation Error", f"Error navigating to next image:\n{str(e)}")
        finally:
            # Safety: immediately enable UI to avoid blocking the app if the
            # singleShot callback doesn't run for any reason. Keep the delayed
            # _enable_navigation as a backup to preserve debounce behavior.
            try:
                self.enable_btns()
                self.ready = True
                self.navigation_in_progress = False
            except Exception as e:
                print(f"Error in finalizing next_image navigation: {e}")

            QtCore.QTimer.singleShot(500, self._enable_navigation)


    def refresh_images_list(self):
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
        image_files = []
        if not self.images_folder or not os.path.exists(self.images_folder):
            return []

        for filename in os.listdir(self.images_folder):
            if filename.lower().endswith(image_extensions) and filename.startswith("._") == False:
                full_path = os.path.join(self.images_folder, filename)
                
                if self.filter_annotated:
                    json_path = full_path + ".json"
                    if os.path.exists(json_path):
                        continue

                image_files.append(full_path)
        
        image_files.sort()
        return image_files
    
    def toggle_filter(self):
        # Clean up old ROIs before loading new image
        if len(self.rois) > 0:
            self.delete_all_rois(self.rois)
        
        self.filter_annotated = not self.filter_annotated
        
        if self.filter_annotated:
            QtWidgets.QMessageBox.information(self, "Filtro Ativado", "Exibindo apenas imagens sem anotações (JSON).")
        else:
            QtWidgets.QMessageBox.information(self, "Filtro Desativado", "Exibindo todas as imagens.")

        self.images_list = self.refresh_images_list()
        
        # Reset to first image if list not empty
        if self.images_list:
            self.image_index = 0
            try:
                image_data = self.load_image_path(self.images_list[0])
                self.label.setImage(image_data)
                self.image_path = self.images_list[0]
                self.load_annotation_file(self.images_list[0])
                # display characters immediately
                self.display_annotation()
            except Exception as e:
                print(f"Error loading filtered image: {e}")
                QtWidgets.QMessageBox.warning(self, "Image Load Error", f"Failed to load first image:\n{str(e)}")
                self.image_index = -1
                self.label.clear()
                self.image_path = None
        else:
            self.image_index = -1
            self.label.clear()
            self.image_path = None
            if len(self.rois) > 0:
                self.delete_all_rois(self.rois)


    def load_annotation_file(self, image_file):
        """Load the JSON annotation for *image_file* and store it in
        ``self.annotation_data``.  This method does **not** draw the ROIs.

        Returns True if an annotation was found, False otherwise.
        """
        try:
            image_annotations = list(Path(self.images_folder).glob(os.path.basename(image_file + ".json")))
            print(f"Searching annotation for image: {image_annotations}")
            if image_annotations:
                with open(image_annotations[0].as_posix(), 'r') as f:
                    self.annotation_data = json.load(f)
                return True
            else:
                # no annotation -> clear stored data and any drawn rois
                self.annotation_data = None
                if len(self.rois) > 0:
                    self.delete_all_rois(self.rois)
                return False
        except Exception as e:
            print(f"Error loading annotation file: {e}")
            self.annotation_data = None
            if len(self.rois) > 0:
                self.delete_all_rois(self.rois)
            return False

    
    def draw_rois_json(self, json_file):
        # kept for compatibility but rarely used directly now
        try:
            with open(json_file, 'r') as f:
                dict_from_file = json.load(f)
            if dict_from_file:
                self.draw_rois_dict(dict_from_file)
        except Exception as e:
            print(f"Error loading annotation file: {e}")




    def draw_rois_dict(self, rois):
        # NOTE: ROIs should already be deleted before this is called (from next_image/previous_image)
        # This check is a safety net but shouldn't be needed anymore
        if len(self.rois) > 0:
            print("[ROI] WARNING: Old ROIs still exist before drawing new ones!")
            self.delete_all_rois(self.rois)
       
        if not rois:
            self.char_list.setText("")  # Clear display if no ROIs
            return
        else:
            _rois = rois.copy()
        
        if not self.rois:   
            self.rois = []

        self.image_chars = ""
        print(f"[ROI] Starting to draw {len(rois)} ROIs...")
        
        try:
            # Step 1: Allow Qt to process any pending cleanup
            print(f"[ROI] Step 1: Processing pending Qt events...")
            QtCore.QCoreApplication.processEvents()
            
            # Step 2: Sort ROIs by x-coordinate (left-to-right position) for correct reading order
            print(f"[ROI] Step 2: Sorting {len(rois)} ROIs by x-coordinate...")
            sorted_roi_items = sorted(_rois.items(), key=lambda item: item[1]['box']['x'])
            print(f"[ROI] Sorted ROI order: {[item[1]['char'] for item in sorted_roi_items]}")
            
            # Step 3: Get the view once
            print(f"[ROI] Step 3: Getting view for ROI display...")
            view = self.label.getView()
            if view is None:
                print("[ROI] ERROR: Could not get view!")
                raise Exception("Could not retrieve ImageView")

            # NOTE: previous implementation cleared the entire view here, which
            # removed the background image and resulted in a blank display.  The
            # delete_all_rois() calls earlier already handle ROI removal, so we
            # no longer clear the whole view.  The image should remain visible.
            if self.debug_roi:
                if self.actual_image is None:
                    print("[ROI] WARNING: actual_image is None; image may not show.")
                else:
                    print(f"[ROI] actual_image shape = {getattr(self.actual_image,'shape',None)}")

            # Step 4: Create ROI objects first, then add them to the view in a second pass
            print(f"[ROI] Step 4: Creating {len(sorted_roi_items)} ROI objects (two-phase)...")
            temp_rois = []  # hold (roi, char)
            for roi_idx, (key, roi_data) in enumerate(sorted_roi_items):
                x = roi_data['box']['x']
                y = roi_data['box']['y']
                w = roi_data['box']['w']
                h = roi_data['box']['h']
                char = roi_data['char']
                print(f"[ROI]   preparing ROI {roi_idx}: '{char}' at ({int(x)},{int(y)}) size {int(w)}x{int(h)}")
                try:
                    img_h, img_w = self.actual_image.shape[:2]
                    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                        print(f"[ROI] WARNING: ROI {roi_idx} has invalid dims/pos - skipping")
                        continue
                    new_roi = pg.ROI(
                        pos=[int(x), int(y)],
                        size=[int(w), int(h)],
                        pen=pg.mkPen('r', width=2),
                        handlePen=pg.mkPen('w', width=1),
                    )
                    new_roi.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
                    new_roi.addScaleHandle([1, 1], [0.5, 0.5])
                    new_roi.addRotateHandle([0, 0], [0.5, 0.5])
                    new_roi.addScaleHandle([.5, 1], [0.2,0.2])
                    new_roi.setZValue(10)
                    temp_rois.append((new_roi, char))
                    self.image_chars += char
                except Exception as e:
                    print(f"[ROI] ERROR preparing ROI {roi_idx}: {e}")
                    print(traceback.format_exc())
                    continue
            print(f"[ROI] Prepared {len(temp_rois)} ROIs, now adding to view")
            for roi_idx, (new_roi, char) in enumerate(temp_rois):
                try:
                    view.addItem(new_roi)
                    after = len(view.addedItems)
                    print(f"[ROI]   view items after add: {after}")
                except Exception as e:
                    print(f"[ROI] ERROR adding ROI {roi_idx} to view: {e}")
                    print(traceback.format_exc())
                try:
                    new_roi.sigRegionChanged.connect(self.on_roi_changed)
                except Exception as e:
                    print(f"[ROI] WARNING: could not connect sigRegionChanged for ROI {roi_idx}: {e}")
                try:
                    new_roi.sigClicked.connect(self.on_roi_selected)
                except Exception as e:
                    print(f"[ROI] WARNING: could not connect sigClicked for ROI {roi_idx}: {e}")
                try:
                    self.rois.append(new_roi)
                except Exception as e:
                    print(f"[ROI] ERROR appending ROI {roi_idx} to internal list: {e}")
                if roi_idx % 3 == 0:
                    QtCore.QCoreApplication.processEvents()
                print(f"[ROI]   added ROI {roi_idx} to view")
            
            # Step 5: Update the character display
            print(f"[ROI] Step 5: Updating character display: '{self.image_chars}'")
            self.char_list.setText(self.image_chars)
            
            print(f"[ROI] ✓ Successfully drew {len(self.rois)} ROIs")

        except Exception as e:
            print(f"[ROI] ✗ CRITICAL ERROR in draw_rois_dict: {e}")
            print(traceback.format_exc())
            # Try to recover
            try:
                self.char_list.setText("[ERROR - Check Console]")
                QtCore.QCoreApplication.processEvents()
            except:
                pass
        

        

            




    def open_image_dialog(self):
        """Opens a file dialog for the user to select an image file."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            config_ini.cam_files_path,
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            # Clean up old ROIs before loading new image
            if len(self.rois) > 0:
                self.delete_all_rois(self.rois)
            
            try:
                image_data = self.load_image_path(file_path)
                self.label.setImage(image_data)
                self.image_path = file_path
                self.images_folder = os.path.dirname(file_path)
                self.image_index = -1
                self.ready = True
                self.navigation_in_progress = False
                # Load annotations for this image
                self.load_annotation_file(file_path)
                # display characters immediately
                self.display_annotation()
            except Exception as e:
                print(f"Error opening image: {e}")
                QtWidgets.QMessageBox.critical(self, "Image Load Error", f"Failed to load image:\n{str(e)}")
                self.image_path = None
                self.ready = False
            


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

    def display_annotation(self):
        """Extract and display characters from the currently loaded annotation.

        This method shows the character string from the annotation JSON without
        drawing any ROI boxes on the image.  It provides a clean view of what
        characters are present without the visual editing aids.
        """
        if not self.annotation_data:
            # no annotation -> clear the display
            self.image_chars = ""
            self.char_list.setText("")
            return

        # extract characters sorted by x-coordinate (left-to-right)
        sorted_items = sorted(
            self.annotation_data.items(),
            key=lambda item: item[1]['box']['x']
        )
        
        chars = ''.join(item[1]['char'] for item in sorted_items)
        self.image_chars = chars
        self.char_list.setText(chars)
        print(f"[ROI] Displayed {len(chars)} characters: '{chars}'")



    def load_image_path(self, file_path):
        """Loads an image from a file using OpenCV and displays it."""
        try:
            # Load image using OpenCV
            # OpenCV loads images as BGR, so we convert to RGB
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError(f"Failed to read image: {file_path}. File may be corrupted or in an unsupported format.")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.actual_image = image
            return image
        
        except Exception as e:
            error_msg = f"Error loading image '{file_path}': {str(e)}"
            print(error_msg)
            raise Exception(error_msg)



    def start_live(self):
        if not self.isLive:
            self.isLive = True
            self.th = Thread(self)
            self.th.index = self.camera_usb_index
            #self.th.finished.connect(self.close)
            self.th.updateFrame.connect(self.setImage)
            self.runWebCam(self.camera_usb_index)
            
            self.live_button.setChecked(True)
        else:
            self.th.status = False
            #self.th.terminate()
            self.th = None
            self.live_button.setChecked(False)
            self.isLive = False
            
    def stop_live(self):
        if self.isLive and self.th:
            self.th.status = False
            self.th.quit()
            self.th.wait()
            self.th = None
            self.isLive = False
            self.live_button.setChecked(False)

    @Slot(QImage)
    def runWebCam(self, idx):
        self.th.start()

    @Slot(QImage)
    def setImage(self, frame):
        if isinstance(frame, QPixmap):
            self.label.setImage(self.load_image_path(self.image_path))
        else:
            self.label.setImage(frame)
            self.actual_image = frame
    
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
        if len(self.rois) > 0:
            self.delete_all_rois(self.rois)

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

            #Add label
            # label = pg.TextItem
            # self.label.addLabel('?', 0, 0, color=(255,0,0), size='12pt', bold=True)

            # Connect the ROI signal to a slot
            self.rois[len(self.rois)-1].sigRegionChanged.connect(self.on_roi_changed)
            #self.rois[len(self.rois)-1].sigClicked.connect(lambda r=self.rois[len(self.rois)-1]: self.on_roi_selected(r))
            roi.sigClicked.connect(self.on_roi_selected)

            #self.on_roi_selected(self.rois[len(self.rois)-1], None)

    def refresh_rois(self, chars):
        """Extract character samples from the annotation data (without ROI objects).
        
        Since we no longer draw ROI boxes, this now uses the annotation_data
        that was loaded by load_annotation_file().
        """
        if not self.annotation_data:
            return {}
        
        if self.actual_image is None or not self.actual_image.any():
            return {}
        
        # Sort annotation items by x-coordinate (left-to-right position)
        sorted_items = sorted(
            self.annotation_data.items(),
            key=lambda item: item[1]['box']['x']
        )
        
        samples = {}
        for idx, (key, roi_data) in enumerate(sorted_items):
            x = int(roi_data['box']['x'])
            y = int(roi_data['box']['y'])
            w = int(roi_data['box']['w'])
            h = int(roi_data['box']['h'])
            char = roi_data['char']
            
            # extract the character image region
            if y + h <= self.actual_image.shape[0] and x + w <= self.actual_image.shape[1]:
                char_img = self.actual_image[y:y+h, x:x+w]
            else:
                char_img = np.zeros((h, w, 3), dtype=np.uint8)  # fallback
            
            samples[str(idx)] = {
                "char": char,
                "box": {"x": x, "y": y, "w": w, "h": h},
                "image": char_img
            }
        
        return samples
    
    def refresh_rois_dict(self, chars):
        
        if len(self.rois) <= 0:
            return
        
        # Sort ROIs by x-coordinate (left-to-right position) for correct reading order
        sorted_rois = sorted(self.rois, key=lambda roi: int(roi.state['pos'].x()))
        
        samples = {}
        i = 0
        for _roi in sorted_rois:
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


    def delete_all_rois(self, rois):
        if not rois:
            print("[ROI] No ROIs to remove")
            self.char_list.setText("")
            return

        print(f"[ROI] Starting deletion of {len(rois)} ROIs...")
        
        try:
            # Step 1: Disconnect signals
            print(f"[ROI] Step 1: Disconnecting signals from {len(rois)} ROIs...")
            for idx, roi in enumerate(rois):
                try:
                    roi.sigRegionChanged.disconnect()
                except Exception as e:
                    print(f"[ROI] Warning: Could not disconnect sigRegionChanged from ROI {idx}: {e}")
                try:
                    roi.sigClicked.disconnect()
                except Exception as e:
                    print(f"[ROI] Warning: Could not disconnect sigClicked from ROI {idx}: {e}")
            
            # Step 2: Get view and collect items to remove
            print(f"[ROI] Step 2: Collecting ROI items from view...")
            view = self.label.getView()
            if view is None:
                print("[ROI] ERROR: Could not get view!")
                rois.clear()
                self.char_list.setText("")
                self.selected_roi = None
                return
            # diagnostic: list current addedItems count
            print(f"[ROI] view.addedItems before deletion: {len(view.addedItems)}")
            items_to_remove = [item for item in view.addedItems if isinstance(item, pg.ROI)]
            print(f"[ROI] Found {len(items_to_remove)} ROI items to remove from view")
            
            # Step 3: Remove items from view
            print(f"[ROI] Step 3: Removing {len(items_to_remove)} items from view...")
            for idx, item in enumerate(items_to_remove):
                try:
                    view.removeItem(item)
                except Exception as e:
                    print(f"[ROI] Warning: Error removing item {idx}: {e}")
            
            # Step 4: Allow Qt to process the removal
            print(f"[ROI] Step 4: Processing Qt events to finalize removal...")
            QtCore.QCoreApplication.processEvents()
            
            # Step 5: Clear the list
            print(f"[ROI] Step 5: Clearing ROI list...")
            rois.clear()
            
            # Step 6: Clear the character display
            print(f"[ROI] Step 6: Clearing character display...")
            self.char_list.setText("")
            
            # Step 7: Clear the selection
            self.selected_roi = None 
            
            print("[ROI] ✓ All ROIs successfully removed")
            
        except Exception as e:
            print(f"[ROI] ✗ CRITICAL ERROR in delete_all_rois: {e}")
            print(traceback.format_exc())
            # Force clear even if there were errors
            try:
                rois.clear()
                self.char_list.setText("")
                self.selected_roi = None
                QtCore.QCoreApplication.processEvents()
            except Exception as e2:
                print(f"[ROI] ✗ CRITICAL ERROR in error recovery: {e2}")


class Thread(QThread):
    updateFrame = Signal(np.ndarray)
    width = 0
    height = 0
    index = -1
    

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.status = True
        self.cap = True
        

    def run(self):
        self.cap = cv2.VideoCapture(self.index)
        
        # Allow the camera to warm up before applying settings
        for _ in range(10):
            self.cap.read()

        # Apply manual settings from config
        self.parent()._apply_camera_settings(self.cap, self.index)
    
        
        while self.status:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Reading the frame, converting it to RGB, and flipping it for correct orientation
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            flipped_frame = cv2.flip(rgb_frame, 1)

            # Emit signal
            self.updateFrame.emit(flipped_frame)
        self.cap.release()
        self.cap = None
        #sys.exit(-1)



if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = CameraView()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec_())