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


import glob
import json
import time
import os
import platform
import traceback
import re
from pathlib import Path
import sys

_IS_MACOS = platform.system() == "Darwin"
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt, QThread, Signal, Slot, QDateTime, QDir
from PySide6.QtCore import Qt, QThread, Signal, Slot, QDateTime, QDir, QObject
from PySide6.QtGui import QImage, QColor
from PySide6.QtGui import QIcon, QPixmap, QImage
from PySide6.QtMultimedia import QMediaDevices
import cv2
import numpy as np

import pv_visionlib
import pyqtgraph as pg
from PySide6.QtWidgets import QProgressDialog, QMessageBox, QDialog, QSlider, QCheckBox, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QApplication, QPushButton


import icons_rc, images_rc, config_ini
color_bg="#b1b5b99f"

def update_config_file(cam_config_index, auto_focus, focus, auto_exp, exposure):
    """
    ATENÇÃO: Esta função modifica diretamente o arquivo config_ini.py.
    É uma abordagem frágil, mas funciona para o propósito atual.
    """
    config_path = 'config_ini.py'
    try:
        with open(config_path, 'r') as f:
            lines = f.readlines()

        def update_list_value(line, key, index, value):
            if line.strip().startswith(key):
                match = re.search(r'\[(.*?)\]', line)
                if match:
                    try:
                        items = json.loads(f'[{match.group(1)}]')
                        if 0 <= index < len(items):
                            items[index] = value
                            new_list_str = json.dumps(items, separators=(', ', ': '))
                            return f"{key} = {new_list_str}\n"
                    except (json.JSONDecodeError, IndexError):
                        pass # Fallback se a análise falhar ou o índice estiver fora
            return line

        with open(config_path, 'w') as f:
            for line in lines:
                if focus != "Não Suportado":
                    line = update_list_value(line, 'cam_focus', cam_config_index, focus)
                if exposure != "Não Suportado":
                    line = update_list_value(line, 'cam_exposure', cam_config_index, exposure)
                
                line = update_list_value(line, 'cam_auto_focus', cam_config_index, auto_focus)
                line = update_list_value(line, 'cam_auto_exposure', cam_config_index, auto_exp)
                f.write(line)
    except Exception as e:
        print(f"Erro ao atualizar o arquivo de configuração: {e}")

class CameraSettingsDialog(QDialog):
    def __init__(self, device_path, current_config_idx, parent=None):
        super().__init__(parent)
        self.device_path = device_path
        self.config_idx = current_config_idx
        self.setWindowTitle("Configurações da Câmera (Manual)")
        self.setMinimumWidth(400)
        
        self.exposure_info = self._get_ctrl_info('exposure_absolute', 3, 5000, 250)
        self.focus_info = self._get_ctrl_info('focus_absolute', 0, 250, 0)
        
        self.init_ui()
        
    def _get_ctrl_info(self, ctrl_name, default_min, default_max, default_val):
        info = {'min': default_min, 'max': default_max, 'val': default_val}
        try:
            import subprocess
            out = subprocess.check_output(['v4l2-ctl', '-d', self.device_path, '-l'], text=True)
            for line in out.split('\n'):
                if ctrl_name in line:
                    import re
                    m_min = re.search(r'min=(\d+)', line)
                    m_max = re.search(r'max=(\d+)', line)
                    m_val = re.search(r'value=(\d+)', line)
                    if m_min: info['min'] = int(m_min.group(1))
                    if m_max: info['max'] = int(m_max.group(1))
                    if m_val: info['val'] = int(m_val.group(1))
                    break
        except Exception:
            pass
        return info
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        import config_ini
        
        # --- Exposição ---
        gb_exp = QGroupBox("Exposição (Luz)")
        exp_layout = QVBoxLayout(gb_exp)
        
        self.chk_auto_exp = QCheckBox("Automático")
        self.sl_exposure = QSlider(Qt.Orientation.Horizontal)
        self.sl_exposure.setRange(self.exposure_info['min'], self.exposure_info['max'])
        
        is_auto_exp = config_ini.cam_auto_exposure[self.config_idx] == 1
        curr_exp = config_ini.cam_exposure[self.config_idx]
        if curr_exp == "Não Suportado": curr_exp = self.exposure_info['val']
        
        self.lbl_exp_val = QLabel(f"Valor: {curr_exp}")
        
        self.chk_auto_exp.setChecked(is_auto_exp)
        self.sl_exposure.setValue(int(curr_exp))
        self.sl_exposure.setEnabled(not is_auto_exp)
        
        self.chk_auto_exp.stateChanged.connect(self._on_auto_exp_changed)
        self.sl_exposure.valueChanged.connect(lambda v: self.lbl_exp_val.setText(f"Valor: {v}"))
        self.sl_exposure.sliderReleased.connect(self._apply_exposure)
        
        exp_layout.addWidget(self.chk_auto_exp)
        exp_layout.addWidget(self.lbl_exp_val)
        exp_layout.addWidget(self.sl_exposure)
        layout.addWidget(gb_exp)
        
        # --- Foco ---
        gb_foc = QGroupBox("Foco (Nitidez)")
        foc_layout = QVBoxLayout(gb_foc)
        
        self.chk_auto_foc = QCheckBox("Automático")
        self.sl_focus = QSlider(Qt.Orientation.Horizontal)
        self.sl_focus.setRange(self.focus_info['min'], self.focus_info['max'])
        
        is_auto_foc = config_ini.cam_auto_focus[self.config_idx] == 1
        curr_foc = config_ini.cam_focus[self.config_idx]
        if curr_foc == "Não Suportado": curr_foc = self.focus_info['val']
        
        self.lbl_foc_val = QLabel(f"Valor: {curr_foc}")
        
        self.chk_auto_foc.setChecked(is_auto_foc)
        self.sl_focus.setValue(int(curr_foc))
        self.sl_focus.setEnabled(not is_auto_foc)
        
        self.chk_auto_foc.stateChanged.connect(self._on_auto_foc_changed)
        self.sl_focus.valueChanged.connect(lambda v: self.lbl_foc_val.setText(f"Valor: {v}"))
        self.sl_focus.sliderReleased.connect(self._apply_focus)
        
        foc_layout.addWidget(self.chk_auto_foc)
        foc_layout.addWidget(self.lbl_foc_val)
        foc_layout.addWidget(self.sl_focus)
        layout.addWidget(gb_foc)
        
        # --- Botoes ---
        btn_layout = QHBoxLayout()
        self.btn_auto_smart = QPushButton("Auto Ajuste (Smart ROI)")
        self.btn_auto_smart.clicked.connect(self.accept_as_smart_auto)
        self.btn_auto_smart.setStyleSheet("background-color: #89b4fa; color: #11111b; font-weight: bold;")
        
        self.btn_save = QPushButton("Salvar Manual")
        self.btn_save.clicked.connect(self.accept)
        self.btn_save.setStyleSheet("background-color: #a6e3a1; color: #11111b; font-weight: bold;")
        
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addWidget(self.btn_auto_smart)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_save)
        
        layout.addLayout(btn_layout)

    def _on_auto_exp_changed(self, state):
        is_auto = (state == 2)
        self.sl_exposure.setEnabled(not is_auto)
        import subprocess
        auto_val = 3 if is_auto else 1
        subprocess.run(['v4l2-ctl', '-d', self.device_path, '-c', f'exposure_auto={auto_val}'], stderr=subprocess.DEVNULL)
        if not is_auto:
            self._apply_exposure()
            
    def _on_auto_foc_changed(self, state):
        is_auto = (state == 2)
        self.sl_focus.setEnabled(not is_auto)
        import subprocess
        auto_val = 1 if is_auto else 0
        subprocess.run(['v4l2-ctl', '-d', self.device_path, '-c', f'focus_auto={auto_val}'], stderr=subprocess.DEVNULL)
        if not is_auto:
            self._apply_focus()
            
    def _apply_exposure(self):
        val = self.sl_exposure.value()
        import subprocess
        subprocess.run(['v4l2-ctl', '-d', self.device_path, '-c', f'exposure_absolute={val}'], stderr=subprocess.DEVNULL)
        subprocess.run(['v4l2-ctl', '-d', self.device_path, '-c', 'exposure_auto_priority=0'], stderr=subprocess.DEVNULL)
        
    def _apply_focus(self):
        val = self.sl_focus.value()
        import subprocess
        subprocess.run(['v4l2-ctl', '-d', self.device_path, '-c', f'focus_absolute={val}'], stderr=subprocess.DEVNULL)
        
    def accept_as_smart_auto(self):
        self.done(2)

class CameraView(QtWidgets.QWidget):
    image_path = None
    actual_image = None
    normal_pen = pg.mkPen('g', width=2)  # Verde para normal
    selected_pen = pg.mkPen('m', width=2) # Amarelo para selecionado
    rois = []
    annotation_data = None          # loaded JSON dict for current image
    auto_adjust_roi = None
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

        self.auto_adjust_button = QtWidgets.QPushButton()
        self.auto_adjust_button.setMaximumHeight(35)
        self.auto_adjust_button.setMaximumWidth(60)
        auto_adjust_icon = QIcon(":icons/icons/zap.svg")
        self.auto_adjust_button.setIcon(auto_adjust_icon)
        self.auto_adjust_button.setToolTip("Auto Ajuste da Câmera (Smart ROI)")

        self.settings_button = QtWidgets.QPushButton()
        self.settings_button.setMaximumHeight(35)
        self.settings_button.setMaximumWidth(60)
        self.settings_button.setIcon(QIcon(":icons/icons/sliders.svg"))
        self.settings_button.setToolTip("Ajuste Manual de Câmera (Sliders)")
        self.settings_button.setProperty("class", "overlay_btn")

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
        self.auto_adjust_button.setProperty("class", "overlay_btn")
        
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
        cam_actions.addWidget(self.auto_adjust_button)
        cam_actions.addWidget(self.settings_button)
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
        self.auto_adjust_button.clicked.connect(self.run_auto_adjust)
        self.settings_button.clicked.connect(self.open_camera_settings)
        self.filter_btn.clicked.connect(self.toggle_filter)
        self.isLive = False
        self.th = None

    def _apply_camera_settings(self, cap, index):
        """Applies camera settings from config_ini to a VideoCapture object."""
        import importlib
        import time
        try:
            importlib.reload(config_ini) # Atualiza as vars com base no arquivo em tempo real
        except Exception as e:
            print(f"Aviso: Falha ao recarregar config_ini.py: {e}")

        # self.camera_usb_index is now the configuration index (0, 1, ...)
        config_idx = self.camera_usb_index

        # Safety check: Ensure config_idx is within bounds of the setting lists
        if not isinstance(config_idx, int) or config_idx >= len(config_ini.cam_auto_exposure):
            print(f"Aviso: Índice de configuração de câmera inválido {config_idx}. Usando 0.")
            config_idx = 0

        device_path = None
        if isinstance(index, str) and index.startswith("/dev/video"):
            device_path = index
        elif isinstance(index, int):
            device_path = f"/dev/video{index}"
            
        if cap and cap.isOpened() and not _IS_MACOS:
            # Aplica via OpenCV (valores V4L2 — apenas Linux)
            auto_exp = 3 if config_ini.cam_auto_exposure[config_idx] else 1
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exp)
            if not config_ini.cam_auto_exposure[config_idx]:
                cap.set(cv2.CAP_PROP_EXPOSURE, config_ini.cam_exposure[config_idx])

            cap.set(cv2.CAP_PROP_AUTO_WB, config_ini.cam_auto_wb[config_idx])
            if not config_ini.cam_auto_wb[config_idx]:
                cap.set(cv2.CAP_PROP_WB_TEMPERATURE, config_ini.cam_wb_temperature[config_idx])

            cap.set(cv2.CAP_PROP_AUTOFOCUS, config_ini.cam_auto_focus[config_idx])
            if not config_ini.cam_auto_focus[config_idx]:
                cap.set(cv2.CAP_PROP_FOCUS, config_ini.cam_focus[config_idx])

        if device_path:
            import subprocess
            auto_exp = 3 if config_ini.cam_auto_exposure[config_idx] else 1
            try:
                # Utilizamos o v4l2-ctl para forçar os parâmetros no driver USB Linux
                subprocess.run(['v4l2-ctl', '-d', device_path, '-c', f'exposure_auto={auto_exp}'], stderr=subprocess.DEVNULL)
                if not config_ini.cam_auto_exposure[config_idx]:
                    subprocess.run(['v4l2-ctl', '-d', device_path, '-c', f'exposure_absolute={config_ini.cam_exposure[config_idx]}'], stderr=subprocess.DEVNULL)
                    subprocess.run(['v4l2-ctl', '-d', device_path, '-c', 'exposure_auto_priority=0'], stderr=subprocess.DEVNULL)
                    subprocess.run(['v4l2-ctl', '-d', device_path, '-c', 'gain_auto=0'], stderr=subprocess.DEVNULL)
                    subprocess.run(['v4l2-ctl', '-d', device_path, '-c', 'autogain=0'], stderr=subprocess.DEVNULL)
                
                wb_auto = config_ini.cam_auto_wb[config_idx]
                subprocess.run(['v4l2-ctl', '-d', device_path, '-c', f'white_balance_temperature_auto={wb_auto}'], stderr=subprocess.DEVNULL)
                if not wb_auto:
                    subprocess.run(['v4l2-ctl', '-d', device_path, '-c', f'white_balance_temperature={config_ini.cam_wb_temperature[config_idx]}'], stderr=subprocess.DEVNULL)
                
                focus_auto = config_ini.cam_auto_focus[config_idx]
                subprocess.run(['v4l2-ctl', '-d', device_path, '-c', f'focus_auto={focus_auto}'], stderr=subprocess.DEVNULL)
                if not focus_auto:
                    subprocess.run(['v4l2-ctl', '-d', device_path, '-c', f'focus_absolute={config_ini.cam_focus[config_idx]}'], stderr=subprocess.DEVNULL)
                
                time.sleep(0.5) # Aguarda a lente/sensor da câmera aplicar as configurações físicas
            except Exception as e:
                print(f"Aviso: v4l2-ctl não funcionou ou não está instalado: {e}")

    def open_camera_settings(self):
        if self.auto_adjust_roi is not None:
            # We are in the middle of a Smart Auto Adjust ROI confirmation!
            self.run_auto_adjust()
            return

        device_path = self.resolve_camera_index()
        v4l2_path = None
        if isinstance(device_path, str) and device_path.startswith('/dev/video'):
            v4l2_path = device_path
        elif isinstance(device_path, int):
            v4l2_path = f"/dev/video{device_path}"
            
        if not v4l2_path and not _IS_MACOS:
            QMessageBox.warning(self, "Erro", "O ajuste só funciona com câmeras V4L2 (ex: /dev/videoX).")
            return

        was_live = self.isLive
        if not self.isLive:
            self.start_live()
            QThread.msleep(500)
            QApplication.processEvents()
            
        dialog = CameraSettingsDialog(v4l2_path, self.camera_usb_index, self)
        result = dialog.exec()
        
        if result == 1: # Accepted / Salvar
            import config_ini
            auto_exp = 1 if dialog.chk_auto_exp.isChecked() else 0
            exp_val = dialog.sl_exposure.value()
            auto_foc = 1 if dialog.chk_auto_foc.isChecked() else 0
            foc_val = dialog.sl_focus.value()
            
            config_ini.cam_auto_exposure[self.camera_usb_index] = auto_exp
            config_ini.cam_exposure[self.camera_usb_index] = exp_val
            config_ini.cam_auto_focus[self.camera_usb_index] = auto_foc
            config_ini.cam_focus[self.camera_usb_index] = foc_val
            
            update_config_file(self.camera_usb_index, auto_foc, foc_val, auto_exp, exp_val)
            QMessageBox.information(self, "Salvo", "Configurações de câmera salvas com sucesso!")
            
        elif result == 2: # Smart Auto Adjust
            self.run_auto_adjust()
            
        else:
            import config_ini
            self._apply_camera_settings(None, v4l2_path)
            
        if not was_live and self.isLive and result != 2:
            self.stop_live()

    def run_auto_adjust(self):
        device_path = self.resolve_camera_index()
        
        v4l2_path = None
        if isinstance(device_path, str) and device_path.startswith('/dev/video'):
            v4l2_path = device_path
        elif isinstance(device_path, int):
            v4l2_path = f"/dev/video{device_path}"
            
        if not v4l2_path and not _IS_MACOS:
            QMessageBox.warning(self, "Erro", "O auto ajuste só funciona com câmeras V4L2 (ex: /dev/videoX).")
            return
            
        if self.auto_adjust_roi is None:
            if self.actual_image is None:
                QMessageBox.warning(self, "Sem Imagem", "Capture ou carregue uma imagem para definir a área de ajuste.")
                return
            
            img_h, img_w = self.actual_image.shape[:2]
            w, h = int(img_w * 0.3), int(img_h * 0.3)
            x, y = int((img_w - w) / 2), int((img_h - h) / 2)
            
            self.auto_adjust_roi = pg.ROI(
                pos=[x, y],
                size=[w, h],
                pen=pg.mkPen('c', width=3, style=Qt.DashLine),
                handlePen=pg.mkPen('c', width=1)
            )
            self.auto_adjust_roi.addScaleHandle([1, 1], [0.5, 0.5])
            self.auto_adjust_roi.addScaleHandle([0, 0], [0.5, 0.5])
            self.auto_adjust_roi.setZValue(20)
            self.label.getView().addItem(self.auto_adjust_roi)
            
            self.auto_adjust_button.setIcon(QIcon(":icons/icons/check-circle.svg"))
            self.auto_adjust_button.setToolTip("Confirmar Área e Iniciar Auto Ajuste")
            
            QMessageBox.information(self, "Área de Ajuste", 
                "Foi adicionada uma caixa tracejada azul na imagem.\n\n"
                "1. Posicione e redimensione-a sobre a área que deseja focar (ex: chassi).\n"
                "2. Clique novamente no botão de Auto Ajuste para confirmar.")
            return

        # Confirmar e extrair coordenadas da ROI
        pos = self.auto_adjust_roi.pos()
        size = self.auto_adjust_roi.size()
        roi_rect = (int(pos.x()), int(pos.y()), int(size.x()), int(size.y()))
        
        self.label.getView().removeItem(self.auto_adjust_roi)
        self.auto_adjust_roi = None
        self.auto_adjust_button.setIcon(QIcon(":icons/icons/zap.svg"))
        self.auto_adjust_button.setToolTip("Auto Ajuste da Câmera (Smart ROI)")

        self.was_live_before_adjust = self.isLive
        if self.isLive:
            self.stop_live()

        self.progress_dialog = QProgressDialog("Ajustando câmera...", "Cancelar", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.show()

        self.thread = QThread()
        self.worker = AutoAdjustWorker(v4l2_path, roi_rect)
        self.worker.moveToThread(self.thread)

        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_auto_adjust_finished)
        self.thread.started.connect(self.worker.run)
        
        # Cleanup
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    @Slot(str, int)
    def update_progress(self, message, value):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.setLabelText(message)
            self.progress_dialog.setValue(value)

    @Slot(dict)
    def on_auto_adjust_finished(self, result):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()

        if 'error' in result:
            QMessageBox.critical(self, "Erro no Auto Ajuste", result['error'])
            return

        focus = result.get('focus')
        exposure = result.get('exposure')

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText("Auto ajuste concluído!")
        msg_box.setInformativeText(f"Valores ótimos encontrados:\n\nFoco: {focus}\nExposição: {exposure}\n\nDeseja salvar estes valores no arquivo config_ini.py?")
        msg_box.setStandardButtons(QMessageBox.Save | QMessageBox.Apply | QMessageBox.Cancel)
        msg_box.setDefaultButton(QMessageBox.Save)
        ret = msg_box.exec()

        if ret == QMessageBox.Save or ret == QMessageBox.Apply:
            try:
                # self.camera_usb_index is already the correct configuration index
                config_idx = self.camera_usb_index
                
                # Atualizar os valores em memória (para a sessão atual)
                if focus != "Não Suportado":
                    config_ini.cam_focus[config_idx] = focus
                    config_ini.cam_auto_focus[config_idx] = 0
                if exposure != "Não Suportado":
                    config_ini.cam_exposure[config_idx] = exposure
                    config_ini.cam_auto_exposure[config_idx] = 0

                if ret == QMessageBox.Save:
                    # Salvar no arquivo
                    save_focus = focus if focus != "Não Suportado" else config_ini.cam_focus[config_idx]
                    save_exp = exposure if exposure != "Não Suportado" else config_ini.cam_exposure[config_idx]
                    update_config_file(config_idx, 0, save_focus, 0, save_exp)
                    QMessageBox.information(self, "Salvo", "Configurações salvas em config_ini.py.")

                # Reaplicar as configurações na câmera para garantir e mostrar o resultado
                if getattr(self, 'was_live_before_adjust', False):
                    self.start_live()
                else:
                    self.capture_single_frame()
            except (ValueError, IndexError) as e:
                QMessageBox.critical(self, "Erro", f"Não foi possível encontrar o índice da câmera no config_ini.py: {e}")

    def take_picture(self):
        """Captures the current frame and saves it to a file."""
        if self.isLive:
            # If live, just stop the feed. The last frame is already in actual_image.
            self.start_live() # Toggles it off
        else:
            # If not live, open camera, grab a single frame, and close
            self.capture_single_frame()

    def resolve_camera_index(self):
        """Resolves the configured camera index to the actual system device index."""
        try:
            import importlib
            importlib.reload(config_ini)
        except Exception:
            pass
            
        try:
            idx = config_ini.cam_usb_index[self.camera_usb_index]
            if isinstance(idx, str) and idx.isdigit():
                idx = int(idx)
            return idx
        except IndexError:
            QMessageBox.warning(
                self, "Erro de Configuração",
                f"Índice de câmera {self.camera_usb_index} está fora dos limites para 'cam_usb_index' no config_ini.py."
            )
            idx = self.camera_usb_index
            if isinstance(idx, str) and idx.isdigit():
                idx = int(idx)
            return idx

    def capture_single_frame(self):
        """Opens the camera, captures a single frame, and displays it."""
        real_index = self.resolve_camera_index()
        # If not live, open camera, grab a single frame, and close
        
        if not _IS_MACOS:
            cap = cv2.VideoCapture(real_index, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(real_index)
        else:
            cap = cv2.VideoCapture(real_index)
        if not cap.isOpened():
            # List available cameras to help debugging
            cameras = QMediaDevices.videoInputs()
            cam_list = "\n".join([f"- {cam.description()} ({cam.id().data().decode()})" for cam in cameras])
            QtWidgets.QMessageBox.warning(
                self, 
                "Camera Error", 
                f"Could not open camera index {real_index} (Config: {self.camera_usb_index}).\n\nAvailable cameras:\n{cam_list}"
            )
            return
        
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Aplica as configurações COM o stream ativo para não serem sobrescritas
        self._apply_camera_settings(cap, real_index)

        # Descarta os quadros escuros e dá tempo ao sensor para estabilizar a luz
        for _ in range(15):
            cap.read()

        try:
            config_idx = self.camera_usb_index

            ret, frame = cap.read()
            cap.release()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if config_idx < len(config_ini.cam_usb_flip) and config_ini.cam_usb_flip[config_idx] == 1:
                    captured_frame = cv2.flip(rgb_frame, 1)
                else:
                    captured_frame = rgb_frame
                    
                # Aplicar tratamento de imagem via software
                try:
                    contrast_list = getattr(config_ini, 'cam_sw_contrast', [1.0, 1.0])
                    brightness_list = getattr(config_ini, 'cam_sw_brightness', [0, 0])
                    sharpen_list = getattr(config_ini, 'cam_sw_sharpen', [0.0, 0.0])
                    
                    contrast = contrast_list[config_idx] if config_idx < len(contrast_list) else 1.0
                    brightness = brightness_list[config_idx] if config_idx < len(brightness_list) else 0
                    sharpen = sharpen_list[config_idx] if config_idx < len(sharpen_list) else 0.0
                    
                    vision_lib = pv_visionlib.pvVisionLib()
                    captured_frame = vision_lib.enhance_image(captured_frame, contrast, brightness, sharpen)
                except Exception as e:
                    print(f"Erro no tratamento de imagem: {e}")
                    
                self.setImage(captured_frame) # Update the view
        except (IndexError, TypeError):
             QMessageBox.warning(self, "Erro de Configuração", f"Índice de câmera {self.camera_usb_index} inválido em config_ini.py.")
             cap.release()

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
        self.settings_button.setEnabled(False)
        self.file_button.setEnabled(False)
        self.previous_button.setEnabled(False)
        self.next_button.setEnabled(False)

    def enable_btns(self):
        self.live_button.setEnabled(True)
        self.picture_button.setEnabled(True)
        self.settings_button.setEnabled(True)
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
            if not file_path:
                print("Aviso: Nenhum caminho de imagem fornecido. Usando imagem em branco.")
                blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
                self.actual_image = blank_image
                return blank_image
                
            # Load image using OpenCV
            # OpenCV loads images as BGR, so we convert to RGB
            image = cv2.imread(file_path)
            if image is None:
                print(f"Aviso: Falha ao ler a imagem: {file_path}. Arquivo inexistente ou formato não suportado.")
                image = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            self.actual_image = image
            return image
        
        except Exception as e:
            error_msg = f"Error loading image '{file_path}': {str(e)}"
            print(error_msg)
            blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
            self.actual_image = blank_image
            return blank_image



    def start_live(self):
        if not self.isLive:
            self.isLive = True
            self.th = Thread(self)
            self.th.index = self.resolve_camera_index()
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

class AutoAdjustWorker(QObject):
    finished = Signal(dict)
    progress = Signal(str, int)

    def __init__(self, device_path, roi_rect=None, parent=None):
        super().__init__(parent)
        self.device_path = device_path
        self.roi_rect = roi_rect

    def _run_v4l2_ctl(self, *args, silent=False):
        import subprocess
        cmd = ['v4l2-ctl', '-d', self.device_path] + list(args)
        try:
            return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        except subprocess.CalledProcessError as e:
            if not silent:
                print(f"Erro ao executar v4l2-ctl: {e.output.strip()}")
            return None
        except FileNotFoundError:
            if not silent:
                print("v4l2-ctl não encontrado. Instale v4l-utils.")
            return None

    def _calculate_sharpness(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @Slot()
    def run(self):
        self.progress.emit("Desabilitando controles automáticos...", 10)
        self._run_v4l2_ctl('-c', 'focus_auto=0', silent=True)
        self._run_v4l2_ctl('-c', 'exposure_auto=1', silent=True) # 1 = Manual
        self._run_v4l2_ctl('-c', 'white_balance_temperature_auto=0', silent=True)
        QThread.msleep(200)
        
        if not _IS_MACOS:
            cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.device_path)
        else:
            cap = cv2.VideoCapture(self.device_path)

        if not cap.isOpened():
            self.finished.emit({'error': 'Não foi possível abrir a câmera.'})
            return

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Aquecimento inicial da câmera
        for _ in range(10): cap.read()

        self.progress.emit("Ajustando exposição...", 10)
        best_exposure = -1
        min_diff = float('inf')
        target_brightness = 110 # Alvo levemente mais escuro para evitar estouro de luz no metal
        
        # Degraus logarítmicos e abrangentes de luz
        exposures = [10, 30, 50, 80, 120, 160, 200, 250, 312, 400, 500, 625, 800, 1000, 1250, 1500, 2000, 2500, 5000]
        
        exposure_supported = self._run_v4l2_ctl('-c', f'exposure_absolute={exposures[0]}', silent=True) is not None
        
        if exposure_supported:
            for idx, exposure in enumerate(exposures):
                self._run_v4l2_ctl('-c', f'exposure_absolute={exposure}', silent=True)
                QThread.msleep(150) # Tempo para o sensor ajustar fisicamente a luz
                for _ in range(3): cap.read() # Limpa quadros velhos do buffer
                ret, frame = cap.read()
                if not ret or frame is None: continue

                eval_frame = frame
                if self.roi_rect:
                    x, y, w, h = self.roi_rect
                    img_h, img_w = frame.shape[:2]
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(img_w, x + w), min(img_h, y + h)
                    if x2 > x1 and y2 > y1:
                        eval_frame = frame[y1:y2, x1:x2]

                mean_brightness = np.mean(cv2.cvtColor(eval_frame, cv2.COLOR_BGR2GRAY))
                if abs(mean_brightness - target_brightness) < min_diff:
                    min_diff = abs(mean_brightness - target_brightness)
                    best_exposure = exposure
                
                progress_percent = 10 + int((idx / len(exposures)) * 40)
                self.progress.emit(f"Ajustando exposição... (Luz: {int(mean_brightness)}/110)", progress_percent)

            if best_exposure != -1:
                self._run_v4l2_ctl('-c', f'exposure_absolute={best_exposure}', silent=True)
                QThread.msleep(200)
                for _ in range(5): cap.read()
            else:
                self.finished.emit({'error': 'Falha ao ajustar a exposição.'})
                cap.release()
                return
        else:
            best_exposure = "Não Suportado"
            self.progress.emit("Exposição manual não suportada, pulando...", 50)
            QThread.msleep(500)

        self.progress.emit("Ajustando foco...", 50)
        best_focus = -1
        max_sharpness = -1
        focus_steps = list(range(0, 256, 15))
        
        focus_supported = self._run_v4l2_ctl('-c', f'focus_absolute={focus_steps[0]}', silent=True) is not None

        if focus_supported:
            for idx, focus in enumerate(focus_steps):
                self._run_v4l2_ctl('-c', f'focus_absolute={focus}', silent=True)
                QThread.msleep(150) # Tempo para a lente física se mover
                for _ in range(3): cap.read()
                ret, frame = cap.read()
                if not ret or frame is None: continue
                
                eval_frame = frame
                if self.roi_rect:
                    x, y, w, h = self.roi_rect
                    img_h, img_w = frame.shape[:2]
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(img_w, x + w), min(img_h, y + h)
                    if x2 > x1 and y2 > y1:
                        eval_frame = frame[y1:y2, x1:x2]
                        
                sharpness = self._calculate_sharpness(eval_frame)
                if sharpness > max_sharpness:
                    max_sharpness = sharpness
                    best_focus = focus
                
                progress_percent = 50 + int((idx / len(focus_steps)) * 40)
                self.progress.emit(f"Ajustando foco... (Nitidez: {int(sharpness)})", progress_percent)

            if best_focus != -1:
                self._run_v4l2_ctl('-c', f'focus_absolute={best_focus}', silent=True)
        else:
            best_focus = "Não Suportado"
            self.progress.emit("Foco manual não suportado, pulando...", 90)
            QThread.msleep(500)

        cap.release()
        self.progress.emit("Ajuste concluído!", 100)
        self.finished.emit({'focus': best_focus, 'exposure': best_exposure})

class Thread(QThread):
    updateFrame = Signal(np.ndarray)
    width = 0
    height = 0
    index = -1
    

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.status = True
        self.cap = True
        self.frame_count = 0
        

    def run(self):
        if not _IS_MACOS:
            self.cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.index)
        else:
            self.cap = cv2.VideoCapture(self.index)
        
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self.parent() and hasattr(self.parent(), '_apply_camera_settings'):
                self.parent()._apply_camera_settings(self.cap, self.index)

            for _ in range(15):
                self.cap.read()
    
        
        while self.status:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            self.frame_count += 1
            if self.frame_count % 15 == 0:
                try:
                    import importlib
                    importlib.reload(config_ini)
                except Exception:
                    pass

            # Reading the frame, converting it to RGB, and flipping it for correct orientation
            try:
                config_idx = self.parent().camera_usb_index

                # Reading the frame, converting it to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if config_idx < len(config_ini.cam_usb_flip) and config_ini.cam_usb_flip[config_idx] == 1:
                    final_frame = cv2.flip(rgb_frame, 1)
                else:
                    final_frame = rgb_frame

                # Aplicar tratamento de imagem via software
                try:
                    contrast_list = getattr(config_ini, 'cam_sw_contrast', [1.0, 1.0])
                    brightness_list = getattr(config_ini, 'cam_sw_brightness', [0, 0])
                    sharpen_list = getattr(config_ini, 'cam_sw_sharpen', [0.0, 0.0])
                    
                    contrast = contrast_list[config_idx] if config_idx < len(contrast_list) else 1.0
                    brightness = brightness_list[config_idx] if config_idx < len(brightness_list) else 0
                    sharpen = sharpen_list[config_idx] if config_idx < len(sharpen_list) else 0.0
                    
                    vision_lib = pv_visionlib.pvVisionLib()
                    final_frame = vision_lib.enhance_image(final_frame, contrast, brightness, sharpen)
                except Exception as e:
                    print(f"Erro no tratamento de imagem (Live): {e}")

                # Emit signal
                self.updateFrame.emit(final_frame)
            except (IndexError, TypeError):
                # This can happen if config is changed while running. Stop the thread.
                print(f"Erro de configuração no thread da câmera, parando. Índice: {self.parent().camera_usb_index}")
                self.status = False
        self.cap.release()
        self.cap = None
        #sys.exit(-1)



if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = CameraView()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec_())