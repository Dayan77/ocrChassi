import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#----- Application ------#
app_title = "Chassi Reader"
app_theme = "light"
app_wnd_mode = "Maximized"
app_wnd_width = 800
app_wnd_height = 800
app_customer_logo = ":images/SWIFT_BRANCO.png"
app_company_logo = ""


#----- Cameras ------#
cam_qty = 2                         # Quantidade de câmeras ativas
cam_type = ["realsense", "realsense"] # "usb" ou "realsense"
cam_usb_index = [ 1, 0]             # USB: índices ou paths. RealSense: ignorado
cam_rs_serial = ["", ""]            # RealSense: número de série (vazio = ordem de detecção)
cam_usb_color = [ 1, 1]             # 1 para RGB/Colorido, 0 para Tons de cinza
cam_usb_flip = [ 0, 0]              # 1 para inverter horizontalmente, 0 para normal

cam_auto_exposure = [0, 0]          # 1 = automático, 0 = manual
cam_exposure = [8500, 8500]         # USB: valor V4L2 (1-5000) | RealSense D455c: microssegundos (1-165000)
cam_auto_wb = [ 0, 0]               # Balanço de Branco: 1 = Automático, 0 = Manual
cam_wb_temperature = [ 4000, 4000]  # Temperatura de cor em Kelvin (2800-6500)
cam_auto_focus = [0, 0]             # 1 = automático, 0 = manual
cam_focus = [49, 241]               # USB: valor V4L2 | RealSense D455c: 0-255
cam_rs_gain = [64, 64]              # RealSense: ganho do sensor de cor (16-248)

#----- Software Image Processing -----#
# Tratamento da imagem via software após a captura
cam_sw_contrast = [0.96, 0.48]
cam_sw_brightness = [50, 10]
cam_sw_sharpen = [1.61, 1.38]

cam_files_path = os.path.join(_BASE_DIR, "models", "sense_chassi_19022026")
default_roi_x = 50
default_roi_y = 50
default_roi_w = 30
default_roi_h = 30





#----- Models ------#
model_name = "sensebike_ocr_model.h5"
model_train_epocs = 30
DATA_DIR = 'dataset' # Directory with your character subfolders
MODEL_SAVE_PATH = 'sensebike_ocr_model.h5'
LABEL_ENCODER_SAVE_PATH = 'sensebike_ocr_model.h5.npy'
IMG_HEIGHT = 128
IMG_WIDTH = 128
EPOCHS = 15

#----- Segmentation -----#
segmentation_threshold = 50
segmentation_diameter = 9
segmentation_sigma = 75
segmentation_space = 75
segmentation_min_a = 0
segmentation_max_a = 100
segmentation_min_w = 3
segmentation_max_w = 100
segmentation_min_h = 1
segmentation_max_h = 100


#----- Programs ------#
program_path = ""





#----- Production ------#
number_of_characters = 10
threshold_char_ocr = 0.7
demo_mode = False
trigger_delay = 1000
production_model_library = "PyTorch" # "PyTorch" or "TensorFlow"
production_cam1_flip = False
production_cam2_flip = False
production_results_inverted = False
production_serial_pattern = "AAA9999999" # "A" para letras, "9" para números, "#" para ignorar correção na posição
production_training_folder = "/home/sense-bike/Área de trabalho/ocrChassi/models/retrain"
