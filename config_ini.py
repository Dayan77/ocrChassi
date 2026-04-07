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
cam_usb_index = [ "/dev/video2", "/dev/video0"] # Índices ou caminhos dos dispositivos (ex: 0, 1, ou "/dev/video0")
cam_usb_color = [ 1, 1]             # 1 para RGB/Colorido, 0 para Tons de cinza (Grayscale)
cam_usb_flip = [ 0, 0]              # 1 para inverter a imagem horizontalmente, 0 para manter normal

cam_auto_exposure = [0, 1]
cam_exposure = [4605, 475]
cam_auto_wb = [ 0, 0]               # Balanço de Branco: 1 para Automático, 0 para Manual
cam_wb_temperature = [ 4000, 4000]  # Temperatura de cor manual em Kelvin (Ex: 2800 para cores quentes, 6500 para cores frias)
cam_auto_focus = [0, 0]
cam_focus = [98, 300]

#----- Software Image Processing -----#
# Tratamento da imagem via software após a captura
cam_sw_contrast = [0.9, 0.9]        # 1.0 = original, > 1.0 aumenta o contraste
cam_sw_brightness = [0, 0]          # 0 = original, > 0 mais claro, < 0 mais escuro
cam_sw_sharpen = [0.0, 0.0]         # 0.0 = original, > 0.0 aplica filtro de nitidez (ex: 1.0 a 3.0)

cam_files_path = "/Users/dayansantos/Dev/ocrChassi/models/sense_chassi_19022026"
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
