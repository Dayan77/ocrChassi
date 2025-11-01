import sys
from PySide6.QtWidgets import QMainWindow, QApplication, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QWidget
from PySide6.QtCore import Slot, QFile, QTextStream

from sidebar_ui import Ui_MainWindow
from views.cameraView  import CameraView
from views.programView import ProgramView 

import config_ini

class MainWindow(QMainWindow):
    program_view = None
    def __init__(self):
        super(MainWindow, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.icon_only_widget.hide()
        self.ui.stackedWidget.setCurrentIndex(6)
        self.ui.home_btn_2.setChecked(True)

    def resizeEvent(self, event):
        # Get the new size from the event object
        new_size = event.size()
        old_size = event.oldSize()

        # Perform your custom actions here
        print(f"Window resized from {old_size.width()}x{old_size.height()} to {new_size.width()}x{new_size.height()}")
        
        if self.program_view:
            self.program_view.resize_views(new_size.width()-150, new_size.height()-40)
        
        # Always call the base class's resizeEvent()
        super().resizeEvent(event)

    ## Function for searching
    def on_search_btn_clicked(self):
        self.ui.stackedWidget.setCurrentIndex(5)
        search_text = self.ui.search_input.text().strip()
        if search_text:
            self.ui.label_9.setText(search_text)

    ## Function for changing page to user page
    def on_user_btn_clicked(self):
        self.ui.stackedWidget.setCurrentIndex(6)

    ## Change QPushButton Checkable status when stackedWidget index changed
    def on_stackedWidget_currentChanged(self, index):
        btn_list = self.ui.icon_only_widget.findChildren(QPushButton) \
                    + self.ui.full_menu_widget.findChildren(QPushButton)
        
        for btn in btn_list:
            if index in [5, 6]:
                btn.setAutoExclusive(False)
                btn.setChecked(False)
            else:
                btn.setAutoExclusive(True)
            
    ## functions for changing menu page
    def on_home_btn_1_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(0)
    
    def on_home_btn_2_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(0)

    @Slot()
    def on_dashboard_btn_1_toggled(self):
        #Cameras view box
        
        entire_view = QWidget()
        entire_widget = QGridLayout(entire_view)
        self.program_view = ProgramView(self.width()-90, self.height()-70)
        
        self.program_view.create_views(config_ini.cam_qty)
        
        self.ui.stackedWidget.addWidget(self.program_view)

    @Slot()
    def on_dashboard_btn_2_toggled(self):
        #self.ui.stackedWidget.setCurrentIndex(1)
        self.on_dashboard_btn_1_toggled()

    def on_orders_btn_1_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(2)

    def on_orders_btn_2_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(2)

    def on_products_btn_1_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(3)

    def on_products_btn_2_toggled(self, ):
        self.ui.stackedWidget.setCurrentIndex(3)

    def on_customers_btn_1_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(4)

    def on_customers_btn_2_toggled(self):
        self.ui.stackedWidget.setCurrentIndex(4)

    


if __name__ == "__main__":
    app = QApplication(sys.argv)

    ## loading style file
    # with open("style.qss", "r") as style_file:
    #     style_str = style_file.read()
    # app.setStyleSheet(style_str)

    ## loading style file, Example 2
    style_file = QFile("style.qss")
    style_file.open(QFile.ReadOnly | QFile.Text)
    style_stream = QTextStream(style_file)
    app.setStyleSheet(style_stream.readAll())


    window = MainWindow()
    window.show()

    sys.exit(app.exec())