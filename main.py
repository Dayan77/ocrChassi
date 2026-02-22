import sys
from PySide6.QtWidgets import QMainWindow, QApplication, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QWidget
from PySide6.QtCore import Slot, QFile, QTextStream, QSettings
from PySide6.QtGui import QPalette, QColor, QIcon, QPixmap

from sidebar_ui import Ui_MainWindow
from views.cameraView  import CameraView
from views.programView import ProgramView
from views.inspection_view import InspectionView

import images_rc
import resource_rc
import icons_rc

import config_ini

class MainWindow(QMainWindow):
    program_view = None
    inspection_view = None
    settings = QSettings("Sense", "OCRChassi")
    
    def __init__(self):
        super(MainWindow, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.icon_only_widget.hide()
        
        self.ui.stackedWidget.setCurrentIndex(6)
        self.ui.home_btn_2.setChecked(True)

        # Set text color for logo_label_3
        _translate = self.ui.centralwidget.tr
        self.ui.logo_label_3.setText(_translate("MainWindow", config_ini.app_title))
        self.setWindowTitle(config_ini.app_title + " - " + config_ini.model_name)
        palette = self.ui.logo_label_3.palette()
        palette.setColor(QPalette.WindowText, QColor(255, 0, 0))
        self.ui.logo_label_3.setPalette(palette)

        self.ui.customers_btn_1.toggled.connect(self.on_customers_btn_toggled)
        self.ui.customers_btn_2.toggled.connect(self.on_customers_btn_toggled)
        
        # Pre-populate stackedWidget with empty pages so indices match
        self.page_home = QWidget()
        self.page_dashboard = QWidget()
        self.page_orders = QWidget()
        self.page_products = QWidget()
        self.page_customers = QWidget()
        self.page_search = QWidget()
        self.page_user = QWidget()

        self.ui.stackedWidget.addWidget(self.page_home)       # 0
        self.ui.stackedWidget.addWidget(self.page_dashboard)  # 1
        self.ui.stackedWidget.addWidget(self.page_orders)     # 2
        self.ui.stackedWidget.addWidget(self.page_products)   # 3
        self.ui.stackedWidget.addWidget(self.page_customers)  # 4
        self.ui.stackedWidget.addWidget(self.page_search)     # 5
        self.ui.stackedWidget.addWidget(self.page_user)       # 6

        # Theme switching
        self.ui.theme_btn.clicked.connect(self.toggle_theme)
        self.load_theme()

    def load_theme(self):
        theme = self.settings.value("theme", "dark")
        self.apply_theme(theme)
        
    def toggle_theme(self):
        current_theme = self.settings.value("theme", "dark")
        new_theme = "light" if current_theme == "dark" else "dark"
        self.apply_theme(new_theme)
        
    def apply_theme(self, theme):
        if theme == "dark":
            style_file = QFile("style_dark.qss")
            self.ui.theme_btn.setIcon(QIcon(":/icon/icon/activity-feed-32.ico")) # Using placeholder, ideally sun icon
            self.ui.theme_btn.setToolTip("Mudar para tema claro")
        else:
            style_file = QFile("style_light.qss")
            self.ui.theme_btn.setIcon(QIcon(":/icon/icon/activity-feed-32.ico")) # Using placeholder, ideally moon icon
            self.ui.theme_btn.setToolTip("Mudar para tema escuro")
            
        if style_file.open(QFile.ReadOnly | QFile.Text):
            style_stream = QTextStream(style_file)
            QApplication.instance().setStyleSheet(style_stream.readAll())
            self.settings.setValue("theme", theme)
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
    @Slot(bool)
    def on_home_btn_1_toggled(self, checked):
        if checked: self.ui.stackedWidget.setCurrentIndex(0)
    
    @Slot(bool)
    def on_home_btn_2_toggled(self, checked):
        if checked: self.ui.stackedWidget.setCurrentIndex(0)

    @Slot(bool)
    def on_dashboard_btn_1_toggled(self, checked=True):  # Used by .clicked in sidebar_ui.py too
        if checked:
            # Lazy load the ProgramView
            if not self.program_view:
                self.program_view = ProgramView(self.width() - 90, self.height() - 70)
                self.program_view.create_views(config_ini.cam_qty)
                self.ui.stackedWidget.removeWidget(self.page_dashboard)
                self.ui.stackedWidget.insertWidget(1, self.program_view)
                self.page_dashboard = self.program_view
            self.ui.stackedWidget.setCurrentIndex(1)

    @Slot(bool)
    def on_dashboard_btn_2_toggled(self, checked=True):
        if checked:
            if not self.program_view:
                self.program_view = ProgramView(self.width() - 90, self.height() - 70)
                self.program_view.create_views(config_ini.cam_qty)
                self.ui.stackedWidget.removeWidget(self.page_dashboard)
                self.ui.stackedWidget.insertWidget(1, self.program_view)
                self.page_dashboard = self.program_view
            self.ui.stackedWidget.setCurrentIndex(1)

    @Slot(bool)
    def on_orders_btn_1_toggled(self, checked):
        if checked: self.ui.stackedWidget.setCurrentIndex(2)

    @Slot(bool)
    def on_orders_btn_2_toggled(self, checked):
        if checked: self.ui.stackedWidget.setCurrentIndex(2)

    @Slot(bool)
    def on_products_btn_1_toggled(self, checked):
        if checked: self.ui.stackedWidget.setCurrentIndex(3)

    @Slot(bool)
    def on_products_btn_2_toggled(self, checked):
        if checked: self.ui.stackedWidget.setCurrentIndex(3)

    @Slot(bool)
    def on_customers_btn_toggled(self, checked=True):
        if checked:
            if not self.inspection_view:
                self.inspection_view = InspectionView()
                self.ui.stackedWidget.removeWidget(self.page_customers)
                self.ui.stackedWidget.insertWidget(4, self.inspection_view)
                self.page_customers = self.inspection_view
            self.ui.stackedWidget.setCurrentIndex(4)

    def closeEvent(self, event):
        if self.inspection_view:
            self.inspection_view.shutdown()
        event.accept()

    


if __name__ == "__main__":
    app = QApplication(sys.argv)

    ## loading style file
    # with open("style.qss", "r") as style_file:
    #     style_str = style_file.read()
    # app.setStyleSheet(style_str)

    ## loading style file, Example 2
    # style_file = QFile("style.qss")
    # style_file.open(QFile.ReadOnly | QFile.Text)
    # style_stream = QTextStream(style_file)
    # app.setStyleSheet(style_stream.readAll())


    window = MainWindow()
    window.show()

    sys.exit(app.exec())