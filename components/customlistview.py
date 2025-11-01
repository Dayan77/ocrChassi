
import sys
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QListWidget,
    QListWidgetItem,
    QWidget,
    QHBoxLayout,
    QLabel,
    QCheckBox
)

class CustomItemWidget(QWidget):
    def __init__(self, image, text, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)

        # Imagem
        self.image_label = QLabel()
        #pixmap = QPixmap(image_path)
        self.image_label.setPixmap(image) #pixmap.scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # Texto
        self.text_label = QLabel(text)

        # Checkbox
        self.checkbox = QCheckBox()

        layout.addWidget(self.checkbox)
        layout.addWidget(self.text_label)
        layout.addWidget(self.image_label)
        
        # layout.addStretch() # Empurra os widgets para a esquerda

        self.setLayout(layout)


class CustomListView(QWidget):
    def __init__(self, title):
        super().__init__()
        self.setWindowTitle(title)#"Lista com Imagem, Texto e Checkbox")
        
        self.list_widget = QListWidget(self)
        self.list_widget.setFixedHeight(310)
        self.list_widget.setFrameStyle
        self.list_widget.setStyleSheet("""
            /* Estilo para o widget QListWidget */
            QListWidget {
                background-color: #2e2e2e;
                border: 2px solid #555;
                border-radius: 8px;
                color: #f0f0f0;
                padding: 5px;
            }
            
            /* Estilo para cada item da lista (mesmo os que não são widgets) */
            QListWidget::item {
                border-bottom: 1px solid #444;
                padding: 8px;
            }

            /* Estilo para itens pares */
            QListWidget::item:alternate {
                background-color: #383838;
            }

            /* Estilo para item selecionado */
            QListWidget::item:selected {
                background-color: #4a4a4a;
                color: #ffffff;
                border-left: 4px solid #00aaff;
            }
            
            /* Estilo para o widget personalizado dentro do QListWidget */
            QListWidget QWidget#customItemWidget {
                background-color: #4a4a4a;
            }

            /* Estilo para o QLabel dentro do widget personalizado */
            QListWidget QWidget#customItemWidget QLabel {
                color: #f0f0f0;
                background-color: #4a4a4a;
            }
            
            /* Estilo para o checkbox */
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            
        """)
        self.list_widget.setAlternatingRowColors(True)


        # Conectar sinal para detectar mudanças no estado do checkbox
        self.list_widget.itemChanged.connect(self.on_item_changed)
        
        # Adicionar itens à lista
        # self.add_list_item("path/to/image1.png", "Item 1")
        # self.add_list_item("path/to/image2.png", "Item 2")
        # self.add_list_item("path/to/image3.png", "Item 3")
    
    def add_list_item(self, image, text):
        # Crie o item da lista
        list_item = QListWidgetItem(self.list_widget)
        
        # Crie o widget personalizado
        item_widget = CustomItemWidget(image, text)
        
        # Defina o tamanho do item da lista para acomodar o widget
        list_item.setSizeHint(item_widget.sizeHint())
        # # Style the list to be partially transparent and smaller
        # list_item.setStyleSheet("""
        #     QListWidgetItem {
        #         background-color: rgba(0, 0, 0, 10); /* Semi-transparent black */
        #         color: white;                       
        #         border: 2px solid white;
        #         border-radius: 5px;
        #         padding: 5px;
        #     }
        # """)
        
        # Associe o widget personalizado ao item
        self.list_widget.addItem(list_item)
        self.list_widget.setItemWidget(list_item, item_widget)
        
    def on_item_changed(self, item):
        # Este método lida com a mudança de estado, caso você prefira não usar o widget personalizado
        # Para o nosso caso, com o widget personalizado, você precisaria de uma conexão de sinal do checkbox.
        pass