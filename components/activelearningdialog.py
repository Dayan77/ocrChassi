"""
Active-learning-lite dialog.

Given a folder of annotated images (image + sidecar .json with characters and
their bounding boxes), runs the current detector+recognizer pipeline on every
image and surfaces the cases worth a human look:

  - predicted label disagrees with the JSON annotation, or
  - prediction confidence is below the chosen threshold.

For each surfaced case the user can:
  - send the crop to the recognition train folder under the *expected* label,
    which directly grows underrepresented classes;
  - or skip it.

This is intentionally lightweight — no DB, no scoring history. Future passes
can layer on top.
"""

import os
import time

import cv2
import numpy as np
import json as _json

from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot, QSize
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QSlider, QCheckBox, QListWidget, QListWidgetItem,
    QWidget, QSizePolicy, QProgressBar, QMessageBox, QGroupBox,
    QFormLayout, QLineEdit, QSpinBox, QPlainTextEdit, QSplitter,
)

try:
    import torch
except ImportError:
    torch = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from components.models import SimpleCNN, EasyOCRCharNet, preprocess_char_crop


def _calculate_iou(box_ann, det_box):
    ax1, ay1 = box_ann['x'], box_ann['y']
    ax2, ay2 = ax1 + box_ann['w'], ay1 + box_ann['h']
    bx1, by1, bx2, by2 = det_box
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    a = (ax2 - ax1) * (ay2 - ay1)
    b = (bx2 - bx1) * (by2 - by1)
    union = a + b - inter
    return inter / float(union) if union > 0 else 0.0


def _preprocess_crop(gray_img, x1, y1, x2, y2, target_w, target_h):
    """Pad-to-square + resize + CLAHE — identical to what the trainer's
    prepare_recognition_dataset and the inference path do, so the
    distribution of crops fed to the CNN matches what it was trained on."""
    crop = gray_img[int(y1):int(y2), int(x1):int(x2)]
    return preprocess_char_crop(crop, target_h, target_w, apply_clahe=True)


class _Worker(QObject):
    progress = Signal(int, int)  # current, total
    log = Signal(str)
    status = Signal(str)         # single-line live status
    finished = Signal(list)  # list of dicts

    def __init__(self, model_data, annotation_dir, conf_threshold):
        super().__init__()
        self.model_data = model_data
        self.annotation_dir = annotation_dir
        self.conf_threshold = float(conf_threshold)
        self._stop = False

    def stop(self):
        self._stop = True

    @Slot()
    def run(self):
        if not torch:
            self.log.emit("PyTorch indisponível.")
            self.finished.emit([])
            return
        if YOLO is None:
            self.log.emit("ultralytics indisponível.")
            self.finished.emit([])
            return

        md = self.model_data
        device = torch.device("cuda" if torch.cuda.is_available() else
                              ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
        try:
            detector = YOLO(md.detector_model_path)
            try:
                detector.to(device)
            except Exception:
                pass
        except Exception as exc:
            self.log.emit(f"Falha ao carregar detector: {exc}")
            self.finished.emit([])
            return

        class_names = sorted(md.model_classes)
        char_to_idx = {c: i for i, c in enumerate(class_names)}
        img_h, img_w = int(md.image_height), int(md.image_width)

        arch = getattr(md, "architecture", "SimpleCNN") or "SimpleCNN"
        model_cls = EasyOCRCharNet if (arch == "EasyOCRCharNet" and EasyOCRCharNet is not None) else SimpleCNN
        cnn = model_cls(len(class_names), img_h, img_w).to(device)
        self.log.emit(f"Reconhecedor: {arch}")
        weights_path = os.path.splitext(md.encoder_filename)[0] + ".pth"
        if not os.path.exists(weights_path):
            self.log.emit(f"Pesos PyTorch não encontrados em {weights_path}.")
            self.finished.emit([])
            return
        try:
            cnn.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        except Exception as exc:
            self.log.emit(f"Falha ao carregar CNN: {exc}")
            self.finished.emit([])
            return
        cnn.eval()

        image_files = sorted([
            os.path.join(self.annotation_dir, f)
            for f in os.listdir(self.annotation_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        results = []
        total = len(image_files)
        t0 = time.time()
        self.log.emit(f"Analisando {total} imagens em {self.annotation_dir}…")
        self.status.emit(f"0/{total} — preparando…")
        self.progress.emit(0, total)

        for i, img_path in enumerate(image_files):
            if self._stop:
                self.log.emit("Interrompido pelo usuário.")
                break
            base = os.path.basename(img_path)
            self.status.emit(
                f"{i + 1}/{total} — {base} — {len(results)} casos suspeitos"
            )

            json_path = img_path + ".json"
            if not os.path.exists(json_path):
                self.progress.emit(i + 1, total)
                continue
            try:
                with open(json_path) as f:
                    annotations = _json.load(f)
            except Exception:
                self.progress.emit(i + 1, total)
                continue
            if not annotations:
                self.progress.emit(i + 1, total)
                continue
            bgr = cv2.imread(img_path)
            if bgr is None:
                self.progress.emit(i + 1, total)
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            try:
                det = detector(bgr, verbose=False)
                boxes = det[0].boxes.xyxy.cpu().numpy()
            except Exception as exc:
                self.log.emit(f"Detector falhou em {base}: {exc}")
                self.progress.emit(i + 1, total)
                continue

            cases_this_image = 0
            ann_list = list(annotations.values())
            for box in boxes:
                best = None
                best_iou = 0.3
                for ann in ann_list:
                    iou = _calculate_iou(ann['box'], box)
                    if iou > best_iou:
                        best_iou = iou
                        best = ann
                if not best:
                    continue
                expected = best['char']
                if expected == '?' or expected not in char_to_idx:
                    continue

                crop = _preprocess_crop(gray, box[0], box[1], box[2], box[3], img_w, img_h)
                if crop is None:
                    continue
                arr = crop.astype(np.float32) / 255.0
                tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = cnn(tensor)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred_idx = torch.max(probs, dim=1)
                predicted = class_names[pred_idx.item()]
                confidence = conf.item() * 100.0

                mismatch = predicted != expected
                low_conf = confidence < self.conf_threshold
                if mismatch or low_conf:
                    results.append({
                        'image_path': img_path,
                        'expected': expected,
                        'predicted': predicted,
                        'confidence': confidence,
                        'mismatch': mismatch,
                        'low_conf': low_conf,
                        'crop': crop,
                    })
                    cases_this_image += 1

            # Throttled log: only when we find something or every 10 images
            if cases_this_image > 0 or (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total - (i + 1)) / rate if rate > 0 else 0
                self.log.emit(
                    f"[{i + 1}/{total}] {base}: +{cases_this_image} caso(s) "
                    f"(total {len(results)}, "
                    f"{rate:.1f} img/s, ETA {eta:.0f}s)"
                )
            self.progress.emit(i + 1, total)

        elapsed = time.time() - t0
        self.log.emit(
            f"Análise concluída em {elapsed:.1f}s. "
            f"{len(results)} casos para revisão."
        )
        self.status.emit(f"Concluído: {len(results)} casos em {total} imagens.")
        self.finished.emit(results)


class ActiveLearningDialog(QDialog):
    """Run inference on annotated images and surface low-confidence /
    disagreement cases so the user can grow the train set where it matters."""

    def __init__(self, model_data, parent=None):
        super().__init__(parent)
        self.model_data = model_data
        self.setWindowTitle("Active Learning — revisão por baixa confiança")
        self.resize(1100, 750)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)

        # Config block
        cfg_group = QGroupBox("Configuração")
        cfg_form = QFormLayout(cfg_group)
        self.folder_edit = QLineEdit(getattr(model_data, "annotation_dataset_path", "") or "")
        browse_btn = QPushButton("...")
        browse_btn.clicked.connect(self._browse)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder_edit)
        folder_row.addWidget(browse_btn)
        cfg_form.addRow("Pasta de anotação:", folder_row)

        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(70)
        self.conf_label = QLabel("Limiar de confiança: 70%")
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_label.setText(f"Limiar de confiança: {v}%")
        )
        conf_row = QHBoxLayout()
        conf_row.addWidget(self.conf_slider)
        conf_row.addWidget(self.conf_label)
        cfg_form.addRow(conf_row)

        self.run_btn = QPushButton("Analisar")
        self.run_btn.clicked.connect(self._run)
        self.stop_btn = QPushButton("Parar")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        cfg_form.addRow(btn_row)

        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m  (%p%)")
        self.progress.setTextVisible(True)
        cfg_form.addRow(self.progress)

        self.status_label = QLabel("Pronto.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "font-weight: bold; padding: 4px; background: rgba(255,255,255,0.05);"
        )
        cfg_form.addRow(self.status_label)

        layout.addWidget(cfg_group)

        # Legend explaining how to read the log entries — collapsible, default
        # visible so first-time users see it; can be hidden to free vertical
        # space once the format is familiar.
        self.legend_toggle = QPushButton("▾  Como ler o log")
        self.legend_toggle.setCheckable(True)
        self.legend_toggle.setChecked(True)
        self.legend_toggle.setStyleSheet(
            "text-align: left; padding: 4px 6px; font-weight: bold;"
        )
        self.legend_toggle.clicked.connect(self._toggle_legend)
        layout.addWidget(self.legend_toggle)

        self.legend_label = QLabel(
            "<div style='padding:6px 10px; line-height:1.5;'>"
            "Cada linha tem o formato:<br>"
            "<code>[N/M] arquivo.png: +X caso(s) (total Y, Z img/s, ETA Ts)</code><br>"
            "<table cellpadding='2' style='margin-top:4px;'>"
            "<tr><td><b>[N/M]</b></td><td>imagem N de M já processadas</td></tr>"
            "<tr><td><b>arquivo.png</b></td><td>arquivo analisado nesse passo</td></tr>"
            "<tr><td><b>+X caso(s)</b></td><td>predições suspeitas <i>nessa imagem</i> "
            "(erradas ou abaixo do limiar de confiança)</td></tr>"
            "<tr><td><b>total Y</b></td><td>acumulado de casos suspeitos no batch inteiro</td></tr>"
            "<tr><td><b>Z img/s</b></td><td>velocidade média de processamento</td></tr>"
            "<tr><td><b>ETA Ts</b></td><td>estimativa de tempo restante em segundos</td></tr>"
            "</table>"
            "<p style='margin-top:6px;'>Linhas aparecem <b>a cada 10 imagens</b> ou "
            "sempre que uma imagem produz ≥1 caso suspeito.<br>"
            "<b>0 casos no final = o modelo acertou tudo</b> acima do limiar — "
            "suba o slider de confiança (ex. 90%) para também listar predições corretas "
            "mas com pouca certeza (boas candidatas para aumentar o treino).</p>"
            "</div>"
        )
        self.legend_label.setTextFormat(Qt.TextFormat.RichText)
        self.legend_label.setWordWrap(True)
        self.legend_label.setStyleSheet(
            "background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); "
            "border-radius: 4px;"
        )
        layout.addWidget(self.legend_label)

        # Live log + result list split — log is the immediate feedback channel
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setPlaceholderText(
            "Log da análise aparecerá aqui em tempo real durante a execução."
        )
        self.log_view.setMinimumHeight(120)
        splitter.addWidget(self.log_view)

        self.results_list = QListWidget()
        self.results_list.setSpacing(4)
        splitter.addWidget(self.results_list)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        layout.addWidget(splitter, 1)

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton("Fechar")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        self._thread = None
        self._worker = None

    def _toggle_legend(self):
        visible = self.legend_toggle.isChecked()
        self.legend_label.setVisible(visible)
        self.legend_toggle.setText(("▾  " if visible else "▸  ") + "Como ler o log")

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Selecione a pasta de anotação")
        if d:
            self.folder_edit.setText(d)

    def _run(self):
        folder = self.folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Pasta inválida", "Selecione uma pasta válida.")
            return
        self.results_list.clear()
        self.log_view.clear()
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)
        self.progress.setMaximum(0)  # busy mode until total is known
        self.status_label.setText("Carregando modelos…")

        self._thread = QThread()
        self._worker = _Worker(self.model_data, folder, self.conf_slider.value())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._on_log)
        self._worker.status.connect(self._on_status)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _stop(self):
        if self._worker:
            self._worker.stop()
        self.stop_btn.setEnabled(False)

    def _on_progress(self, cur, total):
        if total > 0:
            self.progress.setMaximum(total)
            self.progress.setValue(cur)

    def _on_log(self, msg):
        self.log_view.appendPlainText(msg)

    def _on_status(self, msg):
        self.status_label.setText(msg)

    def _on_finished(self, items):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        for item in items:
            row = QListWidgetItem(self.results_list)
            w = _CaseRow(item, self.model_data, self)
            w.removed.connect(lambda r=row: self._remove_row(r))
            row.setSizeHint(QSize(900, 84))
            self.results_list.addItem(row)
            self.results_list.setItemWidget(row, w)
        if not items:
            self.status_label.setText("Nenhum caso suspeito encontrado.")
        else:
            self.status_label.setText(f"{len(items)} casos listados.")

    def _remove_row(self, item):
        row = self.results_list.row(item)
        if row >= 0:
            self.results_list.takeItem(row)


class _CaseRow(QWidget):
    """One row: thumbnail + expected/predicted/conf + action buttons.

    Emits `removed` when the user requests deletion of the source image (the
    row no longer represents valid data and the parent should drop it).
    """

    removed = Signal()

    def __init__(self, case, model_data, parent=None):
        super().__init__(parent)
        self.case = case
        self.model_data = model_data
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        thumb = QLabel()
        crop = case['crop']
        h, w = crop.shape[:2]
        qimg = QImage(crop.data, w, h, w, QImage.Format.Format_Grayscale8)
        thumb.setPixmap(QPixmap.fromImage(qimg).scaled(
            56, 56, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        thumb.setFixedSize(56, 56)
        layout.addWidget(thumb)

        info = QLabel(
            f"<b>{os.path.basename(case['image_path'])}</b> &nbsp;"
            f"esperado: <span style='color: #4cd964;'>{case['expected']}</span> &nbsp;"
            f"previsto: <span style='color: {'#ff4d4d' if case['mismatch'] else '#f4c542'};'>"
            f"{case['predicted']}</span> &nbsp;"
            f"conf: {case['confidence']:.1f}%"
        )
        info.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(info, 1)

        save_btn = QPushButton("Salvar como esperado")
        save_btn.setToolTip("Copia este crop para train/<esperado>/ como nova amostra.")
        save_btn.clicked.connect(self._save_to_train)
        layout.addWidget(save_btn)

        del_ann_btn = QPushButton("Apagar anotação")
        del_ann_btn.setToolTip(
            "Remove apenas a anotação deste caractere do JSON da imagem "
            "(mantém a imagem e demais caracteres)."
        )
        del_ann_btn.clicked.connect(self._delete_annotation)
        layout.addWidget(del_ann_btn)

        del_img_btn = QPushButton("Apagar imagem")
        del_img_btn.setToolTip(
            "Remove a imagem inteira e seu JSON da pasta de anotação. "
            "Use quando a imagem é defeituosa/inaproveitável."
        )
        del_img_btn.setStyleSheet("color: #ff4d4d;")
        del_img_btn.clicked.connect(self._delete_image)
        layout.addWidget(del_img_btn)

        skip_btn = QPushButton("Pular")
        skip_btn.clicked.connect(self._mark_skipped)
        layout.addWidget(skip_btn)

        self._info = info

    def _save_to_train(self):
        train_dir = getattr(self.model_data, "model_train_dataset", "")
        if not train_dir:
            QMessageBox.warning(self, "Sem pasta", "model_train_dataset não está configurado.")
            return
        target_dir = os.path.join(train_dir, self.case['expected'])
        os.makedirs(target_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        fname = f"al_{ts}_{os.path.splitext(os.path.basename(self.case['image_path']))[0]}.png"
        out = os.path.join(target_dir, fname)
        try:
            cv2.imwrite(out, self.case['crop'])
            self._info.setText(self._info.text() + " &nbsp; <i style='color:#4cd964;'>salvo</i>")
        except Exception as exc:
            QMessageBox.warning(self, "Falha ao salvar", str(exc))

    def _delete_annotation(self):
        """Remove this character's entry from the source JSON, keeping the
        rest of the annotations intact. Uses IoU match on the original box to
        find the right entry (since we don't carry roi_id in `case`)."""
        img_path = self.case['image_path']
        json_path = img_path + ".json"
        if not os.path.exists(json_path):
            QMessageBox.warning(self, "Sem anotação", f"JSON não encontrado: {json_path}")
            return
        confirm = QMessageBox.question(
            self,
            "Confirmar",
            f"Remover a anotação do caractere '{self.case['expected']}' "
            f"em {os.path.basename(img_path)}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            with open(json_path) as f:
                annotations = _json.load(f)
        except Exception as exc:
            QMessageBox.warning(self, "JSON inválido", str(exc))
            return

        # Match by expected char + first hit; for richer matching we'd need
        # to thread the original ann key through `case`, but in practice a
        # repeated (char, box) tuple is rare enough that this works.
        target_key = None
        expected = self.case['expected']
        for k, v in annotations.items():
            if v.get('char') == expected:
                target_key = k
                break
        if target_key is None:
            QMessageBox.information(
                self,
                "Não encontrado",
                "Não consegui localizar a anotação correspondente. "
                "Verifique manualmente.",
            )
            return
        annotations.pop(target_key)
        try:
            with open(json_path, "w") as f:
                _json.dump(annotations, f, indent=4)
            self._info.setText(self._info.text() + " &nbsp; <i style='color:#f4c542;'>anotação removida</i>")
        except Exception as exc:
            QMessageBox.warning(self, "Falha ao gravar", str(exc))

    def _delete_image(self):
        img_path = self.case['image_path']
        json_path = img_path + ".json"
        confirm = QMessageBox.question(
            self,
            "Confirmar exclusão",
            f"Apagar permanentemente:\n  {img_path}\n  {json_path}\n\n"
            "Outros caracteres dessa imagem nesta lista ficarão órfãos.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
            if os.path.exists(json_path):
                os.remove(json_path)
            self.removed.emit()
        except Exception as exc:
            QMessageBox.warning(self, "Falha ao apagar", str(exc))

    def _mark_skipped(self):
        self._info.setText(self._info.text() + " &nbsp; <i style='color:#888;'>pulado</i>")
