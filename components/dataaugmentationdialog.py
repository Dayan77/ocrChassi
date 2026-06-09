"""
Data augmentation in-place dialog.

Reads character crops from `model_train_dataset/<class>/*.png` and writes
augmented variants back to the same folders, so the recognition training
set grows on disk. The same augmentation function used at runtime
(`_augment_char_gray` in components.models) is reused here so generated
variants match the distribution the trainer sees.

Two modes:
  - Multiplier: for every source crop, generate N augmented copies.
  - Balance to target: for each class, generate enough variants so the
    class reaches `target_count` (capped per class to avoid blowing up the
    set; the trainer's runtime augmentation handles the rest).

Generated filenames are stamped with a per-run timestamp + an aug index,
so repeated runs never collide.
"""

import os
import time

import cv2
import numpy as np

from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QSpinBox, QRadioButton, QButtonGroup, QGroupBox,
    QFormLayout, QLineEdit, QProgressBar, QMessageBox, QPlainTextEdit,
    QSizePolicy,
)

from components.models import _augment_char_gray, list_class_folders, IMAGE_EXTS


class _AugWorker(QObject):
    progress = Signal(int, int)
    log = Signal(str)
    status = Signal(str)
    finished = Signal(dict)  # summary stats

    def __init__(self, root_dir, mode, multiplier, target_count, max_per_source, seed=42):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode  # 'multiplier' | 'balance'
        self.multiplier = int(multiplier)
        self.target_count = int(target_count)
        self.max_per_source = int(max_per_source)
        self.seed = int(seed)
        self._stop = False

    def stop(self):
        self._stop = True

    @Slot()
    def run(self):
        rng = np.random.default_rng(self.seed)
        # Include sub-second nonce so two runs within the same second still
        # produce unique filenames.
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        stats = {
            'classes_touched': 0,
            'classes_skipped_empty': 0,
            'sources_read': 0,
            'sources_unreadable': 0,
            'variants_written': 0,
            'per_class': {},
        }

        class_names = list_class_folders(self.root_dir)
        if not class_names:
            self.log.emit(f"Nenhuma subpasta de classe encontrada em {self.root_dir}.")
            self.finished.emit(stats)
            return

        # Gather counts per class. Only originals are eligible as augmentation
        # *sources* (files without `_aug` in the name); otherwise repeated
        # runs would augment their own outputs and grow exponentially. For
        # the balance-mode deficit calc, however, we count every image
        # already in the folder (augmenteds included) so target_count
        # reflects the real class size.
        sources_by_class = {}
        total_by_class = {}
        for cls in class_names:
            cls_dir = os.path.join(self.root_dir, cls)
            all_files = [
                f for f in sorted(os.listdir(cls_dir))
                if f.lower().endswith(IMAGE_EXTS)
            ]
            sources = [os.path.join(cls_dir, f) for f in all_files if "_aug" not in f]
            sources_by_class[cls] = sources
            total_by_class[cls] = len(all_files)

        total_to_generate = 0
        plan = {}
        for cls, files in sources_by_class.items():
            n_sources = len(files)
            n_existing = total_by_class[cls]
            if n_sources == 0:
                stats['classes_skipped_empty'] += 1
                plan[cls] = 0
                continue
            if self.mode == 'multiplier':
                planned = n_sources * self.multiplier
            else:  # balance
                deficit = max(0, self.target_count - n_existing)
                planned = deficit
            plan[cls] = planned
            total_to_generate += planned

        self.log.emit(
            f"Plano: {total_to_generate} variantes a gerar "
            f"em {len(class_names)} classes (modo: {self.mode})."
        )
        if total_to_generate == 0:
            self.status.emit("Nada a gerar com a configuração escolhida.")
            self.finished.emit(stats)
            return

        self.status.emit(f"0 / {total_to_generate} variantes geradas")
        self.progress.emit(0, total_to_generate)

        generated = 0
        for cls, files in sources_by_class.items():
            if self._stop:
                self.log.emit("Interrompido pelo usuário.")
                break
            planned = plan[cls]
            if planned == 0:
                continue
            stats['classes_touched'] += 1
            stats['per_class'][cls] = 0
            cls_dir = os.path.join(self.root_dir, cls)

            # In multiplier mode: each source contributes `self.multiplier` variants.
            # In balance mode: distribute `planned` variants across sources, capped
            # at max_per_source so we don't generate 200 from a single image.
            if self.mode == 'multiplier':
                per_source = [self.multiplier] * len(files)
            else:
                base = planned // len(files)
                rem = planned % len(files)
                per_source = [base + (1 if i < rem else 0) for i in range(len(files))]
                per_source = [min(self.max_per_source, n) for n in per_source]

            for src_path, n_variants in zip(files, per_source):
                if self._stop:
                    break
                if n_variants <= 0:
                    continue
                img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    stats['sources_unreadable'] += 1
                    continue
                stats['sources_read'] += 1
                base_name = os.path.splitext(os.path.basename(src_path))[0]

                for j in range(n_variants):
                    if self._stop:
                        break
                    variant = _augment_char_gray(img, rng)
                    out_name = f"{base_name}_aug{j}_{stamp}.png"
                    out_path = os.path.join(cls_dir, out_name)
                    cv2.imwrite(out_path, variant)
                    generated += 1
                    stats['variants_written'] += 1
                    stats['per_class'][cls] = stats['per_class'].get(cls, 0) + 1

                    if generated % 25 == 0 or generated == total_to_generate:
                        self.progress.emit(generated, total_to_generate)
                        self.status.emit(
                            f"{generated} / {total_to_generate} variantes — classe {cls}"
                        )

            self.log.emit(
                f"Classe {cls!r}: +{stats['per_class'].get(cls, 0)} variantes "
                f"(fonte: {len(files)} imgs)"
            )

        self.progress.emit(generated, max(total_to_generate, generated))
        self.status.emit(f"Concluído: {generated} variantes geradas.")
        self.finished.emit(stats)


class DataAugmentationDialog(QDialog):
    """Configure and run on-disk augmentation for the recognition dataset."""

    def __init__(self, model_data, parent=None):
        super().__init__(parent)
        self.model_data = model_data
        self.setWindowTitle("Data Augmentation — expandir dataset de reconhecimento")
        self.resize(900, 650)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)

        # Source folder
        cfg = QGroupBox("Configuração")
        form = QFormLayout(cfg)

        self.folder_edit = QLineEdit(getattr(model_data, "model_train_dataset", "") or "")
        browse = QPushButton("...")
        browse.clicked.connect(self._browse)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder_edit)
        folder_row.addWidget(browse)
        form.addRow("Pasta de classes:", folder_row)

        # Mode
        self.mode_multiplier = QRadioButton("Multiplicador — N variantes por imagem")
        self.mode_balance = QRadioButton(
            "Balancear classes — gerar até atingir um número-alvo por classe"
        )
        self.mode_multiplier.setChecked(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.mode_multiplier, 0)
        self.mode_group.addButton(self.mode_balance, 1)
        mode_box = QVBoxLayout()
        mode_box.addWidget(self.mode_multiplier)
        mode_box.addWidget(self.mode_balance)
        form.addRow("Modo:", mode_box)

        self.multiplier_spin = QSpinBox()
        self.multiplier_spin.setRange(1, 20)
        self.multiplier_spin.setValue(3)
        form.addRow("N variantes por imagem (multiplicador):", self.multiplier_spin)

        self.target_spin = QSpinBox()
        self.target_spin.setRange(1, 5000)
        self.target_spin.setValue(150)
        form.addRow("Alvo por classe (balanceamento):", self.target_spin)

        self.cap_spin = QSpinBox()
        self.cap_spin.setRange(1, 200)
        self.cap_spin.setValue(20)
        form.addRow("Máx. variantes por imagem (balanceamento):", self.cap_spin)

        # Actions
        self.run_btn = QPushButton("Gerar variantes")
        self.run_btn.clicked.connect(self._run)
        self.stop_btn = QPushButton("Parar")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        form.addRow(btn_row)

        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m  (%p%)")
        form.addRow(self.progress)

        self.status_label = QLabel("Pronto.")
        self.status_label.setStyleSheet(
            "font-weight: bold; padding: 4px; background: rgba(255,255,255,0.05);"
        )
        form.addRow(self.status_label)

        layout.addWidget(cfg)

        # Hint about what augmentations are applied
        hint = QLabel(
            "Aplica rotação ±8°, escala ±8%, translação ±5%, brilho/contraste leves "
            "e ruído gaussiano fraco. Mesmo conjunto usado pelo trainer em runtime — "
            "gerar em disco apenas multiplica visivelmente o dataset e permite "
            "inspecionar/curar pelo botão 'Revisar Dados Reconhecimento'."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #cccccc; padding: 4px;")
        layout.addWidget(hint)

        # Log
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setPlaceholderText("Log aparece aqui durante a geração.")
        layout.addWidget(self.log_view, 1)

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton("Fechar")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        self._thread = None
        self._worker = None

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Selecione a pasta de classes")
        if d:
            self.folder_edit.setText(d)

    def _run(self):
        folder = self.folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Pasta inválida", "Selecione uma pasta válida.")
            return

        mode = 'multiplier' if self.mode_multiplier.isChecked() else 'balance'
        self.log_view.clear()
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setMaximum(0)
        self.progress.setValue(0)
        self.status_label.setText("Iniciando…")

        self._thread = QThread()
        self._worker = _AugWorker(
            folder, mode,
            multiplier=self.multiplier_spin.value(),
            target_count=self.target_spin.value(),
            max_per_source=self.cap_spin.value(),
        )
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

    def _on_finished(self, stats):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        per_class = stats.get('per_class', {})
        lines = [
            f"Variantes geradas: {stats.get('variants_written', 0)}",
            f"Classes alteradas: {stats.get('classes_touched', 0)}",
            f"Classes vazias (puladas): {stats.get('classes_skipped_empty', 0)}",
            f"Imagens-fonte lidas: {stats.get('sources_read', 0)}",
            f"Imagens-fonte ilegíveis: {stats.get('sources_unreadable', 0)}",
        ]
        if per_class:
            lines.append("")
            lines.append("Por classe:")
            for k in sorted(per_class):
                lines.append(f"  {k}: +{per_class[k]}")
        summary = "\n".join(lines)
        self.log_view.appendPlainText("\n" + summary)
        QMessageBox.information(self, "Concluído", summary)
