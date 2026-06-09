"""
Modal dialogs to review the datasets produced by the data-prep pipelines:
  - RecognitionReviewDialog: review `model_train_dataset` (recog/<class>/*.png).
  - YoloReviewDialog: review `yolo_dataset_path/images/{train,val}/`, keeping
    each image's matching `labels/{train,val}/<name>.txt` in sync on
    move/delete.

Both reuse DatasetView's grid-per-class UI (thumbnail + context menu +
Apply Changes), so the user gets the same affordances they already know.
"""

import os
import shutil

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QSizePolicy

from components.datasetview import DatasetView


def _wrap_dialog(parent, title, view, width=1200, height=800):
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(8, 8, 8, 8)

    hint = QLabel(
        "Clique direito em um thumbnail para Mover/Flipar/Deletar. "
        "Use 'Apply Changes' para gravar as mudanças no disco."
    )
    hint.setWordWrap(True)
    layout.addWidget(hint)
    layout.addWidget(view)

    dialog.resize(width, height)
    return dialog


class RecognitionReviewDialog:
    """Factory helper: returns a QDialog wrapping a DatasetView pointed at
    the recognition-prep folder. Move/Delete/Flip all work as in the
    'Anotações' tab — the operations target PNG files only, no JSON sidecar
    exists for prepared crops so deletions are simpler.
    """

    @staticmethod
    def build(parent, model_train_dataset_path):
        view = DatasetView()
        if model_train_dataset_path and os.path.isdir(model_train_dataset_path):
            view.load_dataset(model_train_dataset_path)
        return _wrap_dialog(
            parent,
            f"Revisar Dados de Reconhecimento — {model_train_dataset_path}",
            view,
        )


class YoloDatasetView(DatasetView):
    """
    DatasetView variant for a YOLO dataset folder.

    Expected layout (under `yolo_root`):
        images/
          train/*.{png,jpg,jpeg}
          val/*.{png,jpg,jpeg}
        labels/
          train/*.txt
          val/*.txt

    We point the underlying DatasetView at `<yolo_root>/images`, which makes
    the existing "class folder per subfolder" UI display `train` and `val`
    as two groups. Move/Delete are then extended to also move/delete the
    matching label file in `<yolo_root>/labels/<group>/<name>.txt`.
    """

    def __init__(self, yolo_root, parent=None):
        super().__init__(parent)
        self.yolo_root = yolo_root
        self.images_root = os.path.join(yolo_root, "images")
        self.labels_root = os.path.join(yolo_root, "labels")

    def _label_path_for(self, image_path):
        """Return the absolute path of the YOLO label file matching `image_path`,
        or None if we cannot work out a sensible path."""
        if not self.labels_root:
            return None
        # image_path: .../images/<group>/<name>.<ext>
        group = os.path.basename(os.path.dirname(image_path))
        name, _ = os.path.splitext(os.path.basename(image_path))
        return os.path.join(self.labels_root, group, name + ".txt")

    def apply_changes(self):
        """Run the base DatasetView apply, then mirror moves/deletes on the
        corresponding label files. We intercept the lists before calling
        the parent so we can pair them with their labels."""
        moves = list(self.pending_moves)
        deletions = list(self.pending_deletions)

        # Let the base class handle the user confirmation + PNG operations.
        super().apply_changes()

        # `super().apply_changes()` cleared pending_* on success; if the user
        # cancelled, both lists will still be populated (untouched), so we
        # detect cancellation and bail out.
        if self.pending_moves or self.pending_deletions:
            return

        for move in moves:
            src_lbl = self._label_path_for(move["source"])
            dst_lbl = self._label_path_for(move["destination"])
            if src_lbl and dst_lbl and os.path.exists(src_lbl):
                os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)
                try:
                    os.rename(src_lbl, dst_lbl)
                except OSError as exc:
                    print(f"[YOLO review] could not move label {src_lbl}: {exc}")

        for path in deletions:
            lbl = self._label_path_for(path)
            if lbl and os.path.exists(lbl):
                try:
                    os.remove(lbl)
                except OSError as exc:
                    print(f"[YOLO review] could not delete label {lbl}: {exc}")


class YoloReviewDialog:
    """Factory helper that builds a YoloDatasetView and wraps it in a dialog."""

    @staticmethod
    def build(parent, yolo_dataset_path):
        view = YoloDatasetView(yolo_dataset_path)
        images_root = os.path.join(yolo_dataset_path, "images")
        if os.path.isdir(images_root):
            view.load_dataset(images_root)
        return _wrap_dialog(
            parent,
            f"Revisar Dados YOLO — {yolo_dataset_path}",
            view,
        )
