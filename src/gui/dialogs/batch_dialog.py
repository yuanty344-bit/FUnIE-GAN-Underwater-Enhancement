"""
批量处理进度对话框
"""
import logging
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (
    QProgressDialog, QMessageBox, QApplication, QInputDialog, QLineEdit
)

from ...i18n import tr
from ...utils.file_io import get_image_files, read_image, write_image

logger = logging.getLogger(__name__)


class BatchProcessDialog(QProgressDialog):
    """批量处理进度对话框"""

    def __init__(
        self,
        parent,
        input_folder: str,
        enhancer,
        output_dir: Path,
        naming_pattern: str = "{name}_enhanced"
    ):
        super().__init__(tr("批量处理中..."), tr("取消"), 0, 100, parent)

        self.enhancer = enhancer
        self.output_dir = output_dir
        self.naming_pattern = naming_pattern
        self.is_cancelled = False

        self.image_files = get_image_files(input_folder)
        self.total = len(self.image_files)

        if self.total == 0:
            QMessageBox.information(parent, tr("提示"), tr("文件夹中没有找到图像文件"))
            self.close()
            return

        self.setMaximum(self.total)
        self.setWindowTitle(tr("批量处理 (") + str(self.total) + tr(" 个文件)"))
        self.setMinimumDuration(0)
        self.show()
        self._process_batch()

    def _format_output_name(self, input_stem: str, input_suffix: str) -> str:
        now = datetime.now()
        date_str = now.strftime('%Y%m%d')
        time_str = now.strftime('%H%M%S')
        return (
            self.naming_pattern
            .replace("{name}", input_stem)
            .replace("{date}", date_str)
            .replace("{time}", time_str)
            .replace("{ext}", input_suffix)
        ) + input_suffix

    def _process_batch(self) -> None:
        now = datetime.now()
        batch_dir = self.output_dir / f"batch_{now.strftime('%Y%m%d_%H%M%S')}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        for i, path in enumerate(self.image_files):
            if self.wasCanceled():
                self.is_cancelled = True
                break

            self.setLabelText(tr("处理: ") + path.name)
            self.setValue(i)
            QApplication.processEvents()

            if self.wasCanceled():
                self.is_cancelled = True
                break

            try:
                image = read_image(path)
                enhanced = self.enhancer.enhance(image)
                output_name = self._format_output_name(path.stem, path.suffix)
                output_path = batch_dir / output_name
                write_image(enhanced, output_path)
                success_count += 1
            except Exception as e:
                logger.error(f"处理失败 {path}: {e}")

        self.setValue(self.total)
        self.close()

        QMessageBox.information(
            self.parent(), tr("完成"),
            tr("批量处理完成\n成功: ") + f"{success_count}/{self.total}" + tr("\n保存至: ") + str(batch_dir)
        )
