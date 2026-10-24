"""PySide6 replay-only desktop viewer; playback never steps the environment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QTimer
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from relay.runtime.replay_visuals import (
    DEFAULT_OVERLAYS,
    ReplayData,
    export_replay,
    render_replay_frame,
)


class ReplayWindow(QMainWindow):
    def __init__(self, replay: str | Path) -> None:
        super().__init__()
        self.data = ReplayData(replay)
        self.index = 0
        self.playing = False
        self.fps = 4.0
        communication_mode = self.data.environment_config["communication_mode"]
        self.setWindowTitle(f"RELAY {communication_mode} — {self.data.path.name}")
        self.resize(1400, 900)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance)
        self.image_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.image_label)
        self.inspection = QTextEdit(readOnly=True)
        self.inspection.setMinimumWidth(380)
        splitter = QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(self.inspection)
        splitter.setStretchFactor(0, 1)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        transport = QHBoxLayout()
        for label, callback in (
            ("|<", self._restart),
            ("<", self._previous),
            ("Play", self._toggle_play),
            (">", self._next),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            transport.addWidget(button)
            if label == "Play":
                self.play_button = button
        self.jump = QSpinBox(minimum=1, maximum=len(self.data), value=1)
        self.jump.valueChanged.connect(lambda value: self._seek(value - 1))
        transport.addWidget(QLabel("Tick"))
        transport.addWidget(self.jump)
        self.speed = QComboBox()
        for value in (0.25, 0.5, 1, 2, 4):
            self.speed.addItem(f"{value:g}×", value)
        self.speed.setCurrentText("1×")
        self.speed.currentIndexChanged.connect(self._speed_changed)
        transport.addWidget(QLabel("Speed"))
        transport.addWidget(self.speed)
        self.perspective = QComboBox()
        self.perspective.addItems(["global", *self.data.agents])
        self.perspective.currentTextChanged.connect(lambda _: self._render())
        transport.addWidget(QLabel("Perspective"))
        transport.addWidget(self.perspective)
        controls_layout.addLayout(transport)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, len(self.data) - 1)
        self.slider.valueChanged.connect(self._seek)
        controls_layout.addWidget(self.slider)
        overlay_row = QHBoxLayout()
        self.overlay_checks: dict[str, QCheckBox] = {}
        for name in sorted(DEFAULT_OVERLAYS):
            checkbox = QCheckBox(name.replace("_", " ").title())
            checkbox.setChecked(True)
            checkbox.toggled.connect(lambda _: self._render())
            overlay_row.addWidget(checkbox)
            self.overlay_checks[name] = checkbox
        controls_layout.addLayout(overlay_row)
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(splitter, stretch=1)
        layout.addWidget(controls)
        self.setCentralWidget(central)
        toolbar = QToolBar("Replay")
        self.addToolBar(toolbar)
        open_action = QAction("Open", self)
        open_action.triggered.connect(self._open)
        export_action = QAction("Export", self)
        export_action.triggered.connect(self._export)
        generate_action = QAction("Generate replay", self)
        generate_action.triggered.connect(self._generate)
        toolbar.addAction(open_action)
        toolbar.addAction(export_action)
        toolbar.addAction(generate_action)
        self._render()

    def _enabled_overlays(self) -> set[str]:
        return {name for name, checkbox in self.overlay_checks.items() if checkbox.isChecked()}

    def _render(self) -> None:
        image = render_replay_frame(
            self.data,
            self.index,
            perspective=self.perspective.currentText(),
            overlays=self._enabled_overlays(),
        ).convert("RGB")
        raw = image.tobytes("raw", "RGB")
        qimage = QImage(
            raw, image.width, image.height, image.width * 3, QImage.Format.Format_RGB888
        ).copy()
        self.image_label.setPixmap(QPixmap.fromImage(qimage))
        self.inspection.setPlainText(json.dumps(self.data.inspection(self.index), indent=2))
        self.slider.blockSignals(True)
        self.slider.setValue(self.index)
        self.slider.blockSignals(False)
        self.jump.blockSignals(True)
        self.jump.setValue(self.index + 1)
        self.jump.blockSignals(False)

    def _seek(self, index: int) -> None:
        self.index = max(0, min(len(self.data) - 1, index))
        self._render()

    def _restart(self) -> None:
        self._seek(0)

    def _previous(self) -> None:
        self._seek(self.index - 1)

    def _next(self) -> None:
        self._seek(self.index + 1)

    def _advance(self) -> None:
        if self.index >= len(self.data) - 1:
            self._toggle_play()
        else:
            self._next()

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_button.setText("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start(max(20, int(1000 / (self.fps * float(self.speed.currentData())))))
        else:
            self.timer.stop()

    def _speed_changed(self) -> None:
        if self.playing:
            self.timer.start(max(20, int(1000 / (self.fps * float(self.speed.currentData())))))

    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open RELAY replay", filter="Parquet (*.parquet)"
        )
        if path:
            self.replacement = ReplayWindow(path)
            self.replacement.show()
            self.close()

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export replay",
            filter="PNG (*.png);;GIF (*.gif);;MP4 (*.mp4)",
        )
        if not path:
            return
        try:
            export_replay(
                self.data,
                path,
                fps=self.fps,
                perspective=self.perspective.currentText(),
                overlays=self._enabled_overlays(),
            )
            QMessageBox.information(self, "Export complete", str(Path(path).resolve()))
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def _generate(self) -> None:
        config, _ = QFileDialog.getOpenFileName(
            self,
            "Select experiment config",
            filter="YAML (*.yaml)",
        )
        if not config:
            return
        checkpoint, _ = QFileDialog.getOpenFileName(
            self, "Select model checkpoint", filter="PyTorch checkpoint (*.pt)"
        )
        if not checkpoint:
            return
        self.generation_process = QProcess(self)
        self.generation_process.setProgram(sys.executable)
        self.generation_process.setArguments(
            [
                "-m",
                "relay.replay",
                "generate",
                "--config",
                config,
                "--checkpoint",
                checkpoint,
                "--episodes",
                "1",
            ]
        )
        self.generation_process.finished.connect(self._generation_finished)
        self.generation_process.start()

    def _generation_finished(self, exit_code: int) -> None:
        output = bytes(self.generation_process.readAllStandardOutput().data()).decode(
            errors="replace"
        )
        error = bytes(self.generation_process.readAllStandardError().data()).decode(
            errors="replace"
        )
        if exit_code == 0:
            QMessageBox.information(
                self,
                "Replay generation complete",
                "The deterministic evaluation completed. Open its replay artifact "
                "from the run directory.\n\n"
                + output[-1200:],
            )
        else:
            QMessageBox.critical(self, "Replay generation failed", error[-2000:])


def launch_viewer(replay: str | Path) -> int:
    application = QApplication.instance() or QApplication(sys.argv)
    window = ReplayWindow(replay)
    window.show()
    return application.exec()
