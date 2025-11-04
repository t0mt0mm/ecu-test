import math
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import can
import cantools
from PyQt5.QtCore import QObject, QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


OUTPUT_CHANNELS = [f"HS{i}" for i in range(1, 6)]
INPUT_CHANNELS = [
    "DI1",
    "AI1",
    "DI2",
    "AI2",
    "AI3",
]


@dataclass
class OutputState:
    enabled: bool
    pwm: float  # 0.0 - 100.0 percent
    current: float


@dataclass
class ConnectionSettings:
    dbc_path: str
    bustype: str
    channel: str
    bitrate: int


class BackendError(Exception):
    """Raised when a backend cannot complete the requested action."""


class BackendBase:
    """Common interface for ECU backends."""

    name: str = "Base"

    def start(self) -> None:
        """Initialize backend resources."""

    def stop(self) -> None:
        """Tear down backend resources."""

    def set_output(self, channel: str, enabled: bool, pwm: float) -> None:
        raise NotImplementedError

    def read_outputs(self) -> Dict[str, OutputState]:
        raise NotImplementedError

    def read_inputs(self) -> Dict[str, float]:
        raise NotImplementedError

    def update(self, dt: float) -> None:
        """Advance internal simulation and poll hardware."""


class DummyBackend(BackendBase):
    name = "Dummy"

    def __init__(self) -> None:
        self._outputs: Dict[str, OutputState] = {
            ch: OutputState(enabled=False, pwm=0.0, current=0.0) for ch in OUTPUT_CHANNELS
        }
        self._analog_inputs: Dict[str, float] = {"AI1": 0.0, "AI2": 5.0, "AI3": -2.5}
        self._digital_states: Dict[str, bool] = {"DI1": False, "DI2": True}
        self._time = 0.0
        self._tau = 0.5  # time constant for first-order response

    def start(self) -> None:
        self._time = 0.0

    def stop(self) -> None:
        pass

    def set_output(self, channel: str, enabled: bool, pwm: float) -> None:
        if channel not in self._outputs:
            return
        pwm = max(0.0, min(100.0, pwm))
        state = self._outputs[channel]
        state.enabled = enabled
        state.pwm = pwm

    def read_outputs(self) -> Dict[str, OutputState]:
        return {ch: OutputState(s.enabled, s.pwm, s.current) for ch, s in self._outputs.items()}

    def read_inputs(self) -> Dict[str, float]:
        inputs: Dict[str, float] = {}
        inputs.update(self._digital_states)
        inputs.update(self._analog_inputs)
        return {name: float(value) for name, value in inputs.items()}

    def update(self, dt: float) -> None:
        self._time += dt
        # Update digital patterns
        self._digital_states["DI1"] = math.sin(self._time * math.pi / 2) > 0
        self._digital_states["DI2"] = (int(self._time) % 2) == 0

        # Update analog signals
        self._analog_inputs["AI1"] = 5.0 + 2.0 * math.sin(self._time)
        self._analog_inputs["AI2"] = 2.5 + 0.5 * (self._time % 5)
        self._analog_inputs["AI3"] = -1.0 + 0.2 * random.uniform(-1.0, 1.0)

        # Update outputs currents using first-order lag
        for ch, state in self._outputs.items():
            target = 8.0 * (state.pwm / 100.0)
            if not state.enabled:
                target = 0.0
            noise = random.uniform(-0.1, 0.1)
            delta = (target - state.current) * (dt / self._tau)
            state.current += delta + noise
            state.current = max(0.0, state.current)


class RealBackend(QObject, BackendBase):
    name = "Real"

    status_updated = pyqtSignal()

    def __init__(self, settings: ConnectionSettings) -> None:
        QObject.__init__(self)
        self.settings = settings
        self._outputs: Dict[str, OutputState] = {
            ch: OutputState(enabled=False, pwm=0.0, current=0.0) for ch in OUTPUT_CHANNELS
        }
        self._inputs: Dict[str, float] = {ch: 0.0 for ch in INPUT_CHANNELS}
        self._db = None
        self._bus: Optional[can.BusABC] = None
        self._notifier: Optional[can.Notifier] = None
        self._write_message = None
        self._status_message = None
        self._lock = threading.Lock()
        self._listener: Optional["_StatusListener"] = None

    def start(self) -> None:
        if not os.path.exists(self.settings.dbc_path):
            raise BackendError(f"DBC file not found: {self.settings.dbc_path}")

        try:
            self._db = cantools.database.load_file(self.settings.dbc_path)
        except (OSError, cantools.database.errors.ParseError) as exc:
            raise BackendError(f"Failed to load DBC: {exc}")

        try:
            self._write_message = self._db.get_message_by_name("QM_High_side_output_write")
            self._status_message = self._db.get_message_by_name("QM_High_side_output_status")
        except KeyError as exc:
            raise BackendError(f"Required message missing in DBC: {exc}")

        try:
            bus_kwargs = {
                "bustype": self.settings.bustype,
                "channel": self.settings.channel,
            }
            if self.settings.bitrate > 0:
                bus_kwargs["bitrate"] = self.settings.bitrate
            self._bus = can.interface.Bus(**bus_kwargs)
        except (can.CanError, OSError, ValueError) as exc:
            raise BackendError(f"Failed to connect to CAN bus: {exc}")

        self._listener = _StatusListener(self)
        self._notifier = can.Notifier(self._bus, [self._listener])

    def stop(self) -> None:
        if self._notifier is not None:
            self._notifier.stop()
        self._notifier = None
        self._listener = None
        if self._bus is not None:
            try:
                self._bus.shutdown()
            except can.CanError:
                pass
        self._bus = None

    def set_output(self, channel: str, enabled: bool, pwm: float) -> None:
        if channel not in self._outputs:
            return
        if self._bus is None or self._write_message is None:
            raise BackendError("CAN bus is not initialized")

        pwm = max(0.0, min(100.0, pwm))
        with self._lock:
            state = self._outputs[channel]
            state.enabled = enabled
            state.pwm = pwm

            try:
                data = self._build_payload()
                message_data = self._write_message.encode(data)
                tx_msg = can.Message(
                    arbitration_id=self._write_message.frame_id,
                    data=message_data,
                    is_extended_id=self._write_message.is_extended_frame,
                )
                self._bus.send(tx_msg)
            except (ValueError, can.CanError) as exc:
                raise BackendError(f"Failed to send command: {exc}")

    def read_outputs(self) -> Dict[str, OutputState]:
        with self._lock:
            return {ch: OutputState(s.enabled, s.pwm, s.current) for ch, s in self._outputs.items()}

    def read_inputs(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._inputs)

    def update(self, dt: float) -> None:
        # Hardware polling is event-driven via CAN listener.
        _ = dt

    def _build_payload(self) -> Dict[str, float]:
        payload: Dict[str, float] = {}
        for index, channel in enumerate(OUTPUT_CHANNELS, start=1):
            state = self._outputs[channel]
            select_name = f"hs_out{index:02d}_select"
            mode_name = f"hs_out{index:02d}_mode"
            pwm_name = f"hs_out{index:02d}_value_pwm"
            payload[select_name] = 1 if state.enabled else 0
            payload[mode_name] = 1 if state.enabled else 0
            payload[pwm_name] = state.pwm
        return payload

    def _handle_status(self, message: can.Message) -> None:
        if self._status_message is None:
            return
        if message.arbitration_id != self._status_message.frame_id:
            return
        try:
            decoded = self._status_message.decode(message.data, decode_choices=False, scaling=True)
        except (ValueError, KeyError):
            return

        with self._lock:
            for index, channel in enumerate(OUTPUT_CHANNELS, start=1):
                current_key = f"hs_out{index:02d}_current"
                pwm_key = f"hs_out{index:02d}_pwm"
                if current_key in decoded:
                    current_ma = float(decoded[current_key])
                    self._outputs[channel].current = max(0.0, current_ma / 1000.0)
                if pwm_key in decoded:
                    self._outputs[channel].pwm = float(decoded[pwm_key])
            # Expose PWM of status as inputs for visibility
            for index in range(1, len(OUTPUT_CHANNELS) + 1):
                current_key = f"hs_out{index:02d}_current"
                input_key = f"AI{index}"
                if current_key in decoded and input_key in self._inputs:
                    self._inputs[input_key] = float(decoded[current_key]) / 1000.0

        self.status_updated.emit()


class _StatusListener(can.Listener):
    def __init__(self, backend: "RealBackend") -> None:
        super().__init__()
        self._backend = backend

    def on_message_received(self, message: can.Message) -> None:  # type: ignore[override]
        self._backend._handle_status(message)


class OutputWidget(QWidget):
    def __init__(self, channel: str, callback):
        super().__init__()
        self.channel = channel
        self.callback = callback

        layout = QHBoxLayout()

        self.toggle = QCheckBox(channel)
        self.toggle.stateChanged.connect(self._state_changed)
        layout.addWidget(self.toggle)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._slider_changed)
        layout.addWidget(self.slider)

        self.value_label = QLabel("0 %")
        layout.addWidget(self.value_label)

        self.current_label = QLabel("0.00 A")
        layout.addWidget(self.current_label)

        self.setLayout(layout)

    def _state_changed(self, state: int) -> None:
        self.callback(self.channel, state == 2, self.slider.value())

    def _slider_changed(self, value: int) -> None:
        self.value_label.setText(f"{value} %")
        self.callback(self.channel, self.toggle.isChecked(), float(value))

    def update_state(self, output: OutputState) -> None:
        if self.toggle.isChecked() != output.enabled:
            self.toggle.blockSignals(True)
            self.toggle.setChecked(output.enabled)
            self.toggle.blockSignals(False)
        if self.slider.value() != int(output.pwm):
            self.slider.blockSignals(True)
            self.slider.setValue(int(output.pwm))
            self.slider.blockSignals(False)
            self.value_label.setText(f"{int(output.pwm)} %")
        self.current_label.setText(f"{output.current:.2f} A")


class InputWidget(QWidget):
    def __init__(self, channel: str):
        super().__init__()
        self.channel = channel
        layout = QHBoxLayout()
        self.label = QLabel(channel)
        layout.addWidget(self.label)
        self.value_label = QLabel("0.0")
        layout.addWidget(self.value_label)
        self.setLayout(layout)

    def update_value(self, value: float) -> None:
        if self.channel.startswith("DI"):
            rendered = "ON" if value > 0.5 else "OFF"
        else:
            rendered = f"{value:.2f}"
        self.value_label.setText(rendered)


class Logger:
    def __init__(self) -> None:
        self._file = None
        self._path = ""
        self._started = False
        self._signal_names = []
        self._last_error = ""

    def start(self, path: str, signal_names) -> bool:
        self.stop()
        try:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            self._file = open(path, "w", encoding="utf-8")
        except OSError as exc:
            self._file = None
            self._last_error = str(exc)
            return False

        self._signal_names = list(signal_names)
        header = ["timestamp"] + self._signal_names
        self._file.write(",".join(header) + "\n")
        self._file.flush()
        self._path = path
        self._started = True
        self._last_error = ""
        return True

    def stop(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._started = False

    def log_row(self, timestamp: str, values: Dict[str, float]) -> None:
        if not self._started or self._file is None:
            return
        ordered = [timestamp] + [f"{values.get(name, 0.0):.4f}" for name in self._signal_names]
        self._file.write(",".join(ordered) + "\n")
        self._file.flush()

    @property
    def is_running(self) -> bool:
        return self._started

    @property
    def path(self) -> str:
        return self._path

    @property
    def last_error(self) -> str:
        return self._last_error


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ECU Control MVP")

        self.backends = {
            DummyBackend.name: DummyBackend,
            RealBackend.name: RealBackend,
        }
        self.backend: Optional[BackendBase] = None
        self.logger = Logger()
        self._last_tick = time.monotonic()
        self._last_log_time = 0.0

        self.setStatusBar(QStatusBar())

        central = QWidget()
        main_layout = QVBoxLayout()

        main_layout.addLayout(self._build_mode_selector())
        main_layout.addWidget(self._build_connection_group())
        main_layout.addWidget(self._build_output_group())
        main_layout.addWidget(self._build_input_group())
        main_layout.addWidget(self._build_logging_group())

        central.setLayout(main_layout)
        self.setCentralWidget(central)

        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        self._switch_backend(DummyBackend.name)

    def _build_mode_selector(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(self.backends.keys())
        self.mode_combo.currentTextChanged.connect(self._switch_backend)
        layout.addWidget(self.mode_combo)
        layout.addStretch(1)
        return layout

    def _build_connection_group(self) -> QGroupBox:
        group = QGroupBox("Connection")
        layout = QGridLayout()

        layout.addWidget(QLabel("DBC path"), 0, 0)
        self.dbc_path_edit = QLineEdit("ecu-test.dbc")
        layout.addWidget(self.dbc_path_edit, 0, 1)
        dbc_button = QPushButton("Browse")
        dbc_button.clicked.connect(self._browse_dbc)
        layout.addWidget(dbc_button, 0, 2)

        layout.addWidget(QLabel("Bus type"), 1, 0)
        self.bustype_combo = QComboBox()
        self.bustype_combo.addItems(["socketcan", "pcan", "vector"])
        layout.addWidget(self.bustype_combo, 1, 1)

        layout.addWidget(QLabel("Channel"), 2, 0)
        self.channel_edit = QLineEdit("can0")
        layout.addWidget(self.channel_edit, 2, 1)

        layout.addWidget(QLabel("Bitrate"), 3, 0)
        self.bitrate_spin = QSpinBox()
        self.bitrate_spin.setRange(1000, 1000000)
        self.bitrate_spin.setSingleStep(1000)
        self.bitrate_spin.setValue(500000)
        layout.addWidget(self.bitrate_spin, 3, 1)

        layout.setColumnStretch(1, 1)
        group.setLayout(layout)
        return group

    def _build_output_group(self) -> QGroupBox:
        group = QGroupBox("Outputs")
        layout = QVBoxLayout()
        self.output_widgets: Dict[str, OutputWidget] = {}
        for channel in OUTPUT_CHANNELS:
            widget = OutputWidget(channel, self._handle_output_change)
            layout.addWidget(widget)
            self.output_widgets[channel] = widget
        group.setLayout(layout)
        return group

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("Inputs")
        layout = QVBoxLayout()
        self.input_widgets: Dict[str, InputWidget] = {}
        for channel in INPUT_CHANNELS:
            widget = InputWidget(channel)
            layout.addWidget(widget)
            self.input_widgets[channel] = widget
        group.setLayout(layout)
        return group

    def _build_logging_group(self) -> QGroupBox:
        group = QGroupBox("CSV Logging")
        layout = QGridLayout()

        layout.addWidget(QLabel("File path"), 0, 0)
        self.path_edit = QLineEdit("logs.csv")
        layout.addWidget(self.path_edit, 0, 1)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._browse_file)
        layout.addWidget(browse_button, 0, 2)

        layout.addWidget(QLabel("Rate (Hz)"), 1, 0)
        self.rate_spin = QSpinBox()
        self.rate_spin.setRange(1, 50)
        self.rate_spin.setValue(10)
        layout.addWidget(self.rate_spin, 1, 1)

        self.start_button = QPushButton("Start Logging")
        self.start_button.clicked.connect(self._start_logging)
        layout.addWidget(self.start_button, 2, 0)

        self.stop_button = QPushButton("Stop Logging")
        self.stop_button.clicked.connect(self._stop_logging)
        layout.addWidget(self.stop_button, 2, 1)

        group.setLayout(layout)
        return group

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Select log file", self.path_edit.text(), "CSV Files (*.csv)")
        if path:
            self.path_edit.setText(path)

    def _switch_backend(self, name: str) -> None:
        previous = self.backend
        if previous is not None:
            self._disconnect_backend_signals(previous)
            previous.stop()

        try:
            backend = self._create_backend(name)
        except BackendError as exc:
            QMessageBox.critical(self, "Backend", str(exc))
            self.statusBar().showMessage(str(exc), 5000)
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentText(DummyBackend.name)
            self.mode_combo.blockSignals(False)
            backend = self._create_backend(DummyBackend.name)

        self.backend = backend
        self._connect_backend_signals(self.backend)
        self.statusBar().showMessage(f"Active backend: {self.backend.name}", 5000)
        self._refresh_from_backend(force=True)

    def _create_backend(self, name: str) -> BackendBase:
        backend_cls = self.backends.get(name, DummyBackend)
        if backend_cls is RealBackend:
            settings = self._collect_connection_settings()
            backend = backend_cls(settings)
        else:
            backend = backend_cls()
        backend.start()
        return backend

    def _collect_connection_settings(self) -> ConnectionSettings:
        dbc_path = self.dbc_path_edit.text().strip()
        if not dbc_path:
            raise BackendError("Please specify a DBC file path.")
        channel = self.channel_edit.text().strip()
        if not channel:
            raise BackendError("Please provide a CAN channel.")
        bustype = self.bustype_combo.currentText().strip()
        bitrate = int(self.bitrate_spin.value())
        return ConnectionSettings(
            dbc_path=dbc_path,
            bustype=bustype,
            channel=channel,
            bitrate=bitrate,
        )

    def _connect_backend_signals(self, backend: BackendBase) -> None:
        if isinstance(backend, RealBackend):
            backend.status_updated.connect(self._on_backend_status)

    def _disconnect_backend_signals(self, backend: BackendBase) -> None:
        if isinstance(backend, RealBackend):
            try:
                backend.status_updated.disconnect(self._on_backend_status)
            except TypeError:
                pass

    def _on_backend_status(self) -> None:
        self._refresh_from_backend()

    def _handle_output_change(self, channel: str, enabled: bool, pwm: float) -> None:
        if self.backend is None:
            return
        try:
            self.backend.set_output(channel, enabled, pwm)
        except BackendError as exc:
            QMessageBox.critical(self, "Output", str(exc))
            self.statusBar().showMessage(str(exc), 5000)

    def _tick(self) -> None:
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now
        if self.backend is None:
            return
        try:
            self.backend.update(dt)
        except BackendError as exc:
            self.statusBar().showMessage(str(exc), 5000)
            return

        self._refresh_from_backend()

    def _refresh_from_backend(self, force: bool = False) -> None:
        if self.backend is None:
            return
        outputs = self.backend.read_outputs()
        inputs = self.backend.read_inputs()

        for channel, widget in self.output_widgets.items():
            widget.update_state(outputs[channel])

        for channel, widget in self.input_widgets.items():
            value = inputs.get(channel, 0.0)
            widget.update_value(value)

        now = time.monotonic()
        if self.logger.is_running:
            self._log_values(outputs, inputs, now, force)

    def _log_values(
        self,
        outputs: Dict[str, OutputState],
        inputs: Dict[str, float],
        now: float,
        force: bool = False,
    ) -> None:
        rate = self.rate_spin.value()
        if not force and now - self._last_log_time < 1.0 / rate:
            return
        self._last_log_time = now

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))
        values: Dict[str, float] = {}
        for channel, state in outputs.items():
            values[f"{channel}_current"] = state.current
        for channel in INPUT_CHANNELS:
            values[channel] = inputs.get(channel, 0.0)
        self.logger.log_row(timestamp, values)

    def _start_logging(self) -> None:
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Logging", "Please specify a file path for the CSV log.")
            return
        signal_names = [f"{ch}_current" for ch in OUTPUT_CHANNELS] + INPUT_CHANNELS
        if self.logger.start(path, signal_names):
            QMessageBox.information(self, "Logging", f"Logging started: {path}")
        else:
            QMessageBox.critical(self, "Logging", f"Failed to start logging:\n{self.logger.last_error}")

    def _stop_logging(self) -> None:
        if self.logger.is_running:
            self.logger.stop()
            QMessageBox.information(self, "Logging", "Logging stopped.")

    def _browse_dbc(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select DBC file", self.dbc_path_edit.text(), "DBC Files (*.dbc)"
        )
        if path:
            self.dbc_path_edit.setText(path)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(600, 600)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
