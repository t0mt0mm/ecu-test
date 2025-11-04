import math
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import can
import cantools
from cantools.database import errors as cantools_errors
from PyQt5.QtCore import QObject, QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
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


@dataclass
class SignalDefinition:
    message_name: str
    name: str
    unit: str
    minimum: Optional[float]
    maximum: Optional[float]
    scale: float
    offset: float


@dataclass
class SequencerState:
    on_seconds: float = 0.0
    off_seconds: float = 0.0
    total_seconds: float = 0.0
    elapsed_total: float = 0.0
    elapsed_phase: float = 0.0
    running: bool = False
    is_on_phase: bool = True
    target_pwm: float = 0.0
    completed: bool = False


class BackendError(Exception):
    """Raised when a backend cannot complete the requested action."""


class BackendBase:
    """Common interface for ECU backends."""

    name: str = "Base"

    def start(self) -> None:
        """Initialize backend resources."""

    def stop(self) -> None:
        """Tear down backend resources."""

    def apply_database(self, database) -> None:  # pragma: no cover - optional hook
        """Provide the loaded DBC to the backend."""

    def set_output(self, channel: str, enabled: bool, pwm: float) -> None:
        raise NotImplementedError

    def read_outputs(self) -> Dict[str, OutputState]:
        raise NotImplementedError

    def read_inputs(self) -> Dict[str, float]:
        raise NotImplementedError

    def update(self, dt: float) -> None:
        """Advance internal simulation and poll hardware."""

    def read_signal_values(self, signal_names: Iterable[str]) -> Dict[str, float]:
        raise NotImplementedError


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
        self._dbc = None
        self._signal_values: Dict[str, float] = {}

    def start(self) -> None:
        self._time = 0.0

    def stop(self) -> None:
        pass

    def apply_database(self, database) -> None:
        self._dbc = database
        self._signal_values = {}
        if self._dbc is not None:
            for message in self._dbc.messages:
                for signal in message.signals:
                    self._signal_values[signal.name] = 0.0
        self._seed_values()

    def set_output(self, channel: str, enabled: bool, pwm: float) -> None:
        if channel not in self._outputs:
            return
        pwm = max(0.0, min(100.0, pwm))
        state = self._outputs[channel]
        state.enabled = enabled
        state.pwm = pwm
        index = OUTPUT_CHANNELS.index(channel) + 1
        self._update_signal_value(f"hs_out{index:02d}_select", 1 if enabled else 0)
        self._update_signal_value(f"hs_out{index:02d}_mode", 1 if enabled else 0)
        self._update_signal_value(f"hs_out{index:02d}_value_pwm", pwm)

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
            index = OUTPUT_CHANNELS.index(ch) + 1
            current_ma = state.current * 1000.0
            self._update_signal_value(f"hs_out{index:02d}_current", current_ma)
            self._update_signal_value(f"hs_out{index:02d}_value_current", current_ma)
            self._update_signal_value(f"hs_out{index:02d}_pwm", state.pwm)
            self._update_signal_value(f"hs_out{index:02d}_value_pwm", state.pwm)

        # Mirror inputs
        for name, value in self.read_inputs().items():
            self._update_signal_value(name, value)

    def read_signal_values(self, signal_names: Iterable[str]) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for name in signal_names:
            values[name] = float(self._signal_values.get(name, 0.0))
        return values

    def _update_signal_value(self, name: str, value: float) -> None:
        if name in self._signal_values:
            self._signal_values[name] = float(value)

    def _seed_values(self) -> None:
        for channel, state in self._outputs.items():
            index = OUTPUT_CHANNELS.index(channel) + 1
            current_ma = state.current * 1000.0
            self._update_signal_value(f"hs_out{index:02d}_current", current_ma)
            self._update_signal_value(f"hs_out{index:02d}_value_current", current_ma)
            self._update_signal_value(f"hs_out{index:02d}_pwm", state.pwm)
            self._update_signal_value(f"hs_out{index:02d}_value_pwm", state.pwm)
            self._update_signal_value(f"hs_out{index:02d}_select", 1 if state.enabled else 0)
            self._update_signal_value(f"hs_out{index:02d}_mode", 1 if state.enabled else 0)
        for name, value in self.read_inputs().items():
            self._update_signal_value(name, value)


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
        self._signal_cache: Dict[str, float] = {}

    def apply_database(self, database) -> None:
        with self._lock:
            self._signal_cache = {}
            if database is not None:
                for message in getattr(database, "messages", []):
                    for signal in message.signals:
                        self._signal_cache[signal.name] = 0.0

    def start(self) -> None:
        if not os.path.exists(self.settings.dbc_path):
            raise BackendError(f"DBC file not found: {self.settings.dbc_path}")

        try:
            self._db = cantools.database.load_file(self.settings.dbc_path, strict=False)
        except (OSError, cantools_errors.Error) as exc:
            raise BackendError(f"Failed to load DBC: {exc}")

        try:
            self._write_message = self._db.get_message_by_name("QM_High_side_output_write")
            self._status_message = self._db.get_message_by_name("QM_High_side_output_status")
        except KeyError as exc:
            raise BackendError(f"Required message missing in DBC: {exc}")

        self._signal_cache = {}
        for message in self._db.messages:
            for signal in message.signals:
                self._signal_cache[signal.name] = 0.0

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
            self._signal_cache[select_name] = payload[select_name]
            self._signal_cache[mode_name] = payload[mode_name]
            self._signal_cache[pwm_name] = payload[pwm_name]
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
                value_current_key = f"hs_out{index:02d}_value_current"
                value_pwm_key = f"hs_out{index:02d}_value_pwm"
                if current_key in decoded:
                    current_ma = float(decoded[current_key])
                    self._outputs[channel].current = max(0.0, current_ma / 1000.0)
                    self._signal_cache[current_key] = current_ma
                    if value_current_key in self._signal_cache:
                        self._signal_cache[value_current_key] = current_ma
                if pwm_key in decoded:
                    self._outputs[channel].pwm = float(decoded[pwm_key])
                    self._signal_cache[pwm_key] = float(decoded[pwm_key])
                    if value_pwm_key in self._signal_cache:
                        self._signal_cache[value_pwm_key] = float(decoded[pwm_key])
            # Expose PWM of status as inputs for visibility
            for index in range(1, len(OUTPUT_CHANNELS) + 1):
                current_key = f"hs_out{index:02d}_current"
                input_key = f"AI{index}"
                if current_key in decoded and input_key in self._inputs:
                    self._inputs[input_key] = float(decoded[current_key]) / 1000.0
            for name, value in decoded.items():
                self._signal_cache[name] = float(value)

        self.status_updated.emit()

    def read_signal_values(self, signal_names: Iterable[str]) -> Dict[str, float]:
        with self._lock:
            return {name: float(self._signal_cache.get(name, 0.0)) for name in signal_names}


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

    def set_manual_enabled(self, enabled: bool) -> None:
        self.toggle.setEnabled(enabled)
        self.slider.setEnabled(enabled)


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


class SequencerWidget(QWidget):
    def __init__(self, channel: str, start_callback, stop_callback) -> None:
        super().__init__()
        self.channel = channel
        self._start_callback = start_callback
        self._stop_callback = stop_callback

        layout = QHBoxLayout()

        self.label = QLabel(channel)
        layout.addWidget(self.label)

        self.on_spin = QDoubleSpinBox()
        self.on_spin.setSuffix(" s ON")
        self.on_spin.setDecimals(1)
        self.on_spin.setRange(0.1, 3600.0)
        self.on_spin.setValue(5.0)
        layout.addWidget(self.on_spin)

        self.off_spin = QDoubleSpinBox()
        self.off_spin.setSuffix(" s OFF")
        self.off_spin.setDecimals(1)
        self.off_spin.setRange(0.1, 3600.0)
        self.off_spin.setValue(5.0)
        layout.addWidget(self.off_spin)

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setSuffix(" min")
        self.duration_spin.setDecimals(1)
        self.duration_spin.setRange(0.1, 600.0)
        self.duration_spin.setValue(10.0)
        layout.addWidget(self.duration_spin)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self._start)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        layout.addWidget(self.stop_button)

        self.status_label = QLabel("Idle")
        layout.addWidget(self.status_label)

        layout.addStretch(1)
        self.setLayout(layout)

    def _start(self) -> None:
        on_seconds = float(self.on_spin.value())
        off_seconds = float(self.off_spin.value())
        total_minutes = float(self.duration_spin.value())
        self._start_callback(self.channel, on_seconds, off_seconds, total_minutes)

    def _stop(self) -> None:
        self._stop_callback(self.channel)

    def set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

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


class SignalBrowserWidget(QGroupBox):
    def __init__(self) -> None:
        super().__init__("Signal Browser")
        self._signals: Dict[str, List[SignalDefinition]] = {}

        layout = QVBoxLayout()

        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.textChanged.connect(self._apply_filter)
        search_layout.addWidget(self.search_edit)
        layout.addLayout(search_layout)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["Signal", "Unit", "Min", "Max", "Scaling"])
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setRootIsDecorated(True)
        layout.addWidget(self.tree)

        button_layout = QHBoxLayout()
        self.add_button = QPushButton("Add to Watchlist")
        button_layout.addStretch(1)
        button_layout.addWidget(self.add_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def set_signals(self, signals: Dict[str, List[SignalDefinition]]) -> None:
        self._signals = signals
        self.tree.clear()
        for message_name in sorted(self._signals):
            parent = QTreeWidgetItem([message_name, "", "", "", ""])
            parent.setData(0, Qt.UserRole, ("message", message_name))
            for definition in self._signals[message_name]:
                if definition.offset:
                    scaling = f"{definition.scale:g} * raw + {definition.offset:g}"
                else:
                    scaling = f"{definition.scale:g} * raw"
                item = QTreeWidgetItem(
                    [
                        definition.name,
                        definition.unit or "",
                        "" if definition.minimum is None else f"{definition.minimum:g}",
                        "" if definition.maximum is None else f"{definition.maximum:g}",
                        scaling,
                    ]
                )
                item.setData(0, Qt.UserRole, ("signal", definition))
                parent.addChild(item)
            parent.setExpanded(True)
            self.tree.addTopLevelItem(parent)

    def selected_signal_names(self) -> List[str]:
        names: List[str] = []
        for item in self.tree.selectedItems():
            data = item.data(0, Qt.UserRole)
            if isinstance(data, tuple) and data and data[0] == "signal":
                definition: SignalDefinition = data[1]
                names.append(definition.name)
        return names

    def _apply_filter(self, text: str) -> None:
        text = text.lower().strip()
        for i in range(self.tree.topLevelItemCount()):
            message_item = self.tree.topLevelItem(i)
            message_name = message_item.text(0).lower()
            message_match = text in message_name if text else True
            visible_children = 0
            for j in range(message_item.childCount()):
                child = message_item.child(j)
                signal_name = child.text(0).lower()
                unit = child.text(1).lower()
                matches = (
                    not text
                    or text in signal_name
                    or text in unit
                    or text in message_name
                )
                child.setHidden(not matches)
                if matches:
                    visible_children += 1
            message_item.setHidden(not (message_match or visible_children > 0))
            if text and (message_match or visible_children > 0):
                message_item.setExpanded(True)


class WatchlistWidget(QGroupBox):
    def __init__(self) -> None:
        super().__init__("Watchlist")
        self._order: List[str] = []

        layout = QVBoxLayout()

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Signal", "Value"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        layout.addWidget(self.table)

        button_layout = QHBoxLayout()
        self.remove_button = QPushButton("Remove Selected")
        button_layout.addStretch(1)
        button_layout.addWidget(self.remove_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def add_signals(self, names: Iterable[str]) -> List[str]:
        added: List[str] = []
        for name in names:
            if name in self._order:
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem("0.0000"))
            self._order.append(name)
            added.append(name)
        return added

    def remove_selected(self) -> List[str]:
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()}, reverse=True)
        removed: List[str] = []
        for row in rows:
            if 0 <= row < len(self._order):
                removed.append(self._order[row])
                self._order.pop(row)
                self.table.removeRow(row)
        return removed

    def update_values(self, values: Dict[str, float]) -> None:
        for row, name in enumerate(self._order):
            value = values.get(name)
            display = "n/a" if value is None else f"{value:.4f}"
            item = self.table.item(row, 1)
            if item is None:
                item = QTableWidgetItem(display)
                self.table.setItem(row, 1, item)
            else:
                item.setText(display)

    @property
    def signal_names(self) -> List[str]:
        return list(self._order)

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
        self._dbc = None
        self._signals_by_message: Dict[str, List[SignalDefinition]] = {}
        self._watch_values: Dict[str, float] = {}
        self._sequencers: Dict[str, SequencerState] = {
            channel: SequencerState() for channel in OUTPUT_CHANNELS
        }

        self.setStatusBar(QStatusBar())

        central = QWidget()
        main_layout = QVBoxLayout()

        main_layout.addLayout(self._build_mode_selector())
        main_layout.addWidget(self._build_connection_group())
        main_layout.addLayout(self._build_signal_section())
        main_layout.addWidget(self._build_output_group())
        main_layout.addWidget(self._build_sequencer_group())
        main_layout.addWidget(self._build_input_group())
        main_layout.addWidget(self._build_logging_group())

        central.setLayout(main_layout)
        self.setCentralWidget(central)

        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        self._stop_all_sequences()
        self._switch_backend(DummyBackend.name)
        self._load_current_dbc()

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
        load_button = QPushButton("Load")
        load_button.clicked.connect(self._load_current_dbc)
        layout.addWidget(load_button, 0, 3)

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
        layout.setColumnStretch(2, 0)
        layout.setColumnStretch(3, 0)
        group.setLayout(layout)
        return group

    def _build_signal_section(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.signal_browser = SignalBrowserWidget()
        self.signal_browser.add_button.clicked.connect(self._add_selected_signals)
        layout.addWidget(self.signal_browser, 2)

        self.watchlist_widget = WatchlistWidget()
        self.watchlist_widget.remove_button.clicked.connect(self._remove_selected_signals)
        layout.addWidget(self.watchlist_widget, 1)
        return layout

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

    def _build_sequencer_group(self) -> QGroupBox:
        group = QGroupBox("Sequencer")
        layout = QVBoxLayout()
        self.sequencer_widgets: Dict[str, SequencerWidget] = {}
        for channel in OUTPUT_CHANNELS:
            widget = SequencerWidget(channel, self._start_sequence, self._stop_sequence)
            layout.addWidget(widget)
            self.sequencer_widgets[channel] = widget
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

    def _add_selected_signals(self) -> None:
        if self.logger.is_running:
            QMessageBox.warning(self, "Watchlist", "Stop logging before modifying the watchlist.")
            return
        names = self.signal_browser.selected_signal_names()
        if not names:
            return
        added = self.watchlist_widget.add_signals(names)
        if added:
            self.statusBar().showMessage(f"Added to watchlist: {', '.join(added)}", 5000)
            self._refresh_watchlist_values(force=True)

    def _remove_selected_signals(self) -> None:
        if self.logger.is_running:
            QMessageBox.warning(self, "Watchlist", "Stop logging before modifying the watchlist.")
            return
        removed = self.watchlist_widget.remove_selected()
        if removed:
            self.statusBar().showMessage(f"Removed from watchlist: {', '.join(removed)}", 5000)
            for name in removed:
                self._watch_values.pop(name, None)

    def _load_current_dbc(self) -> None:
        path = self.dbc_path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "DBC", "Please specify a DBC file path.")
            return
        if not os.path.exists(path):
            QMessageBox.warning(self, "DBC", f"DBC file not found: {path}")
            return
        try:
            database = cantools.database.load_file(path, strict=False)
        except (OSError, cantools_errors.Error) as exc:
            QMessageBox.critical(self, "DBC", f"Failed to load DBC: {exc}")
            return

        self._dbc = database
        self._signals_by_message = self._extract_signal_definitions(database)
        self.signal_browser.set_signals(self._signals_by_message)
        if self.backend is not None:
            self.backend.apply_database(database)
        self.statusBar().showMessage(f"DBC loaded: {os.path.basename(path)}", 5000)
        self._refresh_watchlist_values(force=True)

    def _extract_signal_definitions(self, database) -> Dict[str, List[SignalDefinition]]:
        catalog: Dict[str, List[SignalDefinition]] = {}
        for message in getattr(database, "messages", []):
            entries: List[SignalDefinition] = []
            for signal in message.signals:
                entries.append(
                    SignalDefinition(
                        message_name=message.name,
                        name=signal.name,
                        unit=signal.unit or "",
                        minimum=signal.minimum,
                        maximum=signal.maximum,
                        scale=signal.scale,
                        offset=signal.offset,
                    )
                )
            if entries:
                catalog[message.name] = entries
        return catalog

    def _switch_backend(self, name: str) -> None:
        self._stop_all_sequences()
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
        if self._dbc is not None:
            backend.apply_database(self._dbc)
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

    def _start_sequence(
        self,
        channel: str,
        on_seconds: float,
        off_seconds: float,
        total_minutes: float,
    ) -> None:
        if channel not in self._sequencers:
            return
        if self.backend is None:
            QMessageBox.warning(self, "Sequencer", "Select a backend before starting a sequence.")
            return
        if self._sequencers[channel].running:
            return
        if on_seconds <= 0.0 or off_seconds <= 0.0 or total_minutes <= 0.0:
            QMessageBox.warning(
                self,
                "Sequencer",
                "Please configure ON time, OFF time, and duration with positive values.",
            )
            return

        state = self._sequencers[channel]
        state.on_seconds = on_seconds
        state.off_seconds = off_seconds
        state.total_seconds = total_minutes * 60.0
        state.elapsed_total = 0.0
        state.elapsed_phase = 0.0
        state.running = True
        state.is_on_phase = True
        state.target_pwm = float(self.output_widgets[channel].slider.value())
        state.completed = False

        widget = self.sequencer_widgets.get(channel)
        if widget is not None:
            widget.set_running(True)
        output_widget = self.output_widgets.get(channel)
        if output_widget is not None:
            output_widget.set_manual_enabled(False)
        if not self._apply_sequencer_output(channel, state):
            self._stop_sequence(channel, completed=False)
            return
        self._update_sequencer_widget(channel)
        self.statusBar().showMessage(
            f"Sequencer started on {channel} (ON {on_seconds:.1f}s / OFF {off_seconds:.1f}s)",
            5000,
        )

    def _stop_sequence(
        self,
        channel: str,
        completed: bool = False,
        disable_output: bool = True,
    ) -> None:
        if channel not in self._sequencers:
            return
        state = self._sequencers[channel]
        was_running = state.running
        state.running = False
        if completed:
            state.completed = True
            state.elapsed_total = state.total_seconds
        else:
            state.completed = False
        state.elapsed_phase = 0.0
        state.is_on_phase = True
        widget = self.sequencer_widgets.get(channel)
        if widget is not None:
            widget.set_running(False)
        output_widget = self.output_widgets.get(channel)
        if output_widget is not None:
            output_widget.set_manual_enabled(True)
        if disable_output and self.backend is not None:
            try:
                self.backend.set_output(channel, False, 0.0)
            except BackendError as exc:
                self.statusBar().showMessage(str(exc), 5000)
        self._update_sequencer_widget(channel)
        if was_running:
            message = "completed" if completed else "stopped"
            self.statusBar().showMessage(f"Sequencer {message} on {channel}", 5000)

    def _apply_sequencer_output(self, channel: str, state: SequencerState) -> bool:
        if self.backend is None:
            return False
        enabled = state.is_on_phase
        pwm = state.target_pwm if enabled else 0.0
        try:
            self.backend.set_output(channel, enabled, pwm)
        except BackendError as exc:
            QMessageBox.critical(self, "Sequencer", str(exc))
            self.statusBar().showMessage(str(exc), 5000)
            return False
        return True

    def _toggle_sequence_phase(
        self, channel: str, state: SequencerState, carry: float = 0.0
    ) -> None:
        state.is_on_phase = not state.is_on_phase
        state.elapsed_phase = max(0.0, carry)
        if not self._apply_sequencer_output(channel, state):
            self._stop_sequence(channel, completed=False)

    def _advance_sequencers(self, dt: float) -> None:
        for channel, state in self._sequencers.items():
            if not state.running:
                continue
            state.elapsed_total += dt
            state.elapsed_phase += dt

            if state.total_seconds > 0.0 and state.elapsed_total >= state.total_seconds:
                self._stop_sequence(channel, completed=True)
                continue

            phase_limit = state.on_seconds if state.is_on_phase else state.off_seconds
            if phase_limit <= 0.0:
                self._toggle_sequence_phase(channel, state)
                self._update_sequencer_widget(channel)
                continue

            toggles = 0
            while state.running and state.elapsed_phase >= phase_limit and toggles < 5:
                leftover = state.elapsed_phase - phase_limit
                self._toggle_sequence_phase(channel, state, leftover)
                phase_limit = state.on_seconds if state.is_on_phase else state.off_seconds
                toggles += 1
                if phase_limit <= 0.0:
                    break

            if toggles >= 5 and state.running:
                state.elapsed_phase = 0.0

            if state.running:
                self._update_sequencer_widget(channel)

    def _update_sequencer_widget(self, channel: str) -> None:
        state = self._sequencers[channel]
        widget = self.sequencer_widgets.get(channel)
        if widget is None:
            return
        if state.running:
            phase = "ON" if state.is_on_phase else "OFF"
            remaining = max(0.0, state.total_seconds - state.elapsed_total)
            widget.set_status(
                f"Running {phase} | {remaining:.1f}s remaining"
            )
        else:
            if state.completed and state.total_seconds > 0.0:
                widget.set_status("Completed")
            else:
                widget.set_status("Idle")

    def _stop_all_sequences(self) -> None:
        for channel in OUTPUT_CHANNELS:
            self._stop_sequence(channel, completed=False)

    def _tick(self) -> None:
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now
        if self.backend is None:
            return
        self._advance_sequencers(dt)
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

        self._refresh_watchlist_values(force)

        now = time.monotonic()
        if self.logger.is_running:
            self._log_values(now, force)

    def _log_values(
        self,
        now: float,
        force: bool = False,
    ) -> None:
        rate = self.rate_spin.value()
        if not force and now - self._last_log_time < 1.0 / rate:
            return
        self._last_log_time = now

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))
        self.logger.log_row(timestamp, self._watch_values)

    def _start_logging(self) -> None:
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Logging", "Please specify a file path for the CSV log.")
            return
        signal_names = self.watchlist_widget.signal_names
        if not signal_names:
            QMessageBox.warning(self, "Logging", "Add signals to the watchlist before starting the log.")
            return
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
            self._load_current_dbc()

    def _refresh_watchlist_values(self, force: bool = False) -> None:
        if self.backend is None:
            self._watch_values = {}
            self.watchlist_widget.update_values(self._watch_values)
            return
        names = self.watchlist_widget.signal_names
        if not names:
            self._watch_values = {}
            self.watchlist_widget.update_values(self._watch_values)
            return
        try:
            values = self.backend.read_signal_values(names)
        except BackendError as exc:
            if force:
                QMessageBox.critical(self, "Watchlist", str(exc))
            self.statusBar().showMessage(str(exc), 5000)
            return
        self._watch_values = values
        self.watchlist_widget.update_values(values)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(600, 600)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
