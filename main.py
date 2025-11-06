"""PyQt5-based ECU control application with dynamic channels."""

from __future__ import annotations

import copy
import datetime
import math
import os
import random
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import can
import cantools
from cantools.database import errors as cantools_errors
import yaml
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")
import pyqtgraph as pg
if pg.Qt.QT_LIB != "PyQt5":
    raise RuntimeError(
        "PyQtGraph loaded Qt binding '" + pg.Qt.QT_LIB + "' but this application requires PyQt5. "
        "Ensure PyQt5 is installed and set PYQTGRAPH_QT_LIB=PyQt5 before launching."
    )
from PyQt5.QtCore import QObject, QSettings, QTimer, Qt, QTime, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTimeEdit,
    QToolButton,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QRadioButton,
    QMenu,
    QAction,
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
PROFILE_DIR = os.path.join(BASE_DIR, "profiles")
WHITELIST_PATH = os.path.join(CONFIG_DIR, "signals.yaml")
CHANNEL_PROFILE_PATH = os.path.join(PROFILE_DIR, "channels.yaml")


@dataclass
class OutputState:
    enabled: bool
    pwm: float
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
class AnalogSimulationProfile:
    generator: str = "hold"
    offset: float = 0.0
    amplitude: float = 1.0
    frequency: float = 0.1
    slope: float = 1.0
    noise: float = 0.0
    hold_value: float = 0.0
    phase: float = 0.0


@dataclass
class DigitalSimulationProfile:
    mode: str = "pattern"
    period: float = 1.0
    duty_cycle: float = 0.5
    high_value: float = 1.0
    low_value: float = 0.0
    manual_value: float = 0.0
    phase: float = 0.0


@dataclass
class SignalSimulationConfig:
    message_name: str
    name: str
    unit: str
    minimum: Optional[float]
    maximum: Optional[float]
    category: str
    analog: Optional[AnalogSimulationProfile] = None
    digital: Optional[DigitalSimulationProfile] = None

    def clone(self) -> "SignalSimulationConfig":
        return SignalSimulationConfig(
            message_name=self.message_name,
            name=self.name,
            unit=self.unit,
            minimum=self.minimum,
            maximum=self.maximum,
            category=self.category,
            analog=copy.deepcopy(self.analog),
            digital=copy.deepcopy(self.digital),
        )

@dataclass
class ChannelMapping:
    message: str = ""
    fields: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, data: Optional[dict]) -> "ChannelMapping":
        if not data:
            return cls()
        message = str(data.get("message", ""))
        raw_fields = data.get("fields") or {}
        normalized: Dict[str, str] = {}
        for key, value in raw_fields.items():
            key_str = str(key)
            value_str = str(value)
            if value_str.lower() in CHANNEL_SEMANTICS:
                normalized[value_str.lower()] = key_str
            else:
                normalized[key_str.lower()] = value_str
        return cls(message=message, fields=normalized)

    def to_yaml(self) -> dict:
        return {
            "message": self.message,
            "fields": {signal: semantic for semantic, signal in self.fields.items()},
        }

    def signal_for(self, semantic: str) -> Optional[str]:
        return self.fields.get(semantic.lower())


@dataclass
class ChannelProfile:
    name: str
    type: str
    write: ChannelMapping = field(default_factory=ChannelMapping)
    status: ChannelMapping = field(default_factory=ChannelMapping)
    sim: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, data: dict) -> "ChannelProfile":
        return cls(
            name=str(data.get("name", "Unnamed")),
            type=str(data.get("type", "HighSide")),
            write=ChannelMapping.from_yaml(data.get("write")),
            status=ChannelMapping.from_yaml(data.get("status")),
            sim=dict(data.get("sim", {})),
        )

    def to_yaml(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "write": self.write.to_yaml(),
            "status": self.status.to_yaml(),
            "sim": dict(self.sim),
        }


class SequenceRepeatMode(Enum):
    OFF = "off"
    ENDLESS = "endless"
    LIMIT = "limit"


@dataclass
class SequenceCfg:
    name: str
    duration_s: int
    pwm: int
    on_s: float
    off_s: float
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "duration_s": int(max(1, self.duration_s)),
            "pwm": int(max(0, min(100, self.pwm))),
            "on_s": float(max(0.01, self.on_s)),
            "off_s": float(max(0.01, self.off_s)),
            "enabled": bool(self.enabled),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SequenceCfg":
        return cls(
            name=str(data.get("name", "Sequence")),
            duration_s=int(max(1, int(data.get("duration_s", 1)))),
            pwm=int(max(0, min(100, int(data.get("pwm", 0))))),
            on_s=float(max(0.01, float(data.get("on_s", 0.01)))),
            off_s=float(max(0.01, float(data.get("off_s", 0.01)))),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass
class ChannelConfig:
    sequences: List[SequenceCfg] = field(default_factory=list)
    repeat_mode: SequenceRepeatMode = SequenceRepeatMode.OFF
    repeat_limit_s: int = 0

    def to_dict(self) -> dict:
        return {
            "sequences": [sequence.to_dict() for sequence in self.sequences],
            "repeat_mode": self.repeat_mode.value,
            "repeat_limit_s": int(max(0, self.repeat_limit_s)),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChannelConfig":
        sequences_data = data.get("sequences") or []
        sequences = [SequenceCfg.from_dict(raw) for raw in sequences_data]
        mode_value = str(data.get("repeat_mode", SequenceRepeatMode.OFF.value))
        try:
            repeat_mode = SequenceRepeatMode(mode_value)
        except ValueError:
            repeat_mode = SequenceRepeatMode.OFF
        repeat_limit_s = int(max(0, int(data.get("repeat_limit_s", 0))))
        return cls(sequences=sequences, repeat_mode=repeat_mode, repeat_limit_s=repeat_limit_s)


class SequenceRunner(QObject):
    progressed = pyqtSignal(int, str, float)
    finished = pyqtSignal()

    def __init__(
        self,
        channel_id: str,
        set_output_cb: Callable[[str, float], None],
        parent: Optional[QObject] = None,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(parent)
        self.channel_id = channel_id
        self._set_output_cb = set_output_cb
        self._now_fn = now_fn
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._advance)
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(500)
        self._progress_timer.timeout.connect(self._emit_progress)
        self._sequences: List[SequenceCfg] = []
        self._enabled_sequences: List[SequenceCfg] = []
        self._repeat_mode = SequenceRepeatMode.OFF
        self._repeat_limit_s = 0
        self._repeat_deadline: Optional[float] = None
        self._sequence_index = -1
        self._phase: str = "off"
        self._phase_end: float = 0.0
        self._sequence_end: float = 0.0
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def sequence_count(self) -> int:
        return len(self._enabled_sequences)

    def load(
        self,
        sequences: List[SequenceCfg],
        repeat_mode: SequenceRepeatMode,
        repeat_limit_s: int,
    ) -> None:
        self.reset()
        self._sequences = list(sequences)
        self._repeat_mode = repeat_mode
        self._repeat_limit_s = int(max(0, repeat_limit_s))
        self._enabled_sequences = [
            seq
            for seq in self._sequences
            if seq.enabled
            and seq.duration_s > 0
            and seq.on_s > 0
            and seq.off_s > 0
        ]

    def start(self) -> bool:
        if self._running:
            return False
        if not self._enabled_sequences:
            return False
        start_time = self._now_fn()
        if self._repeat_mode == SequenceRepeatMode.LIMIT and self._repeat_limit_s > 0:
            self._repeat_deadline = start_time + float(self._repeat_limit_s)
        else:
            self._repeat_deadline = None
        self._running = True
        self._progress_timer.start()
        self._enter_sequence(0, start_time)
        return True

    def stop(self) -> None:
        if not self._running:
            return
        self._timer.stop()
        self._progress_timer.stop()
        self._running = False
        self._sequence_index = -1
        self._phase = "off"
        self._phase_end = 0.0
        self._sequence_end = 0.0
        self._repeat_deadline = None
        self._safe_output(0.0)
        self._emit_progress()

    def reset(self) -> None:
        self.stop()

    def emit_progress(self) -> None:
        self._emit_progress()

    def _enter_sequence(self, index: int, now: float) -> None:
        if not self._enabled_sequences:
            self.stop()
            return
        if index >= len(self._enabled_sequences):
            index = 0
        self._sequence_index = index
        self._phase = "on"
        sequence = self._enabled_sequences[self._sequence_index]
        duration = float(max(0.01, sequence.duration_s))
        self._sequence_end = now + duration
        self._phase_end = min(self._sequence_end, now + float(max(0.01, sequence.on_s)))
        self._apply_phase_output(sequence)
        self._emit_progress()
        self._schedule_next()

    def _advance(self) -> None:
        if not self._running:
            return
        now = self._now_fn()
        if self._repeat_deadline is not None and now >= self._repeat_deadline:
            self.stop()
            self.finished.emit()
            return
        if now >= self._sequence_end:
            if not self._advance_sequence(now):
                self.stop()
                self.finished.emit()
            return
        if now >= self._phase_end:
            self._toggle_phase(now)
            return
        self._schedule_next()

    def _advance_sequence(self, now: float) -> bool:
        if not self._enabled_sequences:
            return False
        next_index = self._sequence_index + 1
        deadline_reached = self._repeat_deadline is not None and now >= self._repeat_deadline
        if next_index >= len(self._enabled_sequences):
            if self._repeat_mode == SequenceRepeatMode.ENDLESS:
                next_index = 0
            elif self._repeat_mode == SequenceRepeatMode.LIMIT and not deadline_reached:
                next_index = 0
            else:
                return False
        if deadline_reached:
            return False
        self._enter_sequence(next_index, now)
        return True

    def _toggle_phase(self, now: float) -> None:
        if not self._enabled_sequences or self._sequence_index < 0:
            return
        sequence = self._enabled_sequences[self._sequence_index]
        if self._phase == "on":
            self._phase = "off"
            off_duration = float(max(0.01, sequence.off_s))
            self._phase_end = min(self._sequence_end, now + off_duration)
        else:
            self._phase = "on"
            on_duration = float(max(0.01, sequence.on_s))
            self._phase_end = min(self._sequence_end, now + on_duration)
        self._apply_phase_output(sequence)
        self._emit_progress()
        self._schedule_next()

    def _schedule_next(self) -> None:
        if not self._running:
            return
        now = self._now_fn()
        targets: List[float] = [self._phase_end, self._sequence_end]
        if self._repeat_deadline is not None:
            targets.append(self._repeat_deadline)
        upcoming = min(targets)
        delay_ms = max(0, int(max(0.0, upcoming - now) * 1000))
        self._timer.start(delay_ms)

    def _apply_phase_output(self, sequence: SequenceCfg) -> None:
        pwm = float(sequence.pwm if self._phase == "on" else 0.0)
        self._safe_output(pwm)

    def _safe_output(self, pwm: float) -> None:
        try:
            self._set_output_cb(self.channel_id, pwm)
        except Exception:
            # Suppress backend errors to avoid crashing the UI; caller handles reporting.
            pass

    def _emit_progress(self) -> None:
        if not self._running or self._sequence_index < 0 or not self._enabled_sequences:
            self.progressed.emit(-1, "off", 0.0)
            return
        remaining = max(0.0, self._phase_end - self._now_fn())
        self.progressed.emit(self._sequence_index, self._phase, remaining)



class BackendError(Exception):
    """Raised when a backend cannot complete the requested action."""


CHANNEL_TYPES = [
    "HighSide",
    "LowSide",
    "HBridge",
    "AO_0_10V",
    "AO_4_20mA",
    "DO",
    "DI",
    "AI_V",
    "AI_I",
]

CHANNEL_SEMANTICS = {
    "select",
    "enable",
    "mode",
    "pwm",
    "setpoint",
    "direction",
    "current",
    "pwm_feedback",
    "state",
    "voltage",
    "current_feedback",
    "diagnostic",
    "value",
    "feedback",
}

CHANNEL_SCHEMA = {
    "HighSide": {
        "write": [
            ("select", "Enable"),
            ("pwm", "PWM (%)"),
            ("mode", "Mode"),
        ],
        "status": [
            ("current", "Current"),
            ("pwm_feedback", "PWM feedback"),
            ("diagnostic", "Diagnostic"),
        ],
    },
    "LowSide": {
        "write": [
            ("select", "Enable"),
            ("pwm", "PWM (%)"),
        ],
        "status": [
            ("current", "Current"),
            ("diagnostic", "Diagnostic"),
        ],
    },
    "HBridge": {
        "write": [
            ("enable", "Enable"),
            ("direction", "Direction"),
            ("pwm", "Duty (%)"),
        ],
        "status": [
            ("current", "Current"),
            ("direction", "Direction feedback"),
        ],
    },
    "AO_0_10V": {
        "write": [("setpoint", "Setpoint (V)")],
        "status": [("feedback", "Feedback (V)")],
    },
    "AO_4_20mA": {
        "write": [("setpoint", "Setpoint (mA)")],
        "status": [("feedback", "Feedback (mA)")],
    },
    "DO": {
        "write": [("state", "Digital state")],
        "status": [("feedback", "Feedback")],
    },
    "DI": {
        "write": [],
        "status": [("state", "State")],
    },
    "AI_V": {
        "write": [],
        "status": [("value", "Value (V)")],
    },
    "AI_I": {
        "write": [],
        "status": [("value", "Value (mA)")],
    },
}


CHANNEL_PLOT_COMMAND_SEMANTICS = {
    "HighSide": ("pwm", "setpoint", "state"),
    "LowSide": ("pwm", "setpoint", "state"),
    "HBridge": ("pwm", "enable", "select"),
    "AO_0_10V": ("setpoint",),
    "AO_4_20mA": ("setpoint",),
    "DO": ("state", "enable"),
    "DI": ("state",),
    "AI_V": ("value",),
    "AI_I": ("value",),
}


CHANNEL_PLOT_FEEDBACK_SEMANTICS = {
    "HighSide": ("current", "pwm_feedback", "feedback", "value"),
    "LowSide": ("current", "feedback", "value"),
    "HBridge": ("current", "feedback", "value"),
    "AO_0_10V": ("feedback", "value"),
    "AO_4_20mA": ("feedback", "value"),
    "DO": ("feedback", "state"),
    "DI": ("state",),
    "AI_V": ("value",),
    "AI_I": ("value",),
}


CHANNEL_SIM_FIELDS = {
    "HighSide": [
        ("tau", "Tau (s)", 0.05, 5.0, 0.05),
        ("current_gain", "Current gain", 0.0, 50.0, 0.1),
        ("noise", "Noise", 0.0, 5.0, 0.01),
        ("overcurrent", "Overcurrent threshold", 0.0, 100.0, 0.1),
    ],
    "LowSide": [
        ("tau", "Tau (s)", 0.05, 5.0, 0.05),
        ("current_gain", "Current gain", 0.0, 50.0, 0.1),
        ("noise", "Noise", 0.0, 5.0, 0.01),
    ],
    "HBridge": [
        ("tau", "Tau (s)", 0.05, 5.0, 0.05),
        ("current_gain", "Current gain", 0.0, 50.0, 0.1),
        ("noise", "Noise", 0.0, 5.0, 0.01),
    ],
    "AO_0_10V": [
        ("tau", "Tau (s)", 0.01, 5.0, 0.01),
        ("noise", "Noise", 0.0, 1.0, 0.01),
        ("target", "Default target", -100.0, 100.0, 0.1),
    ],
    "AO_4_20mA": [
        ("tau", "Tau (s)", 0.01, 5.0, 0.01),
        ("noise", "Noise", 0.0, 1.0, 0.01),
        ("target", "Default target", 0.0, 40.0, 0.1),
    ],
    "DO": [
        ("noise", "Noise", 0.0, 1.0, 0.01),
        ("pattern_period", "Pattern period (s)", 0.1, 10.0, 0.1),
        ("pattern_duty", "Pattern duty", 0.0, 1.0, 0.05),
    ],
    "DI": [
        ("pattern_period", "Pattern period (s)", 0.1, 10.0, 0.1),
        ("pattern_duty", "Pattern duty", 0.0, 1.0, 0.05),
    ],
    "AI_V": [
        ("noise", "Noise", 0.0, 5.0, 0.01),
        ("slope", "Slope", -10.0, 10.0, 0.1),
    ],
    "AI_I": [
        ("noise", "Noise", 0.0, 5.0, 0.01),
        ("slope", "Slope", -10.0, 10.0, 0.1),
    ],
}


def ensure_directories() -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(PROFILE_DIR, exist_ok=True)


def _load_signal_whitelist(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return []
    values = data.get("writable_signals", [])
    return [str(entry) for entry in values]


WRITABLE_SIGNALS = set(_load_signal_whitelist(WHITELIST_PATH))


def is_signal_writable(signal_name: str, message_name: str) -> bool:
    if signal_name in WRITABLE_SIGNALS:
        return True
    lower_name = signal_name.lower()
    lower_message = message_name.lower()
    return "write" in lower_message or "cmd" in lower_message or "cmd" in lower_name

class _SignalSimulator:
    def __init__(self, config: SignalSimulationConfig) -> None:
        self.config = config
        self.override: Optional[float] = None
        self._value = 0.0
        self._ramp_position: float = config.minimum if config.minimum is not None else 0.0

    def apply_profile(self, config: SignalSimulationConfig) -> None:
        self.config = config
        if self.config.category == "analog":
            profile = self.config.analog or AnalogSimulationProfile()
            if profile.generator == "hold":
                self._value = profile.hold_value
            else:
                self._value = profile.offset
            if self.config.minimum is not None:
                self._ramp_position = self.config.minimum
            else:
                self._ramp_position = profile.offset
        else:
            profile = self.config.digital or DigitalSimulationProfile()
            self._value = profile.low_value

    def set_override(self, value: Optional[float]) -> None:
        self.override = value
        if value is not None:
            self._value = float(value)

    def update(self, current_time: float, dt: float) -> float:
        if self.override is not None:
            return float(self.override)
        if self.config.category == "digital":
            result = self._update_digital(current_time)
        else:
            result = self._update_analog(current_time, dt)
        self._value = self._clamp(result)
        return self._value

    def _update_analog(self, current_time: float, dt: float) -> float:
        profile = self.config.analog or AnalogSimulationProfile()
        noise = random.uniform(-profile.noise, profile.noise) if profile.noise > 0.0 else 0.0
        generator = profile.generator.lower()
        if generator == "sine":
            amplitude = profile.amplitude
            frequency = max(profile.frequency, 0.0)
            value = profile.offset + amplitude * math.sin(
                2.0 * math.pi * frequency * current_time + profile.phase
            )
        elif generator == "ramp":
            span = self._span(profile)
            rate = max(profile.slope, span / 20.0 if span > 0.0 else 1.0)
            self._ramp_position += rate * dt
            lower, upper = self._bounds(profile, span)
            width = upper - lower
            if width <= 0.0:
                width = span if span > 0.0 else rate
            self._ramp_position = lower + ((self._ramp_position - lower) % width)
            value = self._ramp_position
        elif generator == "noise":
            amplitude = max(profile.amplitude, 0.0)
            value = profile.offset + random.uniform(-amplitude, amplitude)
        elif generator == "hold":
            value = profile.hold_value
        else:
            value = profile.offset
        return value + noise

    def _update_digital(self, current_time: float) -> float:
        profile = self.config.digital or DigitalSimulationProfile()
        if profile.mode == "manual":
            return profile.manual_value
        period = max(profile.period, 0.1)
        duty = min(max(profile.duty_cycle, 0.0), 1.0)
        phase = (current_time + profile.phase) % period
        threshold = duty * period
        return profile.high_value if phase < threshold else profile.low_value

    def _bounds(self, profile: AnalogSimulationProfile, span: float) -> Tuple[float, float]:
        if self.config.minimum is not None and self.config.maximum is not None:
            return float(self.config.minimum), float(self.config.maximum)
        center = profile.offset
        if span <= 0.0:
            span = max(profile.amplitude * 2.0, 1.0)
        half = span / 2.0
        return center - half, center + half

    def _span(self, profile: AnalogSimulationProfile) -> float:
        if self.config.minimum is not None and self.config.maximum is not None:
            return float(self.config.maximum - self.config.minimum)
        return abs(profile.amplitude) * 2.0

    def _clamp(self, value: float) -> float:
        if self.config.minimum is not None:
            value = max(float(self.config.minimum), value)
        if self.config.maximum is not None:
            value = min(float(self.config.maximum), value)
        return value


class ChannelRuntime:
    def __init__(self, profile: ChannelProfile) -> None:
        self.profile = profile
        self.enabled: bool = False
        self.pwm: float = 0.0
        self.setpoint: float = 0.0
        self.direction: int = 0
        self.last_command: Dict[str, float] = {}
        self.current_value: float = 0.0
        self.status_cache: Dict[str, float] = {}


class BackendBase:
    name: str = "Base"

    def start(self) -> None:
        """Initialize backend resources."""

    def stop(self) -> None:
        """Tear down backend resources."""

    def apply_database(self, database) -> None:
        """Provide the loaded DBC to the backend."""

    def set_channel_profiles(self, profiles: Dict[str, ChannelProfile]) -> None:
        """Provide the channel profiles to the backend."""

    def apply_channel_command(self, channel: str, command: Dict[str, float]) -> None:
        raise NotImplementedError

    def set_output(self, channel: str, enabled: bool, pwm: float) -> None:
        self.apply_channel_command(channel, {"enabled": 1.0 if enabled else 0.0, "pwm": pwm})

    def read_outputs(self) -> Dict[str, OutputState]:
        return {}

    def read_inputs(self) -> Dict[str, float]:
        return {}

    def update(self, dt: float) -> None:
        """Advance internal simulation and poll hardware."""

    def read_signal_values(self, signal_names: Iterable[str]) -> Dict[str, float]:
        raise NotImplementedError

    def simulation_profiles(self) -> Dict[str, SignalSimulationConfig]:
        return {}

    def update_simulation_profile(self, profile: SignalSimulationConfig) -> None:
        raise BackendError("Simulation profiles are not supported by this backend.")

class DummyBackend(BackendBase):
    name = "Dummy"

    def __init__(self) -> None:
        super().__init__()
        self._dbc = None
        self._time = 0.0
        self._signal_values: Dict[str, float] = {}
        self._simulation_configs: Dict[str, SignalSimulationConfig] = {}
        self._simulators: Dict[str, _SignalSimulator] = {}
        self._signal_to_message: Dict[str, str] = {}
        self._overrides: Dict[str, float] = {}
        self._channels: Dict[str, ChannelProfile] = {}
        self._runtimes: Dict[str, ChannelRuntime] = {}

    def start(self) -> None:
        self._time = 0.0

    def stop(self) -> None:
        pass

    def apply_database(self, database) -> None:
        self._dbc = database
        self._signal_values = {}
        self._simulation_configs = {}
        self._simulators = {}
        self._signal_to_message = {}
        self._overrides = {}
        if self._dbc is not None:
            for message in getattr(self._dbc, "messages", []):
                for signal in message.signals:
                    config = self._build_simulation_config(message, signal)
                    self._signal_to_message[signal.name] = message.name
                    self._simulation_configs[signal.name] = config
                    simulator = _SignalSimulator(config.clone())
                    simulator.apply_profile(config.clone())
                    self._simulators[signal.name] = simulator
                    self._signal_values[signal.name] = simulator.update(0.0, 0.0)
        self._refresh_overrides()

    def set_channel_profiles(self, profiles: Dict[str, ChannelProfile]) -> None:
        self._channels = profiles
        self._runtimes = {name: ChannelRuntime(profile) for name, profile in profiles.items()}
        self._refresh_overrides()

    def apply_channel_command(self, channel: str, command: Dict[str, float]) -> None:
        runtime = self._runtimes.get(channel)
        if runtime is None:
            return
        runtime.last_command = {key: float(value) for key, value in command.items()}
        channel_type = runtime.profile.type
        if channel_type in {"HighSide", "LowSide", "HBridge"}:
            enabled = bool(
                command.get("enabled", command.get("select", command.get("state", 0.0)))
                or command.get("enable", 0.0)
            )
            pwm = float(command.get("pwm", runtime.pwm))
            runtime.enabled = enabled
            runtime.pwm = max(0.0, min(100.0, pwm))
            if "direction" in command:
                runtime.direction = int(command.get("direction", runtime.direction))
        elif channel_type == "DO":
            state = float(command.get("state", command.get("enabled", 0.0)))
            runtime.enabled = state > 0.5
        elif channel_type in {"AO_0_10V", "AO_4_20mA"}:
            runtime.setpoint = float(command.get("setpoint", runtime.setpoint))
        elif channel_type == "HBridge":
            runtime.enabled = bool(command.get("enable", command.get("select", 0.0)))
            runtime.pwm = max(0.0, min(100.0, float(command.get("pwm", runtime.pwm))))
            runtime.direction = int(command.get("direction", runtime.direction))
        self._apply_runtime_to_signals(runtime)

    def read_signal_values(self, signal_names: Iterable[str]) -> Dict[str, float]:
        return {name: float(self._signal_values.get(name, 0.0)) for name in signal_names}

    def simulation_profiles(self) -> Dict[str, SignalSimulationConfig]:
        return {name: config.clone() for name, config in self._simulation_configs.items()}

    def update_simulation_profile(self, profile: SignalSimulationConfig) -> None:
        if profile.name not in self._simulation_configs:
            raise BackendError(f"Signal {profile.name} is not available for simulation")
        stored = profile.clone()
        self._simulation_configs[profile.name] = stored
        simulator = self._simulators.get(profile.name)
        if simulator is not None:
            simulator.apply_profile(stored.clone())
        if profile.name not in self._overrides:
            self._signal_values[profile.name] = simulator.update(self._time, 0.0) if simulator else 0.0

    def update(self, dt: float) -> None:
        self._time += dt
        for name, simulator in self._simulators.items():
            value = simulator.update(self._time, dt)
            self._signal_values[name] = value
        for runtime in self._runtimes.values():
            self._update_runtime(runtime, dt)
            self._apply_runtime_to_signals(runtime)

    def _build_simulation_config(self, message, signal) -> SignalSimulationConfig:
        category = "digital" if self._is_digital_signal(signal) else "analog"
        config = SignalSimulationConfig(
            message_name=message.name,
            name=signal.name,
            unit=signal.unit or "",
            minimum=signal.minimum,
            maximum=signal.maximum,
            category=category,
        )
        if category == "digital":
            config.digital = self._default_digital_profile(signal)
        else:
            config.analog = self._default_analog_profile(signal)
        return config

    def _is_digital_signal(self, signal) -> bool:
        if getattr(signal, "choices", None):
            return True
        if signal.length <= 1:
            return True
        if signal.minimum is not None and signal.maximum is not None:
            span = signal.maximum - signal.minimum
            if span <= 1.0 and signal.scale <= 1.0:
                return True
        return False

    def _default_analog_profile(self, signal) -> AnalogSimulationProfile:
        profile = AnalogSimulationProfile()
        minimum = float(signal.minimum) if signal.minimum is not None else 0.0
        maximum = float(signal.maximum) if signal.maximum is not None else minimum + 100.0
        if not math.isfinite(minimum) or not math.isfinite(maximum) or abs(maximum - minimum) > 1e6:
            minimum, maximum = 0.0, 100.0
        span = max(maximum - minimum, 1.0)
        profile.offset = minimum + span / 2.0
        profile.hold_value = profile.offset
        profile.noise = span * 0.01
        name_lower = signal.name.lower()
        if "current" in name_lower:
            profile.generator = "ramp"
            profile.slope = span / 60.0
            profile.noise = span * 0.02
        elif "voltage" in name_lower:
            profile.generator = "hold"
            profile.noise = span * 0.005
        elif "temp" in name_lower:
            profile.generator = "sine"
            profile.amplitude = span * 0.25
            profile.frequency = 0.02
        else:
            profile.generator = "noise"
            profile.amplitude = span * 0.1
        return profile

    def _default_digital_profile(self, signal) -> DigitalSimulationProfile:
        profile = DigitalSimulationProfile()
        minimum = signal.minimum if signal.minimum is not None else 0.0
        maximum = signal.maximum if signal.maximum is not None else 1.0
        if not math.isfinite(minimum) or not math.isfinite(maximum) or abs(float(maximum) - float(minimum)) > 1e6:
            minimum, maximum = 0.0, 1.0
        profile.low_value = float(minimum)
        profile.high_value = float(maximum)
        profile.manual_value = profile.low_value
        profile.period = 2.0
        profile.duty_cycle = 0.5
        profile.phase = random.random()
        return profile

    def _set_override(self, name: str, value: Optional[float]) -> None:
        simulator = self._simulators.get(name)
        if value is None:
            self._overrides.pop(name, None)
            if simulator is not None:
                simulator.set_override(None)
            return
        self._overrides[name] = float(value)
        if simulator is not None:
            simulator.set_override(float(value))
        self._signal_values[name] = float(value)

    def _refresh_overrides(self) -> None:
        for name in list(self._overrides):
            self._set_override(name, self._overrides[name])
        for runtime in self._runtimes.values():
            self._apply_runtime_to_signals(runtime)

    def _update_runtime(self, runtime: ChannelRuntime, dt: float) -> None:
        channel_type = runtime.profile.type
        sim_params = runtime.profile.sim
        if channel_type in {"HighSide", "LowSide"}:
            tau = float(sim_params.get("tau", 0.5))
            gain = float(sim_params.get("current_gain", 8.0))
            noise = float(sim_params.get("noise", 0.1))
            target = gain * (runtime.pwm / 100.0 if runtime.enabled else 0.0)
            runtime.current_value += (target - runtime.current_value) * min(dt / max(tau, 1e-3), 1.0)
            runtime.current_value += random.uniform(-noise, noise)
            runtime.current_value = max(0.0, runtime.current_value)
            runtime.status_cache["current"] = runtime.current_value
            runtime.status_cache["pwm_feedback"] = runtime.pwm if runtime.enabled else 0.0
        elif channel_type == "HBridge":
            tau = float(sim_params.get("tau", 0.3))
            gain = float(sim_params.get("current_gain", 5.0))
            noise = float(sim_params.get("noise", 0.05))
            target = gain * (runtime.pwm / 100.0 if runtime.enabled else 0.0)
            runtime.current_value += (target - runtime.current_value) * min(dt / max(tau, 1e-3), 1.0)
            runtime.current_value += random.uniform(-noise, noise)
            runtime.current_value = max(0.0, runtime.current_value)
            runtime.status_cache["current"] = runtime.current_value
            runtime.status_cache["direction"] = float(runtime.direction)
        elif channel_type in {"AO_0_10V", "AO_4_20mA"}:
            tau = float(sim_params.get("tau", 0.2))
            noise = float(sim_params.get("noise", 0.02))
            if "setpoint" in runtime.last_command:
                target = runtime.setpoint
            else:
                target = float(sim_params.get("target", runtime.current_value))
            runtime.current_value += (target - runtime.current_value) * min(dt / max(tau, 1e-3), 1.0)
            runtime.current_value += random.uniform(-noise, noise)
            runtime.status_cache["feedback"] = runtime.current_value
        elif channel_type == "DO":
            noise = float(sim_params.get("noise", 0.0))
            period = float(sim_params.get("pattern_period", 1.0))
            duty = float(sim_params.get("pattern_duty", 0.5))
            if runtime.enabled:
                value = 1.0
            elif period > 0:
                phase = (self._time % period) / max(period, 1e-6)
                value = 1.0 if phase < duty else 0.0
            else:
                value = 0.0
            value += random.uniform(-noise, noise)
            runtime.status_cache["feedback"] = max(0.0, min(1.0, value))
        elif channel_type == "DI":
            period = float(sim_params.get("pattern_period", 1.0))
            duty = float(sim_params.get("pattern_duty", 0.5))
            if period > 0:
                phase = (self._time % period) / max(period, 1e-6)
                runtime.status_cache["state"] = 1.0 if phase < duty else 0.0
            else:
                runtime.status_cache["state"] = 0.0
        elif channel_type in {"AI_V", "AI_I"}:
            pass

    def _apply_runtime_to_signals(self, runtime: ChannelRuntime) -> None:
        profile = runtime.profile
        for semantic, signal in profile.write.fields.items():
            value = runtime.last_command.get(semantic)
            if value is None:
                if semantic in {"select", "enable"}:
                    value = 1.0 if runtime.enabled else 0.0
                elif semantic == "pwm":
                    value = runtime.pwm
                elif semantic == "direction":
                    value = runtime.direction
                elif semantic == "setpoint":
                    value = runtime.setpoint
                elif semantic == "state":
                    value = 1.0 if runtime.enabled else 0.0
            self._set_override(signal, value)
        for semantic, signal in profile.status.fields.items():
            value = runtime.status_cache.get(semantic)
            if value is not None:
                self._set_override(signal, value)

class RealBackend(QObject, BackendBase):
    name = "Real"

    connection_changed = pyqtSignal(bool, str)
    status_updated = pyqtSignal()

    def __init__(self) -> None:
        QObject.__init__(self)
        BackendBase.__init__(self)
        self._settings = ConnectionSettings("", "socketcan", "vcan0", 500000)
        self._db = None
        self._bus = None
        self._notifier = None
        self._lock = threading.Lock()
        self._signal_cache: Dict[str, float] = {}
        self._signal_to_message: Dict[str, str] = {}
        self._channels: Dict[str, ChannelProfile] = {}
        self._status_handlers: Dict[int, Tuple[str, ChannelProfile]] = {}

    def configure(self, settings: ConnectionSettings) -> None:
        self._settings = settings

    def start(self) -> None:
        if not self._settings.dbc_path:
            raise BackendError("DBC path is required")
        try:
            self._db = cantools.database.load_file(self._settings.dbc_path, strict=False)
        except (OSError, cantools_errors.Error, cantools_errors.ParseError) as exc:
            raise BackendError(str(exc))
        try:
            self._bus = can.interface.Bus(
                bustype=self._settings.bustype,
                channel=self._settings.channel,
                bitrate=self._settings.bitrate,
            )
        except (can.CanError, OSError, ValueError) as exc:
            raise BackendError(str(exc))
        if self._db is None or self._bus is None:
            raise BackendError("Failed to initialize Real backend")
        self.apply_database(self._db)
        listener = _StatusListener(self)
        self._notifier = can.Notifier(self._bus, [listener], 0.1)
        self.connection_changed.emit(True, "Connected")

    def stop(self) -> None:
        if self._notifier is not None:
            self._notifier.stop()
        self._notifier = None
        if self._bus is not None:
            try:
                self._bus.shutdown()
            except can.CanError:
                pass
        self._bus = None
        self.connection_changed.emit(False, "Disconnected")

    def apply_database(self, database) -> None:
        self._db = database
        self._signal_cache = {}
        self._signal_to_message = {}
        if self._db is not None:
            for message in getattr(self._db, "messages", []):
                for signal in message.signals:
                    self._signal_to_message[signal.name] = message.name
        with self._lock:
            self._signal_cache = {name: 0.0 for name in self._signal_to_message}
        self.status_updated.emit()

    def set_channel_profiles(self, profiles: Dict[str, ChannelProfile]) -> None:
        self._channels = profiles
        self._status_handlers = {}
        if not self._db:
            return
        for profile in profiles.values():
            message_name = profile.status.message
            message = self._db.get_message_by_name(message_name) if message_name else None
            if message is None:
                continue
            self._status_handlers[message.frame_id] = (message_name, profile)

    def apply_channel_command(self, channel: str, command: Dict[str, float]) -> None:
        profile = self._channels.get(channel)
        if profile is None or self._db is None or self._bus is None:
            return
        message_name = profile.write.message
        if not message_name:
            raise BackendError(f"Channel {channel} has no write message configured")
        message = self._db.get_message_by_name(message_name)
        if message is None:
            raise BackendError(f"Message {message_name} not found in DBC")
        payload: Dict[str, float] = {}
        for semantic, signal_name in profile.write.fields.items():
            raw_value = command.get(semantic)
            if raw_value is None:
                if semantic in {"select", "enable", "state"}:
                    raw_value = 1.0 if command.get("enabled", 0.0) else 0.0
                elif semantic == "pwm":
                    raw_value = command.get("pwm", 0.0)
                elif semantic == "setpoint":
                    raw_value = command.get("setpoint", 0.0)
                elif semantic == "direction":
                    raw_value = command.get("direction", 0.0)
            if raw_value is None:
                continue
            if not is_signal_writable(signal_name, message_name):
                raise BackendError(f"Signal {signal_name} is not writable")
            payload[signal_name] = float(raw_value)
        if not payload:
            return
        try:
            data = message.encode(payload, scaling=True, strict=True)
            can_message = can.Message(arbitration_id=message.frame_id, data=data, is_extended_id=False)
            self._bus.send(can_message)
            with self._lock:
                for name, value in payload.items():
                    self._signal_cache[name] = float(value)
        except (ValueError, can.CanError) as exc:
            raise BackendError(str(exc))

    def update(self, dt: float) -> None:
        _ = dt

    def read_signal_values(self, signal_names: Iterable[str]) -> Dict[str, float]:
        with self._lock:
            return {name: float(self._signal_cache.get(name, 0.0)) for name in signal_names}

    def _handle_status(self, message: can.Message) -> None:
        if self._db is None:
            return
        handler = self._status_handlers.get(message.arbitration_id)
        if handler is None:
            return
        message_name, _profile = handler
        dbc_message = self._db.get_message_by_name(message_name)
        if dbc_message is None:
            return
        try:
            decoded = dbc_message.decode(message.data, decode_choices=False, scaling=True)
        except (ValueError, KeyError):
            return
        with self._lock:
            for name, value in decoded.items():
                self._signal_cache[name] = float(value)
        self.status_updated.emit()

class _StatusListener(can.Listener):
    def __init__(self, backend: "RealBackend") -> None:
        super().__init__()
        self._backend = backend

    def on_message_received(self, message: can.Message) -> None:  # type: ignore[override]
        self._backend._handle_status(message)


class Logger:
    def __init__(self) -> None:
        self._file = None
        self._path = ""
        self._started = False
        self._signal_names: List[str] = []
        self._last_error = ""

    def start(self, path: str, signal_names: Iterable[str]) -> bool:
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
        ordered = [timestamp] + [f"{values.get(name, 0.0):.6f}" for name in self._signal_names]
        self._file.write(",".join(ordered) + "\n")

    def is_running(self) -> bool:
        return self._started

    @property
    def signal_names(self) -> List[str]:
        return list(self._signal_names)

    @property
    def last_error(self) -> str:
        return self._last_error

class SignalBrowserWidget(QWidget):
    add_requested = pyqtSignal(list)
    plot_requested = pyqtSignal(list)
    simulate_requested = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._signals: Dict[str, List[SignalDefinition]] = {}
        self._allow_simulation = False
        layout = QVBoxLayout(self)
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
        self.tree.itemDoubleClicked.connect(self._on_double_clicked)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self.tree)
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        add_button = QPushButton("Add to Watchlist")
        add_button.clicked.connect(self._emit_selection)
        button_layout.addWidget(add_button)
        layout.addLayout(button_layout)

    def set_signals(self, signals: Dict[str, List[SignalDefinition]]) -> None:
        self._signals = signals
        self.tree.clear()
        for message_name in sorted(self._signals):
            parent = QTreeWidgetItem([message_name, "", "", "", ""])
            parent.setData(0, Qt.UserRole, ("message", message_name))
            for definition in self._signals[message_name]:
                scaling = f"{definition.scale:g} * raw"
                if definition.offset:
                    scaling += f" + {definition.offset:g}"
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

    def _emit_selection(self) -> None:
        names = self.selected_signal_names()
        if names:
            self.add_requested.emit(names)

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

    def _on_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        data = item.data(0, Qt.UserRole)
        if isinstance(data, tuple) and data and data[0] == "signal":
            definition: SignalDefinition = data[1]
            self.add_requested.emit([definition.name])

    def _on_context_menu(self, position) -> None:
        item = self.tree.itemAt(position)
        if not item:
            return
        data = item.data(0, Qt.UserRole)
        if not isinstance(data, tuple) or data[0] != "signal":
            return
        definition: SignalDefinition = data[1]
        menu = QMenu(self)
        plot_action = QAction("→ Plot", self)
        plot_action.triggered.connect(lambda: self.plot_requested.emit([definition.name]))
        menu.addAction(plot_action)
        if self._allow_simulation:
            sim_action = QAction("→ Simulate…", self)
            sim_action.triggered.connect(lambda: self.simulate_requested.emit(definition.name))
            menu.addAction(sim_action)
        menu.exec_(self.tree.viewport().mapToGlobal(position))

    def set_allow_simulation(self, enabled: bool) -> None:
        self._allow_simulation = enabled


class WatchlistWidget(QWidget):
    remove_requested = pyqtSignal(list)
    plot_toggled = pyqtSignal(str, bool)

    def __init__(self) -> None:
        super().__init__()
        self._order: List[str] = []
        self._units: Dict[str, str] = {}
        self._last_update: Dict[str, datetime.datetime] = {}
        self._plot_checkboxes: Dict[str, QCheckBox] = {}
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Signal", "Value", "Unit", "Last update", "Plot"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        layout.addWidget(self.table)
        button_layout = QHBoxLayout()
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self._emit_remove)
        button_layout.addStretch(1)
        button_layout.addWidget(remove_button)
        layout.addLayout(button_layout)

    def set_units(self, units: Dict[str, str]) -> None:
        self._units = dict(units)
        for row, name in enumerate(self._order):
            unit = self._units.get(name, "")
            self._update_cell(row, 2, unit)

    def add_signals(self, names: Iterable[str]) -> List[str]:
        added: List[str] = []
        for name in names:
            if name in self._order:
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._update_cell(row, 0, name)
            self._update_cell(row, 1, "0.0000")
            self._update_cell(row, 2, self._units.get(name, ""))
            self._update_cell(row, 3, "-")
            checkbox = QCheckBox()
            checkbox.stateChanged.connect(lambda state, signal=name: self._on_plot_toggle(signal, state))
            self.table.setCellWidget(row, 4, checkbox)
            self._plot_checkboxes[name] = checkbox
            self._order.append(name)
            added.append(name)
        return added

    def set_plot_enabled(self, name: str, enabled: bool) -> None:
        checkbox = self._plot_checkboxes.get(name)
        if checkbox:
            checkbox.setChecked(enabled)

    def remove_signals(self, names: Iterable[str]) -> None:
        to_remove = {name for name in names}
        rows = [index for index, name in enumerate(self._order) if name in to_remove]
        for row in reversed(rows):
            name = self._order.pop(row)
            self.table.removeRow(row)
            self._last_update.pop(name, None)
            checkbox = self._plot_checkboxes.pop(name, None)
            if checkbox:
                checkbox.deleteLater()

    def update_values(self, values: Dict[str, float]) -> None:
        now = datetime.datetime.now()
        for row, name in enumerate(self._order):
            value = values.get(name)
            display = "n/a" if value is None else f"{value:.4f}"
            self._update_cell(row, 1, display)
            if value is not None:
                self._last_update[name] = now
                self._update_cell(row, 3, now.isoformat(timespec="seconds"))

    def selected_signal_names(self) -> List[str]:
        rows = {index.row() for index in self.table.selectionModel().selectedRows()}
        return [self._order[row] for row in sorted(rows) if 0 <= row < len(self._order)]

    def _update_cell(self, row: int, column: int, text: str) -> None:
        item = self.table.item(row, column)
        if item is None:
            item = QTableWidgetItem(text)
            self.table.setItem(row, column, item)
        else:
            item.setText(text)

    def _emit_remove(self) -> None:
        names = self.selected_signal_names()
        if names:
            self.remove_requested.emit(names)

    def _on_plot_toggle(self, name: str, state: int) -> None:
        self.plot_toggled.emit(name, state == Qt.Checked)

    @property
    def signal_names(self) -> List[str]:
        return list(self._order)

    @property
    def plot_signal_names(self) -> List[str]:
        return [name for name, checkbox in self._plot_checkboxes.items() if checkbox.isChecked()]


class MultiAxisPlotDock(QDockWidget):
    closed = pyqtSignal(int)

    def __init__(self, identifier: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.identifier = identifier
        self.setWindowTitle(f"Plot Window {identifier}")
        self.setObjectName(f"MultiAxisPlotDock{identifier}")
        self._paused = False
        self._signals: Dict[str, dict] = {}
        self._legend = None
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(4, 4, 4, 4)
        controls_layout.addStretch(1)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self._toggle_pause)
        controls_layout.addWidget(self.pause_button)
        self.save_button = QPushButton("Save PNG")
        self.save_button.clicked.connect(self._save_png)
        controls_layout.addWidget(self.save_button)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_all)
        controls_layout.addWidget(self.clear_button)
        layout.addLayout(controls_layout)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMinimumHeight(220)
        self._legend = self.plot_widget.addLegend()
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setMenuEnabled(False)
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.showAxis('right')
        self.right_view = pg.ViewBox()
        self.plot_item.scene().addItem(self.right_view)
        self.plot_item.getAxis('right').linkToView(self.right_view)
        self.right_view.setXLink(self.plot_item.vb)
        self.plot_item.vb.sigResized.connect(self._update_views)
        self._update_views()
        layout.addWidget(self.plot_widget)
        self.setWidget(container)

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self.pause_button.setText("Resume" if self._paused else "Pause")

    def _save_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save plot", os.path.join(BASE_DIR, f"plot_{self.identifier}.png"), "PNG Files (*.png)")
        if not path:
            return
        pixmap = self.plot_widget.grab()
        pixmap.save(path, "PNG")

    def _update_views(self) -> None:
        rect = self.plot_item.vb.sceneBoundingRect()
        if rect:
            self.right_view.setGeometry(rect)
        self.right_view.linkedViewChanged(self.plot_item.vb, self.right_view.XAxis)

    def add_signal(self, name: str, unit: str, side: Optional[str] = None) -> None:
        side = (side or "left").lower()
        if side not in {"left", "right"}:
            side = "left"
        existing = self._signals.get(name)
        if existing:
            if existing["side"] == side:
                return
            self.remove_signal(name)
        pen = pg.intColor(len(self._signals))
        curve = pg.PlotDataItem(pen=pen)
        curve.setZValue(1)
        if side == "left":
            self.plot_item.addItem(curve)
        else:
            self.right_view.addItem(curve)
        if self._legend:
            self._legend.addItem(curve, name)
        self._signals[name] = {
            "buffer": deque(),
            "curve": curve,
            "side": side,
            "unit": unit or "",
        }

    def remove_signal(self, name: str) -> None:
        info = self._signals.pop(name, None)
        if not info:
            return
        curve = info["curve"]
        if info["side"] == "left":
            self.plot_item.removeItem(curve)
        else:
            self.right_view.removeItem(curve)
        if self._legend:
            try:
                self._legend.removeItem(curve)
            except Exception:
                pass

    def clear_all(self) -> None:
        for info in self._signals.values():
            buffer: deque = info["buffer"]
            buffer.clear()
            info["curve"].setData([], [])

    def update(self, timestamp: float, values: Dict[str, Optional[float]]) -> None:
        if not self._signals:
            return
        cutoff = timestamp - 60.0
        base_time: Optional[float] = None
        for name, info in self._signals.items():
            buffer: deque = info["buffer"]
            value = values.get(name)
            if value is not None:
                buffer.append((timestamp, float(value)))
            while buffer and buffer[0][0] < cutoff:
                buffer.popleft()
            if buffer:
                if base_time is None or buffer[0][0] < base_time:
                    base_time = buffer[0][0]
        if base_time is None:
            return
        if self._paused:
            return
        for name, info in self._signals.items():
            buffer: deque = info["buffer"]
            if not buffer:
                continue
            times = [entry[0] - base_time for entry in buffer]
            samples = [entry[1] for entry in buffer]
            if len(times) > 5_000:
                step = max(1, math.ceil(len(times) / 5_000))
                dec_times = times[::step]
                dec_samples = samples[::step]
                if dec_times[-1] != times[-1]:
                    dec_times.append(times[-1])
                    dec_samples.append(samples[-1])
                times, samples = dec_times, dec_samples
            curve = info["curve"]
            curve.setData(times, samples)
        self.plot_item.enableAutoRange('y', True)
        self.right_view.enableAutoRange(axis=pg.ViewBox.YAxis)

    def assigned_signals(self) -> Dict[str, Tuple[str, str]]:
        return {name: (info["unit"], info["side"]) for name, info in self._signals.items()}

    def closeEvent(self, event) -> None:
        self.closed.emit(self.identifier)
        super().closeEvent(event)

class SignalSimulationDialog(QDialog):
    def __init__(self, config: SignalSimulationConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Simulate – {config.name}")
        self._config = config.clone()
        self.result_profile: Optional[SignalSimulationConfig] = None
        layout = QVBoxLayout(self)
        self.form = QFormLayout()
        layout.addLayout(self.form)
        if self._config.category == "analog":
            self._build_analog_controls()
        else:
            self._build_digital_controls()
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _build_analog_controls(self) -> None:
        profile = self._config.analog or AnalogSimulationProfile()
        self._config.analog = profile
        self.analog_generator = QComboBox()
        self.analog_generator.addItems(["hold", "sine", "ramp", "noise"])
        self.analog_generator.setCurrentText(profile.generator)
        self.form.addRow("Generator", self.analog_generator)
        self.analog_offset = self._spin_box(profile.offset, -1_000_000.0, 1_000_000.0, 0.1)
        self.form.addRow("Offset", self.analog_offset)
        self.analog_amplitude = self._spin_box(profile.amplitude, 0.0, 1_000_000.0, 0.1)
        self.form.addRow("Amplitude", self.analog_amplitude)
        self.analog_frequency = self._spin_box(profile.frequency, 0.0, 10.0, 0.01)
        self.form.addRow("Frequency", self.analog_frequency)
        self.analog_slope = self._spin_box(profile.slope, -1_000.0, 1_000.0, 0.1)
        self.form.addRow("Slope", self.analog_slope)
        self.analog_noise = self._spin_box(profile.noise, 0.0, 1_000_000.0, 0.1)
        self.form.addRow("Noise", self.analog_noise)
        self.analog_hold = self._spin_box(profile.hold_value, -1_000_000.0, 1_000_000.0, 0.1)
        self.form.addRow("Hold value", self.analog_hold)
        self.analog_phase = self._spin_box(profile.phase, -math.pi, math.pi, 0.01)
        self.form.addRow("Phase", self.analog_phase)

    def _build_digital_controls(self) -> None:
        profile = self._config.digital or DigitalSimulationProfile()
        self._config.digital = profile
        self.digital_mode = QComboBox()
        self.digital_mode.addItems(["pattern", "manual"])
        self.digital_mode.setCurrentText(profile.mode)
        self.form.addRow("Mode", self.digital_mode)
        self.digital_period = self._spin_box(profile.period, 0.01, 100.0, 0.01)
        self.form.addRow("Period (s)", self.digital_period)
        self.digital_duty = self._spin_box(profile.duty_cycle, 0.0, 1.0, 0.01)
        self.form.addRow("Duty", self.digital_duty)
        self.digital_high = self._spin_box(profile.high_value, -1_000_000.0, 1_000_000.0, 0.1)
        self.form.addRow("High value", self.digital_high)
        self.digital_low = self._spin_box(profile.low_value, -1_000_000.0, 1_000_000.0, 0.1)
        self.form.addRow("Low value", self.digital_low)
        self.digital_manual = self._spin_box(profile.manual_value, -1_000_000.0, 1_000_000.0, 0.1)
        self.form.addRow("Manual value", self.digital_manual)

    def _spin_box(self, value: float, minimum: float, maximum: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(4)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _accept(self) -> None:
        updated = self._config.clone()
        if updated.category == "analog" and updated.analog:
            updated.analog.generator = self.analog_generator.currentText()
            updated.analog.offset = float(self.analog_offset.value())
            updated.analog.amplitude = float(self.analog_amplitude.value())
            updated.analog.frequency = float(self.analog_frequency.value())
            updated.analog.slope = float(self.analog_slope.value())
            updated.analog.noise = float(self.analog_noise.value())
            updated.analog.hold_value = float(self.analog_hold.value())
            updated.analog.phase = float(self.analog_phase.value())
        elif updated.category == "digital" and updated.digital:
            updated.digital.mode = self.digital_mode.currentText()
            updated.digital.period = float(self.digital_period.value())
            updated.digital.duty_cycle = float(self.digital_duty.value())
            updated.digital.high_value = float(self.digital_high.value())
            updated.digital.low_value = float(self.digital_low.value())
            updated.digital.manual_value = float(self.digital_manual.value())
        self.result_profile = updated
        self.accept()

class DummySimulationWidget(QGroupBox):
    profile_changed = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__("Dummy Signal Simulation")
        self._profiles: Dict[str, SignalSimulationConfig] = {}
        self._items_by_signal: Dict[str, QTreeWidgetItem] = {}
        self._current_name: Optional[str] = None
        self._updating = False
        layout = QVBoxLayout(self)
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.textChanged.connect(self._apply_filter)
        search_layout.addWidget(self.search_edit)
        layout.addLayout(search_layout)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Signal", "Type", "Generator/Mode"])
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.tree)
        self.analog_group = QGroupBox("Analog Generator")
        analog_layout = QGridLayout()
        self.analog_generator_combo = QComboBox()
        self.analog_generator_combo.addItems(["hold", "sine", "ramp", "noise"])
        self.analog_generator_combo.currentTextChanged.connect(self._on_analog_generator)
        analog_layout.addWidget(QLabel("Generator"), 0, 0)
        analog_layout.addWidget(self.analog_generator_combo, 0, 1)
        self.analog_offset_spin = self._create_spin(-1_000_000.0, 1_000_000.0, 0.1)
        analog_layout.addWidget(QLabel("Offset"), 1, 0)
        analog_layout.addWidget(self.analog_offset_spin, 1, 1)
        self.analog_amplitude_spin = self._create_spin(0.0, 1_000_000.0, 0.1)
        analog_layout.addWidget(QLabel("Amplitude"), 2, 0)
        analog_layout.addWidget(self.analog_amplitude_spin, 2, 1)
        self.analog_frequency_spin = self._create_spin(0.0, 10.0, 0.01)
        analog_layout.addWidget(QLabel("Frequency (Hz)"), 3, 0)
        analog_layout.addWidget(self.analog_frequency_spin, 3, 1)
        self.analog_slope_spin = self._create_spin(0.0, 1_000_000.0, 1.0)
        analog_layout.addWidget(QLabel("Slope (unit/s)"), 4, 0)
        analog_layout.addWidget(self.analog_slope_spin, 4, 1)
        self.analog_noise_spin = self._create_spin(0.0, 1_000_000.0, 0.1)
        analog_layout.addWidget(QLabel("Noise"), 5, 0)
        analog_layout.addWidget(self.analog_noise_spin, 5, 1)
        self.analog_hold_spin = self._create_spin(-1_000_000.0, 1_000_000.0, 0.1)
        analog_layout.addWidget(QLabel("Hold value"), 6, 0)
        analog_layout.addWidget(self.analog_hold_spin, 6, 1)
        self.analog_phase_spin = self._create_spin(-math.pi, math.pi, 0.01)
        analog_layout.addWidget(QLabel("Phase (rad)"), 7, 0)
        analog_layout.addWidget(self.analog_phase_spin, 7, 1)
        self.analog_group.setLayout(analog_layout)
        layout.addWidget(self.analog_group)
        self.digital_group = QGroupBox("Digital Pattern")
        digital_layout = QGridLayout()
        self.digital_mode_combo = QComboBox()
        self.digital_mode_combo.addItems(["pattern", "manual"])
        self.digital_mode_combo.currentTextChanged.connect(self._on_digital_mode)
        digital_layout.addWidget(QLabel("Mode"), 0, 0)
        digital_layout.addWidget(self.digital_mode_combo, 0, 1)
        self.digital_period_spin = self._create_spin(0.1, 3600.0, 0.1)
        digital_layout.addWidget(QLabel("Period (s)"), 1, 0)
        digital_layout.addWidget(self.digital_period_spin, 1, 1)
        self.digital_duty_spin = self._create_spin(0.0, 1.0, 0.05)
        digital_layout.addWidget(QLabel("Duty"), 2, 0)
        digital_layout.addWidget(self.digital_duty_spin, 2, 1)
        self.digital_high_spin = self._create_spin(-1_000_000.0, 1_000_000.0, 0.1)
        digital_layout.addWidget(QLabel("High value"), 3, 0)
        digital_layout.addWidget(self.digital_high_spin, 3, 1)
        self.digital_low_spin = self._create_spin(-1_000_000.0, 1_000_000.0, 0.1)
        digital_layout.addWidget(QLabel("Low value"), 4, 0)
        digital_layout.addWidget(self.digital_low_spin, 4, 1)
        self.digital_manual_spin = self._create_spin(-1_000_000.0, 1_000_000.0, 0.1)
        digital_layout.addWidget(QLabel("Manual value"), 5, 0)
        digital_layout.addWidget(self.digital_manual_spin, 5, 1)
        self.digital_phase_spin = self._create_spin(-10.0, 10.0, 0.1)
        digital_layout.addWidget(QLabel("Phase (s)"), 6, 0)
        digital_layout.addWidget(self.digital_phase_spin, 6, 1)
        self.digital_group.setLayout(digital_layout)
        layout.addWidget(self.digital_group)
        self._connect_spin_signals()
        self._update_editor_visibility(None)

    def set_profiles(self, profiles: Dict[str, SignalSimulationConfig]) -> None:
        self._profiles = {name: config.clone() for name, config in profiles.items()}
        self._items_by_signal.clear()
        self.tree.clear()
        for name in sorted(self._profiles):
            profile = self._profiles[name]
            item = QTreeWidgetItem([name, profile.category, self._profile_summary(profile)])
            self.tree.addTopLevelItem(item)
            self._items_by_signal[name] = item
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)

    def _create_spin(self, minimum: float, maximum: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.valueChanged.connect(self._on_spin_changed)
        return spin

    def _connect_spin_signals(self) -> None:
        self.analog_offset_spin.valueChanged.connect(self._on_analog_changed)
        self.analog_amplitude_spin.valueChanged.connect(self._on_analog_changed)
        self.analog_frequency_spin.valueChanged.connect(self._on_analog_changed)
        self.analog_slope_spin.valueChanged.connect(self._on_analog_changed)
        self.analog_noise_spin.valueChanged.connect(self._on_analog_changed)
        self.analog_hold_spin.valueChanged.connect(self._on_analog_changed)
        self.analog_phase_spin.valueChanged.connect(self._on_analog_changed)
        self.digital_period_spin.valueChanged.connect(self._on_digital_changed)
        self.digital_duty_spin.valueChanged.connect(self._on_digital_changed)
        self.digital_high_spin.valueChanged.connect(self._on_digital_changed)
        self.digital_low_spin.valueChanged.connect(self._on_digital_changed)
        self.digital_manual_spin.valueChanged.connect(self._on_digital_changed)
        self.digital_phase_spin.valueChanged.connect(self._on_digital_changed)

    def _apply_filter(self, text: str) -> None:
        text = text.lower().strip()
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            signal_name = item.text(0).lower()
            category = item.text(1).lower()
            summary = item.text(2).lower()
            matches = not text or text in signal_name or text in category or text in summary
            item.setHidden(not matches)

    def _on_selection_changed(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            self._current_name = None
            self._update_editor_visibility(None)
            return
        item = items[0]
        name = item.text(0)
        self._current_name = name
        profile = self._profiles.get(name)
        if not profile:
            return
        self._updating = True
        try:
            if profile.category == "analog" and profile.analog is not None:
                analog = profile.analog
                self.analog_generator_combo.setCurrentText(analog.generator)
                self.analog_offset_spin.setValue(analog.offset)
                self.analog_amplitude_spin.setValue(analog.amplitude)
                self.analog_frequency_spin.setValue(analog.frequency)
                self.analog_slope_spin.setValue(analog.slope)
                self.analog_noise_spin.setValue(analog.noise)
                self.analog_hold_spin.setValue(analog.hold_value)
                self.analog_phase_spin.setValue(analog.phase)
            if profile.category == "digital" and profile.digital is not None:
                digital = profile.digital
                self.digital_mode_combo.setCurrentText(digital.mode)
                self.digital_period_spin.setValue(digital.period)
                self.digital_duty_spin.setValue(digital.duty_cycle)
                self.digital_high_spin.setValue(digital.high_value)
                self.digital_low_spin.setValue(digital.low_value)
                self.digital_manual_spin.setValue(digital.manual_value)
                self.digital_phase_spin.setValue(digital.phase)
        finally:
            self._updating = False
        self._update_editor_visibility(profile.category)

    def _update_editor_visibility(self, category: Optional[str]) -> None:
        self.analog_group.setVisible(category == "analog")
        self.digital_group.setVisible(category == "digital")
        self.analog_group.setEnabled(category == "analog")
        self.digital_group.setEnabled(category == "digital")

    def _on_spin_changed(self, _value: float) -> None:
        pass

    def _on_analog_generator(self, text: str) -> None:
        if self._updating or not self._current_name:
            return
        profile = self._profiles.get(self._current_name)
        if profile and profile.analog is not None:
            profile.analog.generator = text
            self._emit_profile_update(profile)

    def _on_analog_changed(self) -> None:
        if self._updating or not self._current_name:
            return
        profile = self._profiles.get(self._current_name)
        if profile and profile.analog is not None:
            profile.analog.offset = self.analog_offset_spin.value()
            profile.analog.amplitude = self.analog_amplitude_spin.value()
            profile.analog.frequency = self.analog_frequency_spin.value()
            profile.analog.slope = self.analog_slope_spin.value()
            profile.analog.noise = self.analog_noise_spin.value()
            profile.analog.hold_value = self.analog_hold_spin.value()
            profile.analog.phase = self.analog_phase_spin.value()
            self._emit_profile_update(profile)

    def _on_digital_mode(self, text: str) -> None:
        if self._updating or not self._current_name:
            return
        profile = self._profiles.get(self._current_name)
        if profile and profile.digital is not None:
            profile.digital.mode = text
            self._emit_profile_update(profile)

    def _on_digital_changed(self) -> None:
        if self._updating or not self._current_name:
            return
        profile = self._profiles.get(self._current_name)
        if profile and profile.digital is not None:
            profile.digital.period = self.digital_period_spin.value()
            profile.digital.duty_cycle = self.digital_duty_spin.value()
            profile.digital.high_value = self.digital_high_spin.value()
            profile.digital.low_value = self.digital_low_spin.value()
            profile.digital.manual_value = self.digital_manual_spin.value()
            profile.digital.phase = self.digital_phase_spin.value()
            self._emit_profile_update(profile)

    def _emit_profile_update(self, profile: SignalSimulationConfig) -> None:
        item = self._items_by_signal.get(profile.name)
        if item is not None:
            item.setText(2, self._profile_summary(profile))
        self.profile_changed.emit(profile.clone())

    def _profile_summary(self, profile: SignalSimulationConfig) -> str:
        if profile.category == "digital" and profile.digital is not None:
            if profile.digital.mode == "manual":
                return "manual"
            return f"pattern {profile.digital.duty_cycle * 100:.0f}%/{profile.digital.period:.1f}s"
        if profile.analog is not None:
            return profile.analog.generator
        return ""

class ChannelCardWidget(QGroupBox):
    command_requested = pyqtSignal(str, dict)
    sequencer_requested = pyqtSignal(str, str)
    sequencer_config_changed = pyqtSignal(str, object)
    plot_visibility_changed = pyqtSignal(str, bool)
    simulation_changed = pyqtSignal(str, dict)

    def __init__(self, profile: ChannelProfile) -> None:
        super().__init__(profile.name)
        self.profile = profile
        self.state_label = QLabel("Idle")
        self.enabled_checkbox = QCheckBox("Enabled")
        self.pwm_slider = QSlider(Qt.Horizontal)
        self.pwm_slider.setRange(0, 100)
        self.pwm_value = QLabel("0 %")
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["Reverse", "Neutral", "Forward"])
        self.setpoint_spin = QDoubleSpinBox()
        self.setpoint_spin.setDecimals(2)
        self.setpoint_spin.setRange(-1_000.0, 1_000.0)
        self.quick_off = QPushButton("OFF")
        self.quick_low = QPushButton("20 %")
        self.quick_mid = QPushButton("50 %")
        self.quick_max = QPushButton("100 %")
        self.apply_button = QPushButton("Apply")
        self.off_button = QPushButton("All OFF")
        self.sequence_table = QTableWidget(0, 7)
        self.sequence_table.setHorizontalHeaderLabels(
            ["Order", "Name", "Duration [s]", "PWM [%]", "On [s]", "Off [s]", "Active"]
        )
        self.sequence_table.verticalHeader().setVisible(False)
        self.sequence_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sequence_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.sequence_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.sequence_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for column in range(2, 6):
            self.sequence_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.sequence_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.sequence_table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.sequence_add = QPushButton("Add")
        self.sequence_delete = QPushButton("Delete")
        self.sequence_duplicate = QPushButton("Duplicate")
        self.sequence_up = QPushButton("Up")
        self.sequence_down = QPushButton("Down")
        self.sequence_start = QPushButton("Start")
        self.sequence_stop = QPushButton("Stop")
        self.sequence_reset = QPushButton("Reset")
        self.sequence_stop.setEnabled(False)
        self.sequence_reset.setEnabled(False)
        self.sequence_status = QLabel("Sequence idle")
        self.repeat_off_radio = QRadioButton("Off")
        self.repeat_endless_radio = QRadioButton("Endless")
        self.repeat_limit_radio = QRadioButton("Limit")
        self.repeat_off_radio.setChecked(True)
        self.repeat_time_edit = QTimeEdit()
        self.repeat_time_edit.setDisplayFormat("HH:mm:ss")
        self.repeat_time_edit.setTime(QTime(0, 5, 0))
        self.repeat_time_edit.setEnabled(False)
        self.values_view = QTextBrowser()
        self.values_view.setFixedHeight(90)
        self.plot_checkbox = QCheckBox("Show plot")
        self.sim_button = QToolButton()
        self.sim_button.setText("⚙ Sim")
        self.sim_button.clicked.connect(self._open_sim_dialog)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMinimumHeight(140)
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addLegend()
        self.plot_widget.hide()
        self.command_curve = self.plot_widget.plot(name="Command", pen=pg.mkPen(QColor("orange"), width=2))
        self.feedback_curve = self.plot_widget.plot(name="Feedback", pen=pg.mkPen(QColor("steelblue"), width=2))
        self._plot_times: deque[float] = deque()
        self._plot_command: deque[float] = deque()
        self._plot_feedback: deque[float] = deque()
        self._updating_sequence_table = False
        self._sequence_running = False
        self._build_layout()
        self._connect_signals()
        self._update_visibility()
        self._update_sequence_buttons()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self.state_label)
        control_layout = QGridLayout()
        row = 0
        control_layout.addWidget(self.enabled_checkbox, row, 0, 1, 2)
        control_layout.addWidget(self.plot_checkbox, row, 2)
        row += 1
        control_layout.addWidget(QLabel("PWM"), row, 0)
        control_layout.addWidget(self.pwm_slider, row, 1)
        control_layout.addWidget(self.pwm_value, row, 2)
        row += 1
        control_layout.addWidget(QLabel("Direction"), row, 0)
        control_layout.addWidget(self.direction_combo, row, 1)
        row += 1
        control_layout.addWidget(QLabel("Setpoint"), row, 0)
        control_layout.addWidget(self.setpoint_spin, row, 1)
        row += 1
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.quick_off)
        buttons_layout.addWidget(self.quick_low)
        buttons_layout.addWidget(self.quick_mid)
        buttons_layout.addWidget(self.quick_max)
        buttons_layout.addStretch(1)
        control_layout.addLayout(buttons_layout, row, 0, 1, 3)
        row += 1
        action_layout = QHBoxLayout()
        action_layout.addWidget(self.apply_button)
        action_layout.addWidget(self.off_button)
        action_layout.addStretch(1)
        action_layout.addWidget(self.sim_button)
        control_layout.addLayout(action_layout, row, 0, 1, 3)
        layout.addLayout(control_layout)
        sequencer_group = QGroupBox("Sequencer")
        seq_layout = QVBoxLayout()
        seq_layout.addWidget(self.sequence_table)
        toolbar_layout = QHBoxLayout()
        toolbar_layout.addWidget(self.sequence_add)
        toolbar_layout.addWidget(self.sequence_delete)
        toolbar_layout.addWidget(self.sequence_duplicate)
        toolbar_layout.addWidget(self.sequence_up)
        toolbar_layout.addWidget(self.sequence_down)
        toolbar_layout.addStretch(1)
        seq_layout.addLayout(toolbar_layout)
        repeat_group = QGroupBox("Repeat")
        repeat_layout = QHBoxLayout()
        repeat_layout.addWidget(self.repeat_off_radio)
        repeat_layout.addWidget(self.repeat_endless_radio)
        repeat_layout.addWidget(self.repeat_limit_radio)
        repeat_layout.addWidget(self.repeat_time_edit)
        repeat_layout.addStretch(1)
        repeat_group.setLayout(repeat_layout)
        seq_layout.addWidget(repeat_group)
        seq_layout.addWidget(self.sequence_status)
        sequence_controls = QHBoxLayout()
        sequence_controls.addWidget(self.sequence_start)
        sequence_controls.addWidget(self.sequence_stop)
        sequence_controls.addWidget(self.sequence_reset)
        sequence_controls.addStretch(1)
        seq_layout.addLayout(sequence_controls)
        sequencer_group.setLayout(seq_layout)
        layout.addWidget(sequencer_group)
        layout.addWidget(QLabel("Status signals"))
        layout.addWidget(self.values_view)
        layout.addWidget(self.plot_widget)

    def _connect_signals(self) -> None:
        self.pwm_slider.valueChanged.connect(lambda value: self.pwm_value.setText(f"{value} %"))
        self.apply_button.clicked.connect(self._emit_command)
        self.off_button.clicked.connect(self._emit_off)
        self.quick_off.clicked.connect(self._emit_off)
        self.quick_low.clicked.connect(lambda: self._set_pwm_slider(20))
        self.quick_mid.clicked.connect(lambda: self._set_pwm_slider(50))
        self.quick_max.clicked.connect(lambda: self._set_pwm_slider(100))
        self.sequence_add.clicked.connect(self._on_add_sequence)
        self.sequence_delete.clicked.connect(self._on_delete_sequence)
        self.sequence_duplicate.clicked.connect(self._on_duplicate_sequence)
        self.sequence_up.clicked.connect(lambda: self._move_sequence(-1))
        self.sequence_down.clicked.connect(lambda: self._move_sequence(1))
        self.sequence_start.clicked.connect(lambda: self._emit_sequence_action("start"))
        self.sequence_stop.clicked.connect(lambda: self._emit_sequence_action("stop"))
        self.sequence_reset.clicked.connect(lambda: self._emit_sequence_action("reset"))
        self.sequence_table.itemChanged.connect(self._on_sequence_item_changed)
        self.repeat_off_radio.toggled.connect(self._on_repeat_mode_changed)
        self.repeat_endless_radio.toggled.connect(self._on_repeat_mode_changed)
        self.repeat_limit_radio.toggled.connect(self._on_repeat_mode_changed)
        self.repeat_time_edit.timeChanged.connect(lambda _time: self._emit_sequence_config())
        self.plot_checkbox.toggled.connect(lambda checked: self._on_plot_toggled(checked))

    def _set_pwm_slider(self, value: int) -> None:
        self.pwm_slider.setValue(value)
        self.pwm_value.setText(f"{value} %")

    def _emit_command(self) -> None:
        command = self._collect_command()
        self.command_requested.emit(self.profile.name, command)

    def _emit_off(self) -> None:
        command = {"enabled": 0.0, "select": 0.0, "pwm": 0.0, "state": 0.0}
        if self.profile.type in {"AO_0_10V", "AO_4_20mA"}:
            command["setpoint"] = 0.0
        self.command_requested.emit(self.profile.name, command)

    def _emit_sequence_action(self, action: str) -> None:
        if action == "start" and not self._has_enabled_sequences():
            return
        self.sequencer_requested.emit(self.profile.name, action)

    def _on_add_sequence(self) -> None:
        self._updating_sequence_table = True
        self._insert_sequence_row(None, self.sequence_table.rowCount())
        self._update_sequence_order()
        self._updating_sequence_table = False
        self._emit_sequence_config()

    def _on_delete_sequence(self) -> None:
        row = self.sequence_table.currentRow()
        if row < 0:
            row = self.sequence_table.rowCount() - 1
        if row < 0:
            return
        self._updating_sequence_table = True
        self.sequence_table.removeRow(row)
        self._update_sequence_order()
        self._updating_sequence_table = False
        self._emit_sequence_config()

    def _on_duplicate_sequence(self) -> None:
        row = self.sequence_table.currentRow()
        if row < 0:
            return
        config = self._sequence_from_row(row)
        self._updating_sequence_table = True
        self._insert_sequence_row(config, row + 1)
        self._update_sequence_order()
        self.sequence_table.selectRow(row + 1)
        self._updating_sequence_table = False
        self._emit_sequence_config()

    def _move_sequence(self, offset: int) -> None:
        row = self.sequence_table.currentRow()
        if row < 0:
            return
        target = row + offset
        if target < 0 or target >= self.sequence_table.rowCount():
            return
        config = self._sequence_from_row(row)
        self._updating_sequence_table = True
        self.sequence_table.removeRow(row)
        self._insert_sequence_row(config, target)
        self._update_sequence_order()
        self.sequence_table.selectRow(target)
        self._updating_sequence_table = False
        self._emit_sequence_config()

    def _on_sequence_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_sequence_table:
            return
        if item.column() not in {1, 6}:
            return
        self._emit_sequence_config()

    def _on_repeat_mode_changed(self) -> None:
        if self._updating_sequence_table:
            return
        self.repeat_time_edit.setEnabled(self.repeat_limit_radio.isChecked())
        self._emit_sequence_config()

    def _insert_sequence_row(self, cfg: Optional[SequenceCfg], row: int) -> None:
        self.sequence_table.insertRow(row)
        order_item = QTableWidgetItem(str(row + 1))
        order_item.setFlags(Qt.ItemIsEnabled)
        self.sequence_table.setItem(row, 0, order_item)
        name = cfg.name if cfg else f"Sequence {row + 1}"
        name_item = QTableWidgetItem(name)
        self.sequence_table.setItem(row, 1, name_item)
        duration_spin = QSpinBox()
        duration_spin.setRange(1, 86_400)
        duration_spin.setValue(cfg.duration_s if cfg else 60)
        duration_spin.valueChanged.connect(self._emit_sequence_config)
        self.sequence_table.setCellWidget(row, 2, duration_spin)
        pwm_spin = QSpinBox()
        pwm_spin.setRange(0, 100)
        pwm_spin.setValue(cfg.pwm if cfg else 50)
        pwm_spin.valueChanged.connect(self._emit_sequence_config)
        self.sequence_table.setCellWidget(row, 3, pwm_spin)
        on_spin = QDoubleSpinBox()
        on_spin.setDecimals(2)
        on_spin.setRange(0.01, 3600.0)
        on_spin.setSingleStep(0.1)
        on_spin.setValue(cfg.on_s if cfg else 1.0)
        on_spin.valueChanged.connect(self._emit_sequence_config)
        self.sequence_table.setCellWidget(row, 4, on_spin)
        off_spin = QDoubleSpinBox()
        off_spin.setDecimals(2)
        off_spin.setRange(0.01, 3600.0)
        off_spin.setSingleStep(0.1)
        off_spin.setValue(cfg.off_s if cfg else 1.0)
        off_spin.valueChanged.connect(self._emit_sequence_config)
        self.sequence_table.setCellWidget(row, 5, off_spin)
        enabled_item = QTableWidgetItem()
        enabled_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        enabled_item.setCheckState(Qt.Checked if (cfg.enabled if cfg else True) else Qt.Unchecked)
        self.sequence_table.setItem(row, 6, enabled_item)

    def _update_sequence_order(self) -> None:
        for index in range(self.sequence_table.rowCount()):
            item = self.sequence_table.item(index, 0)
            if item:
                item.setText(str(index + 1))

    def _sequence_from_row(self, row: int) -> SequenceCfg:
        name_item = self.sequence_table.item(row, 1)
        name = name_item.text().strip() if name_item else f"Sequence {row + 1}"
        duration_spin = self.sequence_table.cellWidget(row, 2)
        pwm_spin = self.sequence_table.cellWidget(row, 3)
        on_spin = self.sequence_table.cellWidget(row, 4)
        off_spin = self.sequence_table.cellWidget(row, 5)
        enabled_item = self.sequence_table.item(row, 6)
        duration = int(duration_spin.value()) if isinstance(duration_spin, QSpinBox) else 60
        pwm = int(pwm_spin.value()) if isinstance(pwm_spin, QSpinBox) else 0
        on_value = float(on_spin.value()) if isinstance(on_spin, QDoubleSpinBox) else 1.0
        off_value = float(off_spin.value()) if isinstance(off_spin, QDoubleSpinBox) else 1.0
        enabled = enabled_item.checkState() == Qt.Checked if enabled_item else True
        return SequenceCfg(name=name, duration_s=duration, pwm=pwm, on_s=on_value, off_s=off_value, enabled=enabled)

    def _collect_sequences(self) -> List[SequenceCfg]:
        return [self._sequence_from_row(row) for row in range(self.sequence_table.rowCount())]

    def _has_enabled_sequences(self) -> bool:
        return any(sequence.enabled for sequence in self._collect_sequences())

    def _emit_sequence_config(self) -> None:
        if self._updating_sequence_table:
            return
        sequences = self._collect_sequences()
        repeat_mode = self._current_repeat_mode()
        repeat_limit = self._repeat_limit_seconds()
        payload = {
            "sequences": sequences,
            "repeat_mode": repeat_mode,
            "repeat_limit_s": repeat_limit,
        }
        self.sequencer_config_changed.emit(self.profile.name, payload)
        self._update_sequence_buttons()

    def _current_repeat_mode(self) -> SequenceRepeatMode:
        if self.repeat_endless_radio.isChecked():
            return SequenceRepeatMode.ENDLESS
        if self.repeat_limit_radio.isChecked():
            return SequenceRepeatMode.LIMIT
        return SequenceRepeatMode.OFF

    def _repeat_limit_seconds(self) -> int:
        if not self.repeat_limit_radio.isChecked():
            return 0
        time_value = self.repeat_time_edit.time()
        return time_value.hour() * 3600 + time_value.minute() * 60 + time_value.second()

    def set_sequencer_config(self, config: ChannelConfig) -> None:
        self._updating_sequence_table = True
        self.sequence_table.setRowCount(0)
        for sequence in config.sequences:
            self._insert_sequence_row(sequence, self.sequence_table.rowCount())
        self._update_sequence_order()
        if config.repeat_mode == SequenceRepeatMode.ENDLESS:
            self.repeat_endless_radio.setChecked(True)
        elif config.repeat_mode == SequenceRepeatMode.LIMIT:
            self.repeat_limit_radio.setChecked(True)
        else:
            self.repeat_off_radio.setChecked(True)
        self.repeat_time_edit.setTime(self._seconds_to_time(config.repeat_limit_s))
        self.repeat_time_edit.setEnabled(config.repeat_mode == SequenceRepeatMode.LIMIT)
        self._updating_sequence_table = False
        self._update_sequence_buttons()

    def _seconds_to_time(self, seconds: int) -> QTime:
        seconds = max(0, int(seconds))
        hours = min(23, seconds // 3600)
        minutes = min(59, (seconds % 3600) // 60)
        secs = min(59, seconds % 60)
        return QTime(hours, minutes, secs)

    def set_sequence_running(self, running: bool) -> None:
        self._sequence_running = running
        self.sequence_stop.setEnabled(running)
        self.sequence_reset.setEnabled(running or self.sequence_table.rowCount() > 0)
        self._set_sequence_controls_enabled(not running)
        if not running:
            self.sequence_status.setText("Sequence idle")
        self._update_sequence_buttons()

    def _set_sequence_controls_enabled(self, enabled: bool) -> None:
        self.sequence_table.setEnabled(enabled)
        for control in (
            self.sequence_add,
            self.sequence_delete,
            self.sequence_duplicate,
            self.sequence_up,
            self.sequence_down,
        ):
            control.setEnabled(enabled)
        for radio in (self.repeat_off_radio, self.repeat_endless_radio, self.repeat_limit_radio):
            radio.setEnabled(enabled)
        self.repeat_time_edit.setEnabled(enabled and self.repeat_limit_radio.isChecked())

    def update_sequence_status(
        self,
        seq_index: Optional[int],
        total_sequences: int,
        phase: Optional[str],
        remaining_s: float,
        running: bool,
    ) -> None:
        self._sequence_running = running
        if running and seq_index is not None and seq_index >= 0 and total_sequences > 0:
            seq_no = seq_index + 1
            phase_label = "On" if phase == "on" else "Off"
            remaining = max(0.0, remaining_s)
            minutes = int(remaining // 60)
            seconds = int(remaining % 60)
            self.sequence_status.setText(
                f"Seq {seq_no}/{total_sequences} • Phase: {phase_label} • Rest: {minutes:02d}:{seconds:02d}"
            )
        else:
            self.sequence_status.setText("Sequence idle")
        self.sequence_stop.setEnabled(running)
        self.sequence_reset.setEnabled((running and self.sequence_table.rowCount() > 0) or (not running and self.sequence_table.rowCount() > 0))
        self._set_sequence_controls_enabled(not running)
        self._update_sequence_buttons()

    def _update_sequence_buttons(self) -> None:
        has_sequences = self.sequence_table.rowCount() > 0
        has_enabled = any(
            self.sequence_table.item(row, 6) and self.sequence_table.item(row, 6).checkState() == Qt.Checked
            for row in range(self.sequence_table.rowCount())
        )
        if not self._sequence_running:
            self.sequence_start.setEnabled(has_sequences and has_enabled)
        else:
            self.sequence_start.setEnabled(False)
        self.sequence_reset.setEnabled(has_sequences)
        if not self._sequence_running:
            for control in (
                self.sequence_delete,
                self.sequence_duplicate,
                self.sequence_up,
                self.sequence_down,
            ):
                control.setEnabled(has_sequences)

    def _collect_command(self) -> Dict[str, float]:
        command: Dict[str, float] = {}
        if self.profile.type in {"HighSide", "LowSide", "HBridge"}:
            command.update(
                {
                    "enabled": 1.0 if self.enabled_checkbox.isChecked() else 0.0,
                    "select": 1.0 if self.enabled_checkbox.isChecked() else 0.0,
                    "pwm": float(self.pwm_slider.value()),
                }
            )
            if self.profile.type == "HBridge":
                command["direction"] = float(self.direction_combo.currentIndex() - 1)
        if self.profile.type in {"AO_0_10V", "AO_4_20mA"}:
            command["setpoint"] = float(self.setpoint_spin.value())
        if self.profile.type in {"DO"}:
            command["state"] = 1.0 if self.enabled_checkbox.isChecked() else 0.0
        return command

    def _update_visibility(self) -> None:
        channel_type = self.profile.type
        is_pwm = channel_type in {"HighSide", "LowSide", "HBridge"}
        is_direction = channel_type == "HBridge"
        is_setpoint = channel_type in {"AO_0_10V", "AO_4_20mA"}
        is_toggle = channel_type in {"DO", "HighSide", "LowSide", "HBridge"}
        self.pwm_slider.setVisible(is_pwm)
        self.pwm_value.setVisible(is_pwm)
        self.direction_combo.setVisible(is_direction)
        self.setpoint_spin.setVisible(is_setpoint)
        self.quick_low.setVisible(is_pwm)
        self.quick_mid.setVisible(is_pwm)
        self.quick_max.setVisible(is_pwm)
        self.quick_off.setVisible(is_pwm or is_setpoint or channel_type == "DO")
        self.enabled_checkbox.setVisible(is_toggle or channel_type == "DO")

    def update_status(self, status: Dict[str, float]) -> None:
        lines = [f"{key}: {value:.4f}" for key, value in sorted(status.items())]
        self.values_view.setPlainText("\n".join(lines))

    def update_state_label(self, text: str) -> None:
        self.state_label.setText(text)

    def set_dummy_mode(self, enabled: bool) -> None:
        self.sim_button.setVisible(enabled)

    def set_plot_checked(self, checked: bool) -> None:
        self.plot_checkbox.setChecked(checked)

    def plot_checked(self) -> bool:
        return self.plot_checkbox.isChecked()

    def record_sample(self, timestamp: float, command_value: Optional[float], feedback_value: Optional[float]) -> None:
        if command_value is None and feedback_value is None:
            return
        if command_value is None:
            command_value = 0.0
        if feedback_value is None:
            feedback_value = 0.0
        self._plot_times.append(timestamp)
        self._plot_command.append(float(command_value))
        self._plot_feedback.append(float(feedback_value))
        cutoff = timestamp - 60.0
        self._trim_history(cutoff)
        if not self.plot_checkbox.isChecked():
            return
        if not self._plot_times:
            return
        base_time = self._plot_times[0]
        times = [value - base_time for value in self._plot_times]
        command = list(self._plot_command)
        feedback = list(self._plot_feedback)
        times, command, feedback = self._decimate_samples(times, command, feedback)
        self.command_curve.setData(times, command)
        self.feedback_curve.setData(times, feedback)
        self.plot_widget.enableAutoRange(axis=pg.ViewBox.YAxis)

    def reset_plot(self) -> None:
        self._plot_times.clear()
        self._plot_command.clear()
        self._plot_feedback.clear()
        self.command_curve.clear()
        self.feedback_curve.clear()

    def _on_plot_toggled(self, checked: bool) -> None:
        self.plot_widget.setVisible(checked)
        if not checked:
            self.reset_plot()
        self.plot_visibility_changed.emit(self.profile.name, checked)

    def _trim_history(self, cutoff: float) -> None:
        while self._plot_times and self._plot_times[0] < cutoff:
            self._plot_times.popleft()
            self._plot_command.popleft()
            self._plot_feedback.popleft()

    def _decimate_samples(
        self,
        times: List[float],
        command: List[float],
        feedback: List[float],
    ) -> Tuple[List[float], List[float], List[float]]:
        if len(times) <= 5_000:
            return times, command, feedback
        step = max(1, math.ceil(len(times) / 5_000))
        dec_times = times[::step]
        dec_command = command[::step]
        dec_feedback = feedback[::step]
        if dec_times[-1] != times[-1]:
            dec_times.append(times[-1])
            dec_command.append(command[-1])
            dec_feedback.append(feedback[-1])
        return dec_times, dec_command, dec_feedback

    def _open_sim_dialog(self) -> None:
        dialog = ChannelSimulationDialog(self.profile, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            updated = dialog.result_parameters
            if updated is not None:
                self.profile.sim.update(updated)
                self.simulation_changed.emit(self.profile.name, dict(self.profile.sim))


class ChannelSimulationDialog(QDialog):
    def __init__(self, profile: ChannelProfile, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Simulation – {profile.name}")
        self.profile = profile
        self.result_parameters: Optional[Dict[str, float]] = None
        self._spins: Dict[str, QDoubleSpinBox] = {}
        layout = QVBoxLayout(self)
        form = QFormLayout()
        for key, label, minimum, maximum, step in CHANNEL_SIM_FIELDS.get(profile.type, []):
            spin = QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            spin.setValue(float(profile.sim.get(key, 0.0)))
            form.addRow(label, spin)
            self._spins[key] = spin
        if not self._spins:
            form.addRow(QLabel("No simulation parameters defined for this channel type."))
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept(self) -> None:
        self.result_parameters = {key: float(spin.value()) for key, spin in self._spins.items()}
        self.accept()


class ChannelDuplicateDialog(QDialog):
    def __init__(
        self,
        source: ChannelProfile,
        suggested_id: str,
        existing_ids: Iterable[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Duplicate Channel")
        self.target_combo = QComboBox()
        self.target_combo.setEditable(True)
        if suggested_id:
            self.target_combo.addItem(suggested_id)
            self.target_combo.setCurrentText(suggested_id)
        for identifier in sorted({str(value) for value in existing_ids}):
            if identifier != suggested_id:
                self.target_combo.addItem(identifier)
        self.name_edit = QLineEdit(suggested_id)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow("Source", QLabel(source.name))
        form.addRow("Target ID", self.target_combo)
        form.addRow("Display name", self.name_edit)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def target_id(self) -> str:
        return self.target_combo.currentText().strip()

    @property
    def display_name(self) -> str:
        return self.name_edit.text().strip()


class ChannelBuilderDialog(QDialog):
    def __init__(self, database, profile: Optional[ChannelProfile] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Channel Builder")
        self.database = database
        self.profile = profile or ChannelProfile("New Channel", "HighSide")
        self.result_profile: Optional[ChannelProfile] = None
        self.name_edit = QLineEdit(self.profile.name)
        self.type_combo = QComboBox()
        self.type_combo.addItems(CHANNEL_TYPES)
        self.type_combo.setCurrentText(self.profile.type)
        self.write_message_combo = QComboBox()
        self.status_message_combo = QComboBox()
        self.write_field_combos: Dict[str, QComboBox] = {}
        self.status_field_combos: Dict[str, QComboBox] = {}
        self._build_ui()
        self._populate_messages()
        self._populate_fields()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow("Name", self.name_edit)
        form.addRow("Type", self.type_combo)
        form.addRow("Write message", self.write_message_combo)
        form.addRow("Status message", self.status_message_combo)
        layout.addLayout(form)
        self.fields_container = QGroupBox("Field mapping")
        self.fields_layout = QFormLayout()
        self.fields_container.setLayout(self.fields_layout)
        layout.addWidget(self.fields_container)
        self.type_combo.currentTextChanged.connect(self._populate_fields)
        self.write_message_combo.currentTextChanged.connect(self._populate_fields)
        self.status_message_combo.currentTextChanged.connect(self._populate_fields)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate_messages(self) -> None:
        self.write_message_combo.clear()
        self.status_message_combo.clear()
        if not self.database:
            return
        messages = sorted(message.name for message in getattr(self.database, "messages", []))
        self.write_message_combo.addItems([""] + messages)
        self.status_message_combo.addItems([""] + messages)
        if self.profile.write.message:
            self.write_message_combo.setCurrentText(self.profile.write.message)
        if self.profile.status.message:
            self.status_message_combo.setCurrentText(self.profile.status.message)

    def _populate_fields(self) -> None:
        channel_type = self.type_combo.currentText()
        schema = CHANNEL_SCHEMA.get(channel_type, {"write": [], "status": []})
        for combo in list(self.write_field_combos.values()):
            combo.deleteLater()
        for combo in list(self.status_field_combos.values()):
            combo.deleteLater()
        self.write_field_combos.clear()
        self.status_field_combos.clear()
        while self.fields_layout.rowCount():
            self.fields_layout.removeRow(0)
        self._create_field_rows(schema["write"], self.write_field_combos, self.profile.write, self.write_message_combo)
        self._create_field_rows(schema["status"], self.status_field_combos, self.profile.status, self.status_message_combo)

    def _create_field_rows(
        self,
        schema_rows: List[Tuple[str, str]],
        combo_store: Dict[str, QComboBox],
        mapping: ChannelMapping,
        message_combo: QComboBox,
    ) -> None:
        message_name = message_combo.currentText()
        signals = self._signals_for_message(message_name)
        for semantic, label in schema_rows:
            combo = QComboBox()
            combo.addItem("", "")
            for signal in signals:
                combo.addItem(signal, signal)
            if mapping.signal_for(semantic):
                combo.setCurrentText(mapping.signal_for(semantic))
            combo_store[semantic] = combo
            self.fields_layout.addRow(f"{label}", combo)

    def _signals_for_message(self, message_name: str) -> List[str]:
        if not message_name or not self.database:
            return []
        message = self.database.get_message_by_name(message_name)
        if not message:
            return []
        return [signal.name for signal in message.signals]

    def _accept(self) -> None:
        name = self.name_edit.text().strip() or "Unnamed"
        channel_type = self.type_combo.currentText()
        write_message = self.write_message_combo.currentText().strip()
        status_message = self.status_message_combo.currentText().strip()
        write_mapping = ChannelMapping(message=write_message)
        status_mapping = ChannelMapping(message=status_message)
        for semantic, combo in self.write_field_combos.items():
            selected = combo.currentText().strip()
            if selected:
                write_mapping.fields[semantic] = selected
        for semantic, combo in self.status_field_combos.items():
            selected = combo.currentText().strip()
            if selected:
                status_mapping.fields[semantic] = selected
        result = ChannelProfile(name=name, type=channel_type, write=write_mapping, status=status_mapping, sim=dict(self.profile.sim))
        self.result_profile = result
        self.accept()

def load_channel_profiles(database) -> Dict[str, ChannelProfile]:
    ensure_directories()
    profiles: Dict[str, ChannelProfile] = {}
    if os.path.exists(CHANNEL_PROFILE_PATH):
        try:
            with open(CHANNEL_PROFILE_PATH, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or []
        except (OSError, yaml.YAMLError):
            data = []
        for entry in data:
            try:
                profile = ChannelProfile.from_yaml(entry)
                profiles[profile.name] = profile
            except Exception:
                continue
    if not profiles:
        profiles = {profile.name: profile for profile in default_channel_profiles(database)}
        save_channel_profiles(list(profiles.values()))
    return profiles


def save_channel_profiles(profiles: List[ChannelProfile]) -> None:
    ensure_directories()
    data = [profile.to_yaml() for profile in profiles]
    try:
        with open(CHANNEL_PROFILE_PATH, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
    except OSError:
        pass


def default_channel_profiles(database) -> List[ChannelProfile]:
    profiles: List[ChannelProfile] = []
    if not database:
        return profiles
    try:
        hs_write = database.get_message_by_name("QM_High_side_output_write")
        hs_status = database.get_message_by_name("QM_High_side_output_status")
    except KeyError:
        hs_write = None
        hs_status = None
    if hs_write and hs_status:
        for index in range(1, 6):
            select = f"hs_out{index:02d}_select"
            mode = f"hs_out{index:02d}_mode"
            pwm = f"hs_out{index:02d}_value_pwm"
            current = f"hs_out{index:02d}_current"
            pwm_feedback = f"hs_out{index:02d}_pwm"
            profile = ChannelProfile(
                name=f"HS{index}",
                type="HighSide",
                write=ChannelMapping(message=hs_write.name, fields={"select": select, "mode": mode, "pwm": pwm}),
                status=ChannelMapping(message=hs_status.name, fields={"current": current, "pwm_feedback": pwm_feedback}),
                sim={"tau": 0.5, "current_gain": 8.0, "noise": 0.1},
            )
            profiles.append(profile)
    try:
        ai_status = database.get_message_by_name("QM_Analog_Input_status")
    except KeyError:
        ai_status = None
    if ai_status:
        for index in range(1, 6):
            voltage = f"an_pin_{index:02d}_voltage"
            current = f"an_pin_{index:02d}_current"
            profile = ChannelProfile(
                name=f"AI{index}",
                type="AI_V",
                status=ChannelMapping(message=ai_status.name, fields={"value": voltage}),
            )
            profiles.append(profile)
            profile_current = ChannelProfile(
                name=f"AI{index}_I",
                type="AI_I",
                status=ChannelMapping(message=ai_status.name, fields={"value": current}),
            )
            profiles.append(profile_current)
    return profiles


def collect_signal_definitions(database) -> Dict[str, List[SignalDefinition]]:
    signals: Dict[str, List[SignalDefinition]] = {}
    if not database:
        return signals
    for message in getattr(database, "messages", []):
        definitions: List[SignalDefinition] = []
        for signal in message.signals:
            definitions.append(
                SignalDefinition(
                    message_name=message.name,
                    name=signal.name,
                    unit=signal.unit or "",
                    minimum=signal.minimum,
                    maximum=signal.maximum,
                    scale=getattr(signal, "scale", 1.0),
                    offset=getattr(signal, "offset", 0.0),
                )
            )
        signals[message.name] = definitions
    return signals

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ECU Control")
        ensure_directories()
        self.backends = {
            DummyBackend.name: DummyBackend,
            RealBackend.name: RealBackend,
        }
        self.backend: Optional[BackendBase] = None
        self.backend_name = DummyBackend.name
        self.logger = Logger()
        self._qt_settings = QSettings("OpenAI", "ECUControl")
        self._dbc = None
        self._signals_by_message: Dict[str, List[SignalDefinition]] = {}
        self._watch_units: Dict[str, str] = {}
        self._channel_profiles: Dict[str, ChannelProfile] = {}
        self._channel_cards: Dict[str, ChannelCardWidget] = {}
        self._channel_status: Dict[str, Dict[str, float]] = {}
        self._channel_commands: Dict[str, Dict[str, float]] = {}
        self._sequencer_configs: Dict[str, ChannelConfig] = {}
        self._sequence_runners: Dict[str, SequenceRunner] = {}
        self._last_tick = time.monotonic()
        self._last_log_time = 0.0
        self._log_interval = 1.0
        self._manual_log_signals: List[str] = []
        self._pending_watchlist: List[str] = []
        self._pending_plot_signals: List[str] = []
        self._channel_plot_settings: Dict[str, bool] = {}
        self._multi_plot_buffers: Dict[str, deque[Tuple[float, float]]] = {}
        self._multi_plot_curves: Dict[str, pg.PlotDataItem] = {}
        self._multi_plot_paused = False
        self._show_dummy_advanced = False
        self._multi_plot_enabled = False
        self._plot_windows: Dict[int, MultiAxisPlotDock] = {}
        self._plot_assignments: Dict[str, Tuple[int, str]] = {}
        self._plot_counter = 0
        self._max_plot_windows = 4
        self._active_plot_id: Optional[int] = None
        self._suspend_plot_assignment = False
        self._restore_settings()
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start()
        self._load_initial_backend()

    # UI construction
    def _build_ui(self) -> None:
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("color: red; font-size: 16pt;")
        self.status_message_label = QLabel("Disconnected")
        status_bar = QStatusBar()
        status_bar.addWidget(self.status_indicator)
        status_bar.addWidget(self.status_message_label, 1)
        self.setStatusBar(status_bar)
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        self._build_dashboard_tab()
        self._build_channels_tab()
        self._build_signals_tab()
        self._build_dummy_tab()
        self._build_logging_tab()
        self._update_dummy_tab_visibility()

    def _build_dashboard_tab(self) -> None:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(self.backends.keys())
        self.mode_combo.setCurrentText(self.backend_name)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        header_layout.addWidget(self.mode_combo)
        header_layout.addStretch(1)
        layout.addLayout(header_layout)
        form = QFormLayout()
        self.dbc_edit = QLineEdit(self._qt_settings.value("dbc_path", os.path.join(BASE_DIR, "ecu-test.dbc")))
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self._browse_dbc)
        dbc_layout = QHBoxLayout()
        dbc_layout.addWidget(self.dbc_edit)
        dbc_layout.addWidget(browse_button)
        dbc_widget = QWidget()
        dbc_widget.setLayout(dbc_layout)
        form.addRow("DBC path", dbc_widget)
        self.bustype_combo = QComboBox()
        self.bustype_combo.addItems(["socketcan", "vector", "pcan"])
        self.bustype_combo.setCurrentText(self._qt_settings.value("bustype", "socketcan"))
        form.addRow("Bus type", self.bustype_combo)
        self.channel_edit = QLineEdit(self._qt_settings.value("channel", "vcan0"))
        form.addRow("Channel", self.channel_edit)
        self.bitrate_spin = QSpinBox()
        self.bitrate_spin.setRange(10_000, 1_000_000)
        self.bitrate_spin.setSingleStep(10_000)
        self.bitrate_spin.setValue(int(self._qt_settings.value("bitrate", 500_000)))
        form.addRow("Bitrate", self.bitrate_spin)
        layout.addLayout(form)
        button_layout = QHBoxLayout()
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self._connect_backend)
        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.clicked.connect(self._disconnect_backend)
        button_layout.addWidget(self.connect_button)
        button_layout.addWidget(self.disconnect_button)
        button_layout.addStretch(1)
        layout.addLayout(button_layout)
        action_layout = QHBoxLayout()
        self.all_off_button = QPushButton("All outputs OFF")
        self.all_off_button.clicked.connect(self._all_outputs_off)
        self.emergency_button = QPushButton("Emergency Stop")
        self.emergency_button.clicked.connect(self._emergency_stop)
        action_layout.addWidget(self.all_off_button)
        action_layout.addWidget(self.emergency_button)
        action_layout.addStretch(1)
        layout.addLayout(action_layout)
        self.show_dummy_checkbox = QCheckBox("Show Dummy Advanced tab")
        self.show_dummy_checkbox.setChecked(self._show_dummy_advanced)
        self.show_dummy_checkbox.stateChanged.connect(self._on_show_dummy_tab_changed)
        layout.addWidget(self.show_dummy_checkbox)
        layout.addStretch(1)
        self.tab_widget.addTab(widget, "Dashboard")

    def _build_channels_tab(self) -> None:
        widget = QWidget()
        outer_layout = QVBoxLayout(widget)
        control_layout = QHBoxLayout()
        self.channel_selector = QComboBox()
        control_layout.addWidget(QLabel("Channels:"))
        control_layout.addWidget(self.channel_selector)
        add_button = QPushButton("Add Channel")
        add_button.clicked.connect(self._add_channel)
        edit_button = QPushButton("Edit Channel")
        edit_button.clicked.connect(self._edit_channel)
        remove_button = QPushButton("Remove Channel")
        remove_button.clicked.connect(self._remove_channel)
        duplicate_button = QPushButton("Duplicate Channel")
        duplicate_button.clicked.connect(self._duplicate_channel)
        control_layout.addWidget(add_button)
        control_layout.addWidget(edit_button)
        control_layout.addWidget(remove_button)
        control_layout.addWidget(duplicate_button)
        control_layout.addStretch(1)
        outer_layout.addLayout(control_layout)
        self.channel_scroll = QScrollArea()
        self.channel_scroll.setWidgetResizable(True)
        self.channel_container = QWidget()
        self.channel_layout = QVBoxLayout(self.channel_container)
        self.channel_layout.addStretch(1)
        self.channel_scroll.setWidget(self.channel_container)
        outer_layout.addWidget(self.channel_scroll)
        self.tab_widget.addTab(widget, "Channels")

    def _build_signals_tab(self) -> None:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        self.signal_browser = SignalBrowserWidget()
        self.watchlist_widget = WatchlistWidget()
        self.watchlist_widget.plot_toggled.connect(self._on_watchlist_plot_toggled)
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.watchlist_widget)
        controls_layout = QHBoxLayout()
        self.multi_plot_enable = QCheckBox("Enable multi-plot")
        self.multi_plot_enable.setChecked(self._multi_plot_enabled)
        self.multi_plot_enable.stateChanged.connect(self._on_multi_plot_enable_changed)
        self.multi_plot_pause = QPushButton("Pause Plot" if not self._multi_plot_paused else "Resume Plot")
        self.multi_plot_pause.clicked.connect(self._toggle_multi_plot_pause)
        self.multi_plot_save = QPushButton("Save PNG")
        self.multi_plot_save.clicked.connect(self._save_multi_plot_png)
        controls_layout.addWidget(self.multi_plot_enable)
        controls_layout.addWidget(self.multi_plot_pause)
        controls_layout.addWidget(self.multi_plot_save)
        self.new_plot_window_button = QPushButton("New Plot Window")
        self.new_plot_window_button.clicked.connect(lambda: self._create_plot_window())
        controls_layout.addWidget(self.new_plot_window_button)
        self.close_plot_windows_button = QPushButton("Close All Plots")
        self.close_plot_windows_button.clicked.connect(lambda: self._close_all_plot_windows())
        controls_layout.addWidget(self.close_plot_windows_button)
        controls_layout.addStretch(1)
        right_layout.addLayout(controls_layout)
        self.multi_plot_widget = pg.PlotWidget()
        self.multi_plot_widget.setMinimumHeight(220)
        self.multi_plot_widget.addLegend()
        self.multi_plot_widget.showGrid(x=True, y=True)
        if not self._multi_plot_enabled:
            self.multi_plot_widget.hide()
        right_layout.addWidget(self.multi_plot_widget, 1)
        layout.addWidget(self.signal_browser, 2)
        layout.addLayout(right_layout, 3)
        self.signal_browser.add_requested.connect(self._on_add_to_watchlist)
        self.signal_browser.plot_requested.connect(self._on_plot_requested)
        self.signal_browser.simulate_requested.connect(self._on_signal_simulate)
        self.watchlist_widget.remove_requested.connect(self._on_remove_from_watchlist)
        self.tab_widget.addTab(widget, "Signals")

    def _create_plot_window(self) -> Optional[int]:
        if len(self._plot_windows) >= self._max_plot_windows:
            QMessageBox.warning(
                self,
                "Plot limit reached",
                f"Cannot open more than {self._max_plot_windows} plot windows.",
            )
            return None
        self._plot_counter += 1
        identifier = self._plot_counter
        dock = MultiAxisPlotDock(identifier, self)
        dock.setFeatures(
            QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable
        )
        dock.closed.connect(self._on_plot_window_closed)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.show()
        self._plot_windows[identifier] = dock
        self._active_plot_id = identifier
        return identifier

    def _close_all_plot_windows(self) -> None:
        for dock in list(self._plot_windows.values()):
            dock.close()
        if not self._plot_windows:
            self._active_plot_id = None

    def _on_plot_window_closed(self, identifier: int) -> None:
        dock = self._plot_windows.pop(identifier, None)
        if dock:
            dock.deleteLater()
        to_clear = [name for name, (win_id, _side) in self._plot_assignments.items() if win_id == identifier]
        for name in to_clear:
            self._plot_assignments.pop(name, None)
        if self._active_plot_id == identifier:
            self._active_plot_id = next(iter(self._plot_windows), None)

    def _assign_signal_via_dialog(self, name: str) -> None:
        if not self._plot_windows:
            created = self._create_plot_window()
            if created is None:
                if not self._plot_windows and not self.multi_plot_enable.isChecked():
                    self.multi_plot_enable.setChecked(True)
                return
        if not self._plot_windows:
            if not self.multi_plot_enable.isChecked():
                self.multi_plot_enable.setChecked(True)
            return
        window_ids = sorted(self._plot_windows.keys())
        default_window = self._active_plot_id if self._active_plot_id in self._plot_windows else window_ids[0]
        suggested_side = self._suggest_axis_side(default_window, name)
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Assign Plot – {name}")
        dialog_layout = QVBoxLayout(dialog)
        form = QFormLayout()
        dialog_layout.addLayout(form)
        window_combo = QComboBox()
        for wid in window_ids:
            dock = self._plot_windows[wid]
            window_combo.addItem(dock.windowTitle(), wid)
        window_combo.setCurrentIndex(window_ids.index(default_window))
        form.addRow("Window", window_combo)
        axis_combo = QComboBox()
        axis_combo.addItems(["Left", "Right"])
        axis_combo.setCurrentIndex(0 if suggested_side == "left" else 1)
        form.addRow("Axis", axis_combo)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        dialog_layout.addWidget(buttons)
        if dialog.exec_() != QDialog.Accepted:
            self.watchlist_widget.set_plot_enabled(name, False)
            return
        window_id = window_combo.currentData()
        axis_side = "left" if axis_combo.currentIndex() == 0 else "right"
        self._assign_signal_to_plot(name, int(window_id), axis_side)

    def _assign_signal_to_plot(self, name: str, window_id: int, side: str) -> None:
        dock = self._plot_windows.get(window_id)
        if not dock:
            return
        previous = self._plot_assignments.get(name)
        if previous and previous == (window_id, side):
            return
        if previous:
            prev_window, _prev_side = previous
            prev_dock = self._plot_windows.get(prev_window)
            if prev_dock:
                prev_dock.remove_signal(name)
        dock.add_signal(name, self._watch_units.get(name, ""), side)
        self._plot_assignments[name] = (window_id, side)
        self._active_plot_id = window_id

    def _suggest_axis_side(self, window_id: int, name: str) -> str:
        unit = (self._watch_units.get(name) or "").strip()
        left_has = False
        right_has = False
        for other, (win_id, side) in self._plot_assignments.items():
            if win_id != window_id:
                continue
            other_unit = (self._watch_units.get(other) or "").strip()
            if unit and other_unit == unit:
                return side
            if side == "left":
                left_has = True
            else:
                right_has = True
        if left_has and right_has:
            return "right"
        return "left"

    def _build_dummy_tab(self) -> None:
        self.dummy_tab = QWidget()
        layout = QVBoxLayout(self.dummy_tab)
        self.simulation_widget = DummySimulationWidget()
        self.simulation_widget.profile_changed.connect(self._apply_simulation_profile)
        layout.addWidget(self.simulation_widget)
        self.tab_widget.addTab(self.dummy_tab, "Dummy Advanced")

    def _build_logging_tab(self) -> None:
        widget = QWidget()
        layout = QFormLayout(widget)
        self.logging_mode_combo = QComboBox()
        self.logging_mode_combo.addItems(["Watchlist", "Manual"])
        layout.addRow("Source", self.logging_mode_combo)
        self.manual_log_edit = QLineEdit()
        self.manual_log_edit.setPlaceholderText("Comma-separated signal names")
        layout.addRow("Manual signals", self.manual_log_edit)
        self.logging_rate_spin = QDoubleSpinBox()
        self.logging_rate_spin.setDecimals(1)
        self.logging_rate_spin.setRange(1.0, 50.0)
        self.logging_rate_spin.setValue(float(self._qt_settings.value("log_rate", 10.0)))
        layout.addRow("Rate (Hz)", self.logging_rate_spin)
        self.logging_path_edit = QLineEdit(self._qt_settings.value("log_path", os.path.join(BASE_DIR, "signals.csv")))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_log_path)
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.logging_path_edit)
        path_layout.addWidget(browse)
        path_widget = QWidget()
        path_widget.setLayout(path_layout)
        layout.addRow("CSV path", path_widget)
        self.logging_button = QPushButton("Start Logging")
        self.logging_button.clicked.connect(self._toggle_logging)
        layout.addRow(self.logging_button)
        self.tab_widget.addTab(widget, "Logging")

    # Backend management
    def _load_initial_backend(self) -> None:
        self._load_dbc(self.dbc_edit.text())
        self._switch_backend(self.backend_name)
        if isinstance(self.backend, DummyBackend):
            self._connect_backend()

    def _switch_backend(self, name: str) -> None:
        if self.backend and self.backend_name == name:
            return
        self._disconnect_backend()
        backend_class = self.backends[name]
        self.backend = backend_class() if backend_class is not RealBackend else backend_class()
        self.backend_name = name
        self._update_dummy_tab_visibility()
        self._apply_backend_configuration()
        if isinstance(self.backend, RealBackend):
            self.status_message_label.setText("Real backend ready")
        else:
            self.status_message_label.setText("Dummy backend ready")
        if isinstance(self.backend, DummyBackend) and self._dbc:
            self.backend.apply_database(self._dbc)
            self.backend.set_channel_profiles(self._channel_profiles)
            self.simulation_widget.set_profiles(self.backend.simulation_profiles())
        self._update_status_indicator(False)
        self._update_channel_card_modes()
        self.signal_browser.set_allow_simulation(isinstance(self.backend, DummyBackend))

    def _apply_backend_configuration(self) -> None:
        if isinstance(self.backend, RealBackend):
            settings = ConnectionSettings(
                dbc_path=self.dbc_edit.text().strip(),
                bustype=self.bustype_combo.currentText(),
                channel=self.channel_edit.text().strip(),
                bitrate=int(self.bitrate_spin.value()),
            )
            self.backend.configure(settings)
            self.backend.connection_changed.connect(self._on_connection_changed)
            self.backend.status_updated.connect(self._on_status_updated)

    def _connect_backend(self) -> None:
        if not self.backend:
            return
        if self._dbc is None:
            self._show_error("Please load a valid DBC file before connecting")
            return
        if isinstance(self.backend, RealBackend):
            try:
                self.backend.configure(
                    ConnectionSettings(
                        dbc_path=self.dbc_edit.text().strip(),
                        bustype=self.bustype_combo.currentText(),
                        channel=self.channel_edit.text().strip(),
                        bitrate=int(self.bitrate_spin.value()),
                    )
                )
                self.backend.start()
                self.backend.apply_database(self._dbc)
                self.backend.set_channel_profiles(self._channel_profiles)
            except BackendError as exc:
                self._show_error(str(exc))
                return
        else:
            try:
                self.backend.start()
                self.backend.apply_database(self._dbc)
                self.backend.set_channel_profiles(self._channel_profiles)
                if isinstance(self.backend, DummyBackend):
                    self.simulation_widget.set_profiles(self.backend.simulation_profiles())
            except BackendError as exc:
                self._show_error(str(exc))
                return
        self._update_status_indicator(True)
        self.status_message_label.setText("Connected")

    def _disconnect_backend(self) -> None:
        if self.backend:
            try:
                self.backend.stop()
            except BackendError:
                pass
        self._update_status_indicator(False)

    def _on_mode_changed(self, name: str) -> None:
        self._switch_backend(name)
        self._save_settings()

    def _on_connection_changed(self, connected: bool, message: str) -> None:
        self._update_status_indicator(connected)
        self.status_message_label.setText(message)

    def _on_status_updated(self) -> None:
        pass

    # Channels management
    def _refresh_channel_cards(self) -> None:
        for card in self._channel_cards.values():
            card.setParent(None)
        self._channel_cards.clear()
        while self.channel_layout.count():
            item = self.channel_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for name, profile in self._channel_profiles.items():
            card = ChannelCardWidget(profile)
            card.command_requested.connect(self._on_channel_command)
            card.sequencer_requested.connect(self._on_sequencer_request)
            card.sequencer_config_changed.connect(self._on_sequencer_config_changed)
            card.plot_visibility_changed.connect(self._on_channel_plot_visibility)
            card.simulation_changed.connect(self._on_channel_simulation)
            self.channel_layout.addWidget(card)
            self._channel_cards[name] = card
            config = self._sequencer_configs.setdefault(name, ChannelConfig())
            card.set_sequencer_config(config)
            runner = self._sequence_runners.get(name)
            if runner is None:
                runner = SequenceRunner(name, self._apply_sequence_output, parent=self)
                runner.progressed.connect(
                    lambda index, phase, remaining, channel=name: self._on_sequence_progress(
                        channel, index, phase, remaining
                    )
                )
                runner.finished.connect(lambda channel=name: self._on_sequence_finished(channel))
                self._sequence_runners[name] = runner
            if not runner.is_running:
                runner.load(config.sequences, config.repeat_mode, config.repeat_limit_s)
            card.set_sequence_running(runner.is_running)
            if runner.is_running:
                runner.emit_progress()
            card.set_plot_checked(self._channel_plot_settings.get(name, False))
        self._update_channel_card_modes()
        self.channel_layout.addStretch(1)
        self.channel_selector.clear()
        self.channel_selector.addItems(sorted(self._channel_profiles))

    def _update_channel_card_modes(self) -> None:
        is_dummy = isinstance(self.backend, DummyBackend)
        for name, card in self._channel_cards.items():
            card.set_dummy_mode(is_dummy)
            if name not in self._channel_plot_settings:
                self._channel_plot_settings[name] = card.plot_checked()

    def _on_channel_command(self, channel: str, command: Dict[str, float]) -> None:
        if not self.backend:
            return
        self._channel_commands[channel] = {key: float(value) for key, value in command.items()}
        try:
            self.backend.apply_channel_command(channel, command)
        except BackendError as exc:
            self._show_error(str(exc))

    def _on_channel_plot_visibility(self, name: str, checked: bool) -> None:
        self._channel_plot_settings[name] = checked
        if not checked:
            card = self._channel_cards.get(name)
            if card:
                card.reset_plot()
        self._save_settings()

    def _on_channel_simulation(self, name: str, parameters: Dict[str, float]) -> None:
        profile = self._channel_profiles.get(name)
        if not profile:
            return
        profile.sim.update(parameters)
        save_channel_profiles(list(self._channel_profiles.values()))
        if isinstance(self.backend, DummyBackend):
            self.backend.set_channel_profiles(self._channel_profiles)
        self._save_settings()

    def _add_channel(self) -> None:
        dialog = ChannelBuilderDialog(self._dbc, parent=self)
        if dialog.exec_() == QDialog.Accepted and dialog.result_profile:
            profile = dialog.result_profile
            self._channel_profiles[profile.name] = profile
            self._channel_profiles = dict(sorted(self._channel_profiles.items()))
            save_channel_profiles(list(self._channel_profiles.values()))
            if self.backend:
                self.backend.set_channel_profiles(self._channel_profiles)
            self._refresh_channel_cards()

    def _edit_channel(self) -> None:
        name = self.channel_selector.currentText()
        profile = self._channel_profiles.get(name)
        if not profile:
            return
        dialog = ChannelBuilderDialog(self._dbc, profile=profile, parent=self)
        if dialog.exec_() == QDialog.Accepted and dialog.result_profile:
            self._channel_profiles[name] = dialog.result_profile
            save_channel_profiles(list(self._channel_profiles.values()))
            if self.backend:
                self.backend.set_channel_profiles(self._channel_profiles)
            self._refresh_channel_cards()

    def _remove_channel(self) -> None:
        name = self.channel_selector.currentText()
        if name and name in self._channel_profiles:
            del self._channel_profiles[name]
            save_channel_profiles(list(self._channel_profiles.values()))
            if self.backend:
                self.backend.set_channel_profiles(self._channel_profiles)
            self._sequencer_configs.pop(name, None)
            runner = self._sequence_runners.pop(name, None)
            if runner:
                runner.stop()
            self._channel_commands.pop(name, None)
            self._channel_status.pop(name, None)
            self._channel_plot_settings.pop(name, None)
            self._refresh_channel_cards()
            self._save_settings()

    def _duplicate_channel(self) -> None:
        source_name = self.channel_selector.currentText()
        source_profile = self._channel_profiles.get(source_name)
        if not source_profile:
            return
        suggested_id = self._suggest_duplicate_target(source_profile.name)
        dialog = ChannelDuplicateDialog(
            source_profile,
            suggested_id,
            self._channel_profiles.keys(),
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        target_id = dialog.target_id
        if not target_id:
            self._show_error("Target channel ID cannot be empty.")
            return
        display_name = dialog.display_name or target_id
        if display_name in self._channel_profiles:
            self._show_error(f"Channel {display_name} already exists.")
            return
        new_profile = copy.deepcopy(source_profile)
        _, source_digits, _ = self._split_channel_name(source_profile.name)
        _, target_digits, _ = self._split_channel_name(target_id)
        if source_digits and target_digits:
            self._remap_channel_profile(new_profile, source_digits, target_digits)
        new_profile.name = display_name
        conflict = self._detect_channel_conflict(new_profile, exclude=source_name)
        if conflict:
            self._show_error(conflict)
            return
        self._channel_profiles[display_name] = new_profile
        self._channel_profiles = dict(sorted(self._channel_profiles.items()))
        save_channel_profiles(list(self._channel_profiles.values()))
        self._sequencer_configs[display_name] = copy.deepcopy(
            self._sequencer_configs.get(source_name, ChannelConfig())
        )
        self._sequence_runners.pop(display_name, None)
        self._channel_commands.setdefault(display_name, {})
        self._channel_status.setdefault(display_name, {})
        self._channel_plot_settings.setdefault(display_name, False)
        if self.backend:
            try:
                self.backend.set_channel_profiles(self._channel_profiles)
            except BackendError as exc:
                self._channel_profiles.pop(display_name, None)
                save_channel_profiles(list(self._channel_profiles.values()))
                self._sequencer_configs.pop(display_name, None)
                self._channel_commands.pop(display_name, None)
                self._channel_status.pop(display_name, None)
                self._channel_plot_settings.pop(display_name, None)
                self._show_error(str(exc))
                return
        self._refresh_channel_cards()
        self._save_settings()

    def _split_channel_name(self, name: str) -> Tuple[str, Optional[str], str]:
        match = re.search(r"(\d+)", name)
        if not match:
            return name, None, ""
        prefix = name[: match.start()]
        digits = match.group(1)
        suffix = name[match.end() :]
        return prefix, digits, suffix

    def _suggest_duplicate_target(self, base_name: str) -> str:
        prefix, digits, suffix = self._split_channel_name(base_name)
        existing = set(self._channel_profiles.keys())
        if digits:
            width = len(digits)
            number = int(digits)
            while True:
                number += 1
                candidate = f"{prefix}{number:0{width}d}{suffix}"
                if candidate not in existing:
                    return candidate
        candidate = f"{base_name}_copy"
        index = 1
        while candidate in existing:
            index += 1
            candidate = f"{base_name}_copy{index}"
        return candidate

    def _remap_channel_profile(self, profile: ChannelProfile, source_digits: str, target_digits: str) -> None:
        if not source_digits:
            return
        target_digits = target_digits or source_digits
        padded_target = target_digits.zfill(len(source_digits))
        source_plain = source_digits.lstrip("0") or source_digits
        target_plain = target_digits.lstrip("0") or target_digits

        def replace_text(text: str) -> str:
            if not text:
                return text
            updated = text.replace(source_digits, padded_target)
            pattern = re.compile(rf"(?<!\\d){re.escape(source_plain)}(?!\\d)")
            return pattern.sub(target_plain, updated)

        profile.write.message = replace_text(profile.write.message)
        profile.write.fields = {semantic: replace_text(value) for semantic, value in profile.write.fields.items()}
        profile.status.message = replace_text(profile.status.message)
        profile.status.fields = {semantic: replace_text(value) for semantic, value in profile.status.fields.items()}

    def _detect_channel_conflict(self, profile: ChannelProfile, exclude: Optional[str] = None) -> Optional[str]:
        for name, existing in self._channel_profiles.items():
            if name == exclude:
                continue
            if profile.write.message and profile.write.message == existing.write.message:
                overlap = set(profile.write.fields.values()) & set(existing.write.fields.values())
                if overlap:
                    return f"Write mapping conflicts with {name}: {', '.join(sorted(overlap))}"
            if profile.status.message and profile.status.message == existing.status.message:
                overlap = set(profile.status.fields.values()) & set(existing.status.fields.values())
                if overlap:
                    return f"Status mapping conflicts with {name}: {', '.join(sorted(overlap))}"
        return None

    def _on_sequencer_request(self, channel: str, action: str) -> None:
        runner = self._sequence_runners.get(channel)
        card = self._channel_cards.get(channel)
        if not runner or not card:
            return
        config = self._sequencer_configs.setdefault(channel, ChannelConfig())
        if action == "start":
            if runner.is_running:
                return
            enabled_sequences = [
                sequence
                for sequence in config.sequences
                if sequence.enabled and sequence.duration_s > 0 and sequence.on_s > 0 and sequence.off_s > 0
            ]
            if not enabled_sequences:
                self._show_error("No enabled sequences available for this channel.")
                return
            runner.load(config.sequences, config.repeat_mode, config.repeat_limit_s)
            if not runner.start():
                self._show_error("Failed to start the sequence. Please verify configuration values.")
                return
            card.set_sequence_running(True)
            runner.emit_progress()
        elif action == "stop":
            runner.stop()
            card.set_sequence_running(False)
        elif action == "reset":
            runner.reset()
            card.set_sequence_running(False)
        self._save_settings()

    def _apply_sequence_output(self, channel: str, pwm: float) -> None:
        pwm_value = max(0.0, min(100.0, float(pwm)))
        enabled = 1.0 if pwm_value > 0.0 else 0.0
        command = {"enabled": enabled, "select": enabled, "pwm": pwm_value}
        self._on_channel_command(channel, command)

    def _on_sequencer_config_changed(self, channel: str, payload: object) -> None:
        runner = self._sequence_runners.get(channel)
        card = self._channel_cards.get(channel)
        if runner and runner.is_running:
            if card:
                card.set_sequencer_config(self._sequencer_configs.get(channel, ChannelConfig()))
                card.set_sequence_running(True)
            self._show_error("Stop the running sequence before editing its configuration.")
            return
        data = payload if isinstance(payload, dict) else {}
        sequences_raw = data.get("sequences", []) if isinstance(data, dict) else []
        sequences: List[SequenceCfg] = []
        for entry in sequences_raw:
            if isinstance(entry, SequenceCfg):
                sequences.append(entry)
            elif isinstance(entry, dict):
                sequences.append(SequenceCfg.from_dict(entry))
        repeat_mode_value = data.get("repeat_mode") if isinstance(data, dict) else None
        if isinstance(repeat_mode_value, SequenceRepeatMode):
            repeat_mode = repeat_mode_value
        else:
            try:
                repeat_mode = SequenceRepeatMode(str(repeat_mode_value))
            except (TypeError, ValueError):
                repeat_mode = SequenceRepeatMode.OFF
        repeat_limit_s = int(max(0, int(data.get("repeat_limit_s", 0)))) if isinstance(data, dict) else 0
        config = self._sequencer_configs.setdefault(channel, ChannelConfig())
        config.sequences = sequences
        config.repeat_mode = repeat_mode
        config.repeat_limit_s = repeat_limit_s
        if runner:
            runner.load(config.sequences, config.repeat_mode, config.repeat_limit_s)
            runner.emit_progress()
        self._save_settings()

    def _on_sequence_progress(self, channel: str, seq_index: int, phase: str, remaining_s: float) -> None:
        card = self._channel_cards.get(channel)
        runner = self._sequence_runners.get(channel)
        if not card or not runner:
            return
        total = runner.sequence_count
        if seq_index < 0:
            card.update_sequence_status(None, total, None, 0.0, False)
        else:
            card.update_sequence_status(seq_index, total, phase, remaining_s, runner.is_running)

    def _on_sequence_finished(self, channel: str) -> None:
        card = self._channel_cards.get(channel)
        if card:
            card.set_sequence_running(False)
        self._save_settings()

    # Signal browser / watchlist
    def _on_add_to_watchlist(self, names: List[str]) -> None:
        added = self.watchlist_widget.add_signals(names)
        for name in added:
            if name not in self._pending_watchlist:
                self._pending_watchlist.append(name)
            if self.multi_plot_enable.isChecked():
                self.watchlist_widget.set_plot_enabled(name, True)
        self._save_settings()

    def _on_remove_from_watchlist(self, names: List[str]) -> None:
        self.watchlist_widget.remove_signals(names)
        self._pending_watchlist = [name for name in self._pending_watchlist if name not in names]
        for name in names:
            assignment = self._plot_assignments.pop(name, None)
            if assignment:
                window_id, _side = assignment
                dock = self._plot_windows.get(window_id)
                if dock:
                    dock.remove_signal(name)
        self._save_settings()

    def _apply_watchlist_plot_settings(self) -> None:
        if not self._pending_plot_signals:
            return
        self._suspend_plot_assignment = True
        try:
            for name in self.watchlist_widget.signal_names:
                self.watchlist_widget.set_plot_enabled(name, name in self._pending_plot_signals)
        finally:
            self._suspend_plot_assignment = False
        self._pending_plot_signals = []

    def _on_watchlist_plot_toggled(self, name: str, enabled: bool) -> None:
        if self._suspend_plot_assignment:
            if not enabled:
                self._multi_plot_buffers.pop(name, None)
                curve = self._multi_plot_curves.pop(name, None)
                if curve and getattr(self, "multi_plot_widget", None):
                    self.multi_plot_widget.removeItem(curve)
            return
        if enabled:
            self._assign_signal_via_dialog(name)
            if not self._plot_windows and not self.multi_plot_enable.isChecked():
                self.multi_plot_enable.setChecked(True)
        else:
            assignment = self._plot_assignments.pop(name, None)
            if assignment:
                window_id, _side = assignment
                dock = self._plot_windows.get(window_id)
                if dock:
                    dock.remove_signal(name)
            self._multi_plot_buffers.pop(name, None)
            curve = self._multi_plot_curves.pop(name, None)
            if curve and getattr(self, "multi_plot_widget", None):
                self.multi_plot_widget.removeItem(curve)
        self._save_settings()

    def _on_multi_plot_enable_changed(self, state: int) -> None:
        self._multi_plot_enabled = state == Qt.Checked
        if self._multi_plot_enabled:
            self.multi_plot_widget.show()
        else:
            self.multi_plot_widget.hide()
            for curve in self._multi_plot_curves.values():
                self.multi_plot_widget.removeItem(curve)
            self._multi_plot_curves.clear()
            self._multi_plot_buffers.clear()
        self._save_settings()

    def _toggle_multi_plot_pause(self) -> None:
        self._multi_plot_paused = not self._multi_plot_paused
        self.multi_plot_pause.setText("Resume Plot" if self._multi_plot_paused else "Pause Plot")
        self._save_settings()

    def _save_multi_plot_png(self) -> None:
        if not getattr(self, "multi_plot_widget", None):
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save plot", os.path.join(BASE_DIR, "watchlist.png"), "PNG Files (*.png)")
        if not path:
            return
        pixmap = self.multi_plot_widget.grab()
        pixmap.save(path, "PNG")

    def _on_plot_requested(self, names: List[str]) -> None:
        self._on_add_to_watchlist(names)
        for name in names:
            self.watchlist_widget.set_plot_enabled(name, True)
        if not self._plot_windows and not self.multi_plot_enable.isChecked():
            self.multi_plot_enable.setChecked(True)

    def _on_signal_simulate(self, name: str) -> None:
        if not isinstance(self.backend, DummyBackend):
            self._show_error("Signal simulation is only available in Dummy mode.")
            return
        profiles = self.backend.simulation_profiles()
        config = profiles.get(name)
        if not config:
            self._show_error(f"Signal {name} is not available for simulation")
            return
        dialog = SignalSimulationDialog(config, parent=self)
        if dialog.exec_() == QDialog.Accepted and dialog.result_profile is not None:
            try:
                self.backend.update_simulation_profile(dialog.result_profile)
            except BackendError as exc:
                self._show_error(str(exc))
                return
            self.simulation_widget.set_profiles(self.backend.simulation_profiles())

    # Dummy simulation profiles
    def _apply_simulation_profile(self, profile: SignalSimulationConfig) -> None:
        if isinstance(self.backend, DummyBackend):
            try:
                self.backend.update_simulation_profile(profile)
            except BackendError as exc:
                self._show_error(str(exc))

    # Logging
    def _browse_log_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Select log file", self.logging_path_edit.text(), "CSV Files (*.csv)")
        if path:
            self.logging_path_edit.setText(path)
            self._save_settings()

    def _toggle_logging(self) -> None:
        if self.logger.is_running():
            self.logger.stop()
            self.logging_button.setText("Start Logging")
            return
        signals = self._collect_logging_signals()
        if not signals:
            self._show_error("No signals selected for logging")
            return
        path = self.logging_path_edit.text().strip()
        if not path:
            self._show_error("Please select a CSV path")
            return
        if self.logger.start(path, signals):
            self.logging_button.setText("Stop Logging")
            self._log_interval = 1.0 / max(self.logging_rate_spin.value(), 1.0)
            self._last_log_time = 0.0
        else:
            self._show_error(self.logger.last_error)

    def _collect_logging_signals(self) -> List[str]:
        mode = self.logging_mode_combo.currentText()
        if mode == "Watchlist":
            return self.watchlist_widget.signal_names
        manual = [name.strip() for name in self.manual_log_edit.text().split(",") if name.strip()]
        self._manual_log_signals = manual
        return manual

    # Timer update
    def _on_timer(self) -> None:
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now
        if not self.backend:
            return
        try:
            self.backend.update(dt)
        except BackendError as exc:
            self._show_error(str(exc))
            return
        self._update_sequencers()
        self._refresh_values(dt)
        self._handle_logging(dt)

    def _update_sequencers(self) -> None:
        for runner in self._sequence_runners.values():
            if runner.is_running:
                runner.emit_progress()

    def _refresh_values(self, _dt: float) -> None:
        if not self.backend:
            return
        requested: set[str] = set()
        for profile in self._channel_profiles.values():
            requested.update(profile.status.fields.values())
        requested.update(self.watchlist_widget.signal_names)
        if self.logger.is_running():
            requested.update(self.logger.signal_names)
        values = self.backend.read_signal_values(requested)
        timestamp = time.monotonic()
        for channel, profile in self._channel_profiles.items():
            status = {semantic: values.get(signal, 0.0) for semantic, signal in profile.status.fields.items()}
            self._channel_status[channel] = status
            card = self._channel_cards.get(channel)
            if card:
                card.update_status(status)
                card.update_state_label(f"Signals: {len(status)}")
                command_value = self._extract_command_value(profile, self._channel_commands.get(channel, {}))
                feedback_value = self._extract_feedback_value(profile, status)
                card.record_sample(timestamp, command_value, feedback_value)
        watch_values = {name: values.get(name) for name in self.watchlist_widget.signal_names}
        self.watchlist_widget.update_values(watch_values)
        self._pending_watchlist = self.watchlist_widget.signal_names
        self._update_multi_plot(timestamp, watch_values)
        for dock in self._plot_windows.values():
            dock.update(timestamp, watch_values)

    def _handle_logging(self, dt: float) -> None:
        if not self.logger.is_running():
            return
        self._last_log_time += dt
        if self._last_log_time < self._log_interval:
            return
        self._last_log_time = 0.0
        signals = self.logger.signal_names
        if not signals:
            return
        if not self.backend:
            return
        values = self.backend.read_signal_values(signals)
        timestamp = datetime.datetime.now().isoformat()
        self.logger.log_row(timestamp, values)

    def _extract_command_value(self, profile: ChannelProfile, command: Dict[str, float]) -> Optional[float]:
        for semantic in CHANNEL_PLOT_COMMAND_SEMANTICS.get(profile.type, ()): 
            if semantic in command:
                return command[semantic]
        return None

    def _extract_feedback_value(self, profile: ChannelProfile, status: Dict[str, float]) -> Optional[float]:
        for semantic in CHANNEL_PLOT_FEEDBACK_SEMANTICS.get(profile.type, ()): 
            if semantic in status:
                return status[semantic]
        return None

    def _update_multi_plot(self, timestamp: float, values: Dict[str, float]) -> None:
        if not getattr(self, "multi_plot_widget", None):
            return
        if not self._multi_plot_enabled:
            return
        active = set(self.watchlist_widget.plot_signal_names)
        for name in list(self._multi_plot_buffers.keys()):
            if name not in active:
                self._multi_plot_buffers.pop(name, None)
                curve = self._multi_plot_curves.pop(name, None)
                if curve:
                    self.multi_plot_widget.removeItem(curve)
        for name in active:
            value = values.get(name)
            if value is None:
                continue
            buffer = self._multi_plot_buffers.setdefault(name, deque())
            buffer.append((timestamp, float(value)))
            cutoff = timestamp - 60.0
            while buffer and buffer[0][0] < cutoff:
                buffer.popleft()
            if name not in self._multi_plot_curves:
                pen = pg.intColor(len(self._multi_plot_curves))
                curve = self.multi_plot_widget.plot(name=name, pen=pen)
                self._multi_plot_curves[name] = curve
        if self._multi_plot_paused:
            return
        if not self._multi_plot_buffers:
            return
        base_time = min((entries[0] for buffer in self._multi_plot_buffers.values() for entries in buffer), default=None)
        if base_time is None:
            return
        for name, buffer in self._multi_plot_buffers.items():
            if not buffer:
                continue
            times = [entry[0] - base_time for entry in buffer]
            samples = [entry[1] for entry in buffer]
            if len(times) > 5_000:
                step = max(1, math.ceil(len(times) / 5_000))
                dec_times = times[::step]
                dec_samples = samples[::step]
                if dec_times[-1] != times[-1]:
                    dec_times.append(times[-1])
                    dec_samples.append(samples[-1])
                times, samples = dec_times, dec_samples
            curve = self._multi_plot_curves.get(name)
            if curve:
                curve.setData(times, samples)
        self.multi_plot_widget.enableAutoRange(axis=pg.ViewBox.YAxis)

    # Helpers
    def _update_status_indicator(self, connected: bool) -> None:
        color = "green" if connected else "red"
        self.status_indicator.setStyleSheet(f"color: {color}; font-size: 16pt;")

    def _show_error(self, message: str) -> None:
        self.status_message_label.setText(message)
        QMessageBox.critical(self, "Error", message)

    def _all_outputs_off(self) -> None:
        for name, profile in self._channel_profiles.items():
            if profile.write.fields:
                self._on_channel_command(name, {"enabled": 0.0, "select": 0.0, "pwm": 0.0, "state": 0.0})
        for runner in self._sequence_runners.values():
            runner.stop()
        for card in self._channel_cards.values():
            card.set_sequence_running(False)

    def _emergency_stop(self) -> None:
        self._all_outputs_off()
        if self.logger.is_running():
            self.logger.stop()
            self.logging_button.setText("Start Logging")
        self.status_message_label.setText("Emergency stop activated")

    def _browse_dbc(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select DBC", self.dbc_edit.text(), "DBC Files (*.dbc)")
        if path:
            self.dbc_edit.setText(path)
            self._load_dbc(path)
            self._save_settings()

    def _load_dbc(self, path: str) -> None:
        if not path:
            return
        try:
            database = cantools.database.load_file(path, strict=False)
        except (OSError, cantools_errors.Error, cantools_errors.ParseError) as exc:
            self._show_error(str(exc))
            return
        self._dbc = database
        self._signals_by_message = collect_signal_definitions(database)
        self.signal_browser.set_signals(self._signals_by_message)
        self._watch_units = {definition.name: definition.unit for defs in self._signals_by_message.values() for definition in defs}
        self.watchlist_widget.set_units(self._watch_units)
        self._channel_profiles = dict(sorted(load_channel_profiles(database).items()))
        self._refresh_channel_cards()
        if self._pending_watchlist:
            self.watchlist_widget.add_signals(self._pending_watchlist)
            self._pending_watchlist = self.watchlist_widget.signal_names
        self._apply_watchlist_plot_settings()
        if self.backend:
            self.backend.apply_database(database)
            self.backend.set_channel_profiles(self._channel_profiles)
            if isinstance(self.backend, DummyBackend):
                self.simulation_widget.set_profiles(self.backend.simulation_profiles())
        self._save_settings()

    def _update_dummy_tab_visibility(self) -> None:
        index = self.tab_widget.indexOf(self.dummy_tab)
        if index >= 0:
            visible = isinstance(self.backend, DummyBackend) and self._show_dummy_advanced
            if hasattr(self.tab_widget, "setTabVisible"):
                self.tab_widget.setTabVisible(index, visible)
            else:
                self.tab_widget.setTabEnabled(index, visible)
                self.dummy_tab.setVisible(visible)

    def _on_show_dummy_tab_changed(self, state: int) -> None:
        self._show_dummy_advanced = state == Qt.Checked
        self._update_dummy_tab_visibility()
        self._save_settings()

    # Settings persistence
    def _restore_settings(self) -> None:
        self.backend_name = self._qt_settings.value("mode", DummyBackend.name)
        watchlist = self._qt_settings.value("watchlist", [])
        if isinstance(watchlist, list):
            self._pending_watchlist = [str(name) for name in watchlist]
        plot_signals = self._qt_settings.value("plot_signals", [])
        if isinstance(plot_signals, list):
            self._pending_plot_signals = [str(name) for name in plot_signals]
        plots = self._qt_settings.value("channel_plots", {})
        if isinstance(plots, dict):
            self._channel_plot_settings = {str(key): bool(value) for key, value in plots.items()}
        self._multi_plot_paused = bool(self._qt_settings.value("multi_plot_paused", False))
        self._show_dummy_advanced = bool(self._qt_settings.value("show_dummy_tab", False))
        self._multi_plot_enabled = bool(self._qt_settings.value("multi_plot_enabled", False))
        sequences_data = self._qt_settings.value("channel_sequences", {})
        if isinstance(sequences_data, dict):
            for key, value in sequences_data.items():
                if isinstance(value, dict):
                    try:
                        self._sequencer_configs[str(key)] = ChannelConfig.from_dict(value)
                    except Exception:
                        continue

    def _save_settings(self) -> None:
        self._qt_settings.setValue("mode", self.backend_name)
        self._qt_settings.setValue("dbc_path", self.dbc_edit.text())
        self._qt_settings.setValue("bustype", self.bustype_combo.currentText())
        self._qt_settings.setValue("channel", self.channel_edit.text())
        self._qt_settings.setValue("bitrate", int(self.bitrate_spin.value()))
        self._qt_settings.setValue("watchlist", self.watchlist_widget.signal_names)
        self._qt_settings.setValue("plot_signals", self.watchlist_widget.plot_signal_names)
        self._qt_settings.setValue("log_rate", self.logging_rate_spin.value())
        self._qt_settings.setValue("log_path", self.logging_path_edit.text())
        self._qt_settings.setValue("channel_plots", self._channel_plot_settings)
        self._qt_settings.setValue("multi_plot_paused", self._multi_plot_paused)
        if hasattr(self, "show_dummy_checkbox"):
            self._qt_settings.setValue("show_dummy_tab", self.show_dummy_checkbox.isChecked())
        if hasattr(self, "multi_plot_enable"):
            self._qt_settings.setValue("multi_plot_enabled", self.multi_plot_enable.isChecked())
        sequence_payload = {name: config.to_dict() for name, config in self._sequencer_configs.items()}
        self._qt_settings.setValue("channel_sequences", sequence_payload)

    def closeEvent(self, event) -> None:
        self._save_settings()
        self._disconnect_backend()
        super().closeEvent(event)

def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    if geometry := window._qt_settings.value("geometry"):
        if isinstance(geometry, bytes):
            window.restoreGeometry(geometry)
    window.show()
    exit_code = app.exec_()
    window._qt_settings.setValue("geometry", window.saveGeometry())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
