"""PyQt5-based ECU control application with dynamic channels."""

from __future__ import annotations

import copy
import datetime
import math
import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import can
import cantools
from cantools.database import errors as cantools_errors
import yaml
from PyQt5.QtCore import QObject, QSettings, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
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
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
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
            runtime.current_value += (runtime.setpoint - runtime.current_value) * min(dt / max(tau, 1e-3), 1.0)
            runtime.current_value += random.uniform(-noise, noise)
            runtime.status_cache["feedback"] = runtime.current_value
        elif channel_type in {"DO", "DI"}:
            runtime.status_cache["state"] = 1.0 if runtime.enabled else 0.0
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

    def __init__(self) -> None:
        super().__init__()
        self._signals: Dict[str, List[SignalDefinition]] = {}
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


class WatchlistWidget(QWidget):
    remove_requested = pyqtSignal(list)

    def __init__(self) -> None:
        super().__init__()
        self._order: List[str] = []
        self._units: Dict[str, str] = {}
        self._last_update: Dict[str, datetime.datetime] = {}
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Signal", "Value", "Unit", "Last update"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
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
            self._order.append(name)
            added.append(name)
        return added

    def remove_signals(self, names: Iterable[str]) -> None:
        to_remove = {name for name in names}
        rows = [index for index, name in enumerate(self._order) if name in to_remove]
        for row in reversed(rows):
            name = self._order.pop(row)
            self.table.removeRow(row)
            self._last_update.pop(name, None)

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

    @property
    def signal_names(self) -> List[str]:
        return list(self._order)

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
    sequencer_requested = pyqtSignal(str, bool, float, float, float)

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
        self.quick_zero = QPushButton("Set 0")
        self.quick_mid = QPushButton("Set 50%")
        self.quick_max = QPushButton("Set 100%")
        self.apply_button = QPushButton("Apply")
        self.off_button = QPushButton("Off")
        self.sequence_on = QDoubleSpinBox()
        self.sequence_on.setRange(0.1, 3600.0)
        self.sequence_on.setValue(5.0)
        self.sequence_off = QDoubleSpinBox()
        self.sequence_off.setRange(0.1, 3600.0)
        self.sequence_off.setValue(5.0)
        self.sequence_duration = QDoubleSpinBox()
        self.sequence_duration.setRange(0.1, 600.0)
        self.sequence_duration.setValue(10.0)
        self.sequence_start = QPushButton("Start Sequence")
        self.sequence_stop = QPushButton("Stop")
        self.sequence_stop.setEnabled(False)
        self.sequence_status = QLabel("Sequence idle")
        self.values_view = QTextBrowser()
        self.values_view.setFixedHeight(90)
        self._build_layout()
        self._connect_signals()
        self._update_visibility()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self.state_label)
        control_layout = QGridLayout()
        row = 0
        control_layout.addWidget(self.enabled_checkbox, row, 0, 1, 2)
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
        buttons_layout.addWidget(self.quick_zero)
        buttons_layout.addWidget(self.quick_mid)
        buttons_layout.addWidget(self.quick_max)
        control_layout.addLayout(buttons_layout, row, 0, 1, 3)
        row += 1
        action_layout = QHBoxLayout()
        action_layout.addWidget(self.apply_button)
        action_layout.addWidget(self.off_button)
        control_layout.addLayout(action_layout, row, 0, 1, 3)
        layout.addLayout(control_layout)
        sequencer_group = QGroupBox("Sequencer")
        seq_layout = QGridLayout()
        seq_layout.addWidget(QLabel("ON (s)"), 0, 0)
        seq_layout.addWidget(self.sequence_on, 0, 1)
        seq_layout.addWidget(QLabel("OFF (s)"), 0, 2)
        seq_layout.addWidget(self.sequence_off, 0, 3)
        seq_layout.addWidget(QLabel("Duration (min)"), 1, 0)
        seq_layout.addWidget(self.sequence_duration, 1, 1)
        seq_layout.addWidget(self.sequence_start, 1, 2)
        seq_layout.addWidget(self.sequence_stop, 1, 3)
        seq_layout.addWidget(self.sequence_status, 2, 0, 1, 4)
        sequencer_group.setLayout(seq_layout)
        layout.addWidget(sequencer_group)
        layout.addWidget(QLabel("Status signals"))
        layout.addWidget(self.values_view)

    def _connect_signals(self) -> None:
        self.pwm_slider.valueChanged.connect(lambda value: self.pwm_value.setText(f"{value} %"))
        self.apply_button.clicked.connect(self._emit_command)
        self.off_button.clicked.connect(self._emit_off)
        self.quick_zero.clicked.connect(lambda: self._set_pwm_slider(0))
        self.quick_mid.clicked.connect(lambda: self._set_pwm_slider(50))
        self.quick_max.clicked.connect(lambda: self._set_pwm_slider(100))
        self.sequence_start.clicked.connect(self._start_sequence)
        self.sequence_stop.clicked.connect(self._stop_sequence)

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

    def _start_sequence(self) -> None:
        self.sequence_start.setEnabled(False)
        self.sequence_stop.setEnabled(True)
        self.sequencer_requested.emit(
            self.profile.name,
            True,
            float(self.sequence_on.value()),
            float(self.sequence_off.value()),
            float(self.sequence_duration.value()),
        )

    def _stop_sequence(self) -> None:
        self.sequence_start.setEnabled(True)
        self.sequence_stop.setEnabled(False)
        self.sequencer_requested.emit(self.profile.name, False, 0.0, 0.0, 0.0)

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
        self.quick_zero.setVisible(is_pwm or is_setpoint)
        self.quick_mid.setVisible(is_pwm)
        self.quick_max.setVisible(is_pwm)
        self.enabled_checkbox.setVisible(is_toggle)

    def update_status(self, status: Dict[str, float]) -> None:
        lines = [f"{key}: {value:.4f}" for key, value in sorted(status.items())]
        self.values_view.setPlainText("\n".join(lines))

    def update_state_label(self, text: str) -> None:
        self.state_label.setText(text)

    def update_sequencer_status(self, text: str, running: bool) -> None:
        self.sequence_status.setText(text)
        self.sequence_start.setEnabled(not running)
        self.sequence_stop.setEnabled(running)

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
        self._sequencers: Dict[str, SequencerState] = {}
        self._last_tick = time.monotonic()
        self._last_log_time = 0.0
        self._log_interval = 1.0
        self._manual_log_signals: List[str] = []
        self._pending_watchlist: List[str] = []
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
        control_layout.addWidget(add_button)
        control_layout.addWidget(edit_button)
        control_layout.addWidget(remove_button)
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
        layout.addWidget(self.signal_browser, 2)
        layout.addWidget(self.watchlist_widget, 3)
        self.signal_browser.add_requested.connect(self._on_add_to_watchlist)
        self.watchlist_widget.remove_requested.connect(self._on_remove_from_watchlist)
        self.tab_widget.addTab(widget, "Signals")

    def _build_dummy_tab(self) -> None:
        self.dummy_tab = QWidget()
        layout = QVBoxLayout(self.dummy_tab)
        self.simulation_widget = DummySimulationWidget()
        self.simulation_widget.profile_changed.connect(self._apply_simulation_profile)
        layout.addWidget(self.simulation_widget)
        self.tab_widget.addTab(self.dummy_tab, "Dummy Sim")

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
            self.channel_layout.addWidget(card)
            self._channel_cards[name] = card
            self._sequencers.setdefault(name, SequencerState())
        self.channel_layout.addStretch(1)
        self.channel_selector.clear()
        self.channel_selector.addItems(sorted(self._channel_profiles))

    def _on_channel_command(self, channel: str, command: Dict[str, float]) -> None:
        if not self.backend:
            return
        try:
            self.backend.apply_channel_command(channel, command)
        except BackendError as exc:
            self._show_error(str(exc))

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
            self._refresh_channel_cards()

    def _on_sequencer_request(self, channel: str, start: bool, on_seconds: float, off_seconds: float, duration_min: float) -> None:
        state = self._sequencers.setdefault(channel, SequencerState())
        if start:
            state.on_seconds = max(on_seconds, 0.1)
            state.off_seconds = max(off_seconds, 0.1)
            state.total_seconds = max(duration_min, 0.1) * 60.0
            state.elapsed_total = 0.0
            state.elapsed_phase = 0.0
            state.running = True
            state.is_on_phase = True
            card = self._channel_cards.get(channel)
            if card:
                state.target_pwm = float(card.pwm_slider.value())
            state.completed = False
        else:
            state.running = False
            state.completed = True
        self._update_card_sequencer(channel)

    def _update_card_sequencer(self, channel: str) -> None:
        card = self._channel_cards.get(channel)
        state = self._sequencers.get(channel)
        if card and state:
            status = "Sequence running" if state.running else "Sequence idle"
            card.update_sequencer_status(status, state.running)

    # Signal browser / watchlist
    def _on_add_to_watchlist(self, names: List[str]) -> None:
        added = self.watchlist_widget.add_signals(names)
        for name in added:
            if name not in self._pending_watchlist:
                self._pending_watchlist.append(name)
        self._save_settings()

    def _on_remove_from_watchlist(self, names: List[str]) -> None:
        self.watchlist_widget.remove_signals(names)
        self._pending_watchlist = [name for name in self._pending_watchlist if name not in names]
        self._save_settings()

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
        self._update_sequencers(dt)
        self._refresh_values(dt)
        self._handle_logging(dt)

    def _update_sequencers(self, dt: float) -> None:
        for channel, state in self._sequencers.items():
            if not state.running:
                continue
            state.elapsed_total += dt
            state.elapsed_phase += dt
            if state.elapsed_total >= state.total_seconds:
                state.running = False
                self._on_channel_command(channel, {"enabled": 0.0, "pwm": 0.0, "select": 0.0})
                self._update_card_sequencer(channel)
                continue
            phase_duration = state.on_seconds if state.is_on_phase else state.off_seconds
            if state.elapsed_phase >= phase_duration:
                state.elapsed_phase = 0.0
                state.is_on_phase = not state.is_on_phase
            if state.is_on_phase:
                self._on_channel_command(channel, {"enabled": 1.0, "select": 1.0, "pwm": state.target_pwm})
            else:
                self._on_channel_command(channel, {"enabled": 0.0, "select": 0.0, "pwm": 0.0})
            self._update_card_sequencer(channel)

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
        for channel, profile in self._channel_profiles.items():
            status = {semantic: values.get(signal, 0.0) for semantic, signal in profile.status.fields.items()}
            self._channel_status[channel] = status
            card = self._channel_cards.get(channel)
            if card:
                card.update_status(status)
                card.update_state_label(f"Signals: {len(status)}")
        watch_values = {name: values.get(name, 0.0) for name in self.watchlist_widget.signal_names}
        self.watchlist_widget.update_values(watch_values)
        self._pending_watchlist = self.watchlist_widget.signal_names

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
        for state in self._sequencers.values():
            state.running = False
        for name in self._channel_profiles:
            self._update_card_sequencer(name)

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
        if self.backend:
            self.backend.apply_database(database)
            self.backend.set_channel_profiles(self._channel_profiles)
            if isinstance(self.backend, DummyBackend):
                self.simulation_widget.set_profiles(self.backend.simulation_profiles())
        self._save_settings()

    def _update_dummy_tab_visibility(self) -> None:
        index = self.tab_widget.indexOf(self.dummy_tab)
        if index >= 0:
            self.tab_widget.setTabEnabled(index, isinstance(self.backend, DummyBackend))

    # Settings persistence
    def _restore_settings(self) -> None:
        self.backend_name = self._qt_settings.value("mode", DummyBackend.name)
        watchlist = self._qt_settings.value("watchlist", [])
        if isinstance(watchlist, list):
            self._pending_watchlist = [str(name) for name in watchlist]

    def _save_settings(self) -> None:
        self._qt_settings.setValue("mode", self.backend_name)
        self._qt_settings.setValue("dbc_path", self.dbc_edit.text())
        self._qt_settings.setValue("bustype", self.bustype_combo.currentText())
        self._qt_settings.setValue("channel", self.channel_edit.text())
        self._qt_settings.setValue("bitrate", int(self.bitrate_spin.value()))
        self._qt_settings.setValue("watchlist", self.watchlist_widget.signal_names)
        self._qt_settings.setValue("log_rate", self.logging_rate_spin.value())
        self._qt_settings.setValue("log_path", self.logging_path_edit.text())

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
