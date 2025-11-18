from __future__ import annotations

import math
import random
import threading
import time
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import can
import cantools
from cantools.database import errors as cantools_errors
from PyQt5.QtCore import QObject, pyqtSignal

from models import (
    AnalogSimulationProfile,
    ChannelProfile,
    ConnectionSettings,
    DigitalSimulationProfile,
    OutputState,
    SignalSimulationConfig,
)


class BackendError(Exception):
    """Raised when a backend cannot complete the requested action."""


def is_signal_writable(signal_name: str, message_name: str) -> bool:
    keywords = ("ctrl", "control", "init", "write", "cmd")
    lower_message = message_name.lower()
    if any(keyword in lower_message for keyword in keywords):
        return True
    lower_name = signal_name.lower()
    return any(keyword in lower_name for keyword in keywords)


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

    def send_message_by_name(
        self,
        message: str,
        payload: Dict[str, Any],
        *,
        repeat: int = 1,
        dt_ms: int = 0,
        force: bool = False,
    ) -> None:
        raise NotImplementedError


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
            tau = float(sim_params.get("tau", 0.1))
            noise = float(sim_params.get("noise", 0.01))
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

    def send_message_by_name(
        self,
        message: str,
        payload: Dict[str, Any],
        *,
        repeat: int = 1,
        dt_ms: int = 0,
        force: bool = False,
    ) -> None:
        _ = repeat, dt_ms, force
        if self._dbc is None:
            raise BackendError("No DBC loaded for Dummy backend")
        try:
            dbc_message = self._dbc.get_message_by_name(message)
        except KeyError:
            raise BackendError(f"Message {message} not found in DBC")
        valid_names = {signal.name for signal in dbc_message.signals}
        overrides: Dict[str, float] = {}
        for name, value in payload.items():
            if name not in valid_names:
                raise BackendError(f"Signal {name} not found in message {message}")
            try:
                overrides[name] = float(value)
            except (TypeError, ValueError):
                raise BackendError(f"Invalid value for signal {name}")
        if not overrides:
            return
        for name, value in overrides.items():
            self._set_override(name, value)


class RealBackend(QObject, BackendBase):
    name = "Real"

    connection_changed = pyqtSignal(bool, str)
    status_updated = pyqtSignal()

    def __init__(self) -> None:
        QObject.__init__(self)
        BackendBase.__init__(self)
        self._settings = ConnectionSettings("", "socketcan", "can0", 500000)
        self._db = None
        self._bus = None
        self._notifier = None
        self._lock = threading.Lock()
        self._signal_cache: Dict[str, float] = {}
        self._signal_to_message: Dict[str, str] = {}
        self._message_by_name: Dict[str, Any] = {}
        self._frame_to_message: Dict[Tuple[int, bool], Any] = {}
        self._channels: Dict[str, ChannelProfile] = {}
        self._status_handlers: Dict[Tuple[int, bool], Tuple[Any, ChannelProfile]] = {}
        self._write_frame_ids: Set[Tuple[int, bool]] = set()
        self._status_signal_names: Set[str] = set()
        self._command_state: Dict[str, Dict[str, float]] = {}

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
            bustype = (self._settings.bustype or "socketcan").lower()
            channel = self._settings.channel or "can0"
            bitrate = int(self._settings.bitrate) if self._settings.bitrate else 500000
            bus_kwargs: Dict[str, Any] = {"bustype": bustype, "channel": channel}
            if bitrate:
                bus_kwargs["bitrate"] = bitrate
            # Prefer explicitly enabling CAN-FD only when the interface type supports it.
            if bustype in {"pcan", "vector"} and bitrate and bitrate > 1000000:
                bus_kwargs["fd"] = True
            self._bus = can.interface.Bus(**bus_kwargs)
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
        self._message_by_name = {}
        self._frame_to_message = {}
        self._write_frame_ids = set()
        self._status_signal_names = set()
        self._command_state = {}
        if self._db is not None:
            for message in getattr(self._db, "messages", []):
                self._message_by_name[message.name] = message
                for signal in message.signals:
                    self._signal_to_message[signal.name] = message.name
                key = (int(message.frame_id), bool(getattr(message, "is_extended_frame", False)))
                self._frame_to_message[key] = message
        with self._lock:
            self._signal_cache = {name: 0.0 for name in self._signal_to_message}
        self.status_updated.emit()

    def set_channel_profiles(self, profiles: Dict[str, ChannelProfile]) -> None:
        self._channels = profiles
        self._status_handlers = {}
        self._write_frame_ids = set()
        self._status_signal_names = set()
        if not self._db:
            return
        for profile in profiles.values():
            write_message_name = profile.write.message
            if write_message_name:
                write_message = self._message_by_name.get(write_message_name)
                if write_message is not None:
                    key = (int(write_message.frame_id), bool(getattr(write_message, "is_extended_frame", False)))
                    self._write_frame_ids.add(key)
            for signal_name in profile.status.fields.values():
                if signal_name:
                    self._status_signal_names.add(signal_name)
            message_name = profile.status.message
            message = self._message_by_name.get(message_name) if message_name else None
            if message is None:
                continue
            key = (int(message.frame_id), bool(getattr(message, "is_extended_frame", False)))
            self._status_handlers[key] = (message, profile)
            for signal in message.signals:
                self._status_signal_names.add(signal.name)

    def _compose_command_payload(self, message: Any, overrides: Dict[str, float]) -> Dict[str, float]:
        with self._lock:
            base = dict(self._command_state.get(message.name, {}))
        for signal in getattr(message, "signals", []):
            if signal.name not in base:
                initial = getattr(signal, "initial", None)
                base[signal.name] = float(initial) if initial is not None else 0.0
        for name, value in overrides.items():
            base[name] = float(value)
        return base

    def _finalize_command(
        self,
        message_name: str,
        complete: Dict[str, float],
        overrides: Dict[str, float],
    ) -> None:
        with self._lock:
            self._command_state[message_name] = {name: float(value) for name, value in complete.items()}
            for name, value in overrides.items():
                if name not in self._status_signal_names:
                    self._signal_cache[name] = float(value)

    def apply_channel_command(self, channel: str, command: Dict[str, float]) -> None:
        profile = self._channels.get(channel)
        if profile is None or self._db is None or self._bus is None:
            return
        message_name = profile.write.message
        if not message_name:
            raise BackendError(f"Channel {channel} has no write message configured")
        message = self._message_by_name.get(message_name)
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
        # The payload currently only contains the mapped fields for this channel
        if not payload:
            return
        try:
            complete = self._compose_command_payload(message, payload)
            data = message.encode(complete, scaling=True, strict=True)

            can_message = can.Message(
                arbitration_id=int(message.frame_id),
                data=data,
                is_extended_id=bool(getattr(message, "is_extended_frame", False)),
                is_fd=True,
                bitrate_switch=True,
            )
            self._bus.send(can_message)

            self._finalize_command(message_name, complete, payload)

        except (ValueError, can.CanError) as exc:
            raise BackendError(str(exc))

    def send_message_by_name(
        self,
        message: str,
        payload: Dict[str, Any],
        *,
        repeat: int = 1,
        dt_ms: int = 0,
        force: bool = False,
    ) -> None:
        _ = force
        if self._db is None or self._bus is None:
            raise BackendError("Real backend is not connected")
        dbc_message = self._message_by_name.get(message)
        if dbc_message is None:
            raise BackendError(f"Message {message} not found in DBC")
        valid_names = {signal.name for signal in dbc_message.signals}
        prepared: Dict[str, float] = {}
        for name, value in payload.items():
            if name not in valid_names:
                raise BackendError(f"Signal {name} not found in message {message}")
            if not is_signal_writable(name, message):
                raise BackendError(f"Signal {name} is not writable")
            try:
                prepared[name] = float(value)
            except (TypeError, ValueError):
                raise BackendError(f"Invalid value for signal {name}")
        if not prepared:
            return
        repeat_count = max(1, int(repeat))
        interval = max(0, int(dt_ms)) / 1000.0
        for attempt_index in range(repeat_count):
            tries = 0
            while tries < 3:
                try:
                    complete = self._compose_command_payload(dbc_message, prepared)
                    data = dbc_message.encode(complete, scaling=True, strict=True)

                    message_obj = can.Message(
                        arbitration_id=int(dbc_message.frame_id),
                        data=data,
                        is_extended_id=bool(getattr(dbc_message, "is_extended_frame", False)),
                        is_fd=True,
                        bitrate_switch=True,
                    )
                    self._bus.send(message_obj)

                    self._finalize_command(message, complete, prepared)
                    break
                except (ValueError, can.CanError) as exc:
                    tries += 1
                    if tries >= 3:
                        raise BackendError(str(exc))
                    if interval > 0.0:
                        time.sleep(interval)

    def update(self, dt: float) -> None:
        _ = dt

    def read_signal_values(self, signal_names: Iterable[str]) -> Dict[str, float]:
        with self._lock:
            return {name: float(self._signal_cache.get(name, 0.0)) for name in signal_names}

    def _handle_status(self, message: can.Message) -> None:
        if self._db is None:
            return
        key = (int(message.arbitration_id), bool(getattr(message, "is_extended_id", False)))
        handler = self._status_handlers.get(key)
        if handler is None:
            if key in self._write_frame_ids:
                return
            dbc_message = self._frame_to_message.get(key)
            if dbc_message is None:
                return
            # Skip decoding for unhandled frames that would overwrite existing
            # status signals, e.g. safety-domain messages duplicating high-side
            # feedback values with different scaling/semantics.
            if any(signal.name in self._status_signal_names for signal in dbc_message.signals):
                return
        else:
            dbc_message, _profile = handler
        if dbc_message is None:
            return
        expected_length = getattr(dbc_message, "length", None)
        if expected_length is not None and len(message.data) != expected_length:
            return
        try:
            decoded = dbc_message.decode(message.data, decode_choices=False, scaling=True)
        except (ValueError, KeyError, cantools_errors.DecodeError):
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
