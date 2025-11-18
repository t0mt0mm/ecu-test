from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


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

    def to_dict(self) -> dict:
        data: Dict[str, Any] = {
            "message_name": self.message_name,
            "name": self.name,
            "unit": self.unit,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "category": self.category,
        }
        if self.analog is not None:
            data["analog"] = asdict(self.analog)
        if self.digital is not None:
            data["digital"] = asdict(self.digital)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalSimulationConfig":
        analog_data = data.get("analog") if isinstance(data, dict) else None
        digital_data = data.get("digital") if isinstance(data, dict) else None
        analog: Optional[AnalogSimulationProfile] = None
        digital: Optional[DigitalSimulationProfile] = None
        if isinstance(analog_data, dict):
            try:
                analog = AnalogSimulationProfile(**analog_data)
            except TypeError:
                analog = AnalogSimulationProfile()
                for key, value in analog_data.items():
                    if hasattr(analog, key):
                        setattr(analog, key, value)
        if isinstance(digital_data, dict):
            try:
                digital = DigitalSimulationProfile(**digital_data)
            except TypeError:
                digital = DigitalSimulationProfile()
                for key, value in digital_data.items():
                    if hasattr(digital, key):
                        setattr(digital, key, value)
        return cls(
            message_name=str(data.get("message_name", "")),
            name=str(data.get("name", "")),
            unit=str(data.get("unit", "")),
            minimum=data.get("minimum"),
            maximum=data.get("maximum"),
            category=str(data.get("category", "analog")),
            analog=analog,
            digital=digital,
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


@dataclass
class StartupGlobalStep:
    message: str
    fields: Dict[str, float]
    repeat: int = 1
    dt_ms: int = 0

    def to_dict(self) -> dict:
        payload = {
            "message": self.message,
            "fields": {name: float(value) for name, value in self.fields.items()},
        }
        if self.repeat != 1:
            payload["repeat"] = int(max(1, self.repeat))
        if self.dt_ms:
            payload["dt_ms"] = int(max(0, self.dt_ms))
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "StartupGlobalStep":
        message = str(data.get("message", ""))
        fields_raw = data.get("fields", {}) if isinstance(data, dict) else {}
        fields: Dict[str, float] = {}
        if isinstance(fields_raw, dict):
            for key, value in fields_raw.items():
                try:
                    fields[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        repeat = data.get("repeat", 1) if isinstance(data, dict) else 1
        dt_ms = data.get("dt_ms", 0) if isinstance(data, dict) else 0
        try:
            repeat_value = int(repeat)
        except (TypeError, ValueError):
            repeat_value = 1
        try:
            delay_value = int(dt_ms)
        except (TypeError, ValueError):
            delay_value = 0
        return cls(message=message, fields=fields, repeat=max(1, repeat_value), dt_ms=max(0, delay_value))


@dataclass
class StartupPerOutputStep:
    channel: str
    message: str
    fields: Dict[str, float]
    repeat: int = 1
    dt_ms: int = 0

    def to_dict(self) -> dict:
        payload = {
            "channel": self.channel,
            "message": self.message,
            "fields": {name: float(value) for name, value in self.fields.items()},
        }
        if self.repeat != 1:
            payload["repeat"] = int(max(1, self.repeat))
        if self.dt_ms:
            payload["dt_ms"] = int(max(0, self.dt_ms))
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "StartupPerOutputStep":
        channel = str(data.get("channel", ""))
        message = str(data.get("message", ""))
        fields_raw = data.get("fields", {}) if isinstance(data, dict) else {}
        fields: Dict[str, float] = {}
        if isinstance(fields_raw, dict):
            for key, value in fields_raw.items():
                try:
                    fields[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        repeat = data.get("repeat", 1) if isinstance(data, dict) else 1
        dt_ms = data.get("dt_ms", 0) if isinstance(data, dict) else 0
        try:
            repeat_value = int(repeat)
        except (TypeError, ValueError):
            repeat_value = 1
        try:
            delay_value = int(dt_ms)
        except (TypeError, ValueError):
            delay_value = 0
        return cls(
            channel=channel,
            message=message,
            fields=fields,
            repeat=max(1, repeat_value),
            dt_ms=max(0, delay_value),
        )


@dataclass
class StartupTeardownStep:
    message: str
    fields: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "message": self.message,
            "fields": {name: float(value) for name, value in self.fields.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StartupTeardownStep":
        message = str(data.get("message", ""))
        fields_raw = data.get("fields", {}) if isinstance(data, dict) else {}
        fields: Dict[str, float] = {}
        if isinstance(fields_raw, dict):
            for key, value in fields_raw.items():
                try:
                    fields[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        return cls(message=message, fields=fields)


@dataclass
class StartupConfig:
    version: int = 1
    globals: List[StartupGlobalStep] = field(default_factory=list)
    per_output: List[StartupPerOutputStep] = field(default_factory=list)
    teardown: List[StartupTeardownStep] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "version": int(self.version),
            "globals": [step.to_dict() for step in self.globals],
            "per_output": [step.to_dict() for step in self.per_output],
            "teardown": [step.to_dict() for step in self.teardown],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StartupConfig":
        if not isinstance(data, dict):
            return cls()
        version_value = data.get("version", 1)
        try:
            version = int(version_value)
        except (TypeError, ValueError):
            version = 1
        globals_section = data.get("globals", []) if isinstance(data, dict) else []
        per_output_section = data.get("per_output", []) if isinstance(data, dict) else []
        teardown_section = data.get("teardown", []) if isinstance(data, dict) else []
        globals_steps = [StartupGlobalStep.from_dict(entry) for entry in globals_section if isinstance(entry, dict)]
        per_output_steps = [StartupPerOutputStep.from_dict(entry) for entry in per_output_section if isinstance(entry, dict)]
        teardown_steps = [StartupTeardownStep.from_dict(entry) for entry in teardown_section if isinstance(entry, dict)]
        return cls(version=version, globals=globals_steps, per_output=per_output_steps, teardown=teardown_steps)


@dataclass
class StartupPreparedStep:
    key: Tuple[str, Optional[str]]
    message: str
    payload: Dict[str, float]
    repeat: int
    dt_ms: int
    channel: Optional[str] = None
