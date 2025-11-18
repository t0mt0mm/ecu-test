from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Literal

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


@dataclass
class StateCondition:
    signal: str
    operator: str = ">="
    value: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateCondition":
        signal = str(data.get("signal", ""))
        operator = str(data.get("operator", ">=")).strip() or ">="
        try:
            value = float(data.get("value", 0.0))
        except (TypeError, ValueError):
            value = 0.0
        return cls(signal=signal, operator=operator, value=value)

    def to_dict(self) -> dict:
        return {"signal": self.signal, "operator": self.operator, "value": float(self.value)}

    def evaluate(self, signal_values: Dict[str, float]) -> bool:
        actual = float(signal_values.get(self.signal, 0.0))
        expected = float(self.value)
        op = self.operator.strip()
        if op == ">":
            return actual > expected
        if op == ">=":
            return actual >= expected
        if op == "<":
            return actual < expected
        if op == "<=":
            return actual <= expected
        if op == "==":
            return actual == expected
        if op == "!=":
            return actual != expected
        return False


@dataclass
class StateAction:
    type: str = "send_message"
    message: str = ""
    fields: Dict[str, float] = field(default_factory=dict)
    channel: str = ""
    command: Dict[str, float] = field(default_factory=dict)
    sequence_mode: str = "none"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateAction":
        action_type = str(data.get("type", "send_message"))
        message = str(data.get("message", "")) if action_type == "send_message" else ""
        channel = str(data.get("channel", "")) if action_type == "set_channel" else ""
        fields_raw = data.get("fields") if isinstance(data, dict) else None
        command_raw = data.get("command") if isinstance(data, dict) else None
        sequence_value = "none"
        if isinstance(data, dict):
            raw_sequence = data.get("sequence") or data.get("sequence_mode")
            if isinstance(raw_sequence, str):
                normalized = raw_sequence.strip().lower()
                if normalized in {"start", "stop", "reset", "none"}:
                    sequence_value = normalized
            elif isinstance(raw_sequence, bool):
                sequence_value = "start" if raw_sequence else "none"
        fields: Dict[str, float] = {}
        if isinstance(fields_raw, dict):
            for key, value in fields_raw.items():
                try:
                    fields[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        command: Dict[str, float] = {}
        if isinstance(command_raw, dict):
            for key, value in command_raw.items():
                try:
                    command[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        return cls(
            type=action_type,
            message=message,
            fields=fields,
            channel=channel,
            command=command,
            sequence_mode=sequence_value,
        )

    def to_dict(self) -> dict:
        payload: Dict[str, Any] = {"type": self.type}
        if self.type == "send_message":
            payload["message"] = self.message
            payload["fields"] = {key: float(value) for key, value in self.fields.items()}
        elif self.type == "set_channel":
            payload["channel"] = self.channel
            payload["command"] = {key: float(value) for key, value in self.command.items()}
            if self.sequence_mode and self.sequence_mode != "none":
                payload["sequence"] = self.sequence_mode
        else:
            payload["data"] = {}
        return payload


@dataclass
class StateTransition:
    name: str
    source: str
    target: str
    conditions: List[StateCondition] = field(default_factory=list)
    actions: List[StateAction] = field(default_factory=list)
    condition_logic: Literal["AND", "OR"] = "AND"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateTransition":
        name = str(
            data.get("name")
            or data.get("title")
            or data.get("label")
            or data.get("id")
            or ""
        )
        source = str(
            data.get("source")
            or data.get("from")
            or data.get("start")
            or data.get("state")
            or ""
        )
        target = str(
            data.get("target")
            or data.get("to")
            or data.get("end")
            or data.get("next")
            or ""
        )
        conditions_raw = None
        actions_raw = None
        if isinstance(data, dict):
            conditions_raw = data.get("conditions") or data.get("when") or data.get("condition")
            actions_raw = data.get("actions") or data.get("do") or data.get("action")
        if conditions_raw is None:
            conditions_raw = []
        if actions_raw is None:
            actions_raw = []
        conditions: List[StateCondition] = []
        if isinstance(conditions_raw, list):
            for entry in conditions_raw:
                if isinstance(entry, dict):
                    conditions.append(StateCondition.from_dict(entry))
        elif isinstance(conditions_raw, dict):
            for key, value in conditions_raw.items():
                if isinstance(value, dict):
                    condition = StateCondition.from_dict({"signal": key, **value})
                    conditions.append(condition)
        actions: List[StateAction] = []
        if isinstance(actions_raw, list):
            for entry in actions_raw:
                if isinstance(entry, dict):
                    actions.append(StateAction.from_dict(entry))
        elif isinstance(actions_raw, dict):
            for key, value in actions_raw.items():
                if isinstance(value, dict):
                    action_payload = dict(value)
                    action_payload.setdefault("type", key)
                    actions.append(StateAction.from_dict(action_payload))
        logic = str(data.get("logic", "AND")).upper()
        logic_value: Literal["AND", "OR"] = "AND" if logic not in {"AND", "OR"} else logic  # type: ignore[assignment]
        return cls(
            name=name,
            source=source,
            target=target,
            conditions=conditions,
            actions=actions,
            condition_logic=logic_value,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "target": self.target,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "actions": [action.to_dict() for action in self.actions],
            "logic": self.condition_logic,
        }


@dataclass
class StateDefinition:
    name: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateDefinition":
        return cls(name=str(data.get("name", "")))

    def to_dict(self) -> dict:
        return {"name": self.name}


@dataclass
class StateMachineConfig:
    name: str
    states: List[StateDefinition] = field(default_factory=list)
    transitions: List[StateTransition] = field(default_factory=list)
    initial_state: Optional[str] = None

    def clone(self) -> "StateMachineConfig":
        return StateMachineConfig(
            name=self.name,
            states=[StateDefinition(name=state.name) for state in self.states],
            transitions=[
                StateTransition(
                    name=transition.name,
                    source=transition.source,
                    target=transition.target,
                    conditions=[StateCondition(signal=c.signal, operator=c.operator, value=c.value) for c in transition.conditions],
                    actions=[
                        StateAction(
                            type=action.type,
                            message=action.message,
                            fields=dict(action.fields),
                            channel=action.channel,
                            command=dict(action.command),
                        )
                        for action in transition.actions
                    ],
                    condition_logic=transition.condition_logic,
                )
                for transition in self.transitions
            ],
            initial_state=self.initial_state,
        )

    @property
    def state_names(self) -> List[str]:
        return [state.name for state in self.states]

    def ensure_initial_state(self) -> None:
        names = [state.name for state in self.states if state.name]
        if not names:
            self.initial_state = None
            return
        if self.initial_state not in names:
            self.initial_state = names[0]

    def to_dict(self) -> dict:
        self.ensure_initial_state()
        return {
            "name": self.name,
            "states": [state.to_dict() for state in self.states],
            "transitions": [transition.to_dict() for transition in self.transitions],
            "initial_state": self.initial_state,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateMachineConfig":
        name = str(data.get("name", "State Machine"))
        states_raw = data.get("states") if isinstance(data, dict) else []
        transitions_raw = data.get("transitions") if isinstance(data, dict) else []
        initial_state = data.get("initial_state") if isinstance(data, dict) else None

        def _append_state(states: List[StateDefinition], value: Any, fallback_name: Optional[str] = None) -> None:
            name_value = ""
            if isinstance(value, dict):
                candidates = [
                    value.get("name"),
                    value.get("title"),
                    value.get("label"),
                    value.get("state"),
                    value.get("id"),
                ]
                for candidate in candidates:
                    if isinstance(candidate, str) and candidate.strip():
                        name_value = candidate.strip()
                        break
                if not name_value and len(value) == 1:
                    key, nested = next(iter(value.items()))
                    if isinstance(nested, dict):
                        nested_name = nested.get("name") or nested.get("title") or nested.get("label")
                        if isinstance(nested_name, str) and nested_name.strip():
                            name_value = nested_name.strip()
                        else:
                            name_value = str(key)
                    else:
                        name_value = str(nested if nested is not None else key)
                if not name_value and fallback_name:
                    name_value = str(fallback_name)
            elif isinstance(value, str):
                name_value = value
            elif value is None and fallback_name:
                name_value = fallback_name
            else:
                name_value = str(value)
            name_value = name_value.strip()
            if not name_value and fallback_name:
                name_value = str(fallback_name).strip()
            if name_value:
                states.append(StateDefinition(name=name_value))

        states: List[StateDefinition] = []
        if isinstance(states_raw, list):
            for entry in states_raw:
                _append_state(states, entry)
        elif isinstance(states_raw, dict):
            for key, value in states_raw.items():
                _append_state(states, value, fallback_name=str(key))

        transitions: List[StateTransition] = []
        if isinstance(transitions_raw, list):
            for entry in transitions_raw:
                if isinstance(entry, dict):
                    transition = StateTransition.from_dict(entry)
                    transitions.append(transition)
        elif isinstance(transitions_raw, dict):
            for key, value in transitions_raw.items():
                if isinstance(value, dict):
                    transition = StateTransition.from_dict(value)
                    if not transition.name:
                        transition.name = str(key)
                    transitions.append(transition)

        config = cls(name=name, states=states, transitions=transitions, initial_state=initial_state)
        config.ensure_initial_state()
        return config


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
