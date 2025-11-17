"""PyQt5-based ECU control application with dynamic channels."""

from __future__ import annotations

import copy
import csv
import datetime
import math
import numpy as np
import os
import random
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, TYPE_CHECKING
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*sipPyTypeDict.*")

import json
import can
import cantools
from cantools.database import errors as cantools_errors
import yaml
import pandas as pd
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")
import pyqtgraph as pg
if pg.Qt.QT_LIB != "PyQt5":
    raise RuntimeError(
        "PyQtGraph loaded Qt binding '" + pg.Qt.QT_LIB + "' but this application requires PyQt5. "
        "Ensure PyQt5 is installed and set PYQTGRAPH_QT_LIB=PyQt5 before launching."
    )
from PyQt5.QtCore import (
    QObject,
    QSettings,
    QTimer,
    Qt,
    QTime,
    QSize,
    QPropertyAnimation,
    QEasingCurve,
    QThread,
    QByteArray,
    QPointF,
    QRectF,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtGui import QColor, QPalette, QPen, QBrush, QPainterPath, QPolygonF, QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCompleter,
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
    QSplitter,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStatusBar,
    QStyle,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTimeEdit,
    QToolBar,
    QToolButton,
    QTextBrowser,
    QListWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QWidgetAction,
    QSizePolicy,
    QFrame,
    QVBoxLayout,
    QWidget,
    QRadioButton,
    QMenu,
    QAction,
    QStackedWidget,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsTextItem,
)

APP_QSS = """
/* Global */
QMainWindow, QDialog, QWidget { background: #ffffff; color: #1f1f1f; }
QAbstractScrollArea,
QAbstractScrollArea > QWidget#qt_scrollarea_viewport { background: #ffffff; }

/* Tables / Trees */
QHeaderView::section { background: #f5f5f7; padding: 6px; border: 1px solid #e6e6e6; }
QTableView, QTreeView {
    background: #ffffff;
    alternate-background-color: #f7f7f9;
    gridline-color: #e0e0e0;
    selection-background-color: #e6f2ff;
    selection-color: #1f1f1f;
}

/* Inputs */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateTimeEdit {
    background: #ffffff; border: 1px solid #d0d0d0; border-radius: 4px; padding: 4px 6px;
}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QDateTimeEdit:focus {
    border-color: #4096ff;
}

/* Buttons */
QPushButton { background: #f7f7f8; border: 1px solid #d0d0d0; border-radius: 6px; padding: 6px 10px; }
QPushButton:hover { background: #fbfbfc; }
QPushButton:pressed { background: #ededee; }
QPushButton:disabled { color: #9aa0a6; background: #f3f4f6; border-color: #e5e7eb; }

/* Gruppen / Tabs / Docks */
QGroupBox { margin-top: 14px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; top: 0px; padding: 0 4px; background: transparent; }
QTabWidget::pane { border: 1px solid #e0e0e0; }
QTabBar::tab { background: #eef1f5; border: 1px solid #e0e0e0; padding: 6px 10px; margin: 1px; }
QTabBar::tab:selected { background: #ffffff; border-bottom-color: #ffffff; }
QMainWindow::separator { background: #d0d0d0; width: 2px; height: 2px; }
QDockWidget { border: 1px solid #d0d0d0; }
QDockWidget::title { padding: 4px 6px; }
QStatusBar, QToolBar, QMenuBar { background: #ffffff; }
"""


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
PROFILE_DIR = os.path.join(BASE_DIR, "profiles")
WHITELIST_PATH = os.path.join(CONFIG_DIR, "signals.yaml")
CHANNEL_PROFILE_PATH = os.path.join(PROFILE_DIR, "channels.yaml")
PRIMARY_SETUP_PATH = os.path.join(PROFILE_DIR, "fccu.yaml")
DEFAULT_SETUP_PATH = os.path.join(PROFILE_DIR, "default_setup.yaml")
DUMMY_SIMULATION_PATH = os.path.join(PROFILE_DIR, "dummy_simulations.json")


FACTORY_DEFAULT_SETUP = {
    "version": 1,
    "backend": {"type": "dummy", "device_id": None},
    "signals": {
        "watch": [],
        "plot": [],
        "multi_plot": {"enabled": False, "paused": False, "windows": []},
    },
    "channels": {"profiles": [], "plot_visibility": {}},
    "sequencer": {"per_channel": {}},
    "state_machines": {"active": None, "definitions": []},
    "dummy": {"sim": {}},
    "startup": {"version": 1, "globals": [], "per_output": [], "teardown": []},
}



PRESETS = {
    "excel_de": {"sep": ";", "decimal": ",", "encoding": "utf-8-sig"},
    "generic_en": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
}

PRESET_LABELS = {
    "excel_de": "Excel (DE)",
    "generic_en": "Generic (EN)",
}


def get_csv_preset_key() -> str:
    settings = QSettings("OpenAI", "ECUControl")
    value = settings.value("csv/preset", "excel_de")
    if isinstance(value, str):
        key = value
    elif value is None:
        key = "excel_de"
    else:
        key = str(value)
    if key not in PRESETS:
        key = "excel_de"
    return key


def get_csv_opts() -> dict:
    key = get_csv_preset_key()
    return PRESETS[key]


def get_csv_preset_label(key: str) -> str:
    return PRESET_LABELS.get(key, PRESET_LABELS["excel_de"])


def write_csv(df: pd.DataFrame, path: str) -> None:
    preset_key = get_csv_preset_key()
    opts = PRESETS[preset_key]
    export = df.copy()
    if preset_key == "excel_de":
        datetime_columns = export.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
        for column in datetime_columns:
            export[column] = export[column].dt.strftime("%Y-%m-%d %H:%M:%S")
    export.to_csv(
        path,
        index=False,
        sep=opts["sep"],
        decimal=opts["decimal"],
        encoding=opts["encoding"],
        quoting=csv.QUOTE_MINIMAL,
    )


def read_csv(path: str) -> pd.DataFrame:
    opts = get_csv_opts()
    frame = pd.read_csv(path, sep=opts["sep"], decimal=opts["decimal"], encoding=opts["encoding"])
    object_columns = list(frame.select_dtypes(include=["object"]).columns)
    if object_columns:
        decimal = opts["decimal"]
        for column in object_columns:
            series = frame[column]
            if decimal != ".":
                normalized = series.astype(str).str.replace(decimal, ".", regex=False)
            else:
                normalized = series
            converted = pd.to_numeric(normalized, errors="coerce")
            if converted.notna().any():
                frame[column] = converted
    return frame



class CollapsibleSection(QWidget):
    """A collapsible container with animated content height."""

    toggled = pyqtSignal(bool)

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._title = title
        self._badge = ""
        self.toggle_button = QToolButton()
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setAutoRaise(True)
        self.toggle_button.toggled.connect(self._on_toggled)
        self.badge_label = QLabel()
        self.badge_label.setStyleSheet("font-weight: 600; color: palette(dark);")
        self._block_signal = False
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)
        header_layout.addWidget(self.toggle_button)
        header_layout.addStretch(1)
        header_layout.addWidget(self.badge_label)
        self.content_area = QFrame()
        self.content_area.setFrameShape(QFrame.NoFrame)
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.content_area.setMaximumHeight(0)
        self.animation = QPropertyAnimation(self.content_area, b"maximumHeight", self)
        self.animation.setDuration(160)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(header_layout)
        layout.addWidget(self.content_area)
        self._update_title()

    def set_content(self, widget: QWidget) -> None:
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.addWidget(widget)
        self.content_area.setLayout(container_layout)

    def set_badge(self, text: str) -> None:
        self._badge = text
        self.badge_label.setText(text)

    def set_collapsed(self, collapsed: bool, animate: bool = False) -> None:
        if self.toggle_button.isChecked() == (not collapsed):
            return
        self._block_signal = True
        self.toggle_button.setChecked(not collapsed)
        self._block_signal = False
        self._apply_toggle(animate=animate)

    def is_collapsed(self) -> bool:
        return not self.toggle_button.isChecked()

    def _on_toggled(self, checked: bool) -> None:
        if self._block_signal:
            return
        self._apply_toggle(animate=True)
        self.toggled.emit(checked)

    def _apply_toggle(self, animate: bool) -> None:
        expanded = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        if self.content_area.layout():
            content_height = self.content_area.layout().sizeHint().height()
        else:
            content_height = 0
        end_value = content_height if expanded else 0
        if animate:
            self.animation.stop()
            self.animation.setStartValue(self.content_area.maximumHeight())
            self.animation.setEndValue(end_value)
            self.animation.start()
        else:
            self.content_area.setMaximumHeight(end_value)

    def _update_title(self) -> None:
        text = self._title
        self.toggle_button.setText(text)

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
        return cls(name=name, source=source, target=target, conditions=conditions, actions=actions)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "target": self.target,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "actions": [action.to_dict() for action in self.actions],
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


class StartupWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, list, str)

    def __init__(
        self,
        backend: BackendBase,
        steps: List[StartupPreparedStep],
        *,
        delay_ms: int = 0,
        force: bool = False,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._backend = backend
        self._steps = steps
        self._delay_ms = max(0, int(delay_ms))
        self._force = force
        self._stop_event = threading.Event()

    def cancel(self) -> None:
        self._stop_event.set()

    @pyqtSlot()
    def run(self) -> None:
        successes: List[Tuple[Tuple[str, Optional[str]], Dict[str, float]]] = []
        errors: List[str] = []
        delay = self._delay_ms / 1000.0 if self._delay_ms > 0 else 0.0
        for index, step in enumerate(self._steps):
            if self._stop_event.is_set():
                errors.append("Startup cancelled")
                break
            description = step.message
            if step.channel:
                description += f" ({step.channel})"
            try:
                self._backend.send_message_by_name(
                    step.message,
                    step.payload,
                    repeat=max(1, step.repeat),
                    dt_ms=max(0, step.dt_ms),
                    force=self._force,
                )
                self.progress.emit(f"Sent {description}: {step.payload}")
                successes.append((step.key, dict(step.payload)))
            except BackendError as exc:
                error_message = f"Failed {description}: {exc}"
                self.progress.emit(error_message)
                errors.append(error_message)
            if delay > 0.0 and index < len(self._steps) - 1:
                time.sleep(delay)
        success = not errors
        summary = "Startup completed" if success else "Startup completed with issues"
        if errors and success:
            summary = "Startup completed with warnings"
        elif errors:
            summary = errors[-1]
        self.finished.emit(success, successes, summary)


class StartupStepDialog(QDialog):
    def __init__(
        self,
        mode: str,
        channels: List[str],
        signals_by_message: Dict[str, List[SignalDefinition]],
        *,
        parent: Optional[QWidget] = None,
        existing: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configure startup step")
        self._mode = mode
        self._channels = channels
        self._signals_by_message = signals_by_message
        self.result_step: Optional[Any] = None
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.message_combo = QComboBox()
        self.message_combo.setEditable(True)
        self.message_combo.setInsertPolicy(QComboBox.NoInsert)
        self.message_combo.blockSignals(True)
        for name in sorted(self._signals_by_message.keys()):
            self.message_combo.addItem(name)
        self.message_combo.blockSignals(False)
        form.addRow("Message", self.message_combo)
        if mode == "per_output":
            self.channel_combo = QComboBox()
            self.channel_combo.addItems(channels)
            self.channel_combo.setEditable(True)
            form.addRow("Channel", self.channel_combo)
        else:
            self.channel_combo = None
        self.repeat_spin = QSpinBox()
        self.repeat_spin.setRange(1, 100)
        self.repeat_spin.setValue(1)
        self.delay_spin = QSpinBox()
        self.delay_spin.setRange(0, 10_000)
        self.delay_spin.setSuffix(" ms")
        if mode != "teardown":
            form.addRow("Repeat", self.repeat_spin)
            form.addRow("Delay", self.delay_spin)
        layout.addLayout(form)
        self.field_table = QTableWidget(0, 2)
        self.field_table.setHorizontalHeaderLabels(["Signal", "Value"])
        self.field_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.field_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.field_table.verticalHeader().setVisible(False)
        self.field_table.setAlternatingRowColors(True)
        self.field_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.field_table.setStyleSheet("QTableView::item { padding: 0px; }")
        layout.addWidget(self.field_table)
        self.message_combo.currentTextChanged.connect(self._on_message_changed)
        field_buttons = QHBoxLayout()
        add_field = QToolButton()
        add_field.setText("Add field")
        add_field.clicked.connect(self._add_field_row)
        remove_field = QToolButton()
        remove_field.setText("Remove field")
        remove_field.clicked.connect(self._remove_field_row)
        field_buttons.addWidget(add_field)
        field_buttons.addWidget(remove_field)
        field_buttons.addStretch(1)
        layout.addLayout(field_buttons)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        if existing is not None:
            self._populate_from_existing(existing)
        else:
            self._add_field_row()
        self._on_message_changed(self.message_combo.currentText())

    def _signal_options(self, message_name: str) -> List[str]:
        definitions = self._signals_by_message.get(message_name, [])
        names: List[str] = []
        for definition in definitions:
            if is_signal_writable(definition.name, message_name):
                names.append(definition.name)
        return names

    def _populate_field_combo(self, combo: QComboBox, selected: str = "") -> None:
        message_name = self.message_combo.currentText().strip()
        options = self._signal_options(message_name) if message_name else []
        current = selected or combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        for name in options:
            combo.addItem(name)
        if current and current not in options:
            combo.addItem(current)
        if current:
            combo.setCurrentText(current)
        elif options:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _add_field_row(self) -> None:
        row = self.field_table.rowCount()
        self.field_table.insertRow(row)
        combo = QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.NoInsert)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._populate_field_combo(combo)
        self.field_table.setCellWidget(row, 0, combo)
        value_item = QTableWidgetItem("0")
        self.field_table.setItem(row, 1, value_item)
        header = self.field_table.verticalHeader()
        combo_height = combo.sizeHint().height()
        header.setMinimumSectionSize(combo_height)
        header.setDefaultSectionSize(combo_height)

    def _remove_field_row(self) -> None:
        row = self.field_table.currentRow()
        if row >= 0:
            self.field_table.removeRow(row)

    def _on_message_changed(self, _text: str) -> None:
        if not hasattr(self, "field_table"):
            return
        for row in range(self.field_table.rowCount()):
            widget = self.field_table.cellWidget(row, 0)
            if isinstance(widget, QComboBox):
                self._populate_field_combo(widget)

    def _populate_from_existing(self, existing: Any) -> None:
        if hasattr(existing, "message"):
            index = self.message_combo.findText(existing.message)
            if index >= 0:
                self.message_combo.setCurrentIndex(index)
            else:
                self.message_combo.setEditText(existing.message)
        if self.channel_combo is not None and hasattr(existing, "channel"):
            index = self.channel_combo.findText(existing.channel)
            if index >= 0:
                self.channel_combo.setCurrentIndex(index)
            elif self.channel_combo.isEditable():
                self.channel_combo.setEditText(existing.channel)
        if hasattr(existing, "repeat"):
            self.repeat_spin.setValue(int(getattr(existing, "repeat", 1)))
        if hasattr(existing, "dt_ms"):
            self.delay_spin.setValue(int(getattr(existing, "dt_ms", 0)))
        if existing.fields:
            for name, value in existing.fields.items():
                row = self.field_table.rowCount()
                self.field_table.insertRow(row)
                combo = QComboBox()
                combo.setEditable(True)
                combo.setInsertPolicy(QComboBox.NoInsert)
                combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
                self.field_table.setCellWidget(row, 0, combo)
                self._populate_field_combo(combo, str(name))
                value_item = QTableWidgetItem(str(value))
                self.field_table.setItem(row, 1, value_item)
                header = self.field_table.verticalHeader()
                combo_height = combo.sizeHint().height()
                header.setMinimumSectionSize(combo_height)
                header.setDefaultSectionSize(combo_height)
        else:
            self._add_field_row()

    def accept(self) -> None:  # type: ignore[override]
        message = self.message_combo.currentText().strip()
        if not message:
            QMessageBox.warning(self, "Invalid input", "Message name is required.")
            return
        fields: Dict[str, float] = {}
        for row in range(self.field_table.rowCount()):
            widget = self.field_table.cellWidget(row, 0)
            value_item = self.field_table.item(row, 1)
            if not isinstance(widget, QComboBox) or value_item is None:
                continue
            signal_name = widget.currentText().strip()
            if not signal_name:
                continue
            try:
                numeric_value = float(value_item.text())
            except (TypeError, ValueError):
                QMessageBox.warning(self, "Invalid input", f"Value for '{signal_name}' is not numeric.")
                return
            fields[signal_name] = numeric_value
        if not fields:
            QMessageBox.warning(self, "Invalid input", "At least one signal/value pair is required.")
            return
        if self._mode == "per_output":
            if self.channel_combo is None:
                QMessageBox.warning(self, "Invalid input", "Channel selection is required.")
                return
            channel = self.channel_combo.currentText().strip()
            if not channel:
                QMessageBox.warning(self, "Invalid input", "Channel name is required.")
                return
            self.result_step = StartupPerOutputStep(
                channel=channel,
                message=message,
                fields=fields,
                repeat=max(1, int(self.repeat_spin.value())),
                dt_ms=max(0, int(self.delay_spin.value())),
            )
        elif self._mode == "teardown":
            self.result_step = StartupTeardownStep(message=message, fields=fields)
        else:
            self.result_step = StartupGlobalStep(
                message=message,
                fields=fields,
                repeat=max(1, int(self.repeat_spin.value())),
                dt_ms=max(0, int(self.delay_spin.value())),
            )
        super().accept()

class MessageParameterWidget(QWidget):
    textChanged = pyqtSignal(str)

    def __init__(self, signals: Iterable[str] = (), text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.signal_combo = QComboBox(self)
        self.signal_combo.setEditable(False)
        self.signal_combo.setInsertPolicy(QComboBox.NoInsert)
        self.signal_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.signal_combo.addItem("Select signal…", "")
        self.signal_combo.currentIndexChanged.connect(self._on_signal_selected)
        self.line_edit = QLineEdit(self)
        self.line_edit.setPlaceholderText("signal=value, comma separated")
        self.line_edit.textChanged.connect(self.textChanged)
        layout.addWidget(self.signal_combo)
        layout.addWidget(self.line_edit, 1)
        self.setFocusProxy(self.line_edit)
        self._signals: List[str] = []
        self.set_signals(signals)
        self.set_text(text)

    def set_signals(self, signals: Iterable[str]) -> None:
        normalized = sorted({str(name).strip() for name in signals if str(name).strip()})
        if normalized == self._signals:
            return
        self._signals = normalized
        current = self.signal_combo.currentData() if self.signal_combo.count() else ""
        self.signal_combo.blockSignals(True)
        self.signal_combo.clear()
        self.signal_combo.addItem("Select signal…", "")
        for name in self._signals:
            self.signal_combo.addItem(name, name)
        if isinstance(current, str) and current in self._signals:
            index = self.signal_combo.findData(current)
            if index >= 0:
                self.signal_combo.setCurrentIndex(index)
            else:
                self.signal_combo.setCurrentIndex(0)
        else:
            self.signal_combo.setCurrentIndex(0)
        self.signal_combo.blockSignals(False)

    def set_text(self, text: str) -> None:
        self.line_edit.blockSignals(True)
        self.line_edit.setText(text)
        self.line_edit.blockSignals(False)

    def text(self) -> str:
        return self.line_edit.text().strip()

    def setToolTip(self, text: str) -> None:  # type: ignore[override]
        super().setToolTip(text)
        self.line_edit.setToolTip(text)
        self.signal_combo.setToolTip("Insert signal name into the parameter list")

    def _on_signal_selected(self, index: int) -> None:
        name = self.signal_combo.itemData(index)
        if not isinstance(name, str) or not name:
            return
        existing = self.text()
        snippet = f"{name}="
        if snippet not in existing:
            updated = f"{existing}, {snippet}" if existing else snippet
            self.set_text(updated)
        self.signal_combo.blockSignals(True)
        self.signal_combo.setCurrentIndex(0)
        self.signal_combo.blockSignals(False)


class StateTransitionDialog(QDialog):
    def __init__(
        self,
        states: List[str],
        *,
        signal_names: Iterable[str] = (),
        message_names: Iterable[str] = (),
        message_signals: Optional[Dict[str, Iterable[str]]] = None,
        channel_names: Iterable[str] = (),
        parent: Optional[QWidget] = None,
        existing: Optional[StateTransition] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configure transition")
        self._states = list(states)
        self._signal_names = sorted({str(name).strip() for name in signal_names if str(name).strip()})
        self._message_names = sorted({str(name).strip() for name in message_names if str(name).strip()})
        self._message_signals: Dict[str, List[str]] = {}
        if message_signals:
            for key, values in message_signals.items():
                message = str(key).strip()
                if not message:
                    continue
                normalized = sorted({str(value).strip() for value in values if str(value).strip()})
                self._message_signals[message] = normalized
        self._channel_names = sorted({str(name).strip() for name in channel_names if str(name).strip()})
        self.result_transition: Optional[StateTransition] = None
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(existing.name if existing else "")
        form.addRow("Name", self.name_edit)
        self.source_combo = QComboBox()
        self.source_combo.addItems(self._states)
        self.source_combo.setEditable(False)
        self.target_combo = QComboBox()
        self.target_combo.addItems(self._states)
        self.target_combo.setEditable(False)
        if existing:
            self.source_combo.setCurrentText(existing.source)
            self.target_combo.setCurrentText(existing.target)
        form.addRow("Source", self.source_combo)
        form.addRow("Target", self.target_combo)
        layout.addLayout(form)
        self.condition_table = QTableWidget(0, 3)
        self.condition_table.setHorizontalHeaderLabels(["Signal", "Operator", "Value"])
        self.condition_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.condition_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.condition_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.condition_table.verticalHeader().setVisible(False)
        self.condition_table.setAlternatingRowColors(True)
        self.condition_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.condition_table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        layout.addWidget(QLabel("Conditions"))
        layout.addWidget(self.condition_table)
        condition_buttons = QHBoxLayout()
        add_condition = QToolButton()
        add_condition.setText("Add condition")
        add_condition.clicked.connect(self._add_condition_row)
        remove_condition = QToolButton()
        remove_condition.setText("Remove condition")
        remove_condition.clicked.connect(self._remove_condition_row)
        condition_buttons.addWidget(add_condition)
        condition_buttons.addWidget(remove_condition)
        condition_buttons.addStretch(1)
        layout.addLayout(condition_buttons)
        self.action_table = QTableWidget(0, 4)
        self.action_table.setHorizontalHeaderLabels(["Type", "Target", "Parameters", "Sequence"])
        header = self.action_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.action_table.verticalHeader().setVisible(False)
        self.action_table.setAlternatingRowColors(True)
        self.action_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.action_table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        layout.addWidget(QLabel("Actions"))
        layout.addWidget(self.action_table)
        action_buttons = QHBoxLayout()
        add_action = QToolButton()
        add_action.setText("Add action")
        add_action.clicked.connect(self._add_action_row)
        remove_action = QToolButton()
        remove_action.setText("Remove action")
        remove_action.clicked.connect(self._remove_action_row)
        action_buttons.addWidget(add_action)
        action_buttons.addWidget(remove_action)
        action_buttons.addStretch(1)
        layout.addLayout(action_buttons)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        if existing:
            for condition in existing.conditions:
                self._add_condition_row(condition.signal, condition.operator, str(condition.value))
            for action in existing.actions:
                if action.type == "send_message":
                    params = ", ".join(f"{key}={value}" for key, value in action.fields.items())
                    self._add_action_row(action.type, action.message, params, action.sequence_mode)
                elif action.type == "set_channel":
                    params = ", ".join(f"{key}={value}" for key, value in action.command.items())
                    self._add_action_row(action.type, action.channel, params, action.sequence_mode)
        else:
            self._add_condition_row()
            self._add_action_row()

    def _add_condition_row(self, signal: str = "", operator: str = ">=", value: str = "0") -> None:
        row = self.condition_table.rowCount()
        self.condition_table.insertRow(row)
        signal_combo = self._create_signal_combo(signal)
        self.condition_table.setCellWidget(row, 0, signal_combo)
        combo = QComboBox()
        combo.addItems([">", ">=", "<", "<=", "==", "!="])
        if operator not in {">", ">=", "<", "<=", "==", "!="}:
            operator = ">="
        combo.setCurrentText(operator)
        self.condition_table.setCellWidget(row, 1, combo)
        self.condition_table.setItem(row, 2, QTableWidgetItem(value))

    def _remove_condition_row(self) -> None:
        row = self.condition_table.currentRow()
        if row >= 0:
            self.condition_table.removeRow(row)

    def _add_action_row(
        self,
        action_type: str = "send_message",
        target: str = "",
        params: str = "",
        sequence_mode: str = "none",
    ) -> None:
        row = self.action_table.rowCount()
        self.action_table.insertRow(row)
        combo = QComboBox()
        combo.addItems(["send_message", "set_channel"])
        if action_type not in {"send_message", "set_channel"}:
            action_type = "send_message"
        combo.setCurrentText(action_type)
        combo.currentTextChanged.connect(lambda value, widget=combo: self._on_action_type_combo_changed(widget, value))
        self.action_table.setCellWidget(row, 0, combo)
        target_editor = self._create_target_editor(action_type, target)
        self.action_table.setCellWidget(row, 1, target_editor)
        current_target = ""
        if isinstance(target_editor, QComboBox):
            current_target = target_editor.currentText().strip()
        self._set_params_editor(row, action_type, current_target, params)
        self.action_table.setCellWidget(row, 3, self._create_sequence_editor(action_type, sequence_mode))

    def _remove_action_row(self) -> None:
        row = self.action_table.currentRow()
        if row >= 0:
            self.action_table.removeRow(row)

    def _set_params_editor(self, row: int, action_type: str, target: str, params: str) -> None:
        if action_type == "send_message":
            existing_widget = self.action_table.cellWidget(row, 2)
            if isinstance(existing_widget, MessageParameterWidget):
                params_text = existing_widget.text()
                self.action_table.removeCellWidget(row, 2)
                existing_widget.deleteLater()
            else:
                item = self.action_table.takeItem(row, 2)
                params_text = params if params else (item.text() if item else "")
                if item is not None:
                    del item
            widget = MessageParameterWidget(self._message_signals.get(target, []), params_text)
            widget.setToolTip(self._params_tooltip(action_type))
            self.action_table.setCellWidget(row, 2, widget)
            self.action_table.resizeRowToContents(row)
        else:
            existing_widget = self.action_table.cellWidget(row, 2)
            if isinstance(existing_widget, MessageParameterWidget):
                params_text = existing_widget.text()
                self.action_table.removeCellWidget(row, 2)
                existing_widget.deleteLater()
            else:
                params_text = params
            item = QTableWidgetItem(params_text)
            item.setToolTip(self._params_tooltip(action_type))
            self.action_table.setItem(row, 2, item)
            self.action_table.resizeRowToContents(row)

    def _extract_params_text(self, row: int) -> str:
        widget = self.action_table.cellWidget(row, 2)
        if isinstance(widget, MessageParameterWidget):
            return widget.text()
        item = self.action_table.item(row, 2)
        if item is not None:
            return item.text().strip()
        return ""

    def _extract_sequence_mode(self, row: int) -> str:
        sequence_widget = self.action_table.cellWidget(row, 3)
        if isinstance(sequence_widget, QComboBox):
            data = sequence_widget.currentData()
            if isinstance(data, str):
                return data
        return "none"

    def _on_action_target_changed(self, combo_widget: QComboBox, value: str) -> None:
        row = self._row_for_widget(combo_widget)
        if row < 0:
            return
        type_widget = self.action_table.cellWidget(row, 0)
        if not isinstance(type_widget, QComboBox) or type_widget.currentText() != "send_message":
            return
        params_widget = self.action_table.cellWidget(row, 2)
        if isinstance(params_widget, MessageParameterWidget):
            params_widget.set_signals(self._message_signals.get(value.strip(), []))

    def accept(self) -> None:  # type: ignore[override]
        if not self._states:
            QMessageBox.warning(self, "Invalid input", "Please create at least one state before defining transitions.")
            return
        name = self.name_edit.text().strip()
        source = self.source_combo.currentText().strip()
        target = self.target_combo.currentText().strip()
        if not source or source not in self._states:
            QMessageBox.warning(self, "Invalid input", "Please select a valid source state.")
            return
        if not target or target not in self._states:
            QMessageBox.warning(self, "Invalid input", "Please select a valid target state.")
            return
        conditions: List[StateCondition] = []
        for row in range(self.condition_table.rowCount()):
            signal_widget = self.condition_table.cellWidget(row, 0)
            value_item = self.condition_table.item(row, 2)
            operator_widget = self.condition_table.cellWidget(row, 1)
            if value_item is None or not isinstance(operator_widget, QComboBox):
                continue
            signal = ""
            if isinstance(signal_widget, QComboBox):
                signal = signal_widget.currentText().strip()
            elif signal_widget is not None:
                text_getter = getattr(signal_widget, "text", None)
                if callable(text_getter):
                    signal = str(text_getter()).strip()
            if not signal:
                continue
            operator = operator_widget.currentText()
            try:
                value = float(value_item.text())
            except (TypeError, ValueError):
                QMessageBox.warning(self, "Invalid input", f"Condition value for '{signal}' is not numeric.")
                return
            conditions.append(StateCondition(signal=signal, operator=operator, value=value))
        actions: List[StateAction] = []
        for row in range(self.action_table.rowCount()):
            type_widget = self.action_table.cellWidget(row, 0)
            target_widget = self.action_table.cellWidget(row, 1)
            sequence_widget = self.action_table.cellWidget(row, 3)
            if not isinstance(type_widget, QComboBox):
                continue
            action_type = type_widget.currentText()
            target_value = ""
            if isinstance(target_widget, QComboBox):
                target_value = target_widget.currentText().strip()
            elif target_widget is not None:
                text_getter = getattr(target_widget, "text", None)
                if callable(text_getter):
                    target_value = str(text_getter()).strip()
            else:
                fallback_item = self.action_table.item(row, 1)
                if fallback_item is not None:
                    target_value = fallback_item.text().strip()
            params_text = self._extract_params_text(row)
            try:
                pairs = self._parse_pairs(params_text)
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid input", str(exc))
                return
            sequence_mode = "none"
            if isinstance(sequence_widget, QComboBox):
                data = sequence_widget.currentData()
                if isinstance(data, str) and data in {"start", "stop", "reset", "none"}:
                    sequence_mode = data
            if action_type == "send_message":
                if not target_value:
                    QMessageBox.warning(self, "Invalid input", "Message name is required for send_message actions.")
                    return
                actions.append(StateAction(type="send_message", message=target_value, fields=pairs))
            elif action_type == "set_channel":
                if not target_value:
                    QMessageBox.warning(self, "Invalid input", "Channel name is required for set_channel actions.")
                    return
                if not pairs and sequence_mode == "none":
                    QMessageBox.warning(
                        self,
                        "Invalid input",
                        "Provide channel command parameters or choose a sequence action.",
                    )
                    return
                actions.append(
                    StateAction(
                        type="set_channel",
                        channel=target_value,
                        command=pairs,
                        sequence_mode=sequence_mode,
                    )
                )
        self.result_transition = StateTransition(name=name, source=source, target=target, conditions=conditions, actions=actions)
        super().accept()

    def _parse_pairs(self, text: str) -> Dict[str, float]:
        pairs: Dict[str, float] = {}
        for chunk in text.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" not in chunk:
                raise ValueError(f"Invalid parameter '{chunk}'. Use key=value format.")
            key, value_text = chunk.split("=", 1)
            key = key.strip()
            value_text = value_text.strip()
            if not key or not value_text:
                raise ValueError("Parameters must be provided as key=value pairs.")
            try:
                value = float(value_text)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid numeric value for '{key}'.")
            pairs[key] = value
        return pairs

    def _create_signal_combo(self, selected: str) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.NoInsert)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        if self._signal_names:
            combo.addItems(self._signal_names)
        if selected and selected not in self._signal_names:
            combo.addItem(selected)
        combo.setCurrentText(selected)
        completer = QCompleter(self._signal_names)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        combo.setCompleter(completer)
        return combo

    def _create_target_editor(self, action_type: str, current: str) -> QComboBox:
        options: List[str] = []
        if action_type == "send_message":
            options = self._message_names
        elif action_type == "set_channel":
            options = self._channel_names
        combo = QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.NoInsert)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        if options:
            combo.addItems(options)
        if current and current not in options:
            combo.addItem(current)
        combo.setCurrentText(current)
        completer = QCompleter(options)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        combo.setCompleter(completer)
        combo.currentTextChanged.connect(lambda value, widget=combo: self._on_action_target_changed(widget, value))
        return combo

    def _create_sequence_editor(self, action_type: str, current: str) -> QWidget:
        if action_type == "set_channel":
            combo = QComboBox()
            options = [
                ("none", "No sequence"),
                ("start", "Start sequence"),
                ("stop", "Stop sequence"),
                ("reset", "Reset sequence"),
            ]
            for value, label in options:
                combo.addItem(label, value)
            if current not in {"start", "stop", "reset", "none"}:
                current = "none"
            index = combo.findData(current)
            combo.setCurrentIndex(max(0, index))
            return combo
        placeholder = QLabel("—")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setEnabled(False)
        return placeholder

    def _on_action_type_combo_changed(self, combo_widget: QComboBox, action_type: str) -> None:
        row = self._row_for_widget(combo_widget)
        if row < 0:
            return
        target_widget = self.action_table.cellWidget(row, 1)
        current_target = ""
        if isinstance(target_widget, QComboBox):
            current_target = target_widget.currentText().strip()
        params_text = self._extract_params_text(row)
        sequence_mode = self._extract_sequence_mode(row)
        editor = self._create_target_editor(action_type, current_target)
        self.action_table.setCellWidget(row, 1, editor)
        self._set_params_editor(row, action_type, current_target, params_text)
        self.action_table.setCellWidget(row, 3, self._create_sequence_editor(action_type, sequence_mode))

    def _row_for_widget(self, widget: QWidget) -> int:
        for row in range(self.action_table.rowCount()):
            if (
                self.action_table.cellWidget(row, 0) is widget
                or self.action_table.cellWidget(row, 1) is widget
                or self.action_table.cellWidget(row, 3) is widget
            ):
                return row
        return -1

    def _params_tooltip(self, action_type: str) -> str:
        if action_type == "set_channel":
            return (
                "Comma separated key=value pairs, e.g. enabled=1.0, pwm=50"
                " (leave empty when only using sequences)"
            )
        return "Comma separated key=value pairs, e.g. signal=1.0"

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

class StateMachineGraphView(QGraphicsView):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        scene = QGraphicsScene(self)
        self.setScene(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setStyleSheet("background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;")
        self.setMinimumHeight(320)
        self._zoom_percent = 120
        self._last_rect = QRectF()

    def set_zoom_percent(self, percent: int) -> None:
        self._zoom_percent = max(60, min(260, percent))
        self._refresh_view()

    def _refresh_view(self) -> None:
        if self._last_rect.isNull():
            return
        self.resetTransform()
        self.fitInView(self._last_rect, Qt.KeepAspectRatio)
        scale_factor = self._zoom_percent / 100.0
        if not math.isclose(scale_factor, 1.0, rel_tol=1e-3):
            self.scale(scale_factor, scale_factor)

    def update_graph(self, config: Optional[StateMachineConfig], active_state: Optional[str]) -> None:
        scene = self.scene()
        if scene is None:
            return
        scene.clear()
        if not config or not config.states:
            self._last_rect = QRectF()
            return
        states = config.state_names
        total = len(states)
        radius = max(120.0, 60.0 + 35.0 * total)
        center = QPointF(0.0, 0.0)
        positions: Dict[str, QPointF] = {}
        for index, name in enumerate(states):
            angle = (2.0 * math.pi * index) / max(total, 1)
            x = center.x() + radius * math.cos(angle)
            y = center.y() + radius * math.sin(angle)
            positions[name] = QPointF(x, y)
        node_radius = 28.0
        base_pen = QPen(QColor("#94a3b8"))
        base_pen.setWidthF(2.0)
        highlight_pen = QPen(QColor("#2563eb"))
        highlight_pen.setWidthF(3.0)
        text_color = QColor("#0f172a")
        edge_pen = QPen(QColor("#cbd5f5"))
        edge_pen.setWidthF(2.0)
        active_edge_pen = QPen(QColor("#6366f1"))
        active_edge_pen.setWidthF(2.5)
        arrow_highlight = QBrush(QColor("#6366f1"))
        label_offsets: Dict[str, int] = {}
        for transition in config.transitions:
            start = positions.get(transition.source)
            end = positions.get(transition.target)
            if start is None or end is None:
                continue
            path = QPainterPath(start)
            path.lineTo(end)
            is_active = active_state == transition.source
            path_item = QGraphicsPathItem(path)
            path_item.setPen(active_edge_pen if is_active else edge_pen)
            scene.addItem(path_item)
            dx = end.x() - start.x()
            dy = end.y() - start.y()
            length = math.hypot(dx, dy)
            if length <= 0.0:
                continue
            ux = dx / length
            uy = dy / length
            tip = QPointF(end.x() - ux * node_radius, end.y() - uy * node_radius)
            arrow_length = 12.0
            arrow_width = 6.0
            left = QPointF(
                tip.x() - ux * arrow_length + (-uy) * arrow_width,
                tip.y() - uy * arrow_length + ux * arrow_width,
            )
            right = QPointF(
                tip.x() - ux * arrow_length - (-uy) * arrow_width,
                tip.y() - uy * arrow_length - ux * arrow_width,
            )
            polygon = QPolygonF([tip, left, right])
            scene.addPolygon(polygon, path_item.pen(), arrow_highlight if is_active else QBrush(path_item.pen().color()))
            label_lines: List[str] = []
            title = transition.name.strip() if isinstance(transition.name, str) else ""
            fallback_title = f"{transition.source} → {transition.target}"
            label_lines.append(title or fallback_title)
            for condition in transition.conditions:
                label_lines.append(f"if {condition.signal} {condition.operator} {condition.value}")
            for action in transition.actions:
                if action.type == "send_message":
                    header = "send message"
                    target_label = action.message.strip() or "message"
                    label_lines.append(f"{header}: {target_label}")
                    for key, value in action.fields.items():
                        label_lines.append(f"  {key}={value}")
                elif action.type == "set_channel":
                    header = "set channel"
                    target_label = action.channel.strip() or "channel"
                    label_lines.append(f"{header}: {target_label}")
                    for key, value in action.command.items():
                        label_lines.append(f"  {key}={value}")
                    if action.sequence_mode and action.sequence_mode != "none":
                        label_lines.append(f"  sequence={action.sequence_mode}")
            label_text = "\n".join(label_lines)
            label_item = scene.addText(label_text)
            font = label_item.font()
            font.setPointSize(8)
            label_item.setFont(font)
            label_item.setDefaultTextColor(QColor("#1e293b" if is_active else "#334155"))
            raw_rect = label_item.boundingRect()
            label_index = label_offsets.get(transition.source, 0)
            perp_index = (label_index // 2) + 1
            perp_sign = -1.0 if label_index % 2 == 0 else 1.0
            label_offsets[transition.source] = label_index + 1
            label_distance = node_radius + 16.0
            perp_spacing = 14.0 * perp_index
            label_anchor = QPointF(
                start.x() + ux * label_distance + (-uy) * perp_spacing * perp_sign,
                start.y() + uy * label_distance + ux * perp_spacing * perp_sign,
            )
            label_item.setPos(label_anchor.x() - raw_rect.width() / 2.0, label_anchor.y() - raw_rect.height() / 2.0)
            backdrop = scene.addRect(
                label_item.boundingRect().translated(label_item.pos()).adjusted(-4.0, -2.0, 4.0, 2.0),
                QPen(Qt.NoPen),
                QBrush(QColor(255, 255, 255, 230)),
            )
            backdrop.setZValue(label_item.zValue() - 1)
        for name in states:
            pos = positions[name]
            scene.addEllipse(
                pos.x() - node_radius,
                pos.y() - node_radius,
                node_radius * 2.0,
                node_radius * 2.0,
                highlight_pen if name == active_state else base_pen,
                QBrush(QColor("#ffffff") if name != active_state else QColor("#fde68a")),
            )
            label = scene.addText(name)
            font = label.font()
            font.setPointSize(10)
            font.setBold(name == active_state)
            label.setFont(font)
            label.setDefaultTextColor(text_color if name != active_state else QColor("#1e293b"))
            rect = label.boundingRect()
            label.setPos(pos.x() - rect.width() / 2.0, pos.y() - rect.height() / 2.0)
        bounds = scene.itemsBoundingRect().adjusted(-40.0, -40.0, 40.0, 40.0)
        scene.setSceneRect(bounds)
        if not bounds.isNull():
            self._last_rect = bounds
            self._refresh_view()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if not self._last_rect.isNull():
            self._refresh_view()


class StateMachineRunner(QObject):
    state_changed = pyqtSignal(str)
    stopped = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._backend: Optional["BackendBase"] = None
        self._config: Optional[StateMachineConfig] = None
        self._transitions_by_source: Dict[str, List[StateTransition]] = {}
        self._running = False
        self._current_state: Optional[str] = None
        self._sequence_controller: Optional[Callable[[str, str], None]] = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def current_state(self) -> Optional[str]:
        return self._current_state

    def set_backend(self, backend: Optional["BackendBase"]) -> None:
        self._backend = backend

    def set_sequence_controller(self, controller: Optional[Callable[[str, str], None]]) -> None:
        self._sequence_controller = controller

    def set_config(self, config: Optional[StateMachineConfig]) -> None:
        self._config = config
        self._transitions_by_source = self._build_transition_map(config) if config else {}
        if self._running and (not config or self._current_state not in (config.state_names if config else [])):
            self.stop()

    def start(self) -> bool:
        if not self._backend or not self._config:
            return False
        self._config.ensure_initial_state()
        initial = self._config.initial_state
        if not initial:
            return False
        if initial not in self._config.state_names:
            return False
        self._current_state = initial
        self._running = True
        self.state_changed.emit(initial)
        return True

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._current_state = None
        self.stopped.emit()

    def step(self) -> None:
        if not self._running or not self._backend or not self._config or not self._current_state:
            return
        transitions = self._transitions_by_source.get(self._current_state, [])
        if not transitions:
            return
        required: Set[str] = set()
        for transition in transitions:
            for condition in transition.conditions:
                if condition.signal:
                    required.add(condition.signal)
        try:
            values = self._backend.read_signal_values(required) if required else {}
        except BackendError as exc:
            self.error_occurred.emit(str(exc))
            self.stop()
            return
        for transition in transitions:
            if not transition.conditions or all(condition.evaluate(values) for condition in transition.conditions):
                try:
                    self._execute_actions(transition.actions)
                except BackendError as exc:
                    self.error_occurred.emit(str(exc))
                    self.stop()
                    return
                self._current_state = transition.target
                if self._config and self._current_state not in self._config.state_names:
                    self._config.states.append(StateDefinition(name=self._current_state))
                    self._transitions_by_source.setdefault(self._current_state, [])
                self.state_changed.emit(self._current_state)
                return

    def _execute_actions(self, actions: Iterable[StateAction]) -> None:
        if not self._backend:
            return
        for action in actions:
            if action.type == "send_message":
                if action.message and action.fields:
                    self._backend.send_message_by_name(action.message, action.fields)
            elif action.type == "set_channel":
                if not action.channel:
                    continue
                if action.command:
                    self._backend.apply_channel_command(action.channel, action.command)
                if (
                    self._sequence_controller
                    and action.sequence_mode
                    and action.sequence_mode != "none"
                ):
                    try:
                        self._sequence_controller(action.channel, action.sequence_mode)
                    except BackendError:
                        raise
                    except Exception as exc:
                        raise BackendError(str(exc)) from exc

    def _build_transition_map(
        self, config: Optional[StateMachineConfig]
    ) -> Dict[str, List[StateTransition]]:
        mapping: Dict[str, List[StateTransition]] = {}
        if not config:
            return mapping
        for transition in config.transitions:
            mapping.setdefault(transition.source, []).append(transition)
        return mapping


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
            self._bus = can.interface.Bus(
                interface="pcan",
                channel=self._settings.channel if hasattr(self._settings, "channel") else "PCAN_USBBUS1",
                state=can.bus.BusState.ACTIVE,
                fd=True,
                # Nominal phase (500k)
                f_clock_mhz=80,
                nom_brp=10, nom_tseg1=12, nom_tseg2=3, nom_sjw=1,
                # Data phase (2M)
                data_brp=4, data_tseg1=7, data_tseg2=2, data_sjw=1,
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


class Logger:
    def __init__(self) -> None:
        self._file = None
        self._path = ""
        self._started = False
        self._signal_names: List[str] = []
        self._last_error = ""
        self._writer: Optional[csv.writer] = None
        self._decimal = "."

    def start(self, path: str, signal_names: Iterable[str]) -> bool:
        self.stop()
        try:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            opts = get_csv_opts()
            self._decimal = opts.get("decimal", ".")
            self._file = open(path, "w", encoding=opts.get("encoding", "utf-8"), newline="")
            self._writer = csv.writer(
                self._file,
                delimiter=opts.get("sep", ","),
                quoting=csv.QUOTE_MINIMAL,
            )
        except OSError as exc:
            self._file = None
            self._writer = None
            self._last_error = str(exc)
            return False
        self._signal_names = list(signal_names)
        header = ["timestamp"] + self._signal_names
        if self._writer is not None:
            self._writer.writerow(header)
            self._file.flush()
        self._path = path
        self._started = True
        self._last_error = ""
        return True

    def stop(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None
        self._started = False

    def log_row(self, timestamp: str, values: Dict[str, float]) -> None:
        if not self._started or self._file is None or self._writer is None:
            return
        ordered: List[str] = [timestamp]
        for name in self._signal_names:
            value = values.get(name, 0.0)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = 0.0
            formatted = f"{numeric:.6f}"
            if self._decimal != ".":
                formatted = formatted.replace(".", self._decimal)
            ordered.append(formatted)
        self._writer.writerow(ordered)

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
        self.table.setAlternatingRowColors(True)
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
        self._measurement_cursors_enabled = False
        self._cursor_lines: Dict[str, pg.InfiniteLine] = {}
        self._cursor_labels: Dict[str, Any] = {}
        self._hud_item = pg.TextItem("", anchor=(1, 0))
        self._hud_item.setZValue(100)
        self.plot_item.addItem(self._hud_item, ignoreBounds=True)
        self._hud_item.hide()
        self._updating_cursors = False
        self._cursor_release_timer = QTimer(self)
        self._cursor_release_timer.setSingleShot(True)
        self._cursor_release_timer.timeout.connect(self._on_cursor_released)
        self._active_curve: Optional[pg.PlotDataItem] = None
        self.plot_item.vb.sigRangeChanged.connect(self._update_hud_position)
        self._cursor_menu_action: Optional[QAction] = None
        self._swap_cursor_action: Optional[QAction] = None
        if hasattr(self.plot_item, "vb") and hasattr(self.plot_item.vb, "menu"):
            menu = getattr(self.plot_item.vb, "menu", None)
        else:
            menu = None
        if menu is not None:
            self._cursor_menu_action = QAction("Measurement cursors", menu)
            self._cursor_menu_action.setCheckable(True)
            self._cursor_menu_action.toggled.connect(self._on_cursor_menu_toggled)
            menu.addAction(self._cursor_menu_action)
            self._swap_cursor_action = QAction("Swap A/B", menu)
            self._swap_cursor_action.triggered.connect(self._swap_cursors)
            self._swap_cursor_action.setEnabled(False)
            menu.addAction(self._swap_cursor_action)
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
        curve.setZValue(len(self._signals) + 2)
        if hasattr(curve, "setCurveClickable"):
            curve.setCurveClickable(True)
        elif hasattr(curve, "setClickable"):
            curve.setClickable(True)
        curve.sigClicked.connect(self._on_curve_clicked)
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
        if self._active_curve is None:
            self._active_curve = curve
        self._update_cursor_info()

    def remove_signal(self, name: str) -> None:
        info = self._signals.pop(name, None)
        if not info:
            return
        curve = info["curve"]
        try:
            curve.sigClicked.disconnect(self._on_curve_clicked)
        except Exception:
            pass
        if info["side"] == "left":
            self.plot_item.removeItem(curve)
        else:
            self.right_view.removeItem(curve)
        if self._legend:
            try:
                self._legend.removeItem(curve)
            except Exception:
                pass
        if self._active_curve is curve:
            self._active_curve = self._select_fallback_curve()
        self._update_cursor_info()

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
        if self._measurement_cursors_enabled:
            self._update_cursor_info()

    def assigned_signals(self) -> Dict[str, Tuple[str, str]]:
        return {name: (info["unit"], info["side"]) for name, info in self._signals.items()}

    def closeEvent(self, event) -> None:
        self.closed.emit(self.identifier)
        super().closeEvent(event)

    def set_measurement_cursors_enabled(self, enabled: bool) -> None:
        if self._measurement_cursors_enabled == enabled:
            if enabled and not self._cursor_lines:
                pass
            else:
                return
        self._measurement_cursors_enabled = enabled
        if self._cursor_menu_action is not None:
            self._cursor_menu_action.blockSignals(True)
            self._cursor_menu_action.setChecked(enabled)
            self._cursor_menu_action.blockSignals(False)
        if self._swap_cursor_action is not None:
            self._swap_cursor_action.setEnabled(enabled)
        if enabled:
            self._ensure_cursor_items()
            self._hud_item.show()
            for line in self._cursor_lines.values():
                line.show()
            for label in self._cursor_labels.values():
                label.setVisible(True)
            self._update_cursor_info()
            self._update_hud_position()
        else:
            for line in self._cursor_lines.values():
                line.hide()
            for label in self._cursor_labels.values():
                label.setVisible(False)
            self._hud_item.hide()

    def _on_cursor_menu_toggled(self, enabled: bool) -> None:
        parent = self.parent()
        if parent is not None and hasattr(parent, "measurement_cursors_action"):
            action = getattr(parent, "measurement_cursors_action")
            if isinstance(action, QAction):
                action.setChecked(enabled)
                return
        self.set_measurement_cursors_enabled(enabled)

    def _ensure_cursor_items(self) -> None:
        if self._cursor_lines:
            if self._measurement_cursors_enabled:
                for line in self._cursor_lines.values():
                    line.show()
                for label in self._cursor_labels.values():
                    label.setVisible(True)
            return
        view_range = self.plot_item.vb.viewRange()
        if view_range:
            x_range = view_range[0]
            xmin, xmax = x_range
            width = xmax - xmin
            if not math.isfinite(width) or width <= 0:
                width = 1.0
            start_a = xmin + width * 0.3
            start_b = xmin + width * 0.6
        else:
            start_a = 0.0
            start_b = 1.0
        self._create_cursor("A", start_a, (255, 215, 0))
        self._create_cursor("B", start_b, (0, 128, 255))
        self._update_cursor_info()

    def _create_cursor(self, label: str, position: float, color: Tuple[int, int, int]) -> None:
        base_pen = pg.mkPen(color, width=1.5)
        hover_pen = pg.mkPen((255, 255, 255), width=2.0)
        line = pg.InfiniteLine(pos=position, angle=90, movable=True, pen=base_pen)
        try:
            line.setHoverPen(hover_pen)
        except AttributeError:
            pass
        line.setZValue(50)
        line.sigPositionChanged.connect(self._on_cursor_moved)
        drag_finished_signal = None
        for attr in ("sigDragFinished", "sigPositionChangeFinished", "sigDragged"):
            drag_finished_signal = getattr(line, attr, None)
            if drag_finished_signal is not None:
                break
        if drag_finished_signal is not None:
            drag_finished_signal.connect(self._on_cursor_released)
        else:
            line.sigPositionChanged.connect(self._on_cursor_release_fallback)
        self.plot_item.addItem(line, ignoreBounds=True)
        self._cursor_lines[label] = line
        try:
            inf_label = pg.InfLineLabel(line, label, position=0.02, movable=False)
        except Exception:
            text = pg.TextItem(label, anchor=(0.5, 1.0), color=pg.mkColor(color))
            text.setParentItem(line)
            inf_label = text
        inf_label.setZValue(60)
        self._cursor_labels[label] = inf_label
        if self._measurement_cursors_enabled:
            line.show()
            inf_label.setVisible(True)
        else:
            line.hide()
            inf_label.setVisible(False)

    def _on_cursor_moved(self) -> None:
        if self._updating_cursors:
            return
        self._update_cursor_info()

    def _on_cursor_release_fallback(self) -> None:
        self._cursor_release_timer.start(75)

    def _on_cursor_released(self) -> None:
        if self._updating_cursors:
            return
        for line in self._cursor_lines.values():
            self._snap_line_to_curve(line)
        self._update_cursor_info()

    def _snap_line_to_curve(self, line: pg.InfiniteLine) -> None:
        curve = self._get_active_curve()
        if curve is None:
            return
        x_data = curve.xData
        if x_data is None or len(x_data) == 0:
            return
        pos = float(line.value())
        idx = int(np.searchsorted(x_data, pos))
        idx = max(0, min(idx, len(x_data) - 1))
        if idx > 0 and abs(pos - x_data[idx - 1]) <= abs(pos - x_data[idx]):
            idx -= 1
        snapped = float(x_data[idx])
        if snapped != pos:
            self._set_line_value(line, snapped)

    def _set_line_value(self, line: pg.InfiniteLine, value: float) -> None:
        self._updating_cursors = True
        try:
            line.setValue(value)
        finally:
            self._updating_cursors = False

    def _update_cursor_info(self) -> None:
        if not self._measurement_cursors_enabled or not self._cursor_lines:
            return
        curve = self._get_active_curve()
        if curve is None:
            self._hud_item.setText("Measurement cursors\nNo active curve")
            return
        x_data = curve.xData
        y_data = curve.yData
        if x_data is None or y_data is None or len(x_data) == 0:
            self._hud_item.setText("Measurement cursors\nNo data")
            return
        pos_a = float(self._cursor_lines["A"].value())
        pos_b = float(self._cursor_lines["B"].value())
        y_a = self._interpolate_value(x_data, y_data, pos_a)
        y_b = self._interpolate_value(x_data, y_data, pos_b)
        delta_t = pos_b - pos_a
        delta_y = y_b - y_a
        integral = self._integrate_between(x_data, y_data, pos_a, pos_b, y_a, y_b)
        unit = self._curve_unit(curve)
        unit_suffix = f" {unit}" if unit else ""
        area_suffix = f" {unit}·s" if unit else ""
        def fmt(value: float) -> str:
            return f"{value:.4g}"
        lines = [
            f"A: t={fmt(pos_a)} s, y={fmt(y_a)}{unit_suffix}",
            f"B: t={fmt(pos_b)} s, y={fmt(y_b)}{unit_suffix}",
            f"Δt: {fmt(delta_t)} s",
            f"Δy: {fmt(delta_y)}{unit_suffix}",
        ]
        if integral is not None:
            lines.append(f"∫y·dt: {fmt(integral)}{area_suffix}")
        else:
            lines.append("∫y·dt: —")
        self._hud_item.setText("\n".join(lines))
        self._update_hud_position()

    def _integrate_between(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        pos_a: float,
        pos_b: float,
        y_a: float,
        y_b: float,
    ) -> Optional[float]:
        left = min(pos_a, pos_b)
        right = max(pos_a, pos_b)
        if not math.isfinite(left) or not math.isfinite(right):
            return None
        if abs(right - left) < 1e-12:
            return 0.0
        mask = np.where((x_data >= left) & (x_data <= right))[0]
        xs = [left]
        ys = [y_a if pos_a <= pos_b else y_b]
        if mask.size:
            for idx in mask:
                x_val = float(x_data[idx])
                y_val = float(y_data[idx])
                if x_val <= left + 1e-12 or x_val >= right - 1e-12:
                    continue
                xs.append(x_val)
                ys.append(y_val)
        xs.append(right)
        ys.append(y_b if pos_a <= pos_b else y_a)
        if len(xs) < 2:
            return None
        xs_array = np.array(xs, dtype=float)
        ys_array = np.array(ys, dtype=float)
        try:
            return float(np.trapz(ys_array, xs_array))
        except Exception:
            return None

    def _interpolate_value(self, x_data: np.ndarray, y_data: np.ndarray, position: float) -> float:
        if len(x_data) == 0:
            return float("nan")
        idx = int(np.searchsorted(x_data, position))
        idx = max(0, min(idx, len(x_data) - 1))
        if idx == 0:
            return float(y_data[0])
        if idx >= len(x_data):
            return float(y_data[-1])
        x0 = float(x_data[idx - 1])
        x1 = float(x_data[idx])
        y0 = float(y_data[idx - 1])
        y1 = float(y_data[idx])
        if x1 == x0:
            return y0
        ratio = (position - x0) / (x1 - x0)
        ratio = max(0.0, min(1.0, ratio))
        return y0 + ratio * (y1 - y0)

    def _curve_unit(self, curve: pg.PlotDataItem) -> str:
        for info in self._signals.values():
            if info["curve"] is curve:
                return info.get("unit", "")
        return ""

    def _update_hud_position(self) -> None:
        if not self._hud_item.isVisible():
            return
        view_range = self.plot_item.vb.viewRange()
        if not view_range:
            return
        x_range, y_range = view_range
        if not x_range or not y_range:
            return
        x_pos = x_range[1]
        y_pos = y_range[1]
        if not math.isfinite(x_pos) or not math.isfinite(y_pos):
            return
        self._hud_item.setPos(x_pos, y_pos)

    def _swap_cursors(self) -> None:
        if not self._cursor_lines:
            return
        pos_a = float(self._cursor_lines["A"].value())
        pos_b = float(self._cursor_lines["B"].value())
        self._set_line_value(self._cursor_lines["A"], pos_b)
        self._set_line_value(self._cursor_lines["B"], pos_a)
        self._update_cursor_info()

    def _on_curve_clicked(self, curve: pg.PlotDataItem, _points) -> None:
        self._set_active_curve(curve)

    def _set_active_curve(self, curve: Optional[pg.PlotDataItem]) -> None:
        if curve is not None and curve not in [info["curve"] for info in self._signals.values()]:
            return
        self._active_curve = curve
        self._update_cursor_info()

    def _get_active_curve(self) -> Optional[pg.PlotDataItem]:
        if self._active_curve in [info["curve"] for info in self._signals.values()]:
            return self._active_curve
        fallback = self._select_fallback_curve()
        self._active_curve = fallback
        return fallback

    def _select_fallback_curve(self) -> Optional[pg.PlotDataItem]:
        curves = [info["curve"] for info in self._signals.values() if info["curve"].xData is not None]
        if not curves:
            curves = [info["curve"] for info in self._signals.values()]
        if not curves:
            return None
        curves.sort(key=lambda c: c.zValue(), reverse=True)
        return curves[0]

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

class ChannelCardWidget(QWidget):
    command_requested = pyqtSignal(str, dict)
    sequencer_requested = pyqtSignal(str, str)
    sequencer_config_changed = pyqtSignal(str, object)
    plot_visibility_changed = pyqtSignal(str, bool)
    simulation_changed = pyqtSignal(str, dict)
    section_collapse_changed = pyqtSignal(str, str, bool)
    duplicate_requested = pyqtSignal(str)
    delete_requested = pyqtSignal(str)

    def __init__(self, profile: ChannelProfile) -> None:
        super().__init__()
        self.setObjectName("channelCard")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            "#channelCard { border: 1px solid palette(mid); border-radius: 4px; margin: 2px; padding: 4px; }"
        )
        self.profile = profile
        self.state_label = QLabel("Idle")
        self.status_led = QLabel("●")
        self.status_led.setStyleSheet("color: red; font-size: 12px;")
        self.name_label = QLabel(profile.name)
        self.name_label.setStyleSheet("font-weight: 600;")
        self.enabled_checkbox = QToolButton()
        self.enabled_checkbox.setCheckable(True)
        self.enabled_checkbox.setAutoRaise(True)
        self.enabled_checkbox.setIcon(self.style().standardIcon(QStyle.SP_DialogYesButton))
        self.enabled_checkbox.setToolTip("Enable channel output")
        self.run_button = QToolButton()
        self.run_button.setAutoRaise(True)
        self.run_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.run_button.setToolTip("Start sequencer")
        self.run_button.hide()
        self.stop_button = QToolButton()
        self.stop_button.setAutoRaise(True)
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.setToolTip("Stop sequencer")
        self.stop_button.hide()
        self.toggle_sequencer_button = QToolButton()
        self.toggle_sequencer_button.setCheckable(True)
        self.toggle_sequencer_button.setAutoRaise(True)
        self.toggle_sequencer_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        self.toggle_sequencer_button.setToolTip("Toggle sequencer editor")
        self.duplicate_button = QToolButton()
        self.duplicate_button.setAutoRaise(True)
        self.duplicate_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogNewFolder))
        self.duplicate_button.setToolTip("Duplicate channel")
        self.plot_checkbox = QToolButton()
        self.plot_checkbox.setCheckable(True)
        self.plot_checkbox.setAutoRaise(True)
        self.plot_checkbox.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        self.plot_checkbox.setToolTip("Toggle channel plot")
        self.delete_button = QToolButton()
        self.delete_button.setAutoRaise(True)
        self.delete_button.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
        self.delete_button.setToolTip("Remove channel")
        self.pwm_slider = QSlider(Qt.Horizontal)
        self.pwm_slider.setRange(0, 100)
        self.pwm_value = QLabel("0 %")
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["Reverse", "Neutral", "Forward"])
        self.setpoint_spin = QDoubleSpinBox()
        self.setpoint_spin.setDecimals(2)
        self.setpoint_spin.setRange(-1_000.0, 1_000.0)
        self.quick_off = QToolButton()
        self.quick_off.setText("0%")
        self.quick_off.setToolTip("Set output to 0%")
        self.quick_low = QToolButton()
        self.quick_low.setText("20%")
        self.quick_low.setToolTip("Set output to 20%")
        self.quick_mid = QToolButton()
        self.quick_mid.setText("50%")
        self.quick_mid.setToolTip("Set output to 50%")
        self.quick_max = QToolButton()
        self.quick_max.setText("100%")
        self.quick_max.setToolTip("Set output to 100%")
        self.apply_button = QToolButton()
        self.apply_button.setText("Apply")
        self.apply_button.setAutoRaise(True)
        self.off_button = QToolButton()
        self.off_button.setText("All OFF")
        self.off_button.setAutoRaise(True)
        for button in (self.quick_off, self.quick_low, self.quick_mid, self.quick_max, self.apply_button, self.off_button):
            button.setAutoRaise(True)
        # nach UI-Erzeugung (Buttons/Slider) ergänzen:
        self._live_timer = QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(60)          # kleines Debounce fürs Schieberegler-Draggen
        self._live_timer.timeout.connect(self._emit_command)

        # Apply-Button nicht mehr nutzen
        self.apply_button.hide()
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
        control_height = max(QSpinBox().sizeHint().height(), QDoubleSpinBox().sizeHint().height())
        self.sequence_table.verticalHeader().setMinimumSectionSize(control_height)
        self.sequence_table.verticalHeader().setDefaultSectionSize(control_height)
        self.sequence_table.setStyleSheet("QTableView::item { padding: 0px; }")
        self.sequence_table.setAlternatingRowColors(True)  # ← HIER
        self.sequence_add = QToolButton()
        self.sequence_add.setText("Add")
        self.sequence_delete = QToolButton()
        self.sequence_delete.setText("Del")
        self.sequence_duplicate = QToolButton()
        self.sequence_duplicate.setText("Dup")
        self.sequence_up = QToolButton()
        self.sequence_up.setText("Up")
        self.sequence_down = QToolButton()
        self.sequence_down.setText("Down")
        self.sequence_start = QToolButton()
        self.sequence_start.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.sequence_start.setToolTip("Start sequence")
        self.sequence_stop = QToolButton()
        self.sequence_stop.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.sequence_stop.setToolTip("Stop sequence")
        self.sequence_reset = QToolButton()
        self.sequence_reset.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.sequence_reset.setToolTip("Reset sequence")
        for button in (
            self.sequence_add,
            self.sequence_delete,
            self.sequence_duplicate,
            self.sequence_up,
            self.sequence_down,
            self.sequence_start,
            self.sequence_stop,
            self.sequence_reset,
        ):
            button.setAutoRaise(True)
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
        self.sim_button = QToolButton()
        self.sim_button.setText("⚙ Sim")
        self.sim_button.clicked.connect(self._open_sim_dialog)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMinimumHeight(40)
        self.plot_widget.setMaximumHeight(80)
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
        self._collapsible_sync = False
        self._build_layout()
        self._connect_signals()
        self._update_visibility()
        self._update_sequence_buttons()
        self.run_button.setEnabled(self._has_enabled_sequences())
        self.stop_button.setEnabled(False)

    def _build_layout(self) -> None:
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        for button in (
            self.enabled_checkbox,
            self.run_button,
            self.stop_button,
            self.toggle_sequencer_button,
            self.duplicate_button,
            self.plot_checkbox,
            self.delete_button,
        ):
            button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        header_layout.addWidget(self.status_led)
        header_layout.addWidget(self.name_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.enabled_checkbox)
        header_layout.addWidget(self.toggle_sequencer_button)
        header_layout.addWidget(self.duplicate_button)
        header_layout.addWidget(self.plot_checkbox)
        header_layout.addWidget(self.delete_button)
        self.toggle_sequencer_button.setChecked(False)
        layout.addLayout(header_layout)
        layout.addWidget(self.state_label)
        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(2)
        pwm_widget = QWidget()
        pwm_layout = QHBoxLayout(pwm_widget)
        pwm_layout.setContentsMargins(0, 0, 0, 0)
        pwm_layout.setSpacing(4)
        pwm_layout.addWidget(self.pwm_slider, 1)
        pwm_layout.addWidget(self.pwm_value)
        form_layout.addRow("PWM", pwm_widget)
        form_layout.addRow("Direction", self.direction_combo)
        form_layout.addRow("Setpoint", self.setpoint_spin)
        quick_widget = QWidget()
        quick_layout = QHBoxLayout(quick_widget)
        quick_layout.setContentsMargins(0, 0, 0, 0)
        quick_layout.setSpacing(2)
        for button in (self.quick_off, self.quick_low, self.quick_mid, self.quick_max):
            button.setToolButtonStyle(Qt.ToolButtonTextOnly)
            quick_layout.addWidget(button)
        quick_layout.addStretch(1)
        form_layout.addRow("Quick", quick_widget)
        action_widget = QWidget()
        action_layout = QHBoxLayout(action_widget)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(4)
        self.apply_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.off_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        action_layout.addWidget(self.apply_button)
        action_layout.addWidget(self.off_button)
        action_layout.addStretch(1)
        action_layout.addWidget(self.sim_button)
        form_layout.addRow("Actions", action_widget)
        layout.addLayout(form_layout)
        self.status_section = CollapsibleSection("Status signals")
        self.status_section.set_content(self.values_view)
        self.status_section.set_collapsed(True, animate=False)
        self.status_section.toggled.connect(self._on_status_section_toggled)
        layout.addWidget(self.status_section)
        sequencer_container = QWidget()
        seq_layout = QVBoxLayout(sequencer_container)
        seq_layout.setContentsMargins(0, 0, 0, 0)
        seq_layout.setSpacing(4)
        self.sequence_table.horizontalHeader().setStretchLastSection(False)
        seq_layout.addWidget(self.sequence_table)
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(2)
        for button in (
            self.sequence_add,
            self.sequence_delete,
            self.sequence_duplicate,
            self.sequence_up,
            self.sequence_down,
        ):
            button.setToolButtonStyle(Qt.ToolButtonTextOnly)
            toolbar_layout.addWidget(button)
        toolbar_layout.addStretch(1)
        seq_layout.addLayout(toolbar_layout)
        repeat_widget = QWidget()
        repeat_layout = QHBoxLayout(repeat_widget)
        repeat_layout.setContentsMargins(0, 0, 0, 0)
        repeat_layout.setSpacing(6)
        repeat_layout.addWidget(self.repeat_off_radio)
        repeat_layout.addWidget(self.repeat_endless_radio)
        repeat_layout.addWidget(self.repeat_limit_radio)
        repeat_layout.addWidget(self.repeat_time_edit)
        repeat_layout.addStretch(1)
        seq_layout.addWidget(repeat_widget)
        seq_layout.addWidget(self.sequence_status)
        sequence_controls = QHBoxLayout()
        sequence_controls.setContentsMargins(0, 0, 0, 0)
        sequence_controls.setSpacing(4)
        sequence_controls.addWidget(self.sequence_start)
        sequence_controls.addWidget(self.sequence_stop)
        sequence_controls.addWidget(self.sequence_reset)
        sequence_controls.addStretch(1)
        seq_layout.addLayout(sequence_controls)
        self.sequencer_section = CollapsibleSection("Sequencer")
        self.sequencer_section.set_content(sequencer_container)
        self.sequencer_section.set_collapsed(True, animate=False)
        self.sequencer_section.toggled.connect(lambda expanded: self._on_section_toggle("sequencer", expanded))
        layout.addWidget(self.sequencer_section)
        layout.addWidget(self.plot_widget)

    def _on_status_section_toggled(self, expanded: bool) -> None:
        self.section_collapse_changed.emit(self.profile.name, "status", not expanded)

    def _on_section_toggle(self, section: str, expanded: bool) -> None:
        if section == "sequencer":
            if not self._collapsible_sync and self.toggle_sequencer_button.isChecked() != expanded:
                self._collapsible_sync = True
                self.toggle_sequencer_button.setChecked(expanded)
                self._collapsible_sync = False
        self.section_collapse_changed.emit(self.profile.name, section, not expanded)

    def _connect_signals(self) -> None:
        #self.pwm_slider.valueChanged.connect(lambda value: self.pwm_value.setText(f"{value} %"))
        #self.apply_button.clicked.connect(self._emit_command)
        #self.off_button.clicked.connect(self._emit_off)
        #self.quick_off.clicked.connect(self._emit_off)
        #self.quick_low.clicked.connect(lambda: self._set_pwm_slider(20))
        #self.quick_mid.clicked.connect(lambda: self._set_pwm_slider(50))
        #self.quick_max.clicked.connect(lambda: self._set_pwm_slider(100))
        # neu:
        self.pwm_slider.valueChanged.connect(self._on_pwm_changed)
        self.quick_off.clicked.connect(lambda: self._on_quick_pwm(0))
        self.quick_low.clicked.connect(lambda: self._on_quick_pwm(20))
        self.quick_mid.clicked.connect(lambda: self._on_quick_pwm(50))
        self.quick_max.clicked.connect(lambda: self._on_quick_pwm(100))
        self.enabled_checkbox.toggled.connect(self._on_enabled_toggled)
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
        self.run_button.clicked.connect(lambda: self._emit_sequence_action("start"))
        self.stop_button.clicked.connect(lambda: self._emit_sequence_action("stop"))
        self.toggle_sequencer_button.toggled.connect(self._on_sequencer_button_toggled)
        self.duplicate_button.clicked.connect(lambda: self.duplicate_requested.emit(self.profile.name))
        self.delete_button.clicked.connect(lambda: self.delete_requested.emit(self.profile.name))

    def _set_pwm_slider(self, value: int) -> None:
        self.pwm_slider.setValue(value)
        self.pwm_value.setText(f"{value} %")

    def _on_sequencer_button_toggled(self, checked: bool) -> None:
        if self._collapsible_sync:
            return
        self._collapsible_sync = True
        self.sequencer_section.set_collapsed(not checked, animate=True)
        self._collapsible_sync = False
        self.section_collapse_changed.emit(self.profile.name, "sequencer", not checked)

    def set_section_collapsed(self, section: str, collapsed: bool) -> None:
        if section == "status":
            self.status_section.set_collapsed(collapsed, animate=False)
        elif section == "sequencer":
            self._collapsible_sync = True
            self.sequencer_section.set_collapsed(collapsed, animate=False)
            self.toggle_sequencer_button.setChecked(not collapsed)
            self._collapsible_sync = False

    def section_collapsed(self, section: str) -> bool:
        if section == "status":
            return self.status_section.is_collapsed()
        if section == "sequencer":
            return self.sequencer_section.is_collapsed()
        return False

    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)
        enable_action = menu.addAction("Enabled")
        enable_action.setCheckable(True)
        enable_action.setChecked(self.enabled_checkbox.isChecked())
        start_action = menu.addAction("Start sequencer")
        stop_action = menu.addAction("Stop sequencer")
        duplicate_action = menu.addAction("Duplicate…")
        plot_action = menu.addAction("Show plot")
        plot_action.setCheckable(True)
        plot_action.setChecked(self.plot_checkbox.isChecked())
        reset_action = menu.addAction("Reset channel…")
        delete_action = menu.addAction("Delete channel")
        selected = menu.exec_(event.globalPos())
        if selected is None:
            return
        if selected == enable_action:
            self.enabled_checkbox.toggle()
        elif selected == start_action:
            self._emit_sequence_action("start")
        elif selected == stop_action:
            self._emit_sequence_action("stop")
        elif selected == duplicate_action:
            self.duplicate_requested.emit(self.profile.name)
        elif selected == plot_action:
            self.plot_checkbox.setChecked(not self.plot_checkbox.isChecked())
        elif selected == reset_action:
            self._emit_off()
        elif selected == delete_action:
            self.delete_requested.emit(self.profile.name)

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
        self.run_button.setEnabled(not running and self._has_enabled_sequences())
        self.stop_button.setEnabled(running)

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
        self.run_button.setEnabled(not running and self._has_enabled_sequences())
        self.stop_button.setEnabled(running)

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
        self.status_section.set_badge(str(len(lines)))

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

    @pyqtSlot(int)
    def _on_pwm_changed(self, value: int) -> None:
        self.pwm_value.setText(f"{value} %")
        if self.enabled_checkbox.isChecked():
            self._live_timer.start()  # sendet nach kurzer Pause automatisch

    def _on_quick_pwm(self, value: int) -> None:
        # DO hat keinen PWM-Slider → Quick "0%" = aus
        if self.profile.type == "DO":
            if value == 0:
                self._emit_off()
            return
        self._set_pwm_slider(value)
        if self.enabled_checkbox.isChecked():
            # Quick-Buttons sollen sofort wirken (ohne Debounce)
            self._emit_command()

    def _on_enabled_toggled(self, checked: bool) -> None:
        if checked:
            # direkt aktuellen PWM/Setpoint übernehmen
            self._emit_command()
        else:
            # sauber ausschalten
            self._emit_off()



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
        self.setDockOptions(
            self.dockOptions()
            | QMainWindow.AllowTabbedDocks
            | QMainWindow.AllowNestedDocks
            | QMainWindow.AnimatedDocks
        )
        self.setDockNestingEnabled(True)
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
        self._state_machine_configs: Dict[str, StateMachineConfig] = {}
        self._active_state_machine: Optional[str] = None
        self._state_machine_runner = StateMachineRunner(self)
        self._state_machine_runner.set_sequence_controller(self._state_machine_sequence_control)
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
        self._max_plot_windows = 8
        self._active_plot_id: Optional[int] = None
        self._last_plot_dock: Optional[QDockWidget] = None
        self._suspend_plot_assignment = False
        self._hardware_apply_required = False
        self._measurement_cursors_enabled = False
        self._updating_state_machine_ui = False
        self._initializing_ui = True
        self.channels_dock: Optional[QDockWidget] = None
        self.signals_dock: Optional[QDockWidget] = None
        self.state_machine_dock: Optional[QDockWidget] = None
        self.state_machine_combo: Optional[QComboBox] = None
        self.state_machine_graph: Optional[StateMachineGraphView] = None
        self.state_machine_start_button: Optional[QPushButton] = None
        self.state_machine_stop_button: Optional[QPushButton] = None
        self.state_machine_status_label: Optional[QLabel] = None
        self.state_table: Optional[QTableWidget] = None
        self.state_name_edit: Optional[QLineEdit] = None
        self.state_initial_combo: Optional[QComboBox] = None
        self.transition_table: Optional[QTableWidget] = None
        self._saved_dock_state: Optional[QByteArray] = None
        self._channel_grid_cols = 2
        self._channel_columns: List[QVBoxLayout] = []
        self._channel_columns_layout: Optional[QHBoxLayout] = None
        self._channel_collapse_state: Dict[str, Dict[str, bool]] = {}
        self._signals_log_visible = True
        self._signals_log_height = 200
        self._toolbar_visible = True
        self._csv_preset_key = "excel_de"
        self._show_startup_tab = False
        self._startup_tab_index: int = -1
        self._startup_config = StartupConfig()
        self._startup_on_connect = False
        self._startup_on_apply = False
        self._startup_delay_ms = 0
        self._startup_only_on_change = False
        self._startup_last_payloads: Dict[Tuple[str, Optional[str]], Dict[str, float]] = {}
        self._startup_worker_thread: Optional[QThread] = None
        self._startup_worker: Optional[StartupWorker] = None
        self._startup_running = False
        self._startup_pending_mode: Optional[str] = None
        self._inline_init_done = False
        self._startup_status_messages: List[str] = []
        self._startup_is_valid = False
        self._active_setup_path: Optional[str] = None
        self._state_machine_runner.state_changed.connect(self._on_state_machine_state_changed)
        self._state_machine_runner.stopped.connect(self._on_state_machine_runner_stopped)
        self._state_machine_runner.error_occurred.connect(self._on_state_machine_error)
        self._restore_settings()
        
        self._build_ui()
        self._load_initial_setup()
        self._update_yaml_status(None)
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start()
        
        self._load_initial_backend()
        self._initializing_ui = False
        #QTimer.singleShot(0, self._inject_demo_sm)
    # UI construction
    def _build_ui(self) -> None:
        #self._initializing_ui = True
        self._build_menu()
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("color: red; font-size: 16pt;")
        self.status_message_label = QLabel("Disconnected")
        status_bar = QStatusBar()
        status_bar.addWidget(self.status_indicator)
        status_bar.addWidget(self.status_message_label, 1)
        self.yaml_path_label = QLabel("YAML: <unsaved>")
        status_bar.addPermanentWidget(self.yaml_path_label)
        self.csv_preset_status_label = QLabel()
        status_bar.addPermanentWidget(self.csv_preset_status_label)
        self.setStatusBar(status_bar)
        self.dashboard_bar = QToolBar("Dashboard", self)
        self.dashboard_bar.setObjectName("Toolbar_Dashboard") 
        self.dashboard_bar.setMovable(False)
        self.dashboard_bar.setIconSize(QSize(16, 16))
        self.addToolBar(Qt.TopToolBarArea, self.dashboard_bar)
        self.dashboard_bar.visibilityChanged.connect(self._on_toolbar_visibility_changed)
        self.dashboard_bar.setVisible(self._toolbar_visible)
        self._build_dashboard_toolbar()
        self.central_stack = QStackedWidget()
        self.central_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._central_placeholder = QWidget()
        self._central_placeholder.setMinimumSize(0, 0)
        self._central_placeholder.setMaximumSize(0, 0)
        self._central_placeholder.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.central_stack.addWidget(self._central_placeholder)
        self.tab_widget = QTabWidget()
        self.central_stack.addWidget(self.tab_widget)
        self.central_stack.setCurrentWidget(self._central_placeholder)
        self.setCentralWidget(self.central_stack)
        self._build_channels_tab()
        self._build_signals_tab()
        self._build_state_machine_dock()
        #self._apply_default_dock_layout()
        #if not self._restore_dock_layout():
        #    self._apply_default_dock_layout()
        
        self._saved_dock_state = None
        self._apply_default_dock_layout()
        self._build_startup_tab()
        self._build_dummy_tab()
        self._update_dummy_tab_visibility()
        self._update_central_stack_visibility()
        self._update_csv_status_label()
        
        #if not self._state_machine_configs:
        #   QTimer.singleShot(0, self._inject_demo_sm)
        #self._initializing_ui = False

    def _clear_state_machine_tables_and_graph(self) -> None:
        """Leert States/Transitions-Tabellen und den Graphen sauber."""
        if getattr(self, "state_table", None):
            self.state_table.setRowCount(0)
        if getattr(self, "transition_table", None):
            self.transition_table.setRowCount(0)
        if getattr(self, "state_graph", None):
            try:
                self.state_graph.update_graph(None, None)
            except Exception:
                pass

    def _ensure_state_machine_table_structure(self) -> None:
        """Stellt sicher, dass beide Tabellen wenigstens 1 Spalte haben und vernünftig aussehen."""
        from PyQt5.QtWidgets import QHeaderView
        if getattr(self, "state_table", None):
            if self.state_table.columnCount() < 1:
                self.state_table.setColumnCount(1)
            self.state_table.setHorizontalHeaderLabels(["State"])
            self.state_table.verticalHeader().setVisible(False)
            self.state_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

        if getattr(self, "transition_table", None):
            # 3 Spalten sind für Name / Source / Target praktisch
            if self.transition_table.columnCount() < 3:
                self.transition_table.setColumnCount(3)
            self.transition_table.setHorizontalHeaderLabels(["Transition", "Source", "Target"])
            self.transition_table.verticalHeader().setVisible(False)
            self.transition_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            self.transition_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            self.transition_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
    def _inject_demo_sm(self):
        demo = StateMachineConfig.from_dict({
            "name": "Injected",
            "initial_state": "State 1",
            "states": [{"name": "State 1"}, {"name": "State 2"}],
            "transitions": [
                {"name": "test", "source": "State 1", "target": "State 2", "conditions": [], "actions": []},
            ],
        })
        self._state_machine_configs = {"Injected": demo}
        self._active_state_machine = "Injected"

        # Refresh auf "Injected" – ohne Signals
        self.state_machine_combo.blockSignals(True)
        try:
            self._refresh_state_machine_combo()
            idx = self.state_machine_combo.findText(self._active_state_machine)
            if idx >= 0:
                self.state_machine_combo.setCurrentIndex(idx)
        finally:
            self.state_machine_combo.blockSignals(False)

        # Falls _refresh_* das Populate nicht schon gemacht hat:
        self._populate_state_machine_editor()
        

        # Jetzt ist der Aufbau fertig
        self._initializing_ui = False

        # Debug
        #print("SM names:", list(self._state_machine_configs.keys()))
        cfg = self._state_machine_configs.get(self._active_state_machine)
        #if cfg:
          #  print("Active:", self._active_state_machine,
            #    "States:", [s.name for s in cfg.states],
            #    "Transitions:", [t.name for t in cfg.transitions])



    def _build_menu(self) -> None:
        menu_bar = self.menuBar()
        setup_menu = menu_bar.addMenu("Setup")
        self.load_yaml_action = QAction("Load YAML…", self)
        self.load_yaml_action.triggered.connect(self._load_yaml)
        setup_menu.addAction(self.load_yaml_action)
        self.save_yaml_action = QAction("Save YAML", self)
        self.save_yaml_action.triggered.connect(self._save_yaml)
        setup_menu.addAction(self.save_yaml_action)
        self.save_yaml_as_action = QAction("Save YAML As…", self)
        self.save_yaml_as_action.triggered.connect(self._save_yaml_as)
        setup_menu.addAction(self.save_yaml_as_action)
        setup_menu.addSeparator()
        self.save_default_action = QAction("Save as Default…", self)
        self.save_default_action.triggered.connect(self._save_as_default)
        setup_menu.addAction(self.save_default_action)
        self.reset_defaults_action = QAction("Reset to Defaults…", self)
        self.reset_defaults_action.triggered.connect(self._prompt_reset_defaults)
        setup_menu.addAction(self.reset_defaults_action)
        setup_menu.addSeparator()
        self.apply_hardware_action = QAction("Apply to hardware", self)
        self.apply_hardware_action.triggered.connect(self._apply_to_hardware)
        setup_menu.addAction(self.apply_hardware_action)
        self._update_apply_action_state()

        view_menu = menu_bar.addMenu("View")
        self.show_log_action = QAction("Show Log", self)
        self.show_log_action.setCheckable(True)
        self.show_log_action.setChecked(self._signals_log_visible)
        self.show_log_action.toggled.connect(self._toggle_log_visibility)
        self.show_dummy_action = QAction("Dummy Advanced", self)
        self.show_dummy_action.setCheckable(True)
        self.show_dummy_action.setChecked(self._show_dummy_advanced)
        self.show_dummy_action.toggled.connect(self._on_show_dummy_tab_changed)
        view_menu.addAction(self.show_log_action)
        view_menu.addAction(self.show_dummy_action)

        self.show_startup_action = QAction("Startup", self)
        self.show_startup_action.setCheckable(True)
        self.show_startup_action.setChecked(self._show_startup_tab)
        self.show_startup_action.toggled.connect(self._toggle_startup_tab_visibility)
        view_menu.addAction(self.show_startup_action)

        view_menu.addSeparator()
        self.measurement_cursors_action = QAction("Measurement cursors", self)
        self.measurement_cursors_action.setCheckable(True)
        self.measurement_cursors_action.setChecked(self._measurement_cursors_enabled)
        self.measurement_cursors_action.setShortcut("C")
        self.measurement_cursors_action.setShortcutContext(Qt.ApplicationShortcut)
        self.measurement_cursors_action.toggled.connect(self._on_toggle_measurement_cursors)
        view_menu.addAction(self.measurement_cursors_action)
        self.addAction(self.measurement_cursors_action)

        self.reset_layout_action = QAction("Reset Layout", self)
        self.reset_layout_action.triggered.connect(self._reset_layout)
        view_menu.addAction(self.reset_layout_action)

    def _build_dashboard_toolbar(self) -> None:
        self.dashboard_bar.clear()
        mode_widget = QWidget()
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(4)
        mode_layout.addWidget(QLabel("Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(self.backends.keys())
        self.mode_combo.setCurrentText(self.backend_name)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        mode_action = QWidgetAction(self.dashboard_bar)
        mode_action.setDefaultWidget(mode_widget)
        self.dashboard_bar.addAction(mode_action)

        dbc_widget = QWidget()
        dbc_layout = QHBoxLayout(dbc_widget)
        dbc_layout.setContentsMargins(0, 0, 0, 0)
        dbc_layout.setSpacing(4)
        dbc_layout.addWidget(QLabel("DBC"))
        self.dbc_edit = QLineEdit(self._qt_settings.value("dbc_path", os.path.join(BASE_DIR, "ecu-test.dbc")))
        self.dbc_edit.setMinimumWidth(160)
        browse_button = QToolButton()
        browse_button.setAutoRaise(True)
        browse_button.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        browse_button.setToolTip("Browse DBC file")
        browse_button.clicked.connect(self._browse_dbc)
        dbc_layout.addWidget(self.dbc_edit)
        dbc_layout.addWidget(browse_button)
        dbc_action = QWidgetAction(self.dashboard_bar)
        dbc_action.setDefaultWidget(dbc_widget)
        self.dashboard_bar.addAction(dbc_action)

        bus_widget = QWidget()
        bus_layout = QHBoxLayout(bus_widget)
        bus_layout.setContentsMargins(0, 0, 0, 0)
        bus_layout.setSpacing(4)
        self.bustype_combo = QComboBox()
        self.bustype_combo.addItems(["socketcan", "vector", "pcan"])
        self.bustype_combo.setCurrentText(self._qt_settings.value("bustype", "socketcan"))
        self.channel_edit = QLineEdit(self._qt_settings.value("channel", "can0"))
        self.channel_edit.setMaximumWidth(90)
        self.bitrate_spin = QSpinBox()
        self.bitrate_spin.setRange(10_000, 1_000_000)
        self.bitrate_spin.setSingleStep(10_000)
        self.bitrate_spin.setValue(int(self._qt_settings.value("bitrate", 500_000)))
        self.bitrate_spin.setMaximumWidth(90)
        bus_layout.setContentsMargins(0, 0, 0, 0)
        bus_layout.setSpacing(6)

        def add_pair(layout, text, widget):
            w = QWidget()
            row = QHBoxLayout(w)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            lab = QLabel(text + ":")
            lab.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
            row.addWidget(lab)
            row.addWidget(widget)
            layout.addWidget(w)

        add_pair(bus_layout, "Bus",     self.bustype_combo)
        add_pair(bus_layout, "Ch",      self.channel_edit)
        # direkt nach dem Erzeugen von self.channel_edit:
        self.channel_edit.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.channel_edit.setMaximumWidth(70)  # oder passend z.B. 60–90 px

        add_pair(bus_layout, "Bitrate", self.bitrate_spin)
        bus_action = QWidgetAction(self.dashboard_bar)
        bus_action.setDefaultWidget(bus_widget)
        self.dashboard_bar.addAction(bus_action)

        self.connect_action = QAction(self.style().standardIcon(QStyle.SP_MediaPlay), "Connect", self)
        self.connect_action.triggered.connect(self._connect_backend)
        self.dashboard_bar.addAction(self.connect_action)
        self.disconnect_action = QAction(self.style().standardIcon(QStyle.SP_MediaStop), "Disconnect", self)
        self.disconnect_action.triggered.connect(self._disconnect_backend)
        self.dashboard_bar.addAction(self.disconnect_action)
        self.dashboard_bar.addSeparator()
        self.all_off_action = QAction("All outputs OFF", self)
        self.all_off_action.triggered.connect(self._all_outputs_off)
        self.dashboard_bar.addAction(self.all_off_action)
        self.emergency_action = QAction("Emergency Stop", self)
        self.emergency_action.triggered.connect(self._emergency_stop)
        self.dashboard_bar.addAction(self.emergency_action)
        self.dashboard_bar.addSeparator()
        self.show_dummy_action = QAction("Dummy Advanced", self)
        self.show_dummy_action.setCheckable(True)
        self.show_dummy_action.setChecked(self._show_dummy_advanced)
        self.show_dummy_action.toggled.connect(self._on_show_dummy_tab_changed)
        #self.dashboard_bar.addAction(self.show_dummy_action)
        #self.dashboard_bar.addAction(self.show_log_action)

    def _build_startup_tab(self) -> None:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        toggle_layout = QHBoxLayout()
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_layout.setSpacing(4)
        self.startup_on_connect_check = QCheckBox("On connect")
        self.startup_on_connect_check.setChecked(self._startup_on_connect)
        self.startup_on_connect_check.toggled.connect(self._on_startup_on_connect_toggled)
        toggle_layout.addWidget(self.startup_on_connect_check)
        self.startup_on_apply_check = QCheckBox("On apply")
        self.startup_on_apply_check.setChecked(self._startup_on_apply)
        self.startup_on_apply_check.toggled.connect(self._on_startup_on_apply_toggled)
        toggle_layout.addWidget(self.startup_on_apply_check)
        self.startup_only_change_check = QCheckBox("Only if changed")
        self.startup_only_change_check.setChecked(self._startup_only_on_change)
        self.startup_only_change_check.toggled.connect(self._on_startup_only_change_toggled)
        toggle_layout.addWidget(self.startup_only_change_check)
        toggle_layout.addStretch(1)
        layout.addLayout(toggle_layout)
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(4)
        control_layout.addWidget(QLabel("Delay"))
        self.startup_delay_spin = QSpinBox()
        self.startup_delay_spin.setRange(0, 10_000)
        self.startup_delay_spin.setSuffix(" ms")
        self.startup_delay_spin.setValue(int(self._startup_delay_ms))
        self.startup_delay_spin.valueChanged.connect(self._on_startup_delay_changed)
        control_layout.addWidget(self.startup_delay_spin)
        self.startup_run_button = QPushButton("Run Startup")
        self.startup_run_button.clicked.connect(self._trigger_manual_startup)
        control_layout.addWidget(self.startup_run_button)
        self.startup_dry_run_button = QPushButton("Dry Run")
        self.startup_dry_run_button.clicked.connect(self._show_startup_dry_run)
        control_layout.addWidget(self.startup_dry_run_button)
        self.startup_status_badge = QLabel("Startup")
        control_layout.addWidget(self.startup_status_badge)
        control_layout.addStretch(1)
        layout.addLayout(control_layout)
        self.startup_tree = QTreeWidget()
        self.startup_tree.setColumnCount(2)
        self.startup_tree.setHeaderLabels(["Step", "Details"])
        self.startup_tree.setRootIsDecorated(True)
        self.startup_tree.setIndentation(14)
        self.startup_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.startup_tree.itemDoubleClicked.connect(lambda *_: self._edit_startup_step())
        self.startup_tree.itemSelectionChanged.connect(self._update_startup_controls)
        self.startup_tree.setMinimumHeight(140)
        layout.addWidget(self.startup_tree)
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(4)
        self.startup_add_global_button = QToolButton()
        self.startup_add_global_button.setText("Add Global")
        self.startup_add_global_button.setAutoRaise(True)
        self.startup_add_global_button.clicked.connect(lambda: self._add_startup_step("global"))
        buttons_layout.addWidget(self.startup_add_global_button)
        self.startup_add_output_button = QToolButton()
        self.startup_add_output_button.setText("Add Output")
        self.startup_add_output_button.setAutoRaise(True)
        self.startup_add_output_button.clicked.connect(lambda: self._add_startup_step("per_output"))
        buttons_layout.addWidget(self.startup_add_output_button)
        self.startup_add_teardown_button = QToolButton()
        self.startup_add_teardown_button.setText("Add Teardown")
        self.startup_add_teardown_button.setAutoRaise(True)
        self.startup_add_teardown_button.clicked.connect(lambda: self._add_startup_step("teardown"))
        buttons_layout.addWidget(self.startup_add_teardown_button)
        self.startup_duplicate_button = QToolButton()
        self.startup_duplicate_button.setText("Duplicate")
        self.startup_duplicate_button.setAutoRaise(True)
        self.startup_duplicate_button.clicked.connect(self._duplicate_startup_step)
        buttons_layout.addWidget(self.startup_duplicate_button)
        self.startup_edit_button = QToolButton()
        self.startup_edit_button.setText("Edit")
        self.startup_edit_button.setAutoRaise(True)
        self.startup_edit_button.clicked.connect(self._edit_startup_step)
        buttons_layout.addWidget(self.startup_edit_button)
        self.startup_remove_button = QToolButton()
        self.startup_remove_button.setText("Remove")
        self.startup_remove_button.setAutoRaise(True)
        self.startup_remove_button.clicked.connect(self._remove_startup_step)
        buttons_layout.addWidget(self.startup_remove_button)
        buttons_layout.addStretch(1)
        layout.addLayout(buttons_layout)
        self._refresh_startup_tree()
        self._update_startup_status_badge()
        self._update_startup_controls()
        self.startup_tab = widget
        index = self.tab_widget.addTab(widget, "Startup")
        self._startup_tab_index = index
        if not self._show_startup_tab:
            try:
                self.tab_widget.setTabVisible(index, False)
            except AttributeError:
                widget.setVisible(False)
        else:
            self.tab_widget.setCurrentIndex(index)
    def _build_channels_tab(self) -> None:
        widget = QWidget()
        outer_layout = QVBoxLayout(widget)
        outer_layout.setContentsMargins(6, 6, 6, 6)
        outer_layout.setSpacing(4)
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(4)
        self.channel_selector = QComboBox()
        control_layout.addWidget(QLabel("Channels:"))
        control_layout.addWidget(self.channel_selector)
        add_button = QToolButton()
        add_button.setText("Add")
        add_button.setAutoRaise(True)
        add_button.clicked.connect(self._add_channel)
        edit_button = QToolButton()
        edit_button.setText("Edit")
        edit_button.setAutoRaise(True)
        edit_button.clicked.connect(self._edit_channel)
        remove_button = QToolButton()
        remove_button.setText("Remove")
        remove_button.setAutoRaise(True)
        remove_button.clicked.connect(self._remove_channel)
        duplicate_button = QToolButton()
        duplicate_button.setText("Duplicate")
        duplicate_button.setAutoRaise(True)
        duplicate_button.clicked.connect(self._duplicate_channel)
        for button in (add_button, edit_button, remove_button, duplicate_button):
            button.setToolButtonStyle(Qt.ToolButtonTextOnly)
            control_layout.addWidget(button)
        control_layout.addStretch(1)
        outer_layout.addLayout(control_layout)
        self.channel_scroll = QScrollArea()
        self.channel_scroll.setWidgetResizable(True)
        self.channel_scroll.setFrameShape(QFrame.NoFrame)
        self.channel_container = QWidget()
        columns_layout = QHBoxLayout(self.channel_container)
        columns_layout.setContentsMargins(0, 0, 0, 0)
        columns_layout.setSpacing(8)
        self._channel_columns_layout = columns_layout
        self._create_channel_columns(max(1, int(self._channel_grid_cols)))
        self.channel_scroll.setWidget(self.channel_container)
        outer_layout.addWidget(self.channel_scroll)
        self.channels_widget = widget
        dock = QDockWidget("Channels", self)
        dock.setObjectName("Dock_Channels")
        dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable
        )
        dock.setWidget(widget)
        self.channels_dock = dock

    def _create_channel_columns(self, count: int) -> None:
        if self._channel_columns_layout is None:
            return
        while self._channel_columns_layout.count():
            item = self._channel_columns_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._channel_columns = []
        column_total = max(1, int(count))
        for _ in range(column_total):
            col_widget = QWidget()
            col_layout = QVBoxLayout(col_widget)
            col_layout.setContentsMargins(0, 0, 0, 0)
            col_layout.setSpacing(8)
            col_layout.addStretch(1)
            self._channel_columns.append(col_layout)
            self._channel_columns_layout.addWidget(col_widget, 1)

    def _build_signals_tab(self) -> None:
        widget = QWidget()
        outer_layout = QVBoxLayout(widget)
        outer_layout.setContentsMargins(6, 6, 6, 6)
        outer_layout.setSpacing(4)
        self.signals_splitter = QSplitter(Qt.Vertical)
        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)
        self.signals_top_splitter = QSplitter(Qt.Horizontal)
        self.signals_top_splitter.setChildrenCollapsible(False)
        self.signal_browser = SignalBrowserWidget()
        self.signals_top_splitter.addWidget(self.signal_browser)
        self.watchlist_widget = WatchlistWidget()
        self.watchlist_widget.plot_toggled.connect(self._on_watchlist_plot_toggled)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        right_layout.addWidget(self.watchlist_widget)
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(4)
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
        self.signals_top_splitter.addWidget(right_widget)
        self.signals_top_splitter.setStretchFactor(0, 2)
        self.signals_top_splitter.setStretchFactor(1, 3)
        top_layout.addWidget(self.signals_top_splitter)
        self.signal_browser.add_requested.connect(self._on_add_to_watchlist)
        self.signal_browser.plot_requested.connect(self._on_plot_requested)
        self.signal_browser.simulate_requested.connect(self._on_signal_simulate)
        self.watchlist_widget.remove_requested.connect(self._on_remove_from_watchlist)
        self.signals_splitter.addWidget(top_container)
        self.logging_container = QWidget()
        self.logging_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        logging_layout = QFormLayout(self.logging_container)
        logging_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        logging_layout.setContentsMargins(0, 0, 0, 0)
        logging_layout.setSpacing(4)
        logging_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.logging_mode_combo = QComboBox()
        self.logging_mode_combo.addItems(["Watchlist", "Manual"])
        logging_layout.addRow("Source", self.logging_mode_combo)
        self.manual_log_edit = QLineEdit()
        self.manual_log_edit.setPlaceholderText("Comma-separated signal names")
        logging_layout.addRow("Manual signals", self.manual_log_edit)
        self.logging_rate_spin = QDoubleSpinBox()
        self.logging_rate_spin.setDecimals(1)
        self.logging_rate_spin.setRange(1.0, 50.0)
        self.logging_rate_spin.setValue(float(self._qt_settings.value("log_rate", 10.0)))
        logging_layout.addRow("Rate (Hz)", self.logging_rate_spin)
        csv_widget = QWidget()
        csv_layout = QHBoxLayout(csv_widget)
        csv_layout.setContentsMargins(0, 0, 0, 0)
        csv_layout.setSpacing(4)
        csv_layout.addWidget(QLabel("Preset"))
        self.csv_preset_combo = QComboBox()
        self.csv_preset_combo.addItem("Excel (DE)", "excel_de")
        self.csv_preset_combo.addItem("Generic (EN)", "generic_en")
        index = max(0, self.csv_preset_combo.findData(self._csv_preset_key))
        self.csv_preset_combo.blockSignals(True)
        self.csv_preset_combo.setCurrentIndex(index)
        self.csv_preset_combo.blockSignals(False)
        self.csv_preset_combo.currentIndexChanged.connect(self._on_csv_preset_changed)
        csv_layout.addWidget(self.csv_preset_combo)
        csv_layout.addStretch(1)
        logging_layout.addRow("Export", csv_widget)
        path_layout = QHBoxLayout()
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.setSpacing(4)
        self.logging_path_edit = QLineEdit(self._qt_settings.value("log_path", os.path.join(BASE_DIR, "signals.csv")))
        browse = QToolButton()
        browse.setAutoRaise(True)
        browse.setText("…")
        browse.clicked.connect(self._browse_log_path)
        path_layout.addWidget(self.logging_path_edit)
        path_layout.addWidget(browse)
        path_widget = QWidget()
        path_widget.setLayout(path_layout)
        logging_layout.addRow("CSV path", path_widget)
        self.logging_button = QPushButton("Start Logging")
        self.logging_button.clicked.connect(self._toggle_logging)
        logging_layout.addRow(self.logging_button)
        self.startup_log_view = QTextBrowser()
        self.startup_log_view.setReadOnly(True)
        self.startup_log_view.setMaximumHeight(140)
        logging_layout.addRow("Startup log", self.startup_log_view)
        self.signals_splitter.addWidget(self.logging_container)
        self.signals_splitter.setStretchFactor(0, 3)
        self.signals_splitter.setStretchFactor(1, 1)
        self.signals_splitter.splitterMoved.connect(self._on_signals_splitter_moved)
        outer_layout.addWidget(self.signals_splitter)
        self.signals_widget = widget
        dock = QDockWidget("Signals", self)
        dock.setObjectName("Dock_Signals")
        dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable
        )
        dock.setWidget(widget)
        self.signals_dock = dock
        self._apply_log_visibility()

    def _build_state_machine_dock(self) -> None:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        top_controls = QHBoxLayout()
        top_controls.addWidget(QLabel("State Machine:"))
        self.state_machine_combo = QComboBox()
        self.state_machine_combo.currentIndexChanged.connect(self._on_state_machine_selected)
        top_controls.addWidget(self.state_machine_combo, 1)
        new_btn = QToolButton()
        new_btn.setText("New")
        new_btn.setAutoRaise(True)
        new_btn.clicked.connect(self._on_state_machine_add)
        duplicate_btn = QToolButton()
        duplicate_btn.setText("Duplicate")
        duplicate_btn.setAutoRaise(True)
        duplicate_btn.clicked.connect(self._on_state_machine_duplicate)
        rename_btn = QToolButton()
        rename_btn.setText("Rename")
        rename_btn.setAutoRaise(True)
        rename_btn.clicked.connect(self._on_state_machine_rename)
        remove_btn = QToolButton()
        remove_btn.setText("Delete")
        remove_btn.setAutoRaise(True)
        remove_btn.clicked.connect(self._on_state_machine_remove)
        for button in (new_btn, duplicate_btn, rename_btn, remove_btn):
            top_controls.addWidget(button)
        layout.addLayout(top_controls)
        runner_controls = QHBoxLayout()
        self.state_machine_start_button = QPushButton("Start")
        self.state_machine_start_button.clicked.connect(self._on_state_machine_start)
        self.state_machine_stop_button = QPushButton("Stop")
        self.state_machine_stop_button.clicked.connect(self._on_state_machine_stop)
        self.state_machine_stop_button.setEnabled(False)
        runner_controls.addWidget(self.state_machine_start_button)
        runner_controls.addWidget(self.state_machine_stop_button)
        runner_controls.addStretch(1)
        self.state_machine_status_label = QLabel("Idle")
        self.state_machine_status_label.setStyleSheet("color: #475569;")
        runner_controls.addWidget(self.state_machine_status_label)
        layout.addLayout(runner_controls)

        self.state_machine_graph = StateMachineGraphView()
        graph_controls = QHBoxLayout()
        graph_controls.addWidget(QLabel("Visualization size"))
        self.state_machine_zoom = QSlider(Qt.Horizontal)
        self.state_machine_zoom.setRange(60, 260)
        self.state_machine_zoom.setValue(120)
        self.state_machine_zoom.setTickPosition(QSlider.TicksBelow)
        self.state_machine_zoom.setTickInterval(20)
        self.state_machine_zoom.valueChanged.connect(self.state_machine_graph.set_zoom_percent)
        graph_controls.addWidget(self.state_machine_zoom, 1)
        zoom_value = QLabel("120%")
        zoom_value.setMinimumWidth(48)
        self.state_machine_zoom.valueChanged.connect(lambda value: zoom_value.setText(f"{value}%"))
        graph_controls.addWidget(zoom_value)
        self.state_machine_graph.set_zoom_percent(self.state_machine_zoom.value())
        layout.addLayout(graph_controls)
        layout.addWidget(self.state_machine_graph)
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.North)

        list_style = (
            "QTableWidget { border: 1px solid #d8dee9; border-radius: 6px; padding: 4px; }"
            " QTableWidget::item { padding: 4px 6px; }"
        )

        states_tab = QWidget()
        states_layout = QVBoxLayout(states_tab)
        states_layout.setContentsMargins(4, 4, 4, 4)
        states_layout.setSpacing(6)
        self.state_table = QTableWidget()
        self.state_table.setColumnCount(1)
        self.state_table.setHorizontalHeaderLabels(["Name"])
        self.state_table.horizontalHeader().setStretchLastSection(True)
        self.state_table.verticalHeader().setVisible(False)
        self.state_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.state_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.state_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.state_table.setAlternatingRowColors(True)
        self.state_table.setMinimumHeight(200)
        self.state_table.setStyleSheet(list_style)
        self.state_table.itemSelectionChanged.connect(self._on_state_selected)
        states_layout.addWidget(self.state_table, 1)
        state_buttons = QHBoxLayout()
        add_state_btn = QToolButton()
        add_state_btn.setText("Add")
        add_state_btn.setAutoRaise(True)
        add_state_btn.clicked.connect(self._on_state_add)
        remove_state_btn = QToolButton()
        remove_state_btn.setText("Remove")
        remove_state_btn.setAutoRaise(True)
        remove_state_btn.clicked.connect(self._on_state_remove)
        state_buttons.addWidget(add_state_btn)
        state_buttons.addWidget(remove_state_btn)
        state_buttons.addStretch(1)
        states_layout.addLayout(state_buttons)
        state_form = QFormLayout()
        state_form.setContentsMargins(0, 0, 0, 0)
        state_form.setSpacing(4)
        self.state_name_edit = QLineEdit()
        self.state_name_edit.setPlaceholderText("Rename selected state")
        self.state_name_edit.editingFinished.connect(self._on_state_name_edited)
        state_form.addRow("Name", self.state_name_edit)
        self.state_initial_combo = QComboBox()
        self.state_initial_combo.currentIndexChanged.connect(self._on_initial_state_changed)
        state_form.addRow("Initial", self.state_initial_combo)
        states_layout.addLayout(state_form)
        tabs.addTab(states_tab, "States")

        transitions_tab = QWidget()
        transitions_layout = QVBoxLayout(transitions_tab)
        transitions_layout.setContentsMargins(4, 4, 4, 4)
        transitions_layout.setSpacing(6)
        self.transition_table = QTableWidget()
        self.transition_table.setColumnCount(3)
        self.transition_table.setHorizontalHeaderLabels(["Name", "Source", "Target"])
        self.transition_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.transition_table.verticalHeader().setVisible(False)
        self.transition_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.transition_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.transition_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.transition_table.setAlternatingRowColors(True)
        self.transition_table.setMinimumHeight(240)
        self.transition_table.setStyleSheet(list_style)
        self.transition_table.itemSelectionChanged.connect(self._on_transition_selected)
        transitions_layout.addWidget(self.transition_table, 1)
        transition_buttons = QHBoxLayout()
        add_transition_btn = QToolButton()
        add_transition_btn.setText("Add")
        add_transition_btn.setAutoRaise(True)
        add_transition_btn.clicked.connect(self._on_transition_add)
        edit_transition_btn = QToolButton()
        edit_transition_btn.setText("Edit")
        edit_transition_btn.setAutoRaise(True)
        edit_transition_btn.clicked.connect(self._on_transition_edit)
        remove_transition_btn = QToolButton()
        remove_transition_btn.setText("Remove")
        remove_transition_btn.setAutoRaise(True)
        remove_transition_btn.clicked.connect(self._on_transition_remove)
        transition_buttons.addWidget(add_transition_btn)
        transition_buttons.addWidget(edit_transition_btn)
        transition_buttons.addWidget(remove_transition_btn)
        transition_buttons.addStretch(1)
        transitions_layout.addLayout(transition_buttons)
        tabs.addTab(transitions_tab, "Transitions")

        layout.addWidget(tabs, 1)
        dock = QDockWidget("State Machine", self)
        dock.setObjectName("Dock_StateMachine")
        dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable
        )
        dock.setWidget(widget)
        self.state_machine_dock = dock
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self._refresh_state_machine_combo()
        self._set_state_machine_buttons()


    def _current_state_machine(self) -> Optional[StateMachineConfig]:
        if self._active_state_machine:
            config = self._state_machine_configs.get(self._active_state_machine)
            if config:
                return config
        # Fallback to the first available config to avoid an empty editor when no
        # active state machine is set (e.g., after loading YAML without an active
        # value).
        if self._state_machine_configs:
            self._active_state_machine = next(iter(self._state_machine_configs))
            return self._state_machine_configs.get(self._active_state_machine)
        return None

    def _state_machine_signal_names(self) -> List[str]:
        names: Set[str] = set()
        for definitions in self._signals_by_message.values():
            for definition in definitions:
                if definition.name:
                    names.add(definition.name)
        return sorted(names)

    def _state_machine_message_names(self) -> List[str]:
        return sorted(self._signals_by_message.keys())

    def _state_machine_channel_names(self) -> List[str]:
        return sorted(self._channel_profiles.keys())

    def _state_machine_message_signals(self) -> Dict[str, List[str]]:
        mapping: Dict[str, Set[str]] = {}
        for message, definitions in self._signals_by_message.items():
            names = mapping.setdefault(message, set())
            for definition in definitions:
                if definition.name:
                    names.add(definition.name)
        return {message: sorted(names) for message, names in mapping.items()}

    def _refresh_state_machine_combo(self) -> None:
        if getattr(self, "_updating_state_machine_ui", False):
            return

        self._updating_state_machine_ui = True
        try:
            combo = getattr(self, "state_machine_combo", None)
            if combo is None:
                return

            combo.blockSignals(True)
            try:
                combo.clear()
                # Namen defensiv normalisieren (ohne führende/nachgestellte Spaces)
                names = [str(n).strip() for n in self._state_machine_configs.keys()]
                # Optional: stabil sortieren
                names = sorted(names)
                combo.addItems(names)

                active = str(self._active_state_machine).strip() if self._active_state_machine else None
                if not active or active not in names:
                    active = names[0] if names else None
                    self._active_state_machine = active

                if active:
                    # Versuche zuerst per findText
                    idx = combo.findText(active)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                    else:
                        # Falls der Combo editierbar ist oder Strings leicht differieren:
                        combo.setCurrentText(active)
            finally:
                combo.blockSignals(False)
        finally:
            self._updating_state_machine_ui = False

        # Editor bewusst (nicht via Signal) aktualisieren
        self._populate_state_machine_editor()


    def _populate_state_machine_editor(self) -> None:
        # --- Struktur / Widgets prüfen und vorbereiten ---
        # Stelle sicher, dass Tabellen existieren und Spalten haben:
        from PyQt5.QtWidgets import QTableWidgetItem, QHeaderView
        from PyQt5.QtCore import Qt

        if not getattr(self, "state_table", None) or not getattr(self, "transition_table", None):
            # Ohne Tabellen macht Populate keinen Sinn
            print("Populate aborted: state/transition table missing")
            return

        # Mindestens 1 Spalte für States
        if self.state_table.columnCount() < 1:
            self.state_table.setColumnCount(1)
            self.state_table.setHorizontalHeaderLabels(["State"])
            self.state_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            self.state_table.verticalHeader().setVisible(False)

        # Mindestens 3 Spalten für Transitions (Name, Source, Target)
        if self.transition_table.columnCount() < 3:
            self.transition_table.setColumnCount(3)
            self.transition_table.setHorizontalHeaderLabels(["Transition", "Source", "Target"])
            header = self.transition_table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.Stretch)
            header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            self.transition_table.verticalHeader().setVisible(False)

        # --- Aktive Konfiguration robust ermitteln (mit Fallback) ---
        print("Populate SM editor for:", self._active_state_machine)
        names = list(self._state_machine_configs.keys()) if hasattr(self, "_state_machine_configs") else []
        config = self._current_state_machine()

        # Fallback if no active state machine is set or the name is invalid
        if not names:
            # Nichts definiert → alles leeren und Graph zurücksetzen
            self.state_table.setRowCount(0)
            self.transition_table.setRowCount(0)
            if getattr(self, "state_initial_combo", None):
                self.state_initial_combo.clear()
            try:
                self._update_state_machine_graph()  # falls diese Methode None/leer behandeln kann
            except Exception:
                pass
            return

        if config is None:
            # auf erste Definition fallen und Combo lautlos anpassen
            first = names[0]
            self._active_state_machine = first
            if getattr(self, "state_machine_combo", None):
                self.state_machine_combo.blockSignals(True)
                try:
                    idx = self.state_machine_combo.findText(first)
                    if idx >= 0:
                        self.state_machine_combo.setCurrentIndex(idx)
                finally:
                    self.state_machine_combo.blockSignals(False)
            config = self._current_state_machine()

        # --- Ab hier: wir haben eine gültige config ---
        self._updating_state_machine_ui = True
        try:
            # Tabellen leeren
            self.state_table.setRowCount(0)
            self.transition_table.setRowCount(0)
            if getattr(self, "state_initial_combo", None):
                self.state_initial_combo.blockSignals(True)
                self.state_initial_combo.clear()
                self.state_initial_combo.blockSignals(False)
            if getattr(self, "state_name_edit", None):
                self.state_name_edit.blockSignals(True)
                self.state_name_edit.clear()
                self.state_name_edit.blockSignals(False)

            # Initial state absichern, bevor wir die Combo setzen
            try:
                config.ensure_initial_state()
            except Exception:
                pass

            # --- States füllen ---
            for state in config.states:
                row = self.state_table.rowCount()
                self.state_table.insertRow(row)
                item = QTableWidgetItem(state.name or "<unnamed>")
                # Selektierbar und Enabled setzen (optional, aber UX-freundlich)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.state_table.setItem(row, 0, item)

            # Erste Zeile selecten, falls noch nichts ausgewählt ist
            if self.state_table.rowCount() and not self.state_table.selectedItems():
                self.state_table.selectRow(0)

            # Initial-State-Combo füllen
            if getattr(self, "state_initial_combo", None):
                self.state_initial_combo.blockSignals(True)
                try:
                    self.state_initial_combo.clear()
                    # 1) Items hinzufügen
                    self.state_initial_combo.addItems(config.state_names)

                    # 2) Zielzustand bestimmen: bevorzugt config.initial_state, sonst 1. State
                    target = config.initial_state if config.initial_state in config.state_names else (
                        config.state_names[0] if config.state_names else None
                    )

                    # 3) Index sicher setzen (und ggf. Model nachziehen)
                    if target:
                        idx = self.state_initial_combo.findText(target)
                        if idx >= 0:
                            self.state_initial_combo.setCurrentIndex(idx)
                        else:
                            # Fallback: auf den ersten Eintrag
                            self.state_initial_combo.setCurrentIndex(0)
                            target = self.state_initial_combo.currentText().strip()

                        # Model (config.initial_state) mit dem tatsächlichen Combo-Wert synchronisieren,
                        # falls ensure_initial_state() noch None gesetzt hatte oder Ziel nicht gefunden wurde.
                        if config.initial_state != target:
                            config.initial_state = target
                            # optional: nicht jedes Mal speichern, aber Status/UI aktualisieren:
                            # self._state_machine_changed()

                    else:
                        # Keine States → leer und deaktivieren
                        self.state_initial_combo.setCurrentIndex(-1)
                        self.state_initial_combo.setEnabled(False)
                finally:
                    self.state_initial_combo.blockSignals(False)

            # --- Transitions füllen ---
            # a) Wenn du deine separate Routine behalten willst (macht Sinn):
            self._populate_transition_list(config)

            # b) Falls _populate_transition_list(config) die QTable nicht befüllt,
            #    kannst du alternativ hier ein Fallback einbauen (auskommentiert lassen):
            # self.transition_table.setRowCount(len(config.transitions))
            # for row, tr in enumerate(config.transitions):
            #     it_name = QTableWidgetItem(tr.name or f"{tr.source} → {tr.target}")
            #     it_src  = QTableWidgetItem(tr.source or "")
            #     it_tgt  = QTableWidgetItem(tr.target or "")
            #     for it in (it_name, it_src, it_tgt):
            #         it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            #     self.transition_table.setItem(row, 0, it_name)
            #     self.transition_table.setItem(row, 1, it_src)
            #     self.transition_table.setItem(row, 2, it_tgt)

        finally:
            self._updating_state_machine_ui = False

        # --- Graph aktualisieren (try/except, soll UI nie blockieren) ---
        try:
            self._update_state_machine_graph()
        except Exception:
            pass

    def _populate_transition_list(self, config: StateMachineConfig) -> None:
        if not self.transition_table:
            return
        self.transition_table.setRowCount(0)
        for transition in config.transitions:
            title_value = (transition.name or "").strip()
            if not title_value:
                source = (transition.source or "").strip()
                target = (transition.target or "").strip()
                if source or target:
                    title_value = f"{source} → {target}".strip()
                else:
                    title_value = "Transition"
            row = self.transition_table.rowCount()
            self.transition_table.insertRow(row)
            self.transition_table.setItem(row, 0, QTableWidgetItem(title_value))
            self.transition_table.setItem(row, 1, QTableWidgetItem(transition.source or ""))
            self.transition_table.setItem(row, 2, QTableWidgetItem(transition.target or ""))
        if self.transition_table.rowCount() and not self.transition_table.selectedItems():
            self.transition_table.selectRow(0)

    def _update_state_machine_graph(self) -> None:
        if not self.state_machine_graph:
            return
        config = self._current_state_machine()
        active = self._state_machine_runner.current_state if self._state_machine_runner.is_running else None
        self.state_machine_graph.update_graph(config, active)

    def _set_state_machine_buttons(self) -> None:
        running = self._state_machine_runner.is_running
        config = self._current_state_machine()
        if self.state_machine_start_button:
            has_config = config is not None and bool(config.states)
            self.state_machine_start_button.setEnabled(not running and has_config and self.backend is not None)
        if self.state_machine_stop_button:
            self.state_machine_stop_button.setEnabled(running)

    def _on_state_machine_add(self) -> None:
        
        if getattr(self, "_initializing_ui", False):
            return

        base = "State Machine"
        counter = 1
        name = base
        while name in self._state_machine_configs:
            counter += 1
            name = f"{base} {counter}"
        state_name = "State 1"
        config = StateMachineConfig(name=name, states=[StateDefinition(name=state_name)], initial_state=state_name)
        self._state_machine_configs[name] = config
        self._active_state_machine = name
        self._state_machine_runner.set_config(config)
        self._state_machine_runner.stop()
        self._refresh_state_machine_combo()
        self._state_machine_changed()

    def _on_state_machine_duplicate(self) -> None:
        
        if getattr(self, "_initializing_ui", False):
            return

        config = self._current_state_machine()
        if not config:
            return
        copy = config.clone()
        base = f"{config.name} Copy"
        name = base
        suffix = 2
        while name in self._state_machine_configs:
            name = f"{base} {suffix}"
            suffix += 1
        copy.name = name
        self._state_machine_configs[name] = copy
        self._active_state_machine = name
        self._state_machine_runner.set_config(copy)
        self._state_machine_runner.stop()
        self._refresh_state_machine_combo()
        self._state_machine_changed()

    def _on_state_machine_rename(self) -> None:
        
        if getattr(self, "_initializing_ui", False):
            return

        config = self._current_state_machine()
        if not config:
            return
        new_name, ok = QInputDialog.getText(self, "Rename state machine", "New name", text=config.name)
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name or new_name == config.name:
            return
        if new_name in self._state_machine_configs and new_name != config.name:
            QMessageBox.warning(self, "Duplicate name", "A state machine with this name already exists.")
            return
        self._state_machine_configs.pop(config.name, None)
        config.name = new_name
        self._state_machine_configs[new_name] = config
        self._active_state_machine = new_name
        self._refresh_state_machine_combo()
        self._state_machine_changed()

    def _on_state_machine_remove(self) -> None:
        config = self._current_state_machine()
        if not config:
            return
        if QMessageBox.question(
            self,
            "Remove state machine",
            f"Delete '{config.name}'?",
            QMessageBox.Yes | QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        self._state_machine_runner.stop()
        self._state_machine_configs.pop(config.name, None)
        self._active_state_machine = next(iter(self._state_machine_configs), None)
        self._refresh_state_machine_combo()
        self._state_machine_changed()

    def _on_state_machine_selected(self, index: int) -> None:
        if getattr(self, "_initializing_ui", False):
            return
        if getattr(self, "_updating_state_machine_ui", False):
            return
        if not self.state_machine_combo:
            return

        name = self.state_machine_combo.itemText(index) if index >= 0 else ""
        if name not in self._state_machine_configs:
            # Ungültige Auswahl – UI bleibt bei der bisherigen active
            return
        if name == self._active_state_machine:
            # Nichts zu tun
            return

        self._active_state_machine = name
        config = self._current_state_machine()
        self._state_machine_runner.set_config(config)
        self._state_machine_runner.stop()
        self._populate_state_machine_editor()
        self._set_state_machine_buttons()
        self._update_state_machine_status("Idle")

    def _on_state_add(self) -> None:
        config = self._current_state_machine()
        if not config:
            return
        base = "State"
        counter = len(config.states) + 1
        name = f"{base} {counter}"
        existing = set(config.state_names)
        while name in existing:
            counter += 1
            name = f"{base} {counter}"
        config.states.append(StateDefinition(name=name))
        config.ensure_initial_state()
        self._populate_state_machine_editor()
        if self.state_table:
            self.state_table.selectRow(len(config.states) - 1)
        self._state_machine_changed()

    def _on_state_remove(self) -> None:
        config = self._current_state_machine()
        if not config or not self.state_table:
            return
        index = self._selected_state_index()
        if index < 0 or index >= len(config.states):
            return
        removed = config.states.pop(index).name
        config.transitions = [
            transition
            for transition in config.transitions
            if transition.source != removed and transition.target != removed
        ]
        config.ensure_initial_state()
        self._populate_state_machine_editor()
        self._state_machine_changed()

    def _selected_state_index(self) -> int:
        if not self.state_table or not self.state_table.selectionModel():
            return -1
        rows = self.state_table.selectionModel().selectedRows()
        return rows[0].row() if rows else -1

    def _on_state_selected(self) -> None:
        if self._updating_state_machine_ui or not self.state_name_edit:
            return
        config = self._current_state_machine()
        index = self._selected_state_index()
        if not config or index < 0 or index >= len(config.states):
            self._updating_state_machine_ui = True
            try:
                self.state_name_edit.clear()
            finally:
                self._updating_state_machine_ui = False
            return
        self._updating_state_machine_ui = True
        try:
            self.state_name_edit.setText(config.states[index].name)
        finally:
            self._updating_state_machine_ui = False

    def _on_state_name_edited(self) -> None:
        if self._updating_state_machine_ui or not self.state_table or not self.state_name_edit:
            return
        config = self._current_state_machine()
        if not config:
            return
        index = self._selected_state_index()
        if index < 0 or index >= len(config.states):
            return
        new_name = self.state_name_edit.text().strip()
        old_name = config.states[index].name
        if not new_name:
            self._updating_state_machine_ui = True
            try:
                self.state_name_edit.setText(old_name)
            finally:
                self._updating_state_machine_ui = False
            return
        if new_name != old_name and new_name in config.state_names:
            QMessageBox.warning(self, "Duplicate state", "Another state already uses this name.")
            self._updating_state_machine_ui = True
            try:
                self.state_name_edit.setText(old_name)
            finally:
                self._updating_state_machine_ui = False
            return
        if new_name == old_name:
            return
        config.states[index].name = new_name
        for transition in config.transitions:
            if transition.source == old_name:
                transition.source = new_name
            if transition.target == old_name:
                transition.target = new_name
        if config.initial_state == old_name:
            config.initial_state = new_name
        self._populate_state_machine_editor()
        self._state_machine_changed()

    def _on_initial_state_changed(self, index: int) -> None:
        if self._updating_state_machine_ui or not self.state_initial_combo:
            return
        config = self._current_state_machine()
        if not config:
            return
        name = self.state_initial_combo.itemText(index) if index >= 0 else ""
        if name and name in config.state_names:
            config.initial_state = name
            self._state_machine_changed()

    def _selected_transition_index(self) -> int:
        if not self.transition_table or not self.transition_table.selectionModel():
            return -1
        rows = self.transition_table.selectionModel().selectedRows()
        return rows[0].row() if rows else -1

    def _on_transition_selected(self) -> None:
        return

    def _on_transition_add(self) -> None:
        config = self._current_state_machine()
        if not config:
            return
        dialog = StateTransitionDialog(
            config.state_names,
            signal_names=self._state_machine_signal_names(),
            message_names=self._state_machine_message_names(),
            message_signals=self._state_machine_message_signals(),
            channel_names=self._state_machine_channel_names(),
            parent=self,
        )
        if dialog.exec_() == QDialog.Accepted and dialog.result_transition:
            config.transitions.append(dialog.result_transition)
            self._populate_state_machine_editor()
            if self.transition_table:
                self.transition_table.selectRow(len(config.transitions) - 1)
            self._state_machine_changed()

    def _on_transition_edit(self) -> None:
        config = self._current_state_machine()
        if not config or not self.transition_table:
            return
        index = self._selected_transition_index()
        if index < 0 or index >= len(config.transitions):
            return
        transition = config.transitions[index]
        dialog = StateTransitionDialog(
            config.state_names,
            signal_names=self._state_machine_signal_names(),
            message_names=self._state_machine_message_names(),
            message_signals=self._state_machine_message_signals(),
            channel_names=self._state_machine_channel_names(),
            parent=self,
            existing=transition,
        )
        if dialog.exec_() == QDialog.Accepted and dialog.result_transition:
            config.transitions[index] = dialog.result_transition
            self._populate_state_machine_editor()
            if self.transition_table:
                self.transition_table.selectRow(index)
            self._state_machine_changed()

    def _on_transition_remove(self) -> None:
        config = self._current_state_machine()
        if not config or not self.transition_table:
            return
        index = self._selected_transition_index()
        if index < 0 or index >= len(config.transitions):
            return
        config.transitions.pop(index)
        self._populate_state_machine_editor()
        self._state_machine_changed()

    def _state_machine_changed(self) -> None:
        self._state_machine_runner.stop()
        config = self._current_state_machine()
        self._state_machine_runner.set_config(config)
        self._set_state_machine_buttons()
        self._update_state_machine_graph()
        self._set_hardware_pending(True, "State machine updated. Save to persist changes.")
        if not self._state_machine_runner.is_running:
            self._update_state_machine_status("Idle")

    def _state_machine_sequence_control(self, channel: str, mode: str) -> None:
        runner = self._sequence_runners.get(channel)
        card = self._channel_cards.get(channel)
        if not runner or not card:
            raise BackendError(f"No sequence available for channel {channel}.")
        config = self._sequencer_configs.get(channel)
        if config is None:
            config = self._sequencer_configs.setdefault(channel, ChannelConfig())
        normalized = (mode or "none").lower()
        if normalized == "start":
            if runner.is_running:
                return
            enabled_sequences = [
                sequence
                for sequence in config.sequences
                if sequence.enabled
                and sequence.duration_s > 0
                and sequence.on_s > 0
                and sequence.off_s > 0
            ]
            if not enabled_sequences:
                raise BackendError(f"No enabled sequence available for {channel}.")
            runner.load(config.sequences, config.repeat_mode, config.repeat_limit_s)
            if not runner.start():
                raise BackendError(f"Failed to start sequence for {channel}.")
            card.set_sequence_running(True)
            runner.emit_progress()
        elif normalized == "stop":
            if runner.is_running:
                runner.stop()
                card.set_sequence_running(False)
        elif normalized == "reset":
            runner.reset()
            card.set_sequence_running(False)

    def _on_state_machine_start(self) -> None:
        config = self._current_state_machine()
        if not config:
            return
        if not config.states:
            QMessageBox.warning(self, "State machine", "Add at least one state before starting the runner.")
            return
        self._state_machine_runner.set_backend(self.backend)
        self._state_machine_runner.set_config(config)
        if not self.backend:
            QMessageBox.warning(self, "State machine", "Connect a backend before starting the state machine.")
            return
        if not self._state_machine_runner.start():
            QMessageBox.warning(self, "State machine", "Unable to start. Ensure initial state is defined.")
            return
        self._set_state_machine_buttons()
        self._update_state_machine_status("Running")

    def _on_state_machine_stop(self) -> None:
        self._state_machine_runner.stop()

    def _on_state_machine_state_changed(self, state: str) -> None:
        self._set_state_machine_buttons()
        self._update_state_machine_status(f"Active: {state}")
        self._update_state_machine_graph()
        self._populate_state_machine_editor()

    def _on_state_machine_runner_stopped(self) -> None:
        self._set_state_machine_buttons()
        self._update_state_machine_status("Idle")
        self._update_state_machine_graph()

    def _on_state_machine_error(self, message: str) -> None:
        self._show_error(message)
        self._on_state_machine_runner_stopped()

    def _update_state_machine_status(self, text: str) -> None:
        if self.state_machine_status_label:
            self.state_machine_status_label.setText(text)

    def _apply_default_dock_layout(self) -> None:
        if not self.channels_dock or not self.signals_dock:
            return
        for dock in (self.channels_dock, self.signals_dock):
            dock.setFloating(False)
            dock.show()
        self.addDockWidget(Qt.LeftDockWidgetArea, self.signals_dock)
           # … und Signals daneben horizontal splitten (statt tabify)
        self.splitDockWidget(self.signals_dock, self.channels_dock, Qt.Horizontal)
        try:
            self.resizeDocks([self.signals_dock, self.channels_dock], [1000, 1000], Qt.Horizontal)
        except Exception:
            pass
        #self.addDockWidget(Qt.LeftDockWidgetArea, self.signals_dock)
        #self.tabifyDockWidget(self.channels_dock, self.signals_dock)
        #self.channels_dock.raise_()

    def _update_central_stack_visibility(self) -> None:
        if (
            not hasattr(self, "central_stack")
            or not hasattr(self, "tab_widget")
            or not hasattr(self, "_central_placeholder")
        ):
            return
        visible = False
        for index in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(index)
            if widget is None:
                continue
            tab_visible = True
            try:
                tab_visible = self.tab_widget.isTabVisible(index)
            except AttributeError:
                tab_visible = not widget.isHidden()
            if tab_visible and not widget.isHidden():
                visible = True
                break
        target = self.tab_widget if visible else self._central_placeholder
        if visible:
            self.central_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        else:
            self.central_stack.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        if self.central_stack.currentWidget() is not target:
            self.central_stack.setCurrentWidget(target)
        self.central_stack.updateGeometry()

    def _restore_dock_layout(self) -> bool:
        if self._saved_dock_state is None:
            return False
        try:
            restored = bool(self.restoreState(self._saved_dock_state))
        except TypeError:
            return False
        return restored

    def _reset_layout(self) -> None:
        if not self.channels_dock or not self.signals_dock:
            return
        self._qt_settings.remove("ui/dock_state")
        self._saved_dock_state = None
        self._apply_default_dock_layout()
        self._update_central_stack_visibility()

    # Startup configuration
    def _refresh_startup_tree(self) -> None:
        if not hasattr(self, "startup_tree"):
            return
        self.startup_tree.blockSignals(True)
        self.startup_tree.clear()
        sections: List[Tuple[str, str, List[Any]]] = [
            ("global", "Global messages", list(self._startup_config.globals)),
            ("per_output", "Per output", list(self._startup_config.per_output)),
            ("teardown", "Teardown", list(self._startup_config.teardown)),
        ]
        for section_key, title, entries in sections:
            parent = QTreeWidgetItem([title, ""])
            parent.setData(0, Qt.UserRole, (section_key, None))
            parent.setFirstColumnSpanned(True)
            has_children = False
            for index, step in enumerate(entries):
                step_title = getattr(step, "message", "")
                if section_key == "per_output" and getattr(step, "channel", ""):
                    step_title = f"{step.channel} → {step.message}"
                details = self._format_startup_details(step)
                item = QTreeWidgetItem([step_title, details])
                item.setData(0, Qt.UserRole, (section_key, index))
                parent.addChild(item)
                has_children = True
                for name, value in sorted(step.fields.items()):
                    field_item = QTreeWidgetItem([name, f"{value}"])
                    item.addChild(field_item)
            if has_children:
                parent.setExpanded(True)
            self.startup_tree.addTopLevelItem(parent)
        self.startup_tree.expandToDepth(0)
        self.startup_tree.blockSignals(False)
        self._update_startup_controls()

    def _format_startup_details(self, step: Any) -> str:
        repeat = getattr(step, "repeat", 1)
        dt_ms = getattr(step, "dt_ms", 0)
        parts: List[str] = []
        if repeat > 1:
            parts.append(f"repeat {repeat}×")
        if dt_ms > 0:
            parts.append(f"dt {dt_ms} ms")
        return ", ".join(parts)

    def _update_startup_controls(self) -> None:
        if not hasattr(self, "startup_run_button"):
            return
        selection = self._selected_startup_step()
        allow_edit = selection is not None
        self.startup_edit_button.setEnabled(allow_edit)
        self.startup_remove_button.setEnabled(allow_edit)
        self.startup_duplicate_button.setEnabled(allow_edit)
        has_steps = bool(
            self._startup_config.globals or self._startup_config.per_output or self._startup_config.teardown
        )
        self.startup_run_button.setEnabled(
            not self._startup_running and self._startup_is_valid and self.backend is not None and has_steps
        )
        self.startup_dry_run_button.setEnabled(self._dbc is not None and has_steps)

    def _selected_startup_step(self) -> Optional[Tuple[str, int]]:
        if not hasattr(self, "startup_tree"):
            return None
        item = self.startup_tree.currentItem()
        if item is None:
            return None
        data = item.data(0, Qt.UserRole)
        if not isinstance(data, tuple) or len(data) != 2:
            return None
        section, index = data
        if index is None:
            return None
        try:
            return str(section), int(index)
        except (TypeError, ValueError):
            return None

    def _add_startup_step(self, section: str) -> None:
        if section not in {"global", "per_output", "teardown"}:
            return
        dialog = StartupStepDialog(
            section,
            sorted(self._channel_profiles.keys()),
            self._signals_by_message if hasattr(self, "_signals_by_message") else {},
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted or dialog.result_step is None:
            return
        step = dialog.result_step
        if isinstance(step, StartupGlobalStep):
            self._startup_config.globals.append(step)
        elif isinstance(step, StartupPerOutputStep):
            self._startup_config.per_output.append(step)
        elif isinstance(step, StartupTeardownStep):
            self._startup_config.teardown.append(step)
        self._on_startup_config_changed()

    def _edit_startup_step(self) -> None:
        selection = self._selected_startup_step()
        if not selection:
            return
        section, index = selection
        existing = None
        if section == "global" and 0 <= index < len(self._startup_config.globals):
            existing = self._startup_config.globals[index]
        elif section == "per_output" and 0 <= index < len(self._startup_config.per_output):
            existing = self._startup_config.per_output[index]
        elif section == "teardown" and 0 <= index < len(self._startup_config.teardown):
            existing = self._startup_config.teardown[index]
        if existing is None:
            return
        dialog = StartupStepDialog(
            section,
            sorted(self._channel_profiles.keys()),
            self._signals_by_message if hasattr(self, "_signals_by_message") else {},
            parent=self,
            existing=existing,
        )
        if dialog.exec_() != QDialog.Accepted or dialog.result_step is None:
            return
        if isinstance(dialog.result_step, StartupGlobalStep) and section == "global":
            self._startup_config.globals[index] = dialog.result_step
        elif isinstance(dialog.result_step, StartupPerOutputStep) and section == "per_output":
            self._startup_config.per_output[index] = dialog.result_step
        elif isinstance(dialog.result_step, StartupTeardownStep) and section == "teardown":
            self._startup_config.teardown[index] = dialog.result_step
        self._on_startup_config_changed()

    def _remove_startup_step(self) -> None:
        selection = self._selected_startup_step()
        if not selection:
            return
        section, index = selection
        if section == "global" and 0 <= index < len(self._startup_config.globals):
            del self._startup_config.globals[index]
        elif section == "per_output" and 0 <= index < len(self._startup_config.per_output):
            del self._startup_config.per_output[index]
        elif section == "teardown" and 0 <= index < len(self._startup_config.teardown):
            del self._startup_config.teardown[index]
        else:
            return
        self._on_startup_config_changed()

    def _duplicate_startup_step(self) -> None:
        selection = self._selected_startup_step()
        if not selection:
            return
        section, index = selection
        if section == "global" and 0 <= index < len(self._startup_config.globals):
            duplicate = copy.deepcopy(self._startup_config.globals[index])
            self._startup_config.globals.insert(index + 1, duplicate)
        elif section == "per_output" and 0 <= index < len(self._startup_config.per_output):
            duplicate = copy.deepcopy(self._startup_config.per_output[index])
            self._startup_config.per_output.insert(index + 1, duplicate)
        elif section == "teardown" and 0 <= index < len(self._startup_config.teardown):
            duplicate = copy.deepcopy(self._startup_config.teardown[index])
            self._startup_config.teardown.insert(index + 1, duplicate)
        else:
            return
        self._on_startup_config_changed()

    def _run_startup_inline_once(self) -> None:
        """
        Führt die in der GUI/Config definierten Init-Steps genau einmal synchron aus.
        Keine festen Messagenamen – es wird 1:1 verwendet, was der User im Startup-Tab konfiguriert hat.
        """
        if self._inline_init_done or not self.backend:
            return
        if not self._startup_is_valid:
            # Nichts configured → als erledigt markieren, damit wir nicht dauernd prüfen
            self._inline_init_done = True
            return

        steps = self._prepare_startup_steps(mode="normal", force=False)
        if not steps:
            self._inline_init_done = True
            return

        errors = []
        for step in steps:
            try:
                self.backend.send_message_by_name(
                    step.message,
                    step.payload,
                    repeat=max(1, int(step.repeat)),
                    dt_ms=max(0, int(step.dt_ms)),
                    force=True,  # hier bewusst: Startup soll sicher ankommen
                )
                # „Last payloads“ für On-Change-Logik aktualisieren:
                self._startup_last_payloads[step.key] = dict(step.payload)
            except BackendError as exc:
                errors.append(f"{step.message}: {exc}")

        self._inline_init_done = True
        if errors:
            self._append_startup_log("Inline Startup had issues:\n  " + "\n  ".join(errors))


    def _on_startup_config_changed(self) -> None:
        self._startup_last_payloads.clear()
        self._refresh_startup_tree()
        self._validate_startup_config()
        self._set_hardware_pending(True, "Startup configuration changed. Apply to hardware when ready.")

    def _on_startup_on_connect_toggled(self, enabled: bool) -> None:
        self._startup_on_connect = enabled
        self._save_settings()

    def _on_startup_on_apply_toggled(self, enabled: bool) -> None:
        self._startup_on_apply = enabled
        self._save_settings()

    def _on_startup_only_change_toggled(self, enabled: bool) -> None:
        self._startup_only_on_change = enabled
        self._save_settings()

    def _on_startup_delay_changed(self, value: int) -> None:
        self._startup_delay_ms = max(0, int(value))
        self._save_settings()

    def _trigger_manual_startup(self) -> None:
        self._run_startup(mode="normal", force=True)

    def _update_startup_status_badge(self) -> None:
        if not hasattr(self, "startup_status_badge"):
            return
        if not self._startup_status_messages and self._startup_is_valid:
            self.startup_status_badge.setText("Startup OK")
            self.startup_status_badge.setStyleSheet("color: green; font-weight: 600;")
            self.startup_status_badge.setToolTip("Startup configuration is valid.")
        elif not self._startup_is_valid:
            text = "Startup ⚠"
            self.startup_status_badge.setText(text)
            self.startup_status_badge.setStyleSheet("color: red; font-weight: 600;")
            tooltip = "\n".join(self._startup_status_messages) if self._startup_status_messages else "Startup configuration is invalid."
            self.startup_status_badge.setToolTip(tooltip)
        else:
            self.startup_status_badge.setText("Startup ⚠")
            self.startup_status_badge.setStyleSheet("color: orange; font-weight: 600;")
            tooltip = "\n".join(self._startup_status_messages) if self._startup_status_messages else "Startup warnings present."
            self.startup_status_badge.setToolTip(tooltip)
        self._update_startup_controls()

    def _validate_startup_config(self) -> None:
        errors: List[str] = []
        warnings: List[str] = []
        if self._dbc is None:
            warnings.append("DBC not loaded")
        else:
            message_cache: Dict[str, Any] = {}
            def ensure_message(name: str) -> Optional[Any]:
                if name in message_cache:
                    return message_cache[name]
                try:
                    message_cache[name] = self._dbc.get_message_by_name(name)
                except KeyError:
                    errors.append(f"Message '{name}' not found in DBC")
                    message_cache[name] = None
                return message_cache[name]

            def validate_step(step: Any, label: str) -> None:
                if not getattr(step, "message", ""):
                    errors.append(f"{label}: message name missing")
                    return
                dbc_message = ensure_message(step.message)
                if dbc_message is None:
                    return
                available = {signal.name for signal in getattr(dbc_message, "signals", [])}
                if not step.fields:
                    warnings.append(f"{label}: no fields specified")
                for signal_name in step.fields:
                    if signal_name not in available:
                        errors.append(f"{label}: signal '{signal_name}' missing")
                    elif not is_signal_writable(signal_name, step.message):
                        errors.append(f"{label}: signal '{signal_name}' not writable")
                repeat_value = getattr(step, "repeat", 1)
                delay_value = getattr(step, "dt_ms", 0)
                if repeat_value < 1:
                    errors.append(f"{label}: repeat must be >= 1")
                if delay_value < 0:
                    errors.append(f"{label}: delay must be >= 0")

            for idx, step in enumerate(self._startup_config.globals):
                validate_step(step, f"Global {idx + 1}")
            for idx, step in enumerate(self._startup_config.per_output):
                if step.channel not in self._channel_profiles:
                    errors.append(f"Per output {idx + 1}: channel '{step.channel}' missing")
                validate_step(step, f"Per output {idx + 1}")
            for idx, step in enumerate(self._startup_config.teardown):
                if not getattr(step, "message", ""):
                    errors.append(f"Teardown {idx + 1}: message name missing")
                else:
                    dbc_message = ensure_message(step.message)
                    if dbc_message is None:
                        continue
                    available = {signal.name for signal in getattr(dbc_message, "signals", [])}
                    for signal_name in step.fields:
                        if signal_name not in available:
                            errors.append(f"Teardown {idx + 1}: signal '{signal_name}' missing")
                        elif not is_signal_writable(signal_name, step.message):
                            errors.append(f"Teardown {idx + 1}: signal '{signal_name}' not writable")
        self._startup_status_messages = errors + warnings
        self._startup_is_valid = not errors and self._dbc is not None
        self._update_startup_status_badge()

    def _prepare_startup_steps(self, mode: str, force: bool) -> List[StartupPreparedStep]:
        steps: List[StartupPreparedStep] = []
        if mode == "teardown":
            for step in self._startup_config.teardown:
                payload = {name: float(value) for name, value in step.fields.items()}
                if not payload and not force:
                    continue
                key = (f"teardown:{step.message}", None)
                steps.append(StartupPreparedStep(key, step.message, payload, 1, 0))
            return steps

        def should_include(step_key: Tuple[str, Optional[str]], payload: Dict[str, float]) -> bool:
            if force or not self._startup_only_on_change:
                return True
            return self._startup_last_payloads.get(step_key) != payload

        for step in self._startup_config.globals:
            payload = {name: float(value) for name, value in step.fields.items()}
            if not payload and not force:
                continue
            key = (step.message, None)
            if should_include(key, payload):
                steps.append(
                    StartupPreparedStep(
                        key=key,
                        message=step.message,
                        payload=payload,
                        repeat=max(1, int(step.repeat)),
                        dt_ms=max(0, int(step.dt_ms)),
                    )
                )
        order_map = {
            "Supply": 0,
            "Power": 0,
            "HBridge": 1,
            "HighSide": 2,
            "LowSide": 2,
            "AO_0_10V": 3,
            "AO_4_20mA": 3,
        }

        def sort_key(step: StartupPerOutputStep) -> Tuple[int, str]:
            profile = self._channel_profiles.get(step.channel)
            channel_type = profile.type if profile else ""
            return (order_map.get(channel_type, 50), step.channel)

        for step in sorted(self._startup_config.per_output, key=sort_key):
            if step.channel not in self._channel_profiles:
                continue
            payload = {name: float(value) for name, value in step.fields.items()}
            if not payload and not force:
                continue
            key = (step.message, step.channel)
            if should_include(key, payload):
                steps.append(
                    StartupPreparedStep(
                        key=key,
                        message=step.message,
                        payload=payload,
                        repeat=max(1, int(step.repeat)),
                        dt_ms=max(0, int(step.dt_ms)),
                        channel=step.channel,
                    )
                )
        return steps

    def _run_startup(self, *, mode: str = "normal", force: bool = False) -> None:
        if not self.backend:
            self._show_error("No backend is configured.")
            return
        if self._startup_running:
            self._show_error("Startup sequence already running.")
            return
        if mode != "teardown" and not self._startup_is_valid:
            self._show_error("Startup configuration is invalid.")
            return
        steps = self._prepare_startup_steps(mode, force)
        if not steps:
            if mode != "teardown":
                self.status_message_label.setText("Startup skipped; no changes required.")
            return
        delay_ms = 0 if mode == "teardown" else max(0, int(self._startup_delay_ms))
        worker_force = force or not self._startup_only_on_change
        worker = StartupWorker(self.backend, steps, delay_ms=delay_ms, force=worker_force)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_startup_progress)
        worker.finished.connect(self._on_startup_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._cleanup_startup_worker)
        self._startup_worker = worker
        self._startup_worker_thread = thread
        self._startup_running = True
        self._startup_pending_mode = mode
        self._append_startup_log(f"Startup initiated ({mode})")
        self._update_startup_controls()
        thread.start()

    def _on_startup_progress(self, message: str) -> None:
        self._append_startup_log(message)

    def _on_startup_finished(self, success: bool, successes: list, summary: str) -> None:
        self._startup_running = False
        mode = self._startup_pending_mode or "normal"
        if mode != "teardown":
            for key, payload in successes:
                self._startup_last_payloads[key] = dict(payload)
        self._append_startup_log(summary)
        if not success:
            self.status_message_label.setText(summary)
        self._startup_pending_mode = None
        self._update_startup_controls()
        if mode == "teardown":
            self._all_outputs_off()

    def _cleanup_startup_worker(self) -> None:
        self._startup_worker_thread = None
        self._startup_worker = None

    def _append_startup_log(self, message: str) -> None:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"{timestamp} • {message}"
        if hasattr(self, "startup_log_view") and self.startup_log_view is not None:
            self.startup_log_view.append(entry)
        self.status_message_label.setText(message)

    def _show_startup_dry_run(self) -> None:
        preview = self._generate_startup_preview()
        dialog = QDialog(self)
        dialog.setWindowTitle("Startup dry run")
        layout = QVBoxLayout(dialog)
        view = QTextBrowser()
        view.setReadOnly(True)
        view.setMinimumSize(420, 240)
        view.setPlainText("\n".join(preview))
        layout.addWidget(view)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec_()

    def _generate_startup_preview(self) -> List[str]:
        if self._dbc is None:
            return ["DBC not loaded"]
        lines: List[str] = []
        steps = self._prepare_startup_steps("normal", force=True)
        if not steps:
            return ["No startup steps defined"]
        for step in steps:
            try:
                message = self._dbc.get_message_by_name(step.message)
            except KeyError:
                lines.append(f"{step.message}: message not found")
                continue
            try:
                encoded = message.encode(step.payload, scaling=True, strict=True)
            except (ValueError, KeyError) as exc:
                lines.append(f"{step.message}: failed to encode ({exc})")
                continue
            data_hex = " ".join(f"{byte:02X}" for byte in encoded)
            info = f"{step.message} (0x{message.frame_id:03X})"
            if step.channel:
                info += f" [{step.channel}]"
            info += f" → {data_hex}"
            lines.append(info)
        return lines

    def _suggest_startup_defaults(self) -> None:
        if self._dbc is None:
            return
        if self._startup_config.globals or self._startup_config.per_output or self._startup_config.teardown:
            return
        suggestions_added = False
        messages = list(getattr(self._dbc, "messages", []))
        message_by_name = {message.name: message for message in messages}

        def add_global(message_name: str, fields: Dict[str, float]) -> bool:
            if any(step.message == message_name for step in self._startup_config.globals):
                return False
            message = message_by_name.get(message_name)
            if message is None:
                return False
            payload: Dict[str, float] = {}
            available = {signal.name for signal in message.signals}
            for field_name, value in fields.items():
                if field_name in available and is_signal_writable(field_name, message_name):
                    payload[field_name] = float(value)
            if not payload:
                return False
            self._startup_config.globals.append(StartupGlobalStep(message=message_name, fields=payload))
            return True

        main_switch_added = add_global(
            "QM_Main_switch_control",
            {
                "enable_sensor_supply": 1.0,
                "enable_actuator_supply": 1.0,
                "enable_ub3": 1.0,
                "enable_ub2": 1.0,
                "enable_ub1": 1.0,
            },
        )
        def default_init_value(signal_name: str) -> float:
            lowered = signal_name.lower()
            if "frequency" in lowered:
                return 1000.0
            if "pwm_max" in lowered:
                return 100.0
            return 0.0

        def add_per_output_step(channel_name: str, message_name: str) -> bool:
            if any(
                step.channel == channel_name and step.message == message_name
                for step in self._startup_config.per_output
            ):
                return False
            message = message_by_name.get(message_name)
            if message is None:
                return False
            payload: Dict[str, float] = {}
            for signal in message.signals:
                if not is_signal_writable(signal.name, message_name):
                    continue
                payload[signal.name] = default_init_value(signal.name)
            if not payload:
                return False
            self._startup_config.per_output.append(
                StartupPerOutputStep(channel=channel_name, message=message_name, fields=payload)
            )
            return True

        suggestions_added = main_switch_added
        high_side_messages = [
            ("HS1", "QM_High_side_output_init_01"),
            ("HS2", "QM_High_side_output_init_02"),
            ("HS3", "QM_High_side_output_init_03"),
        ]
        for channel_hint, message_name in high_side_messages:
            target_channel = channel_hint if channel_hint in self._channel_profiles else message_name
            if add_per_output_step(target_channel, message_name):
                suggestions_added = True

        for message in messages:
            name_lower = message.name.lower()
            if any(step.message == message.name for step in self._startup_config.globals):
                continue
            if "main_switch" in name_lower or "supply" in name_lower:
                fields: Dict[str, float] = {}
                for signal in message.signals:
                    if is_signal_writable(signal.name, message.name):
                        fields[signal.name] = 0.0
                if fields:
                    self._startup_config.globals.append(StartupGlobalStep(message=message.name, fields=fields))
                    suggestions_added = True
                    break
        for channel, profile in self._channel_profiles.items():
            if not profile.write.message:
                continue
            if any(step.channel == channel for step in self._startup_config.per_output):
                continue
            message_name = profile.write.message
            fields: Dict[str, float] = {}
            for semantic, signal_name in profile.write.fields.items():
                if not is_signal_writable(signal_name, message_name):
                    continue
                if semantic in {"enable", "select", "state"}:
                    fields[signal_name] = 0.0
                elif semantic in {"pwm", "setpoint"}:
                    fields[signal_name] = 0.0
                elif semantic == "direction":
                    fields[signal_name] = 0.0
            if fields:
                self._startup_config.per_output.append(
                    StartupPerOutputStep(channel=channel, message=message_name, fields=fields)
                )
                suggestions_added = True
        if suggestions_added:
            self._refresh_startup_tree()


    def _apply_log_visibility(self) -> None:
        if not hasattr(self, "logging_container"):
            return
        if self._signals_log_visible:
            self.logging_container.show()
            if self._signals_log_height <= 0:
                self._signals_log_height = 180
            total = sum(self.signals_splitter.sizes())
            if total <= 0:
                total = max(200, self.signals_splitter.size().height())
            top = max(100, total - self._signals_log_height)
            self.signals_splitter.setSizes([top, self._signals_log_height])
        else:
            sizes = self.signals_splitter.sizes()
            if len(sizes) > 1:
                self._signals_log_height = sizes[1]
            self.logging_container.hide()
            self.signals_splitter.setSizes([self.signals_splitter.size().height(), 0])
        if hasattr(self, "show_log_action"):
            self.show_log_action.blockSignals(True)
            self.show_log_action.setChecked(self._signals_log_visible)
            self.show_log_action.blockSignals(False)

    def _toggle_log_visibility(self, visible: bool) -> None:
        self._signals_log_visible = visible
        self._apply_log_visibility()
        if hasattr(self, "logging_rate_spin"):
            self._save_settings()

    def _on_toggle_measurement_cursors(self, enabled: bool) -> None:
        self._measurement_cursors_enabled = enabled
        for dock in self._plot_windows.values():
            dock.set_measurement_cursors_enabled(enabled)

    def _toggle_startup_tab_visibility(self, visible: bool) -> None:
        self._show_startup_tab = visible
        if hasattr(self, "show_startup_action"):
            self.show_startup_action.blockSignals(True)
            self.show_startup_action.setChecked(visible)
            self.show_startup_action.blockSignals(False)
        if hasattr(self, "tab_widget") and self._startup_tab_index >= 0:
            try:
                self.tab_widget.setTabVisible(self._startup_tab_index, visible)
            except AttributeError:
                if not visible:
                    if self.tab_widget.currentIndex() == self._startup_tab_index:
                        self.tab_widget.setCurrentIndex(0)
                self.tab_widget.widget(self._startup_tab_index).setVisible(visible)
            else:
                if visible:
                    self.tab_widget.setCurrentIndex(self._startup_tab_index)
        self._update_central_stack_visibility()
        if hasattr(self, "logging_rate_spin"):
            self._save_settings()

    def _on_signals_splitter_moved(self, _pos: int, _index: int) -> None:
        sizes = self.signals_splitter.sizes()
        if len(sizes) > 1:
            self._signals_log_height = sizes[1]
            currently_visible = sizes[1] > 0
            if self._signals_log_visible != currently_visible:
                self._signals_log_visible = currently_visible
                if hasattr(self, "show_log_action"):
                    self.show_log_action.blockSignals(True)
                    self.show_log_action.setChecked(currently_visible)
                    self.show_log_action.blockSignals(False)
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
        previous: Optional[QDockWidget] = None
        if self._plot_windows:
            if self._active_plot_id and self._active_plot_id in self._plot_windows:
                previous = self._plot_windows[self._active_plot_id]
            else:
                previous = next(reversed(self._plot_windows.values()))
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        if previous is not None:
            self.tabifyDockWidget(previous, dock)
        dock.show()
        dock.raise_()
        dock.set_measurement_cursors_enabled(self._measurement_cursors_enabled)
        self._plot_windows[identifier] = dock
        self._active_plot_id = identifier
        self._last_plot_dock = dock
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
            if self._last_plot_dock is dock:
                self._last_plot_dock = next(reversed(self._plot_windows.values())) if self._plot_windows else None
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

    # Backend management
    def _load_initial_setup(self) -> None:
        candidates = [PRIMARY_SETUP_PATH, DEFAULT_SETUP_PATH]
        for path in candidates:
            if not path or not os.path.exists(path):
                continue
            payload = self._load_setup_payload(path)
            if payload is None:
                continue
            self._apply_setup_payload(payload)
            self._active_setup_path = path
            self._update_yaml_status(path)
            return
        fallback = copy.deepcopy(FACTORY_DEFAULT_SETUP)
        self._apply_setup_payload(fallback)
        self._active_setup_path = None
        self._update_yaml_status(None)

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
        self._state_machine_runner.set_backend(self.backend)
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
            self._load_dummy_profiles()
        self._update_status_indicator(False)
        self._update_channel_card_modes()
        self.signal_browser.set_allow_simulation(isinstance(self.backend, DummyBackend))
        self._update_apply_action_state()
        self._set_state_machine_buttons()

    def _update_apply_action_state(self) -> None:
        if hasattr(self, "apply_hardware_action"):
            enabled = bool(self.backend) or self._hardware_apply_required
            self.apply_hardware_action.setEnabled(enabled)

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
        if self._startup_on_connect:
            self._run_startup(mode="normal", force=not self._startup_only_on_change)

    def _disconnect_backend(self) -> None:
        if self.backend:
            try:
                self.backend.stop()
            except BackendError:
                pass
        self.backend = None
        self._state_machine_runner.stop()
        self._state_machine_runner.set_backend(None)
        self._set_state_machine_buttons()
        self._update_status_indicator(False)

    def _on_mode_changed(self, name: str) -> None:
        self._switch_backend(name)
        self._save_settings()

    def _on_connection_changed(self, connected: bool, message: str) -> None:
        self._update_status_indicator(connected)
        self.status_message_label.setText(message)
        self._update_startup_controls()

    def _on_status_updated(self) -> None:
        pass

    # Channels management
    def _refresh_channel_cards(self) -> None:
        for card in self._channel_cards.values():
            card.deleteLater()
        self._channel_cards.clear()
        names = list(self._channel_profiles.items())
        max_cols = max(1, int(self._channel_grid_cols))
        self._create_channel_columns(max_cols)
        for index, (name, profile) in enumerate(names):
            card = ChannelCardWidget(profile)
            card.command_requested.connect(self._on_channel_command)
            card.sequencer_requested.connect(self._on_sequencer_request)
            card.sequencer_config_changed.connect(self._on_sequencer_config_changed)
            card.plot_visibility_changed.connect(self._on_channel_plot_visibility)
            card.simulation_changed.connect(self._on_channel_simulation)
            card.section_collapse_changed.connect(self._on_card_section_collapse)
            card.duplicate_requested.connect(self._on_card_duplicate)
            card.delete_requested.connect(self._on_card_delete)
            if not self._channel_columns:
                continue
            column = index % len(self._channel_columns)
            target_layout = self._channel_columns[column]
            target_layout.insertWidget(max(0, target_layout.count() - 1), card)
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
            collapse_state = self._channel_collapse_state.get(name, {})
            card.set_section_collapsed("status", collapse_state.get("status", True))
            card.set_section_collapsed("sequencer", collapse_state.get("sequencer", True))
        self._update_channel_card_modes()
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
        # GUI-konfigurierte Init-Sequenz genau einmal vor dem ersten Schalten
        self._run_startup_inline_once()
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

    def _on_card_section_collapse(self, name: str, section: str, collapsed: bool) -> None:
        state = self._channel_collapse_state.setdefault(name, {})
        state[section] = collapsed
        self._save_settings()

    def _on_card_duplicate(self, name: str) -> None:
        index = self.channel_selector.findText(name)
        if index >= 0:
            self.channel_selector.setCurrentIndex(index)
        self._duplicate_channel()

    def _on_card_delete(self, name: str) -> None:
        index = self.channel_selector.findText(name)
        if index >= 0:
            self.channel_selector.setCurrentIndex(index)
        self._remove_channel()

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

    def _stop_all_sequences(self) -> None:
        for runner in self._sequence_runners.values():
            runner.stop()
        for card in self._channel_cards.values():
            card.set_sequence_running(False)

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
        # _apply_sequence_output(...)
        command = {"enabled": 1.0, "select": 1.0, "pwm": pwm_value}

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
            self._save_dummy_profiles()

    # Dummy simulation profiles
    def _apply_simulation_profile(self, profile: SignalSimulationConfig) -> None:
        if isinstance(self.backend, DummyBackend):
            try:
                self.backend.update_simulation_profile(profile)
            except BackendError as exc:
                self._show_error(str(exc))
                return
            self.simulation_widget.set_profiles(self.backend.simulation_profiles())
            self._save_dummy_profiles()

    # Setup persistence
    def _collect_setup_payload(self) -> dict:
        backend_type = "dummy" if self.backend_name == DummyBackend.name else "real"
        device_id = self.channel_edit.text().strip() if backend_type == "real" else None
        dock_assignments = {name: assignment for name, assignment in self._plot_assignments.items()}
        inline_signals = [
            name
            for name in self.watchlist_widget.plot_signal_names
            if name not in dock_assignments
        ]
        windows: List[dict] = []
        if inline_signals:
            windows.append({"id": 0, "axes": {"left": inline_signals, "right": []}})
        for identifier, dock in sorted(self._plot_windows.items()):
            assigned = dock.assigned_signals()
            left = sorted([name for name, (_unit, side) in assigned.items() if side == "left"])
            right = sorted([name for name, (_unit, side) in assigned.items() if side == "right"])
            windows.append({"id": int(identifier), "axes": {"left": left, "right": right}})
        dummy_profiles: Dict[str, dict] = {}
        if isinstance(self.backend, DummyBackend):
            for name, config in self.backend.simulation_profiles().items():
                dummy_profiles[name] = config.to_dict()
        else:
            dummy_profiles = self._read_stored_dummy_profiles()
        profiles_yaml = [profile.to_yaml() for profile in self._channel_profiles.values()]
        write_map = {profile.name: profile.write.to_yaml() for profile in self._channel_profiles.values()}
        status_map = {profile.name: profile.status.to_yaml() for profile in self._channel_profiles.values()}
        signals_payload = {
            "watch": list(self.watchlist_widget.signal_names),
            "plot": list(self.watchlist_widget.plot_signal_names),
            "multi_plot": {
                "enabled": bool(self._multi_plot_enabled),
                "paused": bool(self._multi_plot_paused),
                "windows": windows,
            },
        }
        channels_payload: Dict[str, Any] = {
            "profiles": profiles_yaml,
            "write": write_map,
            "status": status_map,
            "plot_visibility": dict(self._channel_plot_settings),
        }
        state_payload = {
            "active": self._active_state_machine,
            "definitions": [config.to_dict() for config in self._state_machine_configs.values()],
        }
        payload = {
            "version": 1,
            "backend": {"type": backend_type, "device_id": device_id or None},
            "signals": signals_payload,
            "channels": channels_payload,
            "sequencer": {
                "per_channel": {
                    name: {
                        "list": [sequence.to_dict() for sequence in config.sequences],
                        "repeat_mode": config.repeat_mode.name,
                        "repeat_limit_s": int(config.repeat_limit_s),
                    }
                    for name, config in self._sequencer_configs.items()
                }
            },
            "startup": {
                **self._startup_config.to_dict(),
                "on_connect": bool(self._startup_on_connect),
                "on_apply": bool(self._startup_on_apply),
                "delay_ms": int(self._startup_delay_ms),
                "only_on_change": bool(self._startup_only_on_change),
            },
            "state_machines": state_payload,
        }
        if dummy_profiles:
            payload["dummy"] = {"sim": dummy_profiles}
        return payload

    def _save_yaml(self) -> None:
        if not self._active_setup_path:
            self._save_yaml_as()
            return
        payload = self._collect_setup_payload()
        if self._write_yaml_payload(self._active_setup_path, payload):
            self._update_yaml_status(self._active_setup_path)
            self._set_hardware_pending(True, f"Saved {os.path.basename(self._active_setup_path)}.")

    def _save_yaml_as(self) -> None:
        ensure_directories()
        default_path = PRIMARY_SETUP_PATH if os.path.exists(PRIMARY_SETUP_PATH) else DEFAULT_SETUP_PATH
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save YAML",
            default_path,
            "YAML Files (*.yaml *.yml)",
        )
        if not path:
            return
        payload = self._collect_setup_payload()
        if self._write_yaml_payload(path, payload):
            self._active_setup_path = path
            self._update_yaml_status(path)
            self._set_hardware_pending(True, f"Saved {os.path.basename(path)}.")

    def _load_yaml(self) -> None:
        ensure_directories()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load YAML",
            PROFILE_DIR,
            "YAML Files (*.yaml *.yml)",
        )
        if not path:
            return
        payload = self._load_setup_payload(path)
        if payload is None:
            return
        self._apply_setup_payload(payload)
        self._active_setup_path = path
        self._update_yaml_status(path)
        self._set_hardware_pending(True, f"Loaded {os.path.basename(path)}. Apply to hardware when ready.")

    def _load_setup_payload(self, path: str) -> Optional[dict]:
        data: Optional[dict]
        try:
            with open(path, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
        except (OSError, yaml.YAMLError) as exc:
            self._show_error(f"Failed to load setup: {exc}")
            return None
        if loaded is None:
            data = {}
        elif isinstance(loaded, dict):
            data = loaded
        else:
            self._show_error("Invalid setup file format.")
            return None
        if not isinstance(data, dict):
            self._show_error("Invalid setup file format.")
            return None
        return data

    def _write_yaml_payload(self, path: str, payload: dict) -> bool:
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        except OSError as exc:
            self._show_error(f"Failed to save setup: {exc}")
            return False
        return True

    def _update_yaml_status(self, path: Optional[str]) -> None:
        if not hasattr(self, "yaml_path_label") or self.yaml_path_label is None:
            return
        if path:
            normalized = os.path.abspath(path)
            try:
                rel = os.path.relpath(normalized, PROFILE_DIR)
            except ValueError:
                rel = os.path.basename(normalized)
            display = rel if not rel.startswith("..") else os.path.basename(normalized)
        else:
            display = "<unsaved>"
        self.yaml_path_label.setText(f"YAML: {display}")
        status = f"Active setup: {display}"
        if hasattr(self, "statusBar"):
            self.statusBar().showMessage(status, 5000)

    def _apply_setup_payload(self, payload: dict, scopes: Optional[Set[str]] = None) -> None:
        version_value = payload.get("version") if isinstance(payload, dict) else None
        try:
            version = int(version_value) if version_value is not None else 0
        except (TypeError, ValueError):
            version = 0
        if version != 1:
            self._show_error("Unsupported setup version.")
            return
        target_scopes = scopes or {"backend", "signals", "channels", "sequencer", "dummy", "startup", "state_machines"}
        if "backend" in target_scopes:
            self._apply_backend_section(payload.get("backend", {}))
        if "channels" in target_scopes:
            self._apply_channels_setup(payload.get("channels", {}))
        if "sequencer" in target_scopes:
            self._apply_sequencer_setup(payload.get("sequencer", {}))
        if "signals" in target_scopes:
            self._apply_signals_setup(payload.get("signals", {}))
        if "dummy" in target_scopes:
            self._apply_dummy_setup(payload.get("dummy", {}))
        if "startup" in target_scopes:
            self._apply_startup_setup(payload.get("startup", {}))
        if "state_machines" in target_scopes:
            self._apply_state_machine_setup(payload.get("state_machines", {}))
        self._set_hardware_pending(True, "Setup loaded. Apply to hardware when ready.")
        self._save_settings()

    def _apply_backend_section(self, data: dict) -> None:
        backend_value = str((data or {}).get("type", "dummy")).lower()
        desired_backend = DummyBackend.name if backend_value != "real" else RealBackend.name
        if desired_backend in self.backends and desired_backend != self.backend_name:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentText(desired_backend)
            self.mode_combo.blockSignals(False)
            self._switch_backend(desired_backend)
        device_id = data.get("device_id") if isinstance(data, dict) else None
        if desired_backend == RealBackend.name and isinstance(device_id, str) and device_id:
            current = self.channel_edit.text().strip()
            if current and current != device_id:
                box = QMessageBox(self)
                box.setWindowTitle("Device mismatch")
                box.setText(
                    "The loaded setup targets device "
                    f"'{device_id}', but the current channel is '{current}'.\n"
                    "Do you want to use the stored device identifier?"
                )
                box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                choice = box.exec_()
                if choice == QMessageBox.Yes:
                    self.channel_edit.setText(device_id)
            else:
                self.channel_edit.setText(device_id)

    def _apply_signals_setup(self, data: dict) -> None:
        if not isinstance(data, dict):
            data = {}
        watch_section = data.get("watch") if isinstance(data, dict) else None
        if not isinstance(watch_section, list):
            watch_section = data.get("watchlist", []) if isinstance(data, dict) else []
        watchlist = [str(name) for name in watch_section]
        plot_section = data.get("plot") if isinstance(data, dict) else None
        if not isinstance(plot_section, list):
            plot_section = data.get("plot_signals", []) if isinstance(data, dict) else []
        plot_signals = {str(name) for name in plot_section}
        existing = list(self.watchlist_widget.signal_names)
        if existing:
            self.watchlist_widget.remove_signals(existing)
        if watchlist:
            self.watchlist_widget.add_signals(watchlist)
        self._pending_watchlist = list(watchlist)
        self._pending_plot_signals = list(plot_signals)
        self._suspend_plot_assignment = True
        try:
            for name in self.watchlist_widget.signal_names:
                self.watchlist_widget.set_plot_enabled(name, name in plot_signals)
        finally:
            self._suspend_plot_assignment = False
        multi_plot = data.get("multi_plot", {}) if isinstance(data, dict) else {}
        enabled = bool(multi_plot.get("enabled", False))
        paused = bool(multi_plot.get("paused", False))
        if hasattr(self, "multi_plot_enable"):
            self.multi_plot_enable.blockSignals(True)
            self.multi_plot_enable.setChecked(enabled)
            self.multi_plot_enable.blockSignals(False)
        self._multi_plot_enabled = enabled
        if getattr(self, "multi_plot_widget", None):
            self.multi_plot_widget.clear()
        self._multi_plot_curves.clear()
        self._multi_plot_buffers.clear()
        if hasattr(self, "multi_plot_widget"):
            if enabled:
                self.multi_plot_widget.show()
            else:
                self.multi_plot_widget.hide()
        self._multi_plot_paused = paused
        if hasattr(self, "multi_plot_pause"):
            self.multi_plot_pause.setText("Resume Plot" if paused else "Pause Plot")
        self._close_all_plot_windows()
        self._plot_assignments.clear()
        windows = multi_plot.get("windows", []) if isinstance(multi_plot, dict) else []
        for entry in windows:
            if not isinstance(entry, dict):
                continue
            identifier = int(entry.get("id", 0))
            if identifier <= 0:
                continue
            new_id = self._create_plot_window()
            if new_id is None:
                continue
            axes = entry.get("axes", {}) if isinstance(entry.get("axes", {}), dict) else {}
            left_signals = [str(name) for name in axes.get("left", [])]
            right_signals = [str(name) for name in axes.get("right", [])]
            for name in left_signals:
                if name in plot_signals and name in self.watchlist_widget.signal_names:
                    self._assign_signal_to_plot(name, new_id, "left")
            for name in right_signals:
                if name in plot_signals and name in self.watchlist_widget.signal_names:
                    self._assign_signal_to_plot(name, new_id, "right")

    def _apply_channels_setup(self, data: dict) -> None:
        profiles_section = data.get("profiles", []) if isinstance(data, dict) else []
        if not isinstance(profiles_section, list):
            profiles_section = []
        new_profiles: Dict[str, ChannelProfile] = {}
        for entry in profiles_section:
            if isinstance(entry, ChannelProfile):
                profile = entry
            elif isinstance(entry, dict):
                try:
                    profile = ChannelProfile.from_yaml(entry)
                except Exception:
                    continue
            else:
                continue
            new_profiles[profile.name] = profile
        if not new_profiles and self._dbc:
            new_profiles = load_channel_profiles(self._dbc)
        if new_profiles:
            self._channel_profiles = dict(sorted(new_profiles.items()))
            save_channel_profiles(list(self._channel_profiles.values()))
            if self.backend:
                try:
                    self.backend.set_channel_profiles(self._channel_profiles)
                except BackendError as exc:
                    self._show_error(str(exc))
        plot_visibility = data.get("plot_visibility", {}) if isinstance(data, dict) else {}
        if isinstance(plot_visibility, dict):
            self._channel_plot_settings = {str(key): bool(value) for key, value in plot_visibility.items()}
        else:
            self._channel_plot_settings = {}
        self._refresh_channel_cards()
        self._validate_channel_profiles()

    def _apply_sequencer_setup(self, data: dict) -> None:
        if not isinstance(data, dict):
            data = {}
        per_channel = data.get("per_channel", {}) if isinstance(data, dict) else {}
        self._stop_all_sequences()
        for name in list(self._sequencer_configs.keys()):
            if name not in self._channel_profiles:
                self._sequencer_configs.pop(name, None)
        for channel, config_data in per_channel.items():
            if channel not in self._channel_profiles:
                continue
            sequences_raw = config_data.get("list", []) if isinstance(config_data, dict) else []
            sequences: List[SequenceCfg] = []
            for entry in sequences_raw:
                if isinstance(entry, SequenceCfg):
                    sequences.append(entry)
                elif isinstance(entry, dict):
                    try:
                        sequences.append(SequenceCfg.from_dict(entry))
                    except Exception:
                        continue
            repeat_value = config_data.get("repeat_mode") if isinstance(config_data, dict) else "OFF"
            repeat_mode: SequenceRepeatMode
            if isinstance(repeat_value, SequenceRepeatMode):
                repeat_mode = repeat_value
            else:
                repeat_label = str(repeat_value).upper()
                try:
                    repeat_mode = SequenceRepeatMode[repeat_label]
                except KeyError:
                    try:
                        repeat_mode = SequenceRepeatMode(str(repeat_value).lower())
                    except ValueError:
                        repeat_mode = SequenceRepeatMode.OFF
            repeat_limit = 0
            if isinstance(config_data, dict):
                try:
                    repeat_limit = int(config_data.get("repeat_limit_s", 0))
                except (TypeError, ValueError):
                    repeat_limit = 0
            channel_config = self._sequencer_configs.setdefault(channel, ChannelConfig())
            channel_config.sequences = sequences
            channel_config.repeat_mode = repeat_mode
            channel_config.repeat_limit_s = max(0, int(repeat_limit))
            runner = self._sequence_runners.get(channel)
            if runner:
                runner.load(channel_config.sequences, channel_config.repeat_mode, channel_config.repeat_limit_s)
            card = self._channel_cards.get(channel)
            if card:
                card.set_sequencer_config(channel_config)
                card.set_sequence_running(False)

    def _apply_dummy_setup(self, data: dict) -> None:
        if not isinstance(data, dict):
            data = {}
        simulations = data.get("sim", {}) if isinstance(data, dict) else {}
        if not isinstance(simulations, dict):
            simulations = data.get("simulations", {}) if isinstance(data, dict) else {}
        if not simulations:
            if isinstance(self.backend, DummyBackend):
                self.simulation_widget.set_profiles(self.backend.simulation_profiles())
            return
        if isinstance(self.backend, DummyBackend):
            for name, cfg in simulations.items():
                if not isinstance(cfg, dict):
                    continue
                try:
                    profile = SignalSimulationConfig.from_dict(cfg)
                except Exception:
                    continue
                try:
                    self.backend.update_simulation_profile(profile)
                except BackendError:
                    continue
            self.simulation_widget.set_profiles(self.backend.simulation_profiles())
            self._save_dummy_profiles()
        else:
            ensure_directories()
            try:
                with open(DUMMY_SIMULATION_PATH, "w", encoding="utf-8") as handle:
                    json.dump({str(k): v for k, v in simulations.items()}, handle, indent=2)
            except OSError as exc:
                self._show_error(f"Failed to store dummy simulations: {exc}")

    def _apply_startup_setup(self, data: dict) -> None:
        if not isinstance(data, dict):
            data = {}
        self._startup_config = StartupConfig.from_dict(data)
        self._startup_last_payloads.clear()
        if "on_connect" in data:
            self._startup_on_connect = bool(data.get("on_connect", self._startup_on_connect))
        if "on_apply" in data:
            self._startup_on_apply = bool(data.get("on_apply", self._startup_on_apply))
        if "delay_ms" in data:
            try:
                self._startup_delay_ms = max(0, int(data.get("delay_ms", self._startup_delay_ms)))
            except (TypeError, ValueError):
                self._startup_delay_ms = 0
        if "only_on_change" in data:
            self._startup_only_on_change = bool(data.get("only_on_change", self._startup_only_on_change))
        if hasattr(self, "startup_on_connect_check"):
            self.startup_on_connect_check.blockSignals(True)
            self.startup_on_connect_check.setChecked(self._startup_on_connect)
            self.startup_on_connect_check.blockSignals(False)
        if hasattr(self, "startup_on_apply_check"):
            self.startup_on_apply_check.blockSignals(True)
            self.startup_on_apply_check.setChecked(self._startup_on_apply)
            self.startup_on_apply_check.blockSignals(False)
        if hasattr(self, "startup_only_change_check"):
            self.startup_only_change_check.blockSignals(True)
            self.startup_only_change_check.setChecked(self._startup_only_on_change)
            self.startup_only_change_check.blockSignals(False)
        if hasattr(self, "startup_delay_spin"):
            self.startup_delay_spin.blockSignals(True)
            self.startup_delay_spin.setValue(int(self._startup_delay_ms))
            self.startup_delay_spin.blockSignals(False)
        self._refresh_startup_tree()
        self._validate_startup_config()

    def _apply_state_machine_setup(self, data: dict) -> None:
        if not isinstance(data, dict):
            data = {}
        definitions = data.get("definitions") if isinstance(data, dict) else []
        configs: Dict[str, StateMachineConfig] = {}
        if isinstance(definitions, list):
            for entry in definitions:
                if isinstance(entry, dict):
                    config = StateMachineConfig.from_dict(entry)
                    configs[config.name] = config
        self._state_machine_configs = configs
        active_value = data.get("active") if isinstance(data, dict) else None
        if isinstance(active_value, str) and active_value in configs:
            self._active_state_machine = active_value
        else:
            self._active_state_machine = next(iter(configs), None)
        self._refresh_state_machine_combo()
        config = self._current_state_machine()
        self._state_machine_runner.set_config(config)
        self._state_machine_runner.stop()
        self._populate_state_machine_editor()
        self._set_state_machine_buttons()

    def _save_as_default(self) -> None:
        ensure_directories()
        payload = self._collect_setup_payload()
        if self._write_yaml_payload(DEFAULT_SETUP_PATH, payload):
            self.status_message_label.setText("Default setup saved.")

    def _prompt_reset_defaults(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Reset to Defaults")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        scope_combo = QComboBox()
        scope_combo.addItems(["All", "Signals", "Channels", "Sequencer", "Dummy"])
        form.addRow("Scope", scope_combo)
        layout.addLayout(form)
        notice = QLabel("Outputs remain off until you apply the setup to hardware.")
        notice.setWordWrap(True)
        if isinstance(self.backend, RealBackend):
            notice.setStyleSheet("color: orange;")
        layout.addWidget(notice)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        choice = scope_combo.currentText().lower()
        self._reset_to_defaults(choice)

    def _reset_to_defaults(self, scope: str) -> None:
        payload = self._load_default_setup_payload()
        if not payload:
            payload = copy.deepcopy(FACTORY_DEFAULT_SETUP)
        scope_map = {
            "all": {"backend", "signals", "channels", "sequencer", "dummy", "startup"},
            "signals": {"signals"},
            "channels": {"channels"},
            "sequencer": {"sequencer"},
            "dummy": {"dummy"},
        }
        scopes = scope_map.get(scope, scope_map["all"])
        if "channels" in scopes or "sequencer" in scopes:
            self._stop_all_sequences()
        self._apply_setup_payload(payload, scopes)
        self._set_hardware_pending(True, "Defaults restored. Apply to hardware when ready.")

    def _load_default_setup_payload(self) -> Optional[dict]:
        ensure_directories()
        if os.path.exists(DEFAULT_SETUP_PATH):
            try:
                with open(DEFAULT_SETUP_PATH, "r", encoding="utf-8") as handle:
                    data = yaml.safe_load(handle)
                if isinstance(data, dict):
                    return data
            except (OSError, yaml.YAMLError) as exc:
                self._show_error(f"Failed to load default setup: {exc}")
        return copy.deepcopy(FACTORY_DEFAULT_SETUP)

    def _save_dummy_profiles(self) -> None:
        if not isinstance(self.backend, DummyBackend):
            return
        ensure_directories()
        try:
            payload = {name: config.to_dict() for name, config in self.backend.simulation_profiles().items()}
            with open(DUMMY_SIMULATION_PATH, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except OSError as exc:
            self._show_error(f"Failed to save dummy profiles: {exc}")

    def _read_stored_dummy_profiles(self) -> Dict[str, dict]:
        ensure_directories()
        if not os.path.exists(DUMMY_SIMULATION_PATH):
            return {}
        try:
            with open(DUMMY_SIMULATION_PATH, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(key): value for key, value in data.items() if isinstance(value, dict)}

    def _load_dummy_profiles(self) -> None:
        if not isinstance(self.backend, DummyBackend):
            return
        stored = self._read_stored_dummy_profiles()
        if not stored:
            return
        for name, cfg in stored.items():
            try:
                profile = SignalSimulationConfig.from_dict(cfg)
            except Exception:
                continue
            try:
                self.backend.update_simulation_profile(profile)
            except BackendError:
                continue
        self.simulation_widget.set_profiles(self.backend.simulation_profiles())

    def _load_startup_defaults(self) -> None:
        ensure_directories()
        config_data: Optional[dict] = None
        if os.path.exists(DEFAULT_SETUP_PATH):
            try:
                with open(DEFAULT_SETUP_PATH, "r", encoding="utf-8") as handle:
                    stored = yaml.safe_load(handle)
                if isinstance(stored, dict):
                    section = stored.get("startup")
                    if isinstance(section, dict):
                        config_data = section
            except (OSError, yaml.YAMLError):
                config_data = None
        if config_data is None:
            fallback = FACTORY_DEFAULT_SETUP.get("startup", {})
            if isinstance(fallback, dict):
                config_data = fallback
            else:
                config_data = {"version": 1, "globals": [], "per_output": [], "teardown": []}
        self._startup_config = StartupConfig.from_dict(config_data)
        self._startup_last_payloads.clear()

    def _validate_channel_profiles(self) -> None:
        if not self._dbc:
            return
        issues: List[str] = []
        for profile in self._channel_profiles.values():
            write_missing: List[str] = []
            status_missing: List[str] = []
            if profile.write.message:
                try:
                    write_message = self._dbc.get_message_by_name(profile.write.message)
                except KeyError:
                    issues.append(f"{profile.name}: write message '{profile.write.message}' not found")
                else:
                    available = {signal.name for signal in write_message.signals}
                    write_missing = [signal for signal in profile.write.fields.values() if signal not in available]
            if profile.status.message:
                try:
                    status_message = self._dbc.get_message_by_name(profile.status.message)
                except KeyError:
                    issues.append(f"{profile.name}: status message '{profile.status.message}' not found")
                else:
                    available_status = {signal.name for signal in status_message.signals}
                    status_missing = [signal for signal in profile.status.fields.values() if signal not in available_status]
            if write_missing:
                issues.append(f"{profile.name}: write fields missing {', '.join(sorted(write_missing))}")
            if status_missing:
                issues.append(f"{profile.name}: status fields missing {', '.join(sorted(status_missing))}")
        if issues:
            message = "Some channel mappings could not be validated:\n" + "\n".join(issues[:10])
            if len(issues) > 10:
                message += f"\n… {len(issues) - 10} more entries."
            QMessageBox.warning(
                self,
                "Channel mapping issues",
                message,
            )


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
        if self._state_machine_runner.is_running:
            self._state_machine_runner.step()
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
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
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
    def _set_hardware_pending(self, pending: bool, message: Optional[str] = None) -> None:
        self._hardware_apply_required = pending
        self._update_apply_action_state()
        if pending:
            if message:
                self.status_message_label.setText(message)
            else:
                self.status_message_label.setText("Setup pending hardware apply")
        elif message:
            self.status_message_label.setText(message)

    def _apply_to_hardware(self) -> None:
        if not self.backend:
            self._show_error("No backend is configured.")
            return
        box = QMessageBox(self)
        box.setWindowTitle("Apply to hardware")
        box.setText("Apply the current setup to the active backend?")
        checkbox = QCheckBox("Force 0% outputs on load")
        box.setCheckBox(checkbox)
        box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        if box.exec_() != QMessageBox.Ok:
            return
        try:
            self.backend.set_channel_profiles(self._channel_profiles)
        except BackendError as exc:
            self._show_error(str(exc))
            return
        if isinstance(self.backend, DummyBackend):
            self._save_dummy_profiles()
        if checkbox.isChecked():
            self._all_outputs_off()
        if self._startup_on_apply:
            self._run_startup(mode="normal", force=not self._startup_only_on_change)
            self._set_hardware_pending(False, "Startup sequence triggered.")
        else:
            self._set_hardware_pending(False, "Setup applied to hardware.")

    def _update_status_indicator(self, connected: bool) -> None:
        color = "green" if connected else "red"
        self.status_indicator.setStyleSheet(f"color: {color}; font-size: 16pt;")

    def _update_csv_status_label(self) -> None:
        if hasattr(self, "csv_preset_status_label"):
            self.csv_preset_status_label.setText(f"CSV: {get_csv_preset_label(self._csv_preset_key)}")

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
        self._send_emergency_disables()
        if self._startup_config.teardown:
            self._run_startup(mode="teardown", force=True)
        else:
            self._all_outputs_off()
        if self.logger.is_running():
            self.logger.stop()
            self.logging_button.setText("Start Logging")
        self.status_message_label.setText("Emergency stop activated")

    def _send_emergency_disables(self) -> None:
        if not self.backend:
            return
        payloads: Dict[str, Dict[str, float]] = {}

        def is_enable_field(name: str) -> bool:
            lowered = name.lower()
            return "enable" in lowered or lowered.endswith("_en")

        for step in list(self._startup_config.globals) + list(self._startup_config.per_output):
            message_fields = payloads.setdefault(step.message, {})
            for name in step.fields:
                if is_enable_field(name):
                    message_fields[name] = 0.0

        for message, fields in payloads.items():
            if not fields:
                continue
            try:
                self.backend.send_message_by_name(message, fields, force=True)
                self._append_startup_log(f"Emergency disable {message}: {fields}")
            except BackendError as exc:
                self._append_startup_log(f"Emergency disable failed for {message}: {exc}")
                self.status_message_label.setText(str(exc))

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
        self._suggest_startup_defaults()
        self._validate_startup_config()
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
        if hasattr(self, "show_dummy_action"):
            self.show_dummy_action.blockSignals(True)
            self.show_dummy_action.setChecked(self._show_dummy_advanced)
            self.show_dummy_action.blockSignals(False)
        self._update_central_stack_visibility()

    @staticmethod
    def _to_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    def _on_show_dummy_tab_changed(self, state) -> None:
        self._show_dummy_advanced = self._to_bool(state, False)
        if hasattr(self, "show_dummy_action"):
            self.show_dummy_action.blockSignals(True)
            self.show_dummy_action.setChecked(self._show_dummy_advanced)
            self.show_dummy_action.blockSignals(False)
        self._update_dummy_tab_visibility()
        if hasattr(self, "logging_rate_spin"):
            self._save_settings()


    def _on_csv_preset_changed(self) -> None:
        if not hasattr(self, "csv_preset_combo"):
            return
        key = self.csv_preset_combo.currentData()
        if not isinstance(key, str):
            return
        if key not in PRESETS:
            key = "excel_de"
        if key == self._csv_preset_key:
            return
        self._csv_preset_key = key
        self._qt_settings.setValue("csv/preset", self._csv_preset_key)
        self._update_csv_status_label()
        if hasattr(self, "logging_rate_spin"):
            self._save_settings()

    def _on_toolbar_visibility_changed(self, visible: bool) -> None:
        self._toolbar_visible = visible
        if hasattr(self, "logging_rate_spin"):
            self._save_settings()

    # Settings persistence
    def _restore_settings(self) -> None:
        self.backend_name = str(self._qt_settings.value("mode", DummyBackend.name))
        watchlist = self._qt_settings.value("watchlist", [])
        if isinstance(watchlist, list):
            self._pending_watchlist = [str(name) for name in watchlist]
        plot_signals = self._qt_settings.value("plot_signals", [])
        if isinstance(plot_signals, list):
            self._pending_plot_signals = [str(name) for name in plot_signals]
        plots = self._qt_settings.value("channel_plots", {})
        if isinstance(plots, dict):
            self._channel_plot_settings = {str(key): bool(value) for key, value in plots.items()}
        self._multi_plot_paused = self._to_bool(self._qt_settings.value("multi_plot_paused", False), False)
        self._show_dummy_advanced = self._to_bool(self._qt_settings.value("show_dummy_tab", False), False)
        self._multi_plot_enabled = self._to_bool(self._qt_settings.value("multi_plot_enabled", False), False)
        self._channel_grid_cols = int(self._qt_settings.value("channel_grid_cols", self._channel_grid_cols) or 2)
        collapse_data = self._qt_settings.value("channel_collapse", {})
        if isinstance(collapse_data, dict):
            parsed: Dict[str, Dict[str, bool]] = {}
            for channel, state in collapse_data.items():
                if isinstance(state, dict):
                    parsed[str(channel)] = {str(key): self._to_bool(value, True) for key, value in state.items()}
            self._channel_collapse_state = parsed
        self._signals_log_visible = self._to_bool(
            self._qt_settings.value("signals_log_visible", self._signals_log_visible), True
        )
        self._signals_log_height = int(self._qt_settings.value("signals_log_height", self._signals_log_height) or 200)
        self._toolbar_visible = self._to_bool(self._qt_settings.value("toolbar_visible", self._toolbar_visible), True)
        self._show_startup_tab = self._to_bool(
            self._qt_settings.value("ui/show_startup", self._show_startup_tab), False
        )
        dock_state = self._qt_settings.value("ui/dock_state")
        if isinstance(dock_state, QByteArray):
            self._saved_dock_state = QByteArray(dock_state)
        elif isinstance(dock_state, (bytes, bytearray)):
            self._saved_dock_state = QByteArray(dock_state)
        elif isinstance(dock_state, str):
            try:
                self._saved_dock_state = QByteArray.fromHex(dock_state.encode("ascii"))
            except ValueError:
                self._saved_dock_state = None
        preset_value = self._qt_settings.value("csv/preset", self._csv_preset_key)
        if isinstance(preset_value, str):
            preset_key = preset_value
        elif preset_value is None:
            preset_key = self._csv_preset_key
        else:
            preset_key = str(preset_value)
        if preset_key not in PRESETS:
            preset_key = "excel_de"
        self._csv_preset_key = preset_key
        self._startup_on_connect = self._to_bool(
            self._qt_settings.value("startup/on_connect", self._startup_on_connect), True
        )
        self._startup_on_apply = self._to_bool(
            self._qt_settings.value("startup/on_apply", self._startup_on_apply), True
        )
        try:
            self._startup_delay_ms = int(self._qt_settings.value("startup/delay_ms", self._startup_delay_ms) or 0)
        except (TypeError, ValueError):
            self._startup_delay_ms = 0
        self._startup_only_on_change = self._to_bool(
            self._qt_settings.value("startup/only_on_change", self._startup_only_on_change), False
        )
        sequences_data = self._qt_settings.value("channel_sequences", {})
        if isinstance(sequences_data, dict):
            for key, value in sequences_data.items():
                if isinstance(value, dict):
                    try:
                        self._sequencer_configs[str(key)] = ChannelConfig.from_dict(value)
                    except Exception:
                        continue
        self._load_startup_defaults()

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
        self._qt_settings.setValue("show_dummy_tab", self._show_dummy_advanced)
        if hasattr(self, "multi_plot_enable"):
            self._qt_settings.setValue("multi_plot_enabled", self.multi_plot_enable.isChecked())
        if hasattr(self, "signals_splitter"):
            sizes = self.signals_splitter.sizes()
            if len(sizes) > 1:
                self._signals_log_height = sizes[1]
        self._qt_settings.setValue("channel_grid_cols", int(self._channel_grid_cols))
        self._qt_settings.setValue("channel_collapse", self._channel_collapse_state)
        self._qt_settings.setValue("signals_log_visible", self._signals_log_visible)
        self._qt_settings.setValue("csv/preset", self._csv_preset_key)
        self._qt_settings.setValue("signals_log_height", int(self._signals_log_height))
        self._qt_settings.setValue("toolbar_visible", self._toolbar_visible)
        self._qt_settings.setValue("ui/show_startup", self._show_startup_tab)
        self._qt_settings.setValue("ui/dock_state", self.saveState())
        sequence_payload = {name: config.to_dict() for name, config in self._sequencer_configs.items()}
        self._qt_settings.setValue("channel_sequences", sequence_payload)
        self._qt_settings.setValue("startup/on_connect", self._startup_on_connect)
        self._qt_settings.setValue("startup/on_apply", self._startup_on_apply)
        self._qt_settings.setValue("startup/delay_ms", int(self._startup_delay_ms))
        self._qt_settings.setValue("startup/only_on_change", self._startup_only_on_change)

    def closeEvent(self, event) -> None:
        self._save_settings()
        self._disconnect_backend()
        super().closeEvent(event)

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")          # optional, sorgt für konsistente Metrics
    app.setStyleSheet(APP_QSS)      # QSS aktivieren
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
