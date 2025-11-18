from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Set

from PyQt5.QtCore import QObject, pyqtSignal

from backend import BackendBase, BackendError
from models import StateAction, StateCondition, StateDefinition, StateMachineConfig, StateTransition


class StateMachineRunner(QObject):
    state_changed = pyqtSignal(str)
    stopped = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._backend: Optional[BackendBase] = None
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

    def set_backend(self, backend: Optional[BackendBase]) -> None:
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
            should_trigger = False
            if not transition.conditions:
                should_trigger = True
            else:
                evaluations = [condition.evaluate(values) for condition in transition.conditions]
                if transition.condition_logic == "OR":
                    should_trigger = any(evaluations)
                else:
                    should_trigger = all(evaluations)
            if should_trigger:
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
