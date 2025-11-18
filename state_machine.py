from __future__ import annotations

import time
from typing import Callable, List, Optional

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from models import SequenceCfg, SequenceRepeatMode


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
        self.configure(sequences=sequences, repeat_mode=repeat_mode, repeat_limit_s=repeat_limit_s)

    def configure(
        self,
        *,
        sequences: List[SequenceCfg],
        repeat_mode: SequenceRepeatMode,
        repeat_limit_s: int,
    ) -> None:
        self.stop()
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
            pass

    def _emit_progress(self) -> None:
        if not self._running or self._sequence_index < 0 or not self._enabled_sequences:
            self.progressed.emit(-1, "off", 0.0)
            return
        remaining = max(0.0, self._phase_end - self._now_fn())
        self.progressed.emit(self._sequence_index, self._phase, remaining)
