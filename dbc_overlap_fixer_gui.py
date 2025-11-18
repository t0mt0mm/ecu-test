# -*- coding: utf-8 -*-
"""
DBC Overlap Fixer (GUI) – Auto-Multiplex (No Prompts)
=====================================================

- GUI zum Auswählen einer DBC-Datei.
- Läuft danach **vollautomatisch** durch (keine Rückfragen/Dialogs).
- Strategien:
  1) **Relocation** (nur Startbit), wenn genügend Platz vorhanden ist.
  2) **Auto-Multiplex (Version A, no-prompt)**, wenn kein Platz: 
     - Wählt automatisch ein **vorhandenes 8-bit-Signal** als Multiplexer (M):
       Heuristik: bevorzugt Namen mit "mode" (case-insensitive), sonst das
       8-bit-Signal mit dem kleinsten Startbit.
     - **Nur kollidierende Signale** erhalten eine m-Gruppe (m0, m1, m2, …),
       sodass **innerhalb jeder m-Gruppe keine Overlaps** mehr existieren.
     - **Nicht-kollidierende** Signale bleiben **ohne Mux-Tag** (immer gültig).
     - **Payload-Bits werden nicht geändert.**

Ablage:
- Fixed DBC: <name>_fixed.dbc
- Log:       <name>_overlap_fix_log_<timestamp>.txt

Start:
    python dbc_overlap_fixer_gui.py

Author: M365 Copilot
"""

import os
import re
import datetime
from typing import Dict, List, Optional, Set, Tuple

# --- Regex -----------------------------------------------------------------------------------
BO_RE = re.compile(r"^BO_\s+(\d+)\s+(\S+)\s*:\s*(\d+)\s+(\S+)")
SG_RE = re.compile(
    r"^SG_\s+(?P<name>\S+)(?:\s+(?P<mux>(?:m\d+|M)))?\s*:\s*"
    r"(?P<start>\d+)\|(?P<length>\d+)@(?P<endian>[01])(?P<sign>[+-])\s*(?P<rest>.*)$"
)
REPLACE_START_RE = re.compile(r"(:\s*)(\d+)(\|)")

# --- Bitpositionen --------------------------------------------------------------------------

def intel_positions(start: int, length: int, max_bits: int) -> Optional[List[int]]:
    pos = list(range(start, start + length))
    if pos and (pos[-1] >= max_bits or start < 0):
        return None
    return pos


def motorola_positions(start: int, length: int, max_bits: int) -> Optional[List[int]]:
    pos = []
    bit = start
    for _ in range(length):
        if bit < 0 or bit >= max_bits:
            return None
        pos.append(bit)
        if bit % 8 == 0:
            bit += 15
        else:
            bit -= 1
    return pos


def compute_positions(start: int, length: int, endian: str, max_bits: int) -> Optional[List[int]]:
    return intel_positions(start, length, max_bits) if endian == '1' else motorola_positions(start, length, max_bits)

# --- Datenmodelle ----------------------------------------------------------------------------

class Signal:
    def __init__(self, original_line: str, name: str, mux: Optional[str], start: int,
                 length: int, endian: str, sign: str, rest: str, line_index: int):
        self.original_line = original_line
        self.name = name
        self.mux = mux  # 'M' / 'mX' / None
        self.start = start
        self.length = length
        self.endian = endian
        self.sign = sign
        self.rest = rest
        self.line_index = line_index

    def with_start(self, new_start: int) -> str:
        return REPLACE_START_RE.sub(lambda m: f"{m.group(1)}{new_start}{m.group(3)}", self.original_line, count=1)

    def with_mux_tag(self, tag: Optional[str]) -> str:
        """Return line with mux tag replaced/added after name: ' M' or ' mX' or removed if None."""
        m = SG_RE.match(self.original_line)
        if not m:
            return self.original_line
        name = m.group('name')
        rest_of_line = self.original_line[self.original_line.index(':'):]  # from ':'
        if tag is None:
            # remove any existing mux tag
            return f"SG_ {name} :{rest_of_line[1:]}"
        else:
            return f"SG_ {name} {tag} :{rest_of_line[1:]}"


class Message:
    def __init__(self, bo_line: str, can_id: int, name: str, dlc: int, tx: str, start_index: int):
        self.bo_line = bo_line
        self.can_id = can_id
        self.name = name
        self.dlc = dlc
        self.tx = tx
        self.start_index = start_index
        self.signals: List[Signal] = []

    @property
    def max_bits(self) -> int:
        return self.dlc * 8

# --- Parser ----------------------------------------------------------------------------------

def parse_dbc(lines: List[str]) -> List[Message]:
    messages: List[Message] = []
    current_msg: Optional[Message] = None
    for idx, line in enumerate(lines):
        s = line.rstrip('\n')
        bo = BO_RE.match(s)
        if bo:
            can_id = int(bo.group(1))
            name = bo.group(2)
            dlc = int(bo.group(3))
            tx = bo.group(4)
            current_msg = Message(s, can_id, name, dlc, tx, idx)
            messages.append(current_msg)
            continue
        sg = SG_RE.match(s)
        if sg and current_msg is not None:
            current_msg.signals.append(
                Signal(
                    original_line=s,
                    name=sg.group('name'),
                    mux=sg.group('mux'),
                    start=int(sg.group('start')),
                    length=int(sg.group('length')),
                    endian=sg.group('endian'),
                    sign=sg.group('sign'),
                    rest=sg.group('rest'),
                    line_index=idx,
                )
            )
    return messages

# --- Utilities --------------------------------------------------------------------------------

def first_free_start_for_signal(length: int, endian: str, max_bits: int, occupied: Set[int], preferred_start: int, align_byte: bool) -> Optional[int]:
    def fits(s: int) -> bool:
        pos = compute_positions(s, length, endian, max_bits)
        return pos is not None and all(p not in occupied for p in pos)

    if endian == '1':
        s0 = max(0, preferred_start)
        if align_byte and length % 8 == 0:
            s0 = ((s0 + 7) // 8) * 8
        for s in range(s0, max_bits):
            if fits(s):
                return s
        for s in range(0, s0):
            if align_byte and length % 8 == 0 and (s % 8 != 0):
                continue
            if fits(s):
                return s
        return None
    else:
        def be_iter(start_from: int):
            for s in range(max(0, start_from), max_bits):
                yield s
            for s in range(0, max(0, start_from)):
                yield s
        cands = list(be_iter(preferred_start))
        if align_byte and length % 8 == 0:
            cands2 = [s for s in cands if s % 8 == 7]
            cands = cands2 if cands2 else cands
        for s in cands:
            if fits(s):
                return s
        return None

# --- Core: auto-multiplex only colliding signals --------------------------------------------

def choose_mux_signal(msg: Message) -> Optional[Signal]:
    # Prefer 8-bit signals with 'mode' in name, else first 8-bit by smallest start.
    candidates = [s for s in msg.signals if (s.mux is None or s.mux == 'M') and s.length == 8]
    if not candidates:
        return None
    preferred = [s for s in candidates if 'mode' in s.name.lower()]
    pool = preferred if preferred else candidates
    return sorted(pool, key=lambda s: (s.start, s.name))[0]


def auto_multiplex_only_colliders(msg: Message, original_lines: List[str]) -> Tuple[Dict[int, str], List[str]]:
    """Markiere nur kollidierende Signale als mX; setze genau **ein** Multiplexer (M).
    Nicht-kollidierende Signale bleiben ungetaggt. Bits werden nicht verschoben.
    Liefert (line_index->new_line, log_lines).
    """
    log: List[str] = []
    max_bits = msg.max_bits

    # 1) Baue BASE (untagged) durch Sequentielles Packen, markiere Colliders
    base_occ: Set[int] = set()
    base_indices: Set[int] = set()
    colliders: List[Signal] = []

    def positions(s: Signal) -> Optional[List[int]]:
        return compute_positions(s.start, s.length, s.endian, max_bits)

    for s in msg.signals:
        pos = positions(s)
        if pos is None or any(p in base_occ for p in pos):
            colliders.append(s)
        else:
            base_indices.add(s.line_index)
            for p in pos:
                base_occ.add(p)

    if not colliders:
        log.append("  * Keine Überlappungen – Auto-Multiplex nicht notwendig.")
        return {}, log

    # 2) Mux-Signal wählen (vorhanden lassen falls schon 'M')
    mux_sig = None
    for s in msg.signals:
        if s.mux == 'M':
            mux_sig = s
            break
    if mux_sig is None:
        mux_sig = choose_mux_signal(msg)
        if mux_sig is None:
            log.append("  - ERROR: Kein geeignetes 8-Bit-Signal als Multiplexer verfügbar.")
            return {}, log

    # 3) m-Gruppen bilden: m0, m1, m2, ... nur für Colliders
    groups: List[Set[int]] = []  # bit occupancy per group
    assign: Dict[int, int] = {}  # line_index -> mval

    for s in colliders:
        pos = positions(s)
        if pos is None:
            # Ungültige Pos -> eigene Gruppe
            g_idx = len(groups)
            groups.append(set())
            assign[s.line_index] = g_idx
            log.append(f"  - MUX m{g_idx}: {s.name} (invalid pos)")
            continue
        # Suche erste Gruppe, in der es keine Konflikte gibt
        placed = False
        for g_idx, occ in enumerate(groups):
            if all(p not in occ for p in pos):
                assign[s.line_index] = g_idx
                for p in pos:
                    occ.add(p)
                placed = True
                log.append(f"  - MUX m{g_idx}: {s.name}")
                break
        if not placed:
            g_idx = len(groups)
            groups.append(set(pos))
            assign[s.line_index] = g_idx
            log.append(f"  - MUX m{g_idx}: {s.name}")

    # 4) Zeilen neu aufbauen: 
    #    - Muxor 'M' setzen (falls nicht schon vorhanden)
    #    - Colliders mit 'mX'
    #    - Base ohne Tag (existierende Tags entfernen)
    new_lines: Dict[int, str] = {}

    # Multiplexer-Zeile
    if mux_sig.mux != 'M':
        new_lines[mux_sig.line_index] = mux_sig.with_mux_tag('M')
        log.append(f"  - MUXOR: '{mux_sig.name}' als Multiplexer (M)")

    # Colliders
    for s in colliders:
        mval = assign.get(s.line_index)
        new_lines[s.line_index] = s.with_mux_tag(f"m{mval}")

    # Base (untagged)
    for s in msg.signals:
        if s.line_index in base_indices and s is not mux_sig:
            # Stelle sicher, dass ggf. vorhandene Tags entfernt werden
            if s.mux is not None:
                new_lines[s.line_index] = s.with_mux_tag(None)

    return new_lines, log

# --- Haupt-Logik -----------------------------------------------------------------------------

def process_file(input_path: str, *, respect_mux: bool = True, align_byte: bool = True, auto_mux: bool = True, no_prompts: bool = True) -> Tuple[str, str]:
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    messages = parse_dbc(lines)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(base_dir, f"{base_name}_fixed.dbc")
    log_path = os.path.join(base_dir, f"{base_name}_overlap_fix_log_{timestamp}.txt")

    log_lines: List[str] = []
    log_lines.append(f"DBC Overlap Fixer run at {datetime.datetime.now().isoformat(timespec='seconds')}")
    log_lines.append(f"Input: {input_path}")
    log_lines.append(f"Output: {output_path}")
    log_lines.append("")

    reconstructed_map: Dict[int, str] = {}

    for msg in messages:
        log_lines.append(f"[MSG {msg.can_id} {msg.name}] DLC={msg.dlc} ({msg.max_bits} bits)")

        # 1) Versuche Relocation (nur Startbit) innerhalb DLC
        occupied: Set[int] = set()
        changes = 0
        errors = 0
        overlaps_found = False
        for s in msg.signals:
            pos = compute_positions(s.start, s.length, s.endian, msg.max_bits)
            if pos is not None and all(p not in occupied for p in pos):
                for p in pos:
                    occupied.add(p)
                continue
            overlaps_found = True
            new_start = first_free_start_for_signal(s.length, s.endian, msg.max_bits, occupied, s.start, align_byte)
            if new_start is None:
                errors += 1
            else:
                changes += 1
                reconstructed_map[s.line_index] = s.with_start(new_start)
                for p in compute_positions(new_start, s.length, s.endian, msg.max_bits) or []:
                    occupied.add(p)

        if not overlaps_found:
            log_lines.append("  * Keine Überlappungen.")
            log_lines.append("")
            continue

        if errors == 0:
            # Relocation hat alle Konflikte lösen können
            log_lines.append(f"  * Relocation erfolgreich. Änderungen: {changes}.")
            log_lines.append("")
            continue

        # 2) Relocation konnte nicht alle Konflikte lösen -> Auto-Multiplex (ohne Prompts)
        if auto_mux:
            # Wir verwenden die **Original-Zeilen** der Message (vorherige Relocation-Änderungen
            # am besten verwerfen, um 'Bits unverändert' zu garantieren). Daher entfernen wir
            # ggf. bereits gesetzte Änderungen für diese Message wieder aus reconstructed_map.
            for s in msg.signals:
                reconstructed_map.pop(s.line_index, None)

            new_msg_lines, mux_log = auto_multiplex_only_colliders(msg, lines)
            for ln in mux_log:
                log_lines.append(ln)
            for idx, nl in new_msg_lines.items():
                reconstructed_map[idx] = nl
            if new_msg_lines:
                log_lines.append("  * Auto-Multiplex angewendet (nur kollidierende Signale multiplexed).")
            else:
                log_lines.append("  * Auto-Multiplex nicht angewendet.")
        else:
            log_lines.append("  * Auto-Multiplex deaktiviert; Konflikte verbleiben.")
        log_lines.append("")

    # Write output
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for i, orig in enumerate(lines):
            s = orig.rstrip('\n')
            out_f.write((reconstructed_map.get(i, s)) + '\n')

    with open(log_path, 'w', encoding='utf-8') as lf:
        lf.write('\n'.join(log_lines) + '\n')

    return output_path, log_path

# --- GUI -------------------------------------------------------------------------------------

import tkinter as tk
from tkinter import filedialog, messagebox

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DBC Overlap Fixer – Auto-Multiplex")
        self.geometry("650x220")
        self.resizable(False, False)

        self.file_path_var = tk.StringVar()
        self.align_byte_var = tk.BooleanVar(value=True)

        self._build()

    def _build(self):
        pad = {'padx': 10, 'pady': 6}

        frm_file = tk.Frame(self)
        frm_file.pack(fill='x', **pad)
        tk.Label(frm_file, text="DBC-Datei:").pack(side='left')
        tk.Entry(frm_file, textvariable=self.file_path_var, width=60).pack(side='left', padx=(8, 8))
        tk.Button(frm_file, text="Auswählen…", command=self.on_browse).pack(side='left')

        frm_opts = tk.Frame(self)
        frm_opts.pack(fill='x', **pad)
        tk.Checkbutton(frm_opts, text="Byte-Ausrichtung bevorzugen (bei Längen % 8 == 0)", variable=self.align_byte_var).pack(anchor='w')
        tk.Label(self, text=(
            "Läuft automatisch durch: Erst Relocation, bei Platzmangel Auto-Multiplex.\n"
            "Nur kollidierende Signale werden auf m-Gruppen verteilt; Bits bleiben unverändert."
        )).pack(fill='x', **pad)

        frm_actions = tk.Frame(self)
        frm_actions.pack(fill='x', **pad)
        tk.Button(frm_actions, text="Überlappungen beheben", command=self.on_fix, width=28).pack(side='left')
        tk.Button(frm_actions, text="Beenden", command=self.destroy, width=12).pack(side='right')

    def on_browse(self):
        path = filedialog.askopenfilename(title="DBC auswählen", filetypes=[("DBC files", "*.dbc"), ("Alle Dateien", "*.*")])
        if path:
            self.file_path_var.set(path)

    def on_fix(self):
        path = self.file_path_var.get().strip()
        if not path:
            messagebox.showwarning("Hinweis", "Bitte zuerst eine DBC-Datei auswählen.")
            return
        if not os.path.isfile(path):
            messagebox.showerror("Fehler", f"Datei nicht gefunden:\n{path}")
            return
        try:
            output_path, log_path = process_file(
                path,
                align_byte=self.align_byte_var.get(),
                auto_mux=True,
                no_prompts=True,
            )
            messagebox.showinfo("Fertig", f"Bereinigung abgeschlossen.\n\nOutput:\n{output_path}\n\nLog:\n{log_path}")
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler bei der Verarbeitung:\n{e}")


def main():
    app = App()
    app.mainloop()

if __name__ == '__main__':
    main()
