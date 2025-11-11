# ECU Control GUI

This project provides a PyQt5-based ECU control environment that mirrors identical workflows for a dummy simulation backend and a python-can powered real backend. All CAN interactions originate from a user-selected DBC file so browsing, watching, logging, and channel control always reflect the live database without hardcoded signal names.

## Installation
1. Create a Python 3.10+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the GUI:
   ```bash
   python main.py
   ```
The application starts in Dummy mode so you can explore the UI without hardware.

## Interface Highlights

### Dashboard toolbar
- Pick the backend (Dummy or Real), choose the DBC file, and configure CAN bus settings (bustype/channel/bitrate).
- Connect, disconnect, trigger an “all outputs off”, or perform an emergency stop. The emergency stop now disables every enable signal discovered in the startup configuration before powering outputs down.
- Toggle the Dummy-only Advanced tab or quickly show/hide the logging pane from the same toolbar.

### Channels tab
- Channel cards are generated from `profiles/channels.yaml`. Each card is compact, offers icon-only header controls, an embedded sequencer editor, and an optional miniature plot with a 60 s history.
- Context menus expose duplicate and reset actions, while per-card collapse states persist between sessions.

### Signals tab
- A vertical splitter combines the watchlist, plotting controls, and logging tools. The lower pane hosts the logging form with an integrated “Export” preset selector for Excel (DE) and Generic (EN) CSV formats.
- Watchlist entries feed both the inline multi-plot and any number of dockable dual-axis plot windows (up to eight). Newly opened plot docks are tabified automatically on the right edge of the main window.
- CSV exports honour locale presets and emit timestamps in `YYYY-MM-DD HH:MM:SS,mmm` format.

### Startup tab
- Accessible via **View → Startup** (hidden by default). Configure global and per-channel startup messages, delay/pacing options, and teardown steps. Built-in defaults include `QM_Main_switch_control` and the `QM_High_side_output_init_0x` families so high-side drivers receive deterministic initialisation pulses.
- Manual run and dry-run buttons provide on-demand execution and CAN frame previews.

### Dummy Advanced tab
- Available only when the Dummy backend is active. Exposes generator controls (hold, ramp, sine, noise, patterns) for every DBC signal with immediate effect on watchlists, plots, and logs.

## Startup Automation
- Startup data is stored in `profiles/default_setup.json` (or generated from factory defaults) and round-tripped through a tree editor. Steps are grouped into global, per-output, and teardown sections.
- Each startup message supports repeat/delay parameters plus a dry-run preview that encodes payloads using the active DBC. Manual “Run startup” always executes on a worker thread so the UI remains responsive.
- Discovery heuristics propose `QM_Main_switch_control` along with `QM_High_side_output_init_01/02/03` based on the loaded DBC and available channel profiles.

## CSV Logging
- The logging pane lets you choose between the watchlist or a custom comma-separated list of signals, select the capture rate (1–50 Hz), and browse for a destination path.
- Locale-aware presets drive both export and import: **Excel (DE)** writes semicolon-separated CSV files with comma decimals and a UTF-8 BOM, while **Generic (EN)** keeps comma decimals and UTF-8 without BOM. Timestamp columns are formatted for Excel compatibility.

## Persistence
`QSettings` persists backend selections, DBC path, bus parameters, watchlist contents, per-channel plot toggles, multi-plot options, logging state, Dummy Advanced visibility, startup preferences (including visibility), splitter sizes, and the chosen CSV preset. Channel profiles continue to live in `profiles/channels.yaml` for easy version control.

## Writable Signals
The helper `is_signal_writable()` no longer relies on a static whitelist. A signal is considered writable when either its parent message or the signal name itself contains `ctrl`, `control`, `init`, `write`, or `cmd`. This heuristic keeps the Real backend restricted to control-oriented messages while remaining flexible when new commands appear in the DBC.
