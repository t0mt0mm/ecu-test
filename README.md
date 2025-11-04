# ECU Control GUI

This project provides a PyQt5-based ECU control environment that exposes identical workflows for a dummy simulation backend and a python-can powered real backend. All CAN signals originate from a user-selected DBC file, ensuring that browsing, watching, logging, and channel control reflect the live database without hardcoded names.

## Installation
1. Create a Python 3.10+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the GUI:
   ```bash
   python main.py
   ```
The application opens in Dummy mode so it can be exercised without hardware.

## Interface Overview
The main window is organised into tabs so you can reach common tasks without scrolling:

| Tab | Purpose |
| --- | --- |
| **Dashboard** | Select Dummy or Real mode, choose the DBC, configure bus parameters (bustype/channel/bitrate), and issue connection or emergency-stop actions. A coloured status indicator (green/yellow/red) shows the current connection state. |
| **Channels** | Displays channel cards generated from `profiles/channels.yaml`. Each card adapts to its channel type (HighSide, HBridge, analog output, digital I/O, etc.), exposes quick actions, and integrates a sequencer (ON seconds, OFF seconds, duration). |
| **Signals** | Combines the DBC-driven browser and the watchlist. Double-click or select and press “Add to Watchlist” to monitor any signal. The watchlist table shows value, unit, and last update timestamp. |
| **Dummy Sim** | Available in Dummy mode only. Presents the full signal simulation tree so you can tweak analog (hold, sine, ramp, noise) and digital (pattern/manual) generators that back the dummy telemetry. |
| **Logging** | Configure CSV capture: select whether to log the watchlist or a manual list of signals, set the rate (1–50 Hz), choose the output path, and start/stop logging. CSV files contain ISO timestamps and the selected signals as columns. |

## Channel Profiles
Outputs and inputs are no longer hardcoded. Channel behaviour is defined in YAML profiles located at `profiles/channels.yaml`:

```yaml
- name: HS1
  type: HighSide
  write:
    message: QM_High_side_output_write
    fields:
      hs_out01_select: select
      hs_out01_mode: mode
      hs_out01_value_pwm: pwm
  status:
    message: QM_High_side_output_status
    fields:
      hs_out01_current: current
      hs_out01_pwm: pwm_feedback
  sim:
    tau: 0.5
    current_gain: 8.0
    noise: 0.1
```

The **Channels** tab offers an editor dialog (“Add Channel” / “Edit Channel”) that lets you pick the channel type, associate write/status messages, and map semantics to DBC signals. Profiles are persisted back to the YAML file and reloaded on startup. When the YAML file is absent, default profiles are generated to mirror the previous fixed HS and AI channels so existing workflows keep working.

## Dummy Backend
* Loads the active DBC (strict=False) and creates a simulator for every signal. *
- Commands use the channel profiles to override write signals, while status values follow first-order or generator-based models parameterised by the profile’s `sim` section.
- The Dummy Simulation tab exposes every generated signal so you can adjust ramps, sinewaves, noise levels, or digital patterns with immediate effect on the watchlist and logger.

## Real Backend
* Uses python-can with parameters from the Dashboard tab. *
- Loads the same DBC file and validates writable signals via the `write`/`cmd` heuristic plus the whitelist in `config/signals.yaml`.
- Sends channel commands through the message configured in each profile (for example `QM_High_side_output_write`).
- Listens for the configured status message (for example `QM_High_side_output_status`) via a CAN notifier and populates the shared signal cache so the watchlist and logging paths match the Dummy backend.
- Connection errors or decoding issues surface through dialogs and the status bar instead of terminating the application.

## Signal Browser & Watchlist
All signals from the active DBC are shown in the browser. Search filters both message names and signal metadata. Double-clicking a signal adds it to the watchlist. Watchlist entries refresh at 10 Hz and include the unit and the timestamp of the most recent update. Selected watchlist signals can be logged directly from the Logging tab.

## Sequencer
Every channel card embeds its own sequencer. Provide ON seconds, OFF seconds, and total duration (minutes) and press **Start Sequence**. The backend receives deterministic commands at the UI tick rate while manual controls are locked. **Stop** halts immediately, and switching backends automatically stops all sequencers. Logging continues uninterrupted during sequencer operation.

## CSV Logging
Choose between logging the watchlist or a custom list of signal names. Logging writes ISO 8601 timestamps and the requested signals at the chosen frequency. File creation errors are reported through a dialog and the status bar. Stopping logging flushes and closes the file safely.

## Persistence
`QSettings` stores the selected mode, DBC path, bus parameters, watchlist contents, logging rate/path, and window geometry so the UI restores your previous session. Channel profiles are stored separately in `profiles/channels.yaml` to simplify sharing and version control.

## Writable Signals
The helper `is_signal_writable()` uses two checks before the Real backend transmits a value:
1. If the signal appears in `config/signals.yaml` under `writable_signals`, it is allowed.
2. Otherwise the message or signal name must contain `write` or `cmd`.

Update `config/signals.yaml` when introducing additional writable signals to the hardware backend.
