# ECU Test GUI MVP

This repository contains a PyQt5 application that demonstrates ECU control features with a dual-backend architecture.

## Features
- PyQt5 GUI with Dummy and Real backend selector
- Five high-side outputs (HS1–HS5) with enable toggle and PWM slider
- Five mixed inputs (digital/analog) with live updates at 10 Hz
- DBC-driven signal browser with search, watchlist, CSV logging, and persistent selections across restarts
- Dummy backend loads the DBC and simulates **all** signals with configurable generators that mirror Real-mode names and scaling
- Real backend loads the provided DBC, opens a python-can bus, sends one write command, and decodes one feedback message
- Per-channel sequencer with configurable ON/OFF timing and total duration for HS1–HS5

## Getting Started
1. Create a virtual environment for Python 3.10 or newer.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the GUI:
   ```bash
   python main.py
   ```

The application starts in Dummy mode so it can run without any hardware.

## Backends
- **Dummy Backend**: Loads the active DBC and instantiates a generator for every defined signal. Output commands reuse the same write paths as the Real backend and feed the corresponding status signals. A dedicated panel lets you tweak ramp/sine/noise/hold parameters (analog) or duty/period profiles (digital).
- **Real Backend**: Loads `ecu-test.dbc`, connects to the configured CAN interface, sends the `QM_High_side_output_write` message when outputs change, and listens for `QM_High_side_output_status` to display live current feedback. Writable signals are vetted via a `write`/`cmd` heuristic plus the `config/signals.yaml` whitelist to avoid unintended traffic.

Switch between backends using the dropdown at the top of the main window. If the Real backend cannot open the CAN bus or load the DBC, the application shows a clear error message and falls back to Dummy mode.

### Real backend configuration

Use the **Connection** panel to provide the following settings before selecting **Real** mode:

| Setting     | Description |
|-------------|-------------|
| DBC path    | Path to the DBC file (defaults to `ecu-test.dbc`). |
| Bus type    | python-can backend (e.g., `socketcan`, `pcan`, `vector`). |
| Channel     | Interface channel (examples: `can0`, `PCAN_USBBUS1`, `1`). |
| Bitrate     | Bitrate in bit/s (commonly 125000, 250000, 500000, or 1000000). |

#### Backend-specific hints

- **socketcan**: Ensure your CAN interface is brought up (e.g., `sudo ip link set can0 up type can bitrate 500000`).
- **pcan**: Install Peak drivers and use channel names such as `PCAN_USBBUS1`.
- **vector**: Install the Vector drivers and tools. The channel is typically the device number (for example `0` or `1`).

When an output state changes, the Real backend encodes the PWM and enable signals with cantools and sends a single `QM_High_side_output_write` frame. Incoming `QM_High_side_output_status` frames are decoded asynchronously so the live current is updated immediately in the GUI.

## Dummy signal simulation panel
The **Dummy Signal Simulation** group becomes active in Dummy mode after a DBC is loaded. It lists every message and signal from the database and lets you adjust the generator used by the simulator:

- **Analog signals**: Choose between `hold`, `sine`, `ramp`, and `noise`, and fine-tune offset, amplitude, frequency, slope, noise, hold value, and phase.
- **Digital signals**: Select `pattern` (duty/period) or `manual` mode and edit high/low values, manual value, period, duty, and phase.

Changes apply immediately and affect watchlist values and CSV exports. Default profiles are derived from the signal names (for example, `*_current` ramps, `*_voltage` holds with light noise, temperatures oscillate).

## Signal Browser and Watchlist
1. Load a DBC file with the **Load** button in the connection panel. All messages and signals are parsed dynamically—no signal names are hardcoded.
2. Use the search field to filter messages or signals. Multiple signals can be selected at once.
3. Click **Add to Watchlist** to begin tracking the selected signals. The watchlist refreshes every 100 ms for both Dummy and Real backends.
4. Remove signals with **Remove Selected** when you no longer need them.

The Dummy backend mirrors the Real backend signal names so the watchlist and logger operate identically in either mode. The selected watchlist and logging settings persist via `QSettings`, so the UI reopens with your previous configuration.

## Outputs and Inputs
- Enable or disable each high-side output with the checkbox and adjust the PWM using the slider.
- The live current is shown next to each output.
- Input signals update at 10 Hz. Digital inputs display `ON`/`OFF`, while analog inputs show numeric values.

## Sequencer
- Each output has an independent sequencer panel with ON time, OFF time, and total duration controls.
- Click **Start** to launch the cycle using the current PWM slider value for the ON phase; click **Stop** to halt immediately.
- Sequencers toggle outputs without blocking the UI and can run alongside CSV logging.
- Switching backends or completing the configured duration stops the sequence safely and returns control to the manual widgets.

## CSV Logging
1. Select a destination file path and sampling rate (1–50 Hz).
2. Populate the watchlist with the signals you want to capture.
3. Click **Start Logging**. The CSV header contains `timestamp` plus every signal currently present in the watchlist. Timestamps are written in ISO 8601 UTC format.
4. Click **Stop Logging** to finish and close the file.

If the file cannot be opened, an error message appears and logging does not start.

## Writable signal whitelist

The Real and Dummy backends only transmit values for signals that either belong to a message containing `write`/`cmd` or are explicitly enumerated in `config/signals.yaml`. Adjust the `writable_signals` list to authorise additional command fields when expanding the application.

## Legacy Script
The previous `ecu-test.py` script is still present for reference but is not used by the new GUI. It may require additional dependencies or cleanup before running.
