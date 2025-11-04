# ECU Test GUI MVP

This repository contains a minimal PyQt5 application that demonstrates ECU control features with a dual-backend architecture.

## Features
- PyQt5 GUI with Dummy and Real backend selector
- Five high-side outputs (HS1–HS5) with enable toggle and PWM slider
- Five mixed inputs (digital/analog) with live updates at 10 Hz
- Dummy backend simulates first-order current response with noise and dynamic inputs
- DBC-driven signal browser with search, watchlist, and CSV logging for any selected signals
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
- **Dummy Backend**: Implements the required simulation, reacting to PWM commands with a first-order current model and generating synthetic digital/analog inputs.
- **Real Backend**: Loads `ecu-test.dbc`, connects to the configured CAN interface, sends the `QM_High_side_output_write` message when outputs change, and listens for `QM_High_side_output_status` to display live current feedback.

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

## Signal Browser and Watchlist
1. Load a DBC file with the **Load** button in the connection panel. All messages and signals are parsed dynamically—no signal names are hardcoded.
2. Use the search field to filter messages or signals. Multiple signals can be selected at once.
3. Click **Add to Watchlist** to begin tracking the selected signals. The watchlist refreshes every 100 ms for both Dummy and Real backends.
4. Remove signals with **Remove Selected** when you no longer need them.

The Dummy backend mirrors the Real backend signal names so the watchlist and logger operate identically in either mode.

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
3. Click **Start Logging**. The CSV header contains `timestamp` plus every signal currently present in the watchlist.
4. Click **Stop Logging** to finish and close the file.

If the file cannot be opened, an error message appears and logging does not start.

## Legacy Script
The previous `ecu-test.py` script is still present for reference but is not used by the new GUI. It may require additional dependencies or cleanup before running.
