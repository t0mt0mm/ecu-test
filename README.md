# ECU Test GUI MVP

This repository contains a minimal PyQt5 application that demonstrates ECU control features with a dual-backend architecture.

## Features
- PyQt5 GUI with Dummy and Real backend selector
- Five high-side outputs (HS1–HS5) with enable toggle and PWM slider
- Five mixed inputs (digital/analog) with live updates at 10 Hz
- Dummy backend simulates first-order current response with noise and dynamic inputs
- CSV logging for all ten signals with ISO timestamps and configurable rate
- Real backend loads the provided DBC, opens a python-can bus, sends one write command, and decodes one feedback message

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

## Outputs and Inputs
- Enable or disable each high-side output with the checkbox and adjust the PWM using the slider.
- The live current is shown next to each output.
- Input signals update at 10 Hz. Digital inputs display `ON`/`OFF`, while analog inputs show numeric values.

## CSV Logging
1. Choose the destination file path and the sampling rate (1–50 Hz).
2. Click **Start Logging**. The CSV file includes the header `timestamp`, the currents for all five outputs, and the five input values.
3. Click **Stop Logging** to finish and close the file.

If the file cannot be opened, an error message appears and logging does not start.

## Legacy Script
The previous `ecu-test.py` script is still present for reference but is not used by the new GUI. It may require additional dependencies or cleanup before running.
