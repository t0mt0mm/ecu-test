# ECU Test GUI MVP

This repository contains a minimal PyQt5 application that demonstrates ECU control features with a dual-backend architecture.

## Features
- PyQt5 GUI with Dummy and Real backend selector
- Five high-side outputs (HS1–HS5) with enable toggle and PWM slider
- Five mixed inputs (digital/analog) with live updates at 10 Hz
- Dummy backend simulates first-order current response with noise and dynamic inputs
- CSV logging for all ten signals with ISO timestamps and configurable rate
- Real backend stub prepared for future CAN implementation

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
- **Real Backend**: Placeholder that maintains state locally. Extend the methods in `RealBackend` to integrate python-can, cantools, and your DBC file.

Switch between backends using the dropdown at the top of the main window. The Real backend currently does not communicate with hardware.

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
