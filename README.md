# traejctory-reconstruction-2026

This project reconstructs racing lines from iRacing telemetry and compares a target lap against a reference lap on a corner-by-corner basis.

The core idea is simple: iRacing does not provide a directly usable GPS racing line in the telemetry stream, so this project rebuilds the trajectory from physical telemetry channels and then compares that reconstructed line against GPS-derived trajectories for validation and visual inspection.

The output is:
- corner-by-corner advice labels for `entry`, `apex`, and `exit`
- physical trajectory plots
- GPS vs physical comparison plots for each corner

## Why This Project 

In iRacing, telemetry is available in real time, but native position data is not exposed in a way that directly gives a clean racing line for driver comparison.

This project solves that by:
- reconstructing a physical trajectory from telemetry only
- using the reconstructed line to compare two drivers or two laps
- generating interpretable corner advice such as:
  - `aligned`
  - `slightly wider`
  - `slightly tighter`
  - `wider`
  - `tighter`

This makes it possible to perform racing line analysis and driver coaching even when GPS is not part of the simulator telemetry workflow.

## What Each File Does
- curveloop.py

contains the core reconstruction and corner advice logic
exposes the main function used to evaluate a single curve segment


- run_analysis.py

loads the two CSV files
loops over all defined turns
prints corner-by-corner advice
shows physical and GPS comparison plots


## Required Input Data
The code expects CSV telemetry exports containing at least these columns:

LapDistPct
Speed
Yaw or YawRate

Notes:
Lat and Lon are used for GPS comparison and plotting
the physical trajectory reconstruction does not use GPS in the core reconstruction logic


## Installation
Clone the repository and install dependencies:
pip install -r requirements.txt


## How to Use
Open run_analysis.py and configure these values:

REF_FILE = "data/ref.csv"

ME_FILE = "data/me.csv"

TRACK_NAME = "Your Track Name"

TRACK_LENGTH_M = track length (m)

TURNS = [ .. ]

You need to provide:

path to the reference CSV
path to the lap being compared
track length in meters
the list of turns as (start_pct, end_pct, "Turn Name")

Then run:
python run_analysis.py









