# traejctory-reconstruction-2026

This project reconstructs racing lines from iRacing telemetry and compares a target lap against a reference lap on a corner-by-corner basis.

iRacing does not provide a directly usable GPS racing line in the telemetry stream, so this project rebuilds the trajectory from physical telemetry channels and then compares that reconstructed line against GPS-derived trajectories for validation and visual inspection.

The output is:
 advice labels at every curve for `entry`, `apex`, and `exit`
- physical trajectory plots
- GPS vs physical comparison plots for each corner

## Why This Project is useful 

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

## Files
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


## Mathematical Formulation

The local arc-length increment is estimated from lap progress as:

$$
\Delta s_i = (p_i - p_{i-1})\,L
$$

The local time increment is approximated from distance and speed:

$$
\Delta t_i = \frac{\Delta s_i}{v_i}
$$

If absolute yaw is available, heading is modeled as:

$$
\psi_i^{(y)} = \mathrm{unwrap}(\mathrm{yaw}_i)
$$

If yaw rate is available, heading is reconstructed by trapezoidal integration:

$$
\psi_i^{(r)} = \psi_{i-1}^{(r)} + \frac{1}{2}\left(\dot{\psi}_{i-1} + \dot{\psi}_i\right)\Delta t_i
$$

When both estimates are available, the final heading is obtained by weighted fusion:

$$
\psi_i = w_y\,\psi_i^{(y)} + w_r\,\psi_i^{(r)}
$$

The midpoint heading between two consecutive samples is:

$$
\psi_i^{\text{mid}} = \frac{\psi_{i-1} + \psi_i}{2}
$$

The planar trajectory is then reconstructed as:

$$
x_i = x_{i-1} + \Delta s_i \cos\left(\psi_i^{\text{mid}}\right)
$$

$$
y_i = y_{i-1} + \Delta s_i \sin\left(\psi_i^{\text{mid}}\right)
$$

When GPS is available, latitude and longitude are projected to a local Cartesian frame as:

$$
x_i^{\mathrm{gps}} = R\,\lambda_i \cos(\phi_0)
$$

$$
y_i^{\mathrm{gps}} = R\,\phi_i
$$

To compare two trajectories, they are aligned geometrically. After centering:

$$
P_c = P - \bar{P}, \qquad Q_c = Q - \bar{Q}
$$

the cross-covariance matrix is:

$$
H = Q_c^\top P_c
$$

with singular value decomposition:

$$
H = U\Sigma V^\top
$$

The optimal planar rotation is:

$$
R_a = U V^\top
$$

and the aligned trajectory is:

$$
Q_{\mathrm{aligned}} = Q_c R_a + \bar{P}
$$

At a keypoint, the local tangent direction of the reference curve is approximated by:

$$
\mathbf{t}_i =
\frac{\mathbf{r}_{i+h} - \mathbf{r}_{i-h}}
{\left\lVert \mathbf{r}_{i+h} - \mathbf{r}_{i-h} \right\rVert}
$$

The corresponding local normal direction is:

$$
\mathbf{n}_i = (-t_{y,i},\, t_{x,i})
$$

If $\mathbf{p}_i$ is the reference keypoint and $\mathbf{q}_i$ is the locally projected point on the compared trajectory, the signed lateral offset is:

$$
d_i = (\mathbf{q}_i - \mathbf{p}_i)\cdot \mathbf{n}_i
$$

Turn direction is inferred from the sign of planar curvature:

$$
\kappa(s) \propto x'(s)y''(s) - y'(s)x''(s)
$$

The final line classification is threshold-based:

$$
|d_i| \le d_0 \;\Rightarrow\; \text{aligned}
$$

$$
d_0 < |d_i| \le d_1 \;\Rightarrow\; \text{slightly wider / slightly tighter}
$$

$$
|d_i| > d_1 \;\Rightarrow\; \text{wider / tighter}
$$

## Legend

- $i$: sample index
- $L$: total track length
- $p_i$: lap progress (`LapDistPct`) at sample $i$
- $\Delta s_i$: local arc-length increment
- $v_i$: vehicle speed at sample $i$
- $\Delta t_i$: local time increment
- $\psi_i$: final fused heading estimate
- $\psi_i^{(y)}$: heading estimated from yaw
- $\psi_i^{(r)}$: heading estimated from yaw-rate integration
- $\dot{\psi}_i$: yaw rate at sample $i$
- $w_y, w_r$: fusion weights for yaw-based and rate-based heading
- $x_i, y_i$: reconstructed planar physical coordinates
- $x_i^{\mathrm{gps}}, y_i^{\mathrm{gps}}$: projected GPS coordinates
- $R$: Earth radius
- $\phi_i$: latitude in radians
- $\lambda_i$: longitude in radians
- $\phi_0$: reference latitude used for longitude scaling
- $P, Q$: reference and compared trajectories
- $\bar{P}, \bar{Q}$: centroids of the two trajectories
- $P_c, Q_c$: centered trajectories
- $H$: cross-covariance matrix used for alignment
- $U, \Sigma, V^\top$: singular value decomposition terms
- $R_a$: optimal alignment rotation matrix
- $\mathbf{r}_i$: position vector at sample $i$
- $\mathbf{t}_i$: local tangent direction
- $\mathbf{n}_i$: local normal direction
- $\mathbf{p}_i$: reference keypoint
- $\mathbf{q}_i$: projected comparison point
- $d_i$: signed lateral offset
- $\kappa(s)$: signed planar curvature
- $d_0$: alignment threshold
- $d_1$: slight-deviation threshold
