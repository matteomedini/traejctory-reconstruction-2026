from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from curveloop import build_curve_advice_labels_object


# =========================
# USER CONFIG
# =========================
REF_FILE = "data/ref.csv"
ME_FILE = "data/me.csv"

TRACK_NAME = "Your Track Name"
TRACK_LENGTH_M = 5220.0

TURNS = [
    (0.038, 0.098, "First Corner"),
    (0.115, 0.175, "Williams"),
    (0.190, 0.267, "Moss S"),
    (0.280, 0.360, "Attwood"),
    (0.525, 0.580, "Hairpin"),
    (0.585, 0.619, "Revolver"),
    (0.622, 0.670, "Piper"),
    (0.705, 0.750, "Redman"),
    (0.755, 0.800, "Hobbs"),
    (0.820, 0.920, "Mike Knight and last corner"),
]

N_INTERP = 4000
SHOW_PLOTS = True
PLOT_CONTEXT_M = 20.0



CFG = {
    "min_points_turn": 25,
    "speed_win": 5,
    "ds_win": 3,
    "yaw_win": 7,
    "yr_win": 9,
    "rate_head_win": 7,
    "heading_win": 5,
    "w_yaw": 0.55,
    "w_rate": 0.45,
    "speed_floor": 0.8,
    "ds_max_p95": 2.8,
    "ds_max_p50": 4.2,
    "entry_threshold_factor": 0.23,
    "exit_threshold_factor": 0.32,
    "hold_samples": 6,
    "entry_shift_pct": 0.015,
    "pct_window_entry": 0.0070,
    "pct_window_apex": 0.0055,
    "pct_window_exit": 0.0060,
    "dead_zone_m": 0.18,
    "slight_max_m": 0.45,
}


def _fill_nan_1d(a):
    a = np.asarray(a, dtype=float).copy()
    if a.size == 0:
        return a
    m = np.isfinite(a)
    if np.all(m):
        return a
    if not np.any(m):
        return np.zeros_like(a)
    idx = np.arange(a.size)
    a[~m] = np.interp(idx[~m], idx[m], a[m])
    return a


def ensure_speed_ms(speed):
    s = _fill_nan_1d(speed)
    return s / 3.6 if np.nanmean(s) > 80 else s


def smooth_1d(x, win=9):
    x = _fill_nan_1d(x)
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3 or win < 3:
        return x
    if win % 2 == 0:
        win += 1
    if win > n:
        win = n if n % 2 == 1 else n - 1
    if win < 3:
        return x
    k = np.ones(win, dtype=float) / win
    return np.convolve(x, k, mode="same")


def normalize_yaw_rate(yaw_rate):
    y = _fill_nan_1d(yaw_rate)
    if y.size == 0:
        return y
    if np.nanmedian(np.abs(y)) > 3.5:
        return np.radians(y)
    return y


def normalize_yaw(yaw):
    y = _fill_nan_1d(yaw)
    if y.size == 0:
        return y
    if np.nanmedian(np.abs(y)) > 6.5:
        y = np.radians(y)
    return np.unwrap(y)


def gps_to_xy(lat_deg, lon_deg, base_lat_deg):
    R = 6378137.0
    lat_r = np.radians(np.asarray(lat_deg, dtype=float))
    lon_r = np.radians(np.asarray(lon_deg, dtype=float))
    x = lon_r * R * math.cos(math.radians(float(base_lat_deg)))
    y = lat_r * R
    x -= x[0]
    y -= y[0]
    return x, y


def align_similarity(x_ref, y_ref, x_me, y_me):
    P = np.stack([_fill_nan_1d(x_ref), _fill_nan_1d(y_ref)], axis=1)
    Q = np.stack([_fill_nan_1d(x_me), _fill_nan_1d(y_me)], axis=1)

    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)

    H = Qc.T @ Pc
    if np.linalg.matrix_rank(H) < 2:
        t = P.mean(axis=0) - Q.mean(axis=0)
        Qa = Q + t
        return Qa[:, 0], Qa[:, 1]

    U, _, Vt = np.linalg.svd(H)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt

    Qa = Qc @ Rm
    t = P.mean(axis=0) - Qa.mean(axis=0)
    Qa = Qa + t
    return Qa[:, 0], Qa[:, 1]


def local_normal_from_curve(x, y, idx, half_window=4):
    i0 = max(0, idx - half_window)
    i1 = min(len(x) - 1, idx + half_window)

    tx = float(x[i1] - x[i0])
    ty = float(y[i1] - y[i0])

    nt = (tx * tx + ty * ty) ** 0.5
    if nt < 1e-12:
        dx = np.gradient(x)
        dy = np.gradient(y)
        tx = float(dx[idx])
        ty = float(dy[idx])
        nt = (tx * tx + ty * ty) ** 0.5
        if nt < 1e-12:
            return 0.0, 1.0

    tx /= nt
    ty /= nt
    return -ty, tx


def project_point_to_segment(px, py, ax, ay, bx, by):
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay

    vv = vx * vx + vy * vy
    if vv < 1e-12:
        t = 0.0
    else:
        t = (wx * vx + wy * vy) / vv
        t = max(0.0, min(1.0, t))

    qx = ax + t * vx
    qy = ay + t * vy
    d2 = (qx - px) ** 2 + (qy - py) ** 2
    return qx, qy, d2


def best_projection_local_pct(px, py, me_x, me_y, me_pct, pct_ref, pct_window=0.004, min_pts=20):
    me_x = _fill_nan_1d(me_x)
    me_y = _fill_nan_1d(me_y)
    me_pct = _fill_nan_1d(me_pct)

    start = int(np.searchsorted(me_pct, pct_ref - pct_window, side="left"))
    end = int(np.searchsorted(me_pct, pct_ref + pct_window, side="right"))

    if end - start < min_pts:
        pad = (min_pts - (end - start)) // 2 + 1
        start = max(0, start - pad)
        end = min(len(me_pct), end + pad)

    if end - start < 2:
        start = 0
        end = len(me_pct)

    seg_start = max(0, start - 1)
    seg_end = min(len(me_x) - 2, end - 1)

    best_d2 = float("inf")
    best_qx = float(me_x[0])
    best_qy = float(me_y[0])

    for i in range(seg_start, seg_end + 1):
        qx, qy, d2 = project_point_to_segment(
            px, py,
            float(me_x[i]), float(me_y[i]),
            float(me_x[i + 1]), float(me_y[i + 1]),
        )
        if d2 < best_d2:
            best_d2 = d2
            best_qx, best_qy = qx, qy

    return best_qx, best_qy, float(np.sqrt(best_d2))


def safe_interp_xy(pct0, pct, x, y):
    pct = np.asarray(pct, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(pct) & np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(m) < 2:
        return (np.nan, np.nan)

    pp = pct[m]
    xx = x[m]
    yy = y[m]

    order = np.argsort(pp)
    pp = pp[order]
    xx = xx[order]
    yy = yy[order]

    pp_u, idx_u = np.unique(pp, return_index=True)
    xx_u = xx[idx_u]
    yy_u = yy[idx_u]

    if len(pp_u) == 1:
        return (float(xx_u[0]), float(yy_u[0]))

    p = float(np.clip(pct0, pp_u[0], pp_u[-1]))
    return (float(np.interp(p, pp_u, xx_u)), float(np.interp(p, pp_u, yy_u)))


def keypoint_xy_on_curve(pct0, pct_curve, x_curve, y_curve, pct_window=0.010):
    pct_curve = np.asarray(pct_curve, dtype=float)
    x_curve = np.asarray(x_curve, dtype=float)
    y_curve = np.asarray(y_curve, dtype=float)

    px, py = safe_interp_xy(pct0, pct_curve, x_curve, y_curve)
    if np.isfinite(px) and np.isfinite(py):
        qx, qy, _ = best_projection_local_pct(
            px, py,
            x_curve, y_curve, pct_curve,
            float(pct0),
            pct_window=pct_window,
            min_pts=30,
        )
        return float(qx), float(qy)

    m = np.isfinite(pct_curve) & np.isfinite(x_curve) & np.isfinite(y_curve)
    if np.count_nonzero(m) == 0:
        return (np.nan, np.nan)

    pp = pct_curve[m]
    xx = x_curve[m]
    yy = y_curve[m]
    i = int(np.argmin(np.abs(pp - pct0)))
    return float(xx[i]), float(yy[i])


def expanded_axis_limits(curves, margin_ratio=0.35, min_neg_y_ratio=0.20):
    xs = []
    ys = []

    for x, y in curves:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if np.any(m):
            xs.append(x[m])
            ys.append(y[m])

    if not xs:
        return (-1.0, 1.0), (-1.0, 1.0)

    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)

    xmin, xmax = float(np.min(x_all)), float(np.max(x_all))
    ymin, ymax = float(np.min(y_all)), float(np.max(y_all))

    span = max(xmax - xmin, ymax - ymin, 1.0)
    half = 0.5 * span * (1.0 + 2.0 * margin_ratio)

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    xlim = (cx - half, cx + half)
    ylim = (cy - half, cy + half)

    min_neg = -min_neg_y_ratio * (2.0 * half)
    if ylim[0] > min_neg:
        shift = ylim[0] - min_neg
        ylim = (ylim[0] - shift, ylim[1] - shift)

    return xlim, ylim


def scatter_if_finite(ax, pt, **kwargs):
    if pt is None:
        return
    x, y = pt
    if np.isfinite(x) and np.isfinite(y):
        ax.scatter([x], [y], **kwargs)


def detect_keypoints_from_steer(
    steer, n, entry_threshold_factor=0.26, exit_threshold_factor=0.35, hold_samples=8
):
    sa = np.abs(np.asarray(steer, dtype=float))
    sa = np.nan_to_num(sa, nan=0.0, posinf=0.0, neginf=0.0)

    if len(sa) == 0:
        return n // 6, n // 2, 5 * n // 6

    p95 = float(np.percentile(sa, 95))
    thr_entry = max(entry_threshold_factor * p95, 0.01)
    thr_exit = max(exit_threshold_factor * p95, 0.01)

    active = np.where(sa >= thr_entry)[0]
    if len(active) < 8:
        return n // 6, n // 2, 5 * n // 6

    entry = int(active[0])
    apex = entry + int(np.argmax(sa[entry:]))

    exit_ = n - 1
    end_scan = max(apex + 1, n - hold_samples)
    found = False
    for i in range(apex, end_scan):
        if np.all(sa[i:i + hold_samples] <= thr_exit):
            exit_ = i
            found = True
            break

    if not found:
        exit_ = min(n - 1, apex + max(10, n // 10))

    if exit_ <= apex:
        exit_ = min(n - 1, apex + 5)

    return entry, apex, exit_


def turn_sign_from_geometry(x, y, idx, half_window=12):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    a = max(2, idx - half_window)
    b = min(len(x) - 3, idx + half_window)
    if b <= a:
        return 0

    xx = x[a:b + 1]
    yy = y[a:b + 1]
    dx = np.gradient(xx)
    dy = np.gradient(yy)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    kappa_sign = dx * ddy - dy * ddx
    m = float(np.nanmedian(kappa_sign))
    if not np.isfinite(m) or abs(m) < 1e-9:
        return 0
    return 1 if m > 0 else -1


def turn_sign_local(heading, idx, steer=None, half_window=20):
    h = np.asarray(heading, dtype=float)
    a = max(1, idx - half_window)
    b = min(len(h) - 1, idx + half_window)
    if b <= a:
        m = 0.0
    else:
        m = float(np.nanmean(np.gradient(h[a:b + 1])))

    if abs(m) < 1e-8 and steer is not None:
        s = np.asarray(steer, dtype=float)
        a2 = max(0, idx - half_window)
        b2 = min(len(s), idx + half_window + 1)
        m = float(np.nanmean(s[a2:b2]))

    return -1 if m < 0 else 1


def standardize_df(df):
    out = df.copy()

    ren = {
        "SpeedGps": "Speed",
        "speed": "Speed",
        "latitude": "Lat",
        "longitude": "Lon",
    }
    for s, d in ren.items():
        if s in out.columns and d not in out.columns:
            out.rename(columns={s: d}, inplace=True)

    req = ["LapDistPct", "Speed"]
    for c in req:
        if c not in out.columns:
            raise ValueError(f"Missing required column: {c}")

    if "Yaw" not in out.columns and "YawRate" not in out.columns:
        raise ValueError("Need Yaw or YawRate")

    out = out.sort_values("LapDistPct").reset_index(drop=True)
    return out


def reconstruct_turn_no_gps(df, start, end, track_length_m, cfg):
    sub = df[(df["LapDistPct"] >= start) & (df["LapDistPct"] <= end)].copy()
    if len(sub) < cfg["min_points_turn"]:
        return None

    lap = np.asarray(sub["LapDistPct"], dtype=float)
    lap = np.clip(lap, 0.0, 1.0)
    lap = np.maximum.accumulate(lap)
    for i in range(1, len(lap)):
        if lap[i] <= lap[i - 1]:
            lap[i] = lap[i - 1] + 1e-9

    speed = ensure_speed_ms(sub["Speed"].to_numpy(dtype=float))
    speed = np.clip(speed, 0.0, None)
    speed = smooth_1d(speed, win=cfg["speed_win"])
    speed = np.maximum(speed, cfg["speed_floor"])

    raw_ds = np.diff(lap, prepend=lap[0]) * float(track_length_m)
    if len(raw_ds) > 6:
        p50 = float(np.nanpercentile(raw_ds[1:], 50))
        p95 = float(np.nanpercentile(raw_ds[1:], 95))
        ds_max = max(cfg["ds_max_p95"] * p95, cfg["ds_max_p50"] * p50, 1e-3)
    else:
        ds_max = 1e-3

    ds = np.clip(raw_ds, 0.0, ds_max)
    ds = smooth_1d(ds, win=cfg["ds_win"])
    ds = np.clip(ds, 0.0, None)
    ds[0] = 0.0

    target_len = float(max((lap[-1] - lap[0]) * track_length_m, 1e-6))
    cur_len = float(np.sum(ds[1:]))
    if cur_len > 1e-9:
        ds[1:] *= (target_len / cur_len)

    dt_est = ds / np.maximum(speed, cfg["speed_floor"])

    heading_yaw = None
    heading_rate = None

    if "Yaw" in sub.columns:
        heading_yaw = normalize_yaw(sub["Yaw"].to_numpy(dtype=float))
        heading_yaw = smooth_1d(heading_yaw, win=cfg["yaw_win"])

    if "YawRate" in sub.columns:
        yr = normalize_yaw_rate(sub["YawRate"].to_numpy(dtype=float))
        yr = smooth_1d(yr, win=cfg["yr_win"])

        heading_rate = np.zeros(len(sub), dtype=float)
        for i in range(1, len(sub)):
            heading_rate[i] = heading_rate[i - 1] + 0.5 * (yr[i - 1] + yr[i]) * dt_est[i]
        heading_rate = np.unwrap(heading_rate)
        heading_rate = smooth_1d(heading_rate, win=cfg["rate_head_win"])

    if heading_yaw is not None and heading_rate is not None:
        heading_rate = heading_rate + (heading_yaw[0] - heading_rate[0])
        heading = np.unwrap(cfg["w_yaw"] * heading_yaw + cfg["w_rate"] * heading_rate)
    elif heading_yaw is not None:
        heading = heading_yaw
    elif heading_rate is not None:
        heading = heading_rate
    else:
        return None

    heading_final = smooth_1d(heading, win=cfg["heading_win"])

    x = np.zeros(len(sub), dtype=float)
    y = np.zeros(len(sub), dtype=float)
    for i in range(1, len(sub)):
        h_mid = 0.5 * (heading_final[i - 1] + heading_final[i])
        x[i] = x[i - 1] + ds[i] * math.cos(h_mid)
        y[i] = y[i - 1] + ds[i] * math.sin(h_mid)

    x -= x[0]
    y -= y[0]

    steer_col = "SteeringWheelAngle" if "SteeringWheelAngle" in sub.columns else (
        "SteeringAngle" if "SteeringAngle" in sub.columns else None
    )
    if steer_col is None:
        steer = np.gradient(heading_final)
    else:
        steer = _fill_nan_1d(sub[steer_col].to_numpy(dtype=float))

    return {
        "lap": lap,
        "x_phys": x,
        "y_phys": y,
        "heading": heading_final,
        "steer": steer,
    }


def reconstruct_gps_turn(df, start, end, base_lat):
    if "Lat" not in df.columns or "Lon" not in df.columns:
        return None

    sub = df[(df["LapDistPct"] >= start) & (df["LapDistPct"] <= end)].copy()
    if len(sub) < 2:
        return None

    lap = sub["LapDistPct"].to_numpy(dtype=float)
    gx, gy = gps_to_xy(
        sub["Lat"].to_numpy(dtype=float),
        sub["Lon"].to_numpy(dtype=float),
        base_lat,
    )
    return {"lap": lap, "x_gps": gx, "y_gps": gy}


def interp_turn(turn, start, end, n=800):
    pct = np.linspace(start, end, n)
    lap = np.asarray(turn["lap"], dtype=float)

    def interp_clean(v):
        v = np.asarray(v, dtype=float)
        m = np.isfinite(lap) & np.isfinite(v)
        if np.count_nonzero(m) < 2:
            return np.full_like(pct, np.nan, dtype=float)

        ll = lap[m]
        vv = v[m]

        order = np.argsort(ll)
        ll = ll[order]
        vv = vv[order]

        ll_u, idx_u = np.unique(ll, return_index=True)
        vv_u = vv[idx_u]

        if len(ll_u) == 1:
            return np.full_like(pct, vv_u[0], dtype=float)

        return np.interp(pct, ll_u, vv_u)

    out = {"pct": pct}
    for k in turn:
        if k != "lap":
            out[k] = interp_clean(turn[k])
    return out


def main():
    ref_path = Path(REF_FILE)
    me_path = Path(ME_FILE)

    if not ref_path.exists():
        raise FileNotFoundError(f"REF non trovato: {REF_FILE}")
    if not me_path.exists():
        raise FileNotFoundError(f"ME non trovato: {ME_FILE}")

    df_ref = standardize_df(pd.read_csv(ref_path))
    df_me = standardize_df(pd.read_csv(me_path))

    gps_available = all(c in df_ref.columns for c in ["Lat", "Lon"]) and all(
        c in df_me.columns for c in ["Lat", "Lon"]
    )
    base_lat_global = float(df_ref["Lat"].iloc[0]) if gps_available else None

    print(f"\nTrack: {TRACK_NAME}")
    print(f"REF file: {REF_FILE}")
    print(f"ME  file: {ME_FILE}")
    print(f"GPS available: {gps_available}")
    print("")

    advice_by_turn = {}

    for start, end, name in TURNS:
        df_ref_curve = df_ref[(df_ref["LapDistPct"] >= start) & (df_ref["LapDistPct"] <= end)].copy()
        df_me_curve = df_me[(df_me["LapDistPct"] >= start) & (df_me["LapDistPct"] <= end)].copy()

        advice = build_curve_advice_labels_object(
            df_ref=df_ref_curve,
            df_me=df_me_curve,
            track_length_m=TRACK_LENGTH_M,
            n_interp=N_INTERP,
        )
        advice_by_turn[name] = advice

        print(name)
        print("  entry:", advice["entry"])
        print("  apex :", advice["apex"])
        print("  exit :", advice["exit"])
        print("")

        if not SHOW_PLOTS:
            continue

        ctx_pct = PLOT_CONTEXT_M / float(TRACK_LENGTH_M)
        plot_start = max(0.0, start - ctx_pct)
        plot_end = end

        ref_turn = reconstruct_turn_no_gps(df_ref, plot_start, plot_end, TRACK_LENGTH_M, CFG)
        me_turn = reconstruct_turn_no_gps(df_me, plot_start, plot_end, TRACK_LENGTH_M, CFG)
        if ref_turn is None or me_turn is None:
            continue

        n_plot = int(max(N_INTERP, N_INTERP * (plot_end - plot_start) / max(end - start, 1e-6)))

        ref_p = interp_turn(ref_turn, plot_start, plot_end, n=n_plot)
        me_p = interp_turn(me_turn, plot_start, plot_end, n=n_plot)
        me_p["x_phys"], me_p["y_phys"] = align_similarity(
            ref_p["x_phys"], ref_p["y_phys"], me_p["x_phys"], me_p["y_phys"]
        )

        ref_turn_core = reconstruct_turn_no_gps(df_ref, start, end, TRACK_LENGTH_M, CFG)
        if ref_turn_core is None:
            continue
        ref_i = interp_turn(ref_turn_core, start, end, n=N_INTERP)

        k_entry, k_apex, k_exit = detect_keypoints_from_steer(
            ref_i["steer"],
            len(ref_i["pct"]),
            entry_threshold_factor=CFG["entry_threshold_factor"],
            exit_threshold_factor=CFG["exit_threshold_factor"],
            hold_samples=CFG["hold_samples"],
        )

        entry_shift = max(6, int(CFG["entry_shift_pct"] * len(ref_i["pct"])))
        k_entry_eval = min(k_entry + entry_shift, k_apex - 3)
        if k_entry_eval <= k_entry:
            k_entry_eval = k_entry

        e_pct = float(np.clip(ref_i["pct"][k_entry_eval], ref_p["pct"][0], ref_p["pct"][-1]))
        a_pct = float(np.clip(ref_i["pct"][k_apex], ref_p["pct"][0], ref_p["pct"][-1]))
        x_pct = float(np.clip(ref_i["pct"][k_exit], ref_p["pct"][0], ref_p["pct"][-1]))

        e_ref_phy = keypoint_xy_on_curve(e_pct, ref_p["pct"], ref_p["x_phys"], ref_p["y_phys"], pct_window=0.010)
        a_ref_phy = keypoint_xy_on_curve(a_pct, ref_p["pct"], ref_p["x_phys"], ref_p["y_phys"], pct_window=0.010)
        x_ref_phy = keypoint_xy_on_curve(x_pct, ref_p["pct"], ref_p["x_phys"], ref_p["y_phys"], pct_window=0.010)

        gps_ok = False
        if gps_available:
            ref_gps_turn = reconstruct_gps_turn(df_ref, plot_start, plot_end, base_lat_global)
            me_gps_turn = reconstruct_gps_turn(df_me, plot_start, plot_end, base_lat_global)

            if ref_gps_turn is not None and me_gps_turn is not None:
                gps_ok = True
                ref_g = interp_turn(ref_gps_turn, plot_start, plot_end, n=n_plot)
                me_g = interp_turn(me_gps_turn, plot_start, plot_end, n=n_plot)
                me_g["x_gps"], me_g["y_gps"] = align_similarity(
                    ref_g["x_gps"], ref_g["y_gps"], me_g["x_gps"], me_g["y_gps"]
                )

                e_ref_gps = keypoint_xy_on_curve(e_pct, ref_g["pct"], ref_g["x_gps"], ref_g["y_gps"], pct_window=0.010)
                a_ref_gps = keypoint_xy_on_curve(a_pct, ref_g["pct"], ref_g["x_gps"], ref_g["y_gps"], pct_window=0.010)
                x_ref_gps = keypoint_xy_on_curve(x_pct, ref_g["pct"], ref_g["x_gps"], ref_g["y_gps"], pct_window=0.010)

                fig, axes = plt.subplots(1, 2, figsize=(21, 8.5))

                axes[0].plot(ref_g["x_gps"], ref_g["y_gps"], label="REF GPS", color="tab:blue", linewidth=2.7)
                axes[0].plot(me_g["x_gps"], me_g["y_gps"], label="ME GPS", color="tab:orange", linewidth=2.7)
                scatter_if_finite(axes[0], e_ref_gps, color="green", s=105, label="Entry", zorder=6)
                scatter_if_finite(axes[0], a_ref_gps, color="red", s=105, label="Apex", zorder=6)
                scatter_if_finite(axes[0], x_ref_gps, color="purple", s=105, label="Exit", zorder=6)
                axes[0].set_title(f"{name} - GPS", fontsize=14)
                axes[0].grid(alpha=0.25)

                gps_pts_x = np.array([e_ref_gps[0], a_ref_gps[0], x_ref_gps[0]], dtype=float)
                gps_pts_y = np.array([e_ref_gps[1], a_ref_gps[1], x_ref_gps[1]], dtype=float)
                xlim_gps, ylim_gps = expanded_axis_limits(
                    [(ref_g["x_gps"], ref_g["y_gps"]), (me_g["x_gps"], me_g["y_gps"]), (gps_pts_x, gps_pts_y)],
                    margin_ratio=0.38,
                    min_neg_y_ratio=0.22,
                )
                axes[0].set_xlim(*xlim_gps)
                axes[0].set_ylim(*ylim_gps)
                axes[0].set_aspect("equal", adjustable="box")
                axes[0].legend(fontsize=10)

                axes[1].plot(ref_p["x_phys"], ref_p["y_phys"], label="REF Physical", color="tab:blue", linewidth=2.7)
                axes[1].plot(me_p["x_phys"], me_p["y_phys"], label="ME Physical", color="tab:orange", linewidth=2.7)
                scatter_if_finite(axes[1], e_ref_phy, color="green", s=105, label="Entry", zorder=6)
                scatter_if_finite(axes[1], a_ref_phy, color="red", s=105, label="Apex", zorder=6)
                scatter_if_finite(axes[1], x_ref_phy, color="purple", s=105, label="Exit", zorder=6)
                axes[1].set_title(f"{name} - Physical NO GPS", fontsize=14)
                axes[1].grid(alpha=0.25)

                phy_pts_x = np.array([e_ref_phy[0], a_ref_phy[0], x_ref_phy[0]], dtype=float)
                phy_pts_y = np.array([e_ref_phy[1], a_ref_phy[1], x_ref_phy[1]], dtype=float)
                xlim_phy, ylim_phy = expanded_axis_limits(
                    [(ref_p["x_phys"], ref_p["y_phys"]), (me_p["x_phys"], me_p["y_phys"]), (phy_pts_x, phy_pts_y)],
                    margin_ratio=0.38,
                    min_neg_y_ratio=0.22,
                )
                axes[1].set_xlim(*xlim_phy)
                axes[1].set_ylim(*ylim_phy)
                axes[1].set_aspect("equal", adjustable="box")
                axes[1].legend(fontsize=10)

                plt.tight_layout()
                plt.show()

        if not gps_ok:
            fig, ax = plt.subplots(1, 1, figsize=(10.5, 8.0))
            ax.plot(ref_p["x_phys"], ref_p["y_phys"], label="REF Physical", color="tab:blue", linewidth=2.7)
            ax.plot(me_p["x_phys"], me_p["y_phys"], label="ME Physical", color="tab:orange", linewidth=2.7)
            scatter_if_finite(ax, e_ref_phy, color="green", s=105, label="Entry", zorder=6)
            scatter_if_finite(ax, a_ref_phy, color="red", s=105, label="Apex", zorder=6)
            scatter_if_finite(ax, x_ref_phy, color="purple", s=105, label="Exit", zorder=6)
            ax.set_title(f"{name} - Physical NO GPS", fontsize=14)
            ax.grid(alpha=0.25)

            phy_pts_x = np.array([e_ref_phy[0], a_ref_phy[0], x_ref_phy[0]], dtype=float)
            phy_pts_y = np.array([e_ref_phy[1], a_ref_phy[1], x_ref_phy[1]], dtype=float)
            xlim_phy, ylim_phy = expanded_axis_limits(
                [(ref_p["x_phys"], ref_p["y_phys"]), (me_p["x_phys"], me_p["y_phys"]), (phy_pts_x, phy_pts_y)],
                margin_ratio=0.38,
                min_neg_y_ratio=0.22,
            )
            ax.set_xlim(*xlim_phy)
            ax.set_ylim(*ylim_phy)
            ax.set_aspect("equal", adjustable="box")
            ax.legend(fontsize=10)

            plt.tight_layout()
            plt.show()

    print("\nFinal advice dictionary:\n")
    print(json.dumps(advice_by_turn, indent=2))


if __name__ == "__main__":
    main()
