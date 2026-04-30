def build_curve_advice_labels_object(
    df_ref,
    df_me,
    track_length_m=5220.0,
    n_interp=4000,
):
    """
    Input:
    - df_ref: DataFrame REF della singola curva (gia ritagliato)
    - df_me: DataFrame ME della singola curva (gia ritagliato)

    Output:
    {
      "entry": "wider" / "aligned" / "tighter" / "slightly wider" / "slightly tighter",
      "apex":  "...",
      "exit":  "..."
    }

    NOTE:
    - NO GPS nei calcoli physical
    - Parametri fissi preset B_u_turn
    """
    import math
    import numpy as np
    import pandas as pd

    # Preset fisso B_u_turn
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

    def signed_offset_keypoint_local_normal(ref_x, ref_y, ref_pct, me_x, me_y, me_pct, idx_ref, pct_window=0.004):
        px = float(ref_x[idx_ref])
        py = float(ref_y[idx_ref])
        pct_ref = float(ref_pct[idx_ref])

        nx, ny = local_normal_from_curve(ref_x, ref_y, idx_ref, half_window=4)
        qx, qy, _ = best_projection_local_pct(px, py, me_x, me_y, me_pct, pct_ref, pct_window=pct_window)

        dx = qx - px
        dy = qy - py
        d_signed = dx * nx + dy * ny
        if abs(d_signed) < 1e-12:
            return 0.0
        return float(d_signed)

    def detect_keypoints_from_steer(steer, n, entry_threshold_factor, exit_threshold_factor, hold_samples):
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

    def classify_lateral(offset_m, turn_sign, dead_zone_m, slight_max_m):
        ao = abs(offset_m)
        if ao <= dead_zone_m:
            return "allineato"

        if turn_sign < 0:
            wider = offset_m > 0
        else:
            wider = offset_m < 0

        if ao <= slight_max_m:
            return "leggermente piu largo" if wider else "leggermente piu stretto"
        return "piu largo" if wider else "piu stretto"

    def to_en_label(label_it):
        m = {
            "allineato": "aligned",
            "leggermente piu largo": "slightly wider",
            "leggermente piu stretto": "slightly tighter",
            "piu largo": "wider",
            "piu stretto": "tighter",
        }
        return m.get(label_it, "aligned")

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

        return {"lap": lap, "x_phys": x, "y_phys": y, "heading": heading_final, "steer": steer}

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

        return {
            "pct": pct,
            "x_phys": interp_clean(turn["x_phys"]),
            "y_phys": interp_clean(turn["y_phys"]),
            "heading": interp_clean(turn["heading"]),
            "steer": interp_clean(turn["steer"]),
        }

    df_ref = standardize_df(df_ref)
    df_me = standardize_df(df_me)

    start_pct = float(df_ref["LapDistPct"].min())
    end_pct = float(df_ref["LapDistPct"].max())

    ref_turn = reconstruct_turn_no_gps(df_ref, start_pct, end_pct, track_length_m, CFG)
    me_turn = reconstruct_turn_no_gps(df_me, start_pct, end_pct, track_length_m, CFG)

    if ref_turn is None or me_turn is None:
        return {"entry": "insufficient data", "apex": "insufficient data", "exit": "insufficient data"}

    ref_i = interp_turn(ref_turn, start_pct, end_pct, n=n_interp)
    me_i = interp_turn(me_turn, start_pct, end_pct, n=n_interp)

    me_i["x_phys"], me_i["y_phys"] = align_similarity(
        ref_i["x_phys"], ref_i["y_phys"], me_i["x_phys"], me_i["y_phys"]
    )

    k_entry, k_apex, k_exit = detect_keypoints_from_steer(
        ref_i["steer"],
        len(ref_i["pct"]),
        entry_threshold_factor=CFG["entry_threshold_factor"],
        exit_threshold_factor=CFG["exit_threshold_factor"],
        hold_samples=CFG["hold_samples"],
    )

    n_ref = len(ref_i["pct"])
    k_entry = int(np.clip(k_entry, 0, n_ref - 1))
    k_apex = int(np.clip(k_apex, 0, n_ref - 1))
    k_exit = int(np.clip(k_exit, 0, n_ref - 1))

    if k_apex <= k_entry:
        k_apex = min(n_ref - 2, max(k_entry + 1, n_ref // 2))
    if k_exit <= k_apex:
        k_exit = min(n_ref - 1, k_apex + max(5, n_ref // 15))

    entry_shift = max(6, int(CFG["entry_shift_pct"] * n_ref))
    entry_shift = min(entry_shift, max(1, (k_apex - k_entry) // 2))
    k_entry_eval = min(k_entry + entry_shift, k_apex - 1)
    if k_entry_eval <= k_entry:
        k_entry_eval = k_entry

    s_apex = turn_sign_from_geometry(ref_i["x_phys"], ref_i["y_phys"], k_apex)
    if s_apex == 0:
        s_apex = turn_sign_local(ref_i["heading"], k_apex, steer=ref_i["steer"])

    p_entry = signed_offset_keypoint_local_normal(
        ref_i["x_phys"], ref_i["y_phys"], ref_i["pct"],
        me_i["x_phys"], me_i["y_phys"], me_i["pct"],
        k_entry_eval, pct_window=CFG["pct_window_entry"]
    )
    p_apex = signed_offset_keypoint_local_normal(
        ref_i["x_phys"], ref_i["y_phys"], ref_i["pct"],
        me_i["x_phys"], me_i["y_phys"], me_i["pct"],
        k_apex, pct_window=CFG["pct_window_apex"]
    )
    p_exit = signed_offset_keypoint_local_normal(
        ref_i["x_phys"], ref_i["y_phys"], ref_i["pct"],
        me_i["x_phys"], me_i["y_phys"], me_i["pct"],
        k_exit, pct_window=CFG["pct_window_exit"]
    )

    return {
        "entry": to_en_label(classify_lateral(p_entry, s_apex, CFG["dead_zone_m"], CFG["slight_max_m"])),
        "apex": to_en_label(classify_lateral(p_apex, s_apex, CFG["dead_zone_m"], CFG["slight_max_m"])),
        "exit": to_en_label(classify_lateral(p_exit, s_apex, CFG["dead_zone_m"], CFG["slight_max_m"])),
    }
