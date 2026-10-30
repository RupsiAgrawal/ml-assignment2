"""Aggregate ball-by-ball deliveries into one row per innings.

Target `high_tempo` = 1 when the innings run rate (runs per legal ball) is at or
above the median run rate of all innings, else 0.  The run rate itself, total
runs and the singles/twos counts are NOT given to the models, so the label is
not a direct formula of the inputs.

Usage: python prepare_data.py --src deliveries.csv --out innings_dataset.csv
"""
import argparse

import numpy as np
import pandas as pd


def build(src):
    d = pd.read_csv(src, low_memory=False)
    d["legal"] = (~d["extra_type"].isin(["wide", "no-ball"])).astype(int)
    d["is_four"] = (d["batsman_runs"] == 4).astype(int)
    d["is_six"] = (d["batsman_runs"] == 6).astype(int)
    d["is_dot"] = (d["total_runs"] == 0).astype(int)
    d["wicket"] = d["is_wicket"].astype(int)
    for k in ["wide", "no-ball", "bye", "leg-bye"]:
        d["x_" + k] = (d["extra_type"] == k).astype(int)
    for k in ["caught", "bowled", "lbw", "run out", "stumped"]:
        d["w_" + k] = ((d["dismissal_type"] == k)).astype(int)
    d["powerplay_ball"] = (d["over"] <= 6).astype(int)
    d["death_ball"] = (d["over"] >= 16).astype(int)

    g = d.groupby(["match_id", "innings"], sort=False)
    s = g.agg(
        batting_team=("batting_team", "first"),
        bowling_team=("bowling_team", "first"),
        balls_bowled=("delivery_id", "size"),
        legal_balls=("legal", "sum"),
        wickets=("wicket", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
        dot_balls=("is_dot", "sum"),
        wides=("x_wide", "sum"),
        no_balls=("x_no-ball", "sum"),
        byes=("x_bye", "sum"),
        leg_byes=("x_leg-bye", "sum"),
        caught=("w_caught", "sum"),
        bowled=("w_bowled", "sum"),
        run_outs=("w_run out", "sum"),
        stumped=("w_stumped", "sum"),
        lbw=("w_lbw", "sum"),
        powerplay_balls=("powerplay_ball", "sum"),
        death_balls=("death_ball", "sum"),
        distinct_bowlers=("bowler", "nunique"),
        distinct_batters=("striker", "nunique"),
        total_runs=("total_runs", "sum"),
    ).reset_index()

    s["boundary_pct"] = (s["fours"] + s["sixes"]) / s["balls_bowled"] * 100
    s["dot_ball_pct"] = s["dot_balls"] / s["balls_bowled"] * 100
    s["extras_pct"] = (s["wides"] + s["no_balls"] + s["byes"] + s["leg_byes"]) / s["balls_bowled"] * 100
    s["wickets_per_over"] = s["wickets"] / np.maximum(s["legal_balls"], 1) * 6

    rate = s["total_runs"] / np.maximum(s["legal_balls"], 1)
    s["high_tempo"] = (rate >= rate.median()).astype(int)

    keep = ["match_id", "innings", "batting_team", "bowling_team", "legal_balls", "wickets",
            "fours", "sixes", "boundary_pct", "dot_ball_pct", "wides", "no_balls", "byes",
            "leg_byes", "extras_pct", "caught", "bowled", "run_outs", "stumped", "lbw",
            "wickets_per_over", "powerplay_balls", "death_balls", "distinct_bowlers",
            "distinct_batters", "high_tempo"]
    return s[keep], rate.median()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="deliveries.csv")
    ap.add_argument("--out", default="innings_dataset.csv")
    a = ap.parse_args()
    out, med = build(a.src)
    out.to_csv(a.out, index=False)
    print(f"Saved {a.out}: {len(out)} innings, {out.shape[1] - 2} features (+match_id, target)")
    print(f"Median run rate used as cut-off: {med:.4f} runs per legal ball")
    print(out["high_tempo"].value_counts().to_dict())
