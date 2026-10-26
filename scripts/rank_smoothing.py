"""Shrink each rank's remaining growth toward its neighbours' on the normal total board.

Errors after the opening are mostly board-wide heat, so a single rank whose growth
log(prediction/current) departs from both neighbours is more likely noise than signal.
Replay of the main-model holdout events (cn/en/tw/kr 181-182, jp 220-221): 60-85%
6.26 -> 5.95, 85-100% 2.46 -> 2.29 mean |err|, better on 9 of 10 events in the last stage.
The weight ramps in from 30% to 40% progress so the layer never switches on as a step.
"""
import math
import statistics

VERSION = "rank_smoothing_v1"
RAMP = (0.30, 0.40)
WEIGHT = 0.5


def apply(rows):
    board = sorted((r for r in rows if r.get("series_type") == "normal" and int(r.get("character_id") or 0) == 0
                    and not r.get("synthetic_rank") and not r.get("event_finished_prediction")
                    and float(r.get("progress") or 0) > RAMP[0]
                    and float(r.get("current_score") or 0) > 0 and float(r.get("model_prediction") or 0) > 0),
                   key=lambda r: int(r["rank"]))
    if len(board) < 3:
        return 0
    growth = [math.log(float(r["model_prediction"]) / float(r["current_score"])) for r in board]
    for i, row in enumerate(board):
        neighbour = statistics.median(growth[max(0, i - 1):i + 2])
        weight = WEIGHT * min(1.0, (float(row["progress"]) - RAMP[0]) / (RAMP[1] - RAMP[0]))
        value = float(row["current_score"]) * math.exp((1 - weight) * growth[i] + weight * neighbour)
        row["rank_smoothing_input_prediction"] = row["model_prediction"]
        row["model_prediction"] = row["pressure_adjusted_prediction"] = max(float(row["current_score"]), value)
    # Synthetic and finished rows keep their values; restore the rank order around them.
    floor = 0.0
    for row in sorted((r for r in rows if int(r.get("character_id") or 0) == 0 and not r.get("event_finished_prediction")),
                      key=lambda r: int(r["rank"]), reverse=True):
        if float(row["model_prediction"]) < floor:
            row["model_prediction"] = row["pressure_adjusted_prediction"] = floor
        floor = max(floor, float(row["model_prediction"]))
    for row in board:
        end = row.get("end_timestamp")
        hours = max(0.0, (float(end) - float(row["current_timestamp"])) / 3600) if end and row.get("current_timestamp") else 0.0
        row["forecast_required_pace"] = (float(row["model_prediction"]) - float(row["current_score"])) / hours if hours else None
    return len(board)
