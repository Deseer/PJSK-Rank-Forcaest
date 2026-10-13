#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


DEFAULT_CACHE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "cache"
    / "world_bloom_rank_timeseries_cn_jp_100_1000.csv.gz"
)
DEFAULT_OUTPUT_JSON = Path(__file__).resolve().parents[1] / "results" / "wl_character_heat_validation.json"
DEFAULT_OUTPUT_TSV = Path(__file__).resolve().parents[1] / "results" / "wl_character_heat_validation.tsv"
JP_WL2_EVENTS = frozenset({163, 167, 170, 171, 176, 179})
CN_WL1_EVENTS = frozenset({112, 118, 124, 130, 137, 140})
VS_CHARACTER_IDS = frozenset({21, 22, 23, 24, 25, 26})
MASTERDATA_ROOT = Path("/Users/deseer/PycharmProjects/haruki-sekai-master/master")


def parse_rank_weights(value: str) -> dict[int, float]:
    weights: dict[int, float] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        rank, weight = item.split(":", 1)
        weights[int(rank.strip())] = float(weight.strip())
    if not weights:
        raise ValueError("at least one rank weight is required")
    return weights


def load_rows(path: Path) -> list[dict[str, int | str]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({
                "region": row["region"],
                "event_id": int(row["event_id"]),
                "character_id": int(row["character_id"]),
                "rank": int(row["rank"]),
                "timestamp": int(row["timestamp"]),
                "score": int(row["score"]),
            })
    return rows


def load_character_meta() -> dict[int, dict[str, str]]:
    meta: dict[int, dict[str, str]] = {}
    for row in json.loads((MASTERDATA_ROOT / "gameCharacters.json").read_text(encoding="utf-8")):
        character_id = int(row["id"])
        name = f"{row.get('firstName') or ''}{row.get('givenName') or ''}".strip()
        meta[character_id] = {
            "name": name or str(character_id),
            "unit": row.get("unit") or "none",
        }
    return meta


def pearson(xs: list[float], ys: list[float]) -> float:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((value - mean_x) ** 2 for value in xs)
    var_y = sum((value - mean_y) ** 2 for value in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return float("nan")
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / math.sqrt(var_x * var_y)


def ordinal_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(values):
        end = index
        while end + 1 < len(values) and values[order[end + 1]] == values[order[index]]:
            end += 1
        rank = (index + end) / 2.0 + 1.0
        for rank_index in range(index, end + 1):
            ranks[order[rank_index]] = rank
        index = end + 1
    return ranks


def coefficient_metrics(rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    xs = [float(row["jp_coef"]) for row in rows]
    ys = [float(row["cn_coef"]) for row in rows]
    return {
        "characters": len(rows),
        "pearson": pearson(xs, ys),
        "spearman": pearson(ordinal_ranks(xs), ordinal_ranks(ys)),
        "sign_same": sum((x >= 0.0) == (y >= 0.0) for x, y in zip(xs, ys, strict=True)) / len(rows),
        "mae_log_coef": sum(abs(y - x) for x, y in zip(xs, ys, strict=True)) / len(rows),
    }


def build_final_rows(rows: list[dict[str, int | str]]) -> dict[tuple[str, int, int, int], dict[str, int | str]]:
    finals: dict[tuple[str, int, int, int], dict[str, int | str]] = {}
    for row in rows:
        key = (str(row["region"]), int(row["event_id"]), int(row["character_id"]), int(row["rank"]))
        if key not in finals or int(row["timestamp"]) > int(finals[key]["timestamp"]):
            finals[key] = row
    return finals


def build_rank_residuals(
    finals: dict[tuple[str, int, int, int], dict[str, int | str]],
    rank_weights: dict[int, float],
) -> dict[tuple[str, int, int], list[float]]:
    by_event_rank: dict[tuple[str, int, int], list[dict[str, int | str]]] = defaultdict(list)
    for (region, event_id, _character_id, rank), row in finals.items():
        if rank not in rank_weights:
            continue
        if region == "jp" and event_id in JP_WL2_EVENTS:
            by_event_rank[(region, event_id, rank)].append(row)
        if region == "cn" and event_id in CN_WL1_EVENTS:
            by_event_rank[(region, event_id, rank)].append(row)

    residuals: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for (region, _event_id, rank), event_rows in by_event_rank.items():
        event_log_scores = [math.log(max(int(row["score"]), 1)) for row in event_rows]
        event_median = statistics.median(event_log_scores)
        for row in event_rows:
            character_id = int(row["character_id"])
            residuals[(region, character_id, rank)].append(
                math.log(max(int(row["score"]), 1)) - event_median
            )
    return residuals


def blend_coefficients(
    residuals: dict[tuple[str, int, int], list[float]],
    rank_weights: dict[int, float],
) -> dict[tuple[str, int], float]:
    coefficients: dict[tuple[str, int], float] = {}
    for region in ("jp", "cn"):
        character_ids = sorted({
            character_id
            for current_region, character_id, _rank in residuals
            if current_region == region
        })
        for character_id in character_ids:
            weighted_values = []
            weight_total = 0.0
            for rank, weight in rank_weights.items():
                values = residuals.get((region, character_id, rank), [])
                if not values:
                    continue
                weighted_values.append(statistics.mean(values) * weight)
                weight_total += weight
            if weight_total > 0.0:
                coefficients[(region, character_id)] = sum(weighted_values) / weight_total
    return coefficients


def build_character_rows(coefficients: dict[tuple[str, int], float]) -> list[dict[str, float | int | str]]:
    meta = load_character_meta()
    common_character_ids = sorted({
        character_id
        for region, character_id in coefficients
        if region == "jp"
    } & {
        character_id
        for region, character_id in coefficients
        if region == "cn"
    })
    rows: list[dict[str, float | int | str]] = []
    for character_id in common_character_ids:
        jp_coef = coefficients[("jp", character_id)]
        cn_coef = coefficients[("cn", character_id)]
        character_meta = meta.get(character_id, {"name": str(character_id), "unit": "none"})
        rows.append({
            "character_id": character_id,
            "name": character_meta["name"],
            "unit": character_meta["unit"],
            "jp_coef": jp_coef,
            "jp_ratio": math.exp(jp_coef),
            "cn_coef": cn_coef,
            "cn_ratio": math.exp(cn_coef),
            "delta": cn_coef - jp_coef,
        })
    for field in ("jp_coef", "cn_coef"):
        rank_field = field.replace("_coef", "_rank")
        for rank, row in enumerate(sorted(rows, key=lambda item: float(item[field]), reverse=True), 1):
            row[rank_field] = rank
    return rows


def summarize(rows: list[dict[str, float | int | str]], rank_weights: dict[int, float]) -> dict[str, object]:
    def filter_rows(kind: str) -> list[dict[str, float | int | str]]:
        if kind == "all":
            return rows
        if kind == "vs":
            return [row for row in rows if int(row["character_id"]) in VS_CHARACTER_IDS]
        if kind == "non_vs":
            return [row for row in rows if int(row["character_id"]) not in VS_CHARACTER_IDS]
        return [row for row in rows if row["unit"] == kind]

    groups = {
        name: coefficient_metrics(group_rows)
        for name in ("all", "non_vs", "vs", "light_sound", "idol", "street", "theme_park", "school_refusal")
        if len(group_rows := filter_rows(name)) >= 2
    }
    return {
        "definition": "coef = weighted mean of log(final_score/event_median_score) within each WL event",
        "rank_weights": rank_weights,
        "jp_events": sorted(JP_WL2_EVENTS),
        "cn_events": sorted(CN_WL1_EVENTS),
        "groups": groups,
        "top_jp": sorted(rows, key=lambda row: float(row["jp_coef"]), reverse=True)[:8],
        "top_cn": sorted(rows, key=lambda row: float(row["cn_coef"]), reverse=True)[:8],
        "largest_delta": sorted(rows, key=lambda row: abs(float(row["delta"])), reverse=True)[:8],
    }


def write_outputs(rows: list[dict[str, float | int | str]], summary: dict[str, object], output_json: Path, output_tsv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=[
                "character_id",
                "name",
                "unit",
                "jp_coef",
                "jp_ratio",
                "jp_rank",
                "cn_coef",
                "cn_ratio",
                "cn_rank",
                "delta",
            ],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda item: float(item["jp_coef"]), reverse=True):
            writer.writerow({
                key: f"{value:.6f}" if isinstance(value, float) else value
                for key, value in row.items()
            })


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate JP WL2 character heat coefficients against CN WL1.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--rank-weights", default="100:0.6,1000:0.4")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-tsv", type=Path, default=DEFAULT_OUTPUT_TSV)
    args = parser.parse_args()

    rank_weights = parse_rank_weights(args.rank_weights)
    finals = build_final_rows(load_rows(args.cache))
    residuals = build_rank_residuals(finals, rank_weights)
    coefficients = blend_coefficients(residuals, rank_weights)
    rows = build_character_rows(coefficients)
    summary = summarize(rows, rank_weights)
    write_outputs(rows, summary, args.output_json, args.output_tsv)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
