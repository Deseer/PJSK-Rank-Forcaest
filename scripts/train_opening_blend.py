"""Build the opening_blend artifact from finished normal and World Link boards."""
import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import forecast_ops  # noqa: E402
import opening_blend as ob  # noqa: E402


def collect_lines(config):
    os.environ.update(forecast_ops.env_from_config(config))
    import train_baseline as module
    module = importlib.reload(module)
    data_root = Path(config["data_root"])
    meta = module.load_event_meta(data_root, module.load_finished_event_ids())
    contexts = module.load_event_feature_contexts()
    normalized = module.load_normalized_rank_timeseries_event_keys(data_root)
    legacy = module.load_legacy_event_files(data_root, normalized)
    ranks = set(module.COMMON_RANKS) | {int(r) for r in config.get("event_tracker", {}).get("ranks", [])}
    horizon = (ob.GRID_HOURS[-1] + 1) * 3600
    lines = {}

    def add(region, event_id, char, rank, t, score):
        em = meta.get((region, event_id))
        if em is None or score < 0:
            return
        if em.series_type == "normal":
            if char:
                return
            kind, (start, end) = ob.NORMAL, (em.start_timestamp, em.end_timestamp)
        elif em.series_type == "world_bloom":
            if len(em.chapter_windows) <= 1:
                return  # finales have no earlier counterpart to learn from yet
            if char:
                window = em.chapter_windows.get(int(char))
                if not window:
                    return
                kind, (start, end) = ob.CHAPTER, window
            else:
                kind, (start, end) = ob.TOTAL, (em.start_timestamp, em.end_timestamp)
        else:
            return
        if t < start:
            return
        key = (region, event_id, char, rank)
        line = lines.get(key)
        if line is None:
            ctx = contexts.get((region, event_id))
            line = lines[key] = {"kind": kind, "start": start, "end": end, "chapters": len(em.chapter_windows),
                                 "condition": ctx.condition if ctx else "other",
                                 "units": list(ctx.unit_labels) if ctx else [], "points": {}, "final": None}
        if line["final"] is None or t >= line["final"][0]:
            line["final"] = (t, score)
        if t - start <= horizon:
            # Sources are not chronological; keep the latest value per slot.
            slot = ob.point_slot(t - start)
            old = line["points"].get(slot)
            if old is None or t >= old[0]:
                line["points"][slot] = (t, score)

    series = frozenset({"normal", "world_bloom"})
    history_dir = Path(config.get("event_tracker", {}).get("history_dir", str(SCRIPT_DIR.parent / "live_history")))
    for region in module.REGIONS:
        for row in module.iter_all_rank_timeseries_rows(data_root, region, ranks, legacy, series):
            add(region, row["event_id"], int(row["character_id"] or 0), row["rank"], row["timestamp"], row["score"])
        # World Link total boards only exist in the tracker's total histories.
        for path in history_dir.glob(f"{region}_*_total_history.tsv"):
            event_id = int(path.name.split("_")[1])
            em = meta.get((region, event_id))
            if em is None or em.series_type != "world_bloom":
                continue
            with path.open() as handle:
                for raw in handle:
                    parts = raw.rstrip("\n").split("\t")
                    if len(parts) == 3 and int(parts[0]) in ranks:
                        add(region, event_id, 0, int(parts[0]), int(parts[1]), int(parts[2]))
        print("scanned", region, flush=True)
    out = []
    for (region, event_id, char, rank), line in lines.items():
        final_t, final = line["final"]
        if final_t < line["end"] - 120 or final <= 0:
            continue
        duration = line["end"] - line["start"]
        item = {"r": region, "e": event_id, "k": rank, "h": char, "t": line["kind"], "n": line["chapters"],
                "s": line["start"], "d": duration, "c": line["condition"], "u": line["units"], "f": final,
                "v": ob.curve_from_points(line["points"].values(), line["start"], duration)}
        if line["kind"] == ob.NORMAL:
            item["o"] = ob.checkpoint_observations(line["points"].values(), line["start"], duration)
        out.append(item)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    lines = collect_lines(config)
    beta = ob.fit_cross_server_beta(lines)
    coefficients = ob.fit_coefficients(lines, beta=beta)
    kinds = {kind: len({(v["r"], v["e"]) for v in lines if v["t"] == kind}) for kind in (ob.NORMAL, ob.CHAPTER, ob.TOTAL)}
    artifact = {"version": ob.VERSION, "built_at": int(time.time()), "grid_hours": list(ob.GRID_HOURS),
                "cross_server_beta": beta, "coefficients": coefficients, "lines": lines,
                "events": len({(v["r"], v["e"]) for v in lines}), "events_by_kind": kinds}
    if len(coefficients) < len(ob.CHECKPOINTS) // 2:
        raise SystemExit(f"too few fitted checkpoints: {sorted(coefficients)}")
    ob.write_atomic(args.output, artifact)
    print(json.dumps({"lines": len(lines), "events_by_kind": kinds, "cross_server_beta": beta,
                      "coefficients": coefficients}), flush=True)


if __name__ == "__main__":
    main()
