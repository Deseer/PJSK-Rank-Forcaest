"""WL output circuit breaker. Bounds calibration, not observed score growth."""
import math


def reference(row):
    current = float(row['current_score'])
    values = [float(row.get(k) or current) for k in
              ('shape_first_prediction', 'hybrid_projection', 'pace_6h_projection')]
    if row.get("opening_prior_applied"):
        values.append(float(row.get("opening_prior_score") or current))
    return max([current]+[v for v in values if math.isfinite(v)])


def protect(rows, previous=(), elapsed=None):
    old = {(r.get('character_id', 0), r['rank']): r for r in previous}
    groups = {}
    for r in rows:
        if r.get('series_type') not in ('world_bloom', 'world_bloom_total'):
            continue
        c = float(r['current_score'])
        base = reference(r)
        candidate = float(r['model_prediction'])
        # Maximum extra amplification is 25% of uncalibrated remaining growth.
        cap = c + 1.25 * (base-c)
        temporal = None
        prior = old.get((r.get('character_id', 0), r['rank']))
        if (prior and elapsed is not None and 0 < elapsed <= 1800
                and prior.get('end_timestamp') == r.get('end_timestamp')
                and prior.get('forecast_safety_policy') == 'wl_guard_v1'):
            # Actual pace/shape acceleration passes through, calibration-only jumps do not.
            ratio = max(1., base / max(reference(prior), 1.))
            # Compare with the previous chain value; the opening blend runs after this guard.
            previous_chain = float(prior.get('opening_blend_chain_prediction') or prior['model_prediction'])
            temporal = (previous_chain * ratio
                        * math.exp(math.log(1.03)*elapsed/300)
                        + max(0., c-float(prior['current_score'])))
            cap = min(cap, max(c, temporal))
        final = c if r.get('event_finished_prediction') else max(c, min(candidate, cap))
        if not math.isfinite(final):
            final = c
        r.update(forecast_safety_policy='wl_guard_v1', forecast_safety_input=candidate,
                 forecast_safety_reference=base, forecast_safety_cap=cap,
                 forecast_safety_temporal_cap=temporal, model_prediction=final,
                 pressure_adjusted_prediction=final)
        groups.setdefault(r.get('character_id', 0), []).append(r)
    # Downward rank reconciliation never propagates a bad tail prediction upward.
    for group in groups.values():
        ceiling = math.inf
        for r in sorted(group, key=lambda r:r['rank']):
            final = max(float(r['current_score']), min(r['model_prediction'], ceiling))
            r['forecast_safety_rank_clamped'] = final != r['model_prediction']
            r['model_prediction'] = r['pressure_adjusted_prediction'] = final
            r['forecast_safety_applied'] = final != r['forecast_safety_input']
            hours = max(0., (float(r.get('end_timestamp') or 0)
                             - float(r.get('current_timestamp') or 0))/3600)
            r['forecast_required_pace'] = (final-float(r['current_score']))/hours if hours else None
            r['forecast_input_required_pace'] = (r['forecast_safety_input']-float(r['current_score']))/hours if hours else None
            # Never mislabel an empirical envelope as a theoretical game limit.
            r['forecast_theoretical_pace_limit'] = None
            r['forecast_theoretical_limit_verified'] = False
            ceiling = final
