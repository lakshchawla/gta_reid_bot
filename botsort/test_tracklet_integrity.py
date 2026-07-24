"""
Standalone regression checks for botsort/tracklet_integrity.py.

Not pytest-based (matches this repo's standalone test style, see
botsort/test_gta.py) - run: `python3 -m botsort.test_tracklet_integrity`.

Two layers: pure detection unit tests (feed synthetic per-frame feature
sequences straight into a _TrackHistory, no BoT-SORT needed) and one
integration test driving a real BoTSORT instance to verify the live split
surgery doesn't corrupt tracker state.
"""

from __future__ import annotations

import numpy as np

from .basetrack import TrackState
from .bot_sort import BoTSORT
from .global_tracklet_association import GTA
from .tracklet_integrity import TrackletIntegrityChecker, _TrackHistory


def _unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def _make_checker(**kwargs):
    defaults = dict(min_samples_for_gmm=20, check_interval_frames=1,
                     min_run_length=5, bic_margin=10.0,
                     min_component_weight=0.2, min_separation=0.3)
    defaults.update(kwargs)
    return TrackletIntegrityChecker(**defaults)


def _fill_history(hist: _TrackHistory, samples):
    for frame_id, feat in samples:
        hist.append(frame_id, feat)


def test_clean_bimodal_sequence_fires_at_correct_changepoint() -> bool:
    """First 60 frames orbit identity A, next 60 orbit a well-separated
    identity B (a sustained single swap - the occlusion-crossing case this
    module exists to catch). Must fire, and localize the changepoint at
    frame 60 within a small tolerance."""
    rng = np.random.RandomState(0)
    checker = _make_checker()
    a = _unit(rng.randn(128))
    b = _unit(rng.randn(128))
    b = _unit(b - float(np.dot(b, a)) * a)   # orthogonalize: cos_dist(a, b) = 1.0

    hist = _TrackHistory(maxlen=1000)
    samples = [(f, _unit(a + rng.randn(128).astype(np.float32) * 0.03)) for f in range(0, 60)]
    samples += [(f, _unit(b + rng.randn(128).astype(np.float32) * 0.03)) for f in range(60, 120)]
    _fill_history(hist, samples)

    event = checker._check_bimodal(101, hist)
    ok = (event is not None and abs(event.changepoint_frame - 60) <= 5
          and event.pre_count > 40 and event.post_count > 40
          and event.separation > 0.3)
    print(f"[clean_bimodal_fires_at_correct_changepoint] "
          f"event={'None' if event is None else event.changepoint_frame} "
          f"(expect ~60) -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_single_mode_noise_does_not_fire() -> bool:
    """120 frames all around one identity with ordinary noise - must NOT be
    flagged as bimodal; a real tracker's appearance naturally drifts a bit
    over a long dwell and that alone can't be evidence of an identity swap."""
    rng = np.random.RandomState(1)
    checker = _make_checker()
    a = _unit(rng.randn(128))

    hist = _TrackHistory(maxlen=1000)
    samples = [(f, _unit(a + rng.randn(128).astype(np.float32) * 0.05)) for f in range(0, 120)]
    _fill_history(hist, samples)

    event = checker._check_bimodal(102, hist)
    ok = event is None
    print(f"[single_mode_noise_does_not_fire] event={'None' if event is None else 'fired'} "
          f"(expect None) -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_multi_flip_sequence_declines_to_split() -> bool:
    """Feature alternates A/B/A/B in blocks - genuinely two well-separated
    populations, but flickering rather than one clean swap. This module only
    handles the single-changepoint case; a multi-flip signal must decline
    rather than guess which switch is 'the' split."""
    rng = np.random.RandomState(2)
    checker = _make_checker()
    a = _unit(rng.randn(128))
    b = _unit(rng.randn(128))
    b = _unit(b - float(np.dot(b, a)) * a)

    hist = _TrackHistory(maxlen=1000)
    samples = []
    f = 0
    for block, base in enumerate([a, b, a, b]):
        for _ in range(30):
            samples.append((f, _unit(base + rng.randn(128).astype(np.float32) * 0.03)))
            f += 1
    _fill_history(hist, samples)

    event = checker._check_bimodal(103, hist)
    ok = event is None
    print(f"[multi_flip_sequence_declines_to_split] event={'None' if event is None else 'fired'} "
          f"(expect None - declines) -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_live_splice_on_real_botsort_instance() -> bool:
    """Drive a real BoTSORT.update() loop (not fakes) with a track whose
    embedding swaps mid-track while its bounding box barely moves (high IOU
    keeps BoT-SORT's own association on the same track_id - exactly the
    occlusion-crossing scenario this module targets). The swap is a moderate
    cosine distance (0.5), not a near-total flip: distinct enough for the GMM
    to eventually resolve two clusters, but under BoT-SORT's own
    appearance_veto_thresh (0.7 default) so its per-frame IOU-fused
    association doesn't already reject the match itself - a genuinely SILENT
    swap is the case this module exists for; if BoT-SORT's own veto already
    caught it, there'd be nothing left for this module to do. Confirms the
    split surgery leaves tracker.tracked_stracks/removed_stracks/id_assigner in
    a consistent state - not just that detection logic works in isolation."""
    rng = np.random.RandomState(3)
    a = _unit(rng.randn(256))
    orth = _unit(rng.randn(256))
    orth = _unit(orth - float(np.dot(orth, a)) * a)
    b = _unit(0.5 * a + np.sqrt(1 - 0.5 ** 2) * orth)   # cos_dist(a, b) = 0.5

    gta = GTA(solver_interval_frames=10_000, min_tracklet_len=1)   # never ticks; just need deactivate_track bookkeeping
    tracker = BoTSORT(cam_source=1, track_buffer=50, with_reid=True,
                       new_track_thresh=0.3, track_high_thresh=0.5, track_low_thresh=0.1,
                       registry=gta)
    checker = _make_checker(min_samples_for_gmm=20, check_interval_frames=1, min_run_length=5)

    def det(feat, conf=0.9):
        return {"bbox": np.array([100, 100, 50, 150], dtype=np.float32),
                "det_confidence": conf, "obj_meta": False, "reid_vector": feat}

    first_track_id = None
    split_events = []
    for f in range(1, 41):
        base = a if f <= 20 else b
        tracker.update([det(base + rng.randn(256).astype(np.float32) * 0.02)])
        if first_track_id is None and tracker.tracked_stracks:
            first_track_id = tracker.tracked_stracks[0].track_id
        gta.step(tracker, f)   # real pipeline calls this (via mct.update_batch) every tick too
        events = checker.observe(tracker, f, registry=gta)
        if events:
            split_events.extend(events)

    live_ids = [t.track_id for t in tracker.tracked_stracks]
    removed_ids = [t.track_id for t in tracker.removed_stracks]
    old_removed_correctly = (first_track_id in removed_ids
                              and any(t.track_id == first_track_id and t.state == TrackState.Removed
                                      for t in tracker.removed_stracks))
    # solver_interval_frames=10_000 means the tracklet never got a chance to
    # resolve an identity before the split - it's still sitting in the
    # pending pool. deactivate_track() must have marked it dead (closed
    # bookkeeping) without leaving a stale live claim behind.
    old_node_closed = (first_track_id in gta._pending and gta._pending[first_track_id].dead
                        and gta._pending[first_track_id].track_ref is None)

    ok = (len(split_events) == 1
          and len(live_ids) == 1
          and live_ids[0] != first_track_id
          and old_removed_correctly
          and old_node_closed)
    print(f"[live_splice_on_real_botsort_instance] splits={len(split_events)} "
          f"first_id={first_track_id} live_ids={live_ids} removed_ids={removed_ids} "
          f"old_node_closed={old_node_closed} -> {'PASS' if ok else 'FAIL'}")
    return ok


def run_all() -> bool:
    results = [
        test_clean_bimodal_sequence_fires_at_correct_changepoint(),
        test_single_mode_noise_does_not_fire(),
        test_multi_flip_sequence_declines_to_split(),
        test_live_splice_on_real_botsort_instance(),
    ]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    return all(results)


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_all() else 1)
