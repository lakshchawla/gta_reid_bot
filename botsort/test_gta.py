"""
Standalone regression checks for the tick-based tracklet matching, convergence,
and pruning logic in botsort/global_tracklet_association.py.

Not pytest-based (matches this repo's standalone test style, see
botsort/test_interval_solver.py) - run: `python3 -m botsort.test_gta`.

Drives GTA.step()/deactivate_track() directly with duck-typed fake
tracks/trackers (only the attributes step() actually reads) so tracklet
timing and feature similarity are exact and deterministic - what's under
test is purely GTA's tick-boundary decision policy, not Kalman/IOU
association noise.
"""

from __future__ import annotations

import numpy as np

from .global_tracklet_association import GTA


def _unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


class FakeTrack:
    def __init__(self, tid, feat, tracklet_len=100):
        self.track_id = tid
        self.smooth_feat = _unit(np.asarray(feat, dtype=np.float32))
        self.t_global_id = 0
        self.t_identity_since_frame = 0
        self.is_touching_edge = False
        self.tracklet_len = tracklet_len


class FakeTracker:
    def __init__(self, cam_source, tracks=()):
        self.cam_source = cam_source
        self.tracked_stracks = list(tracks)


def _make_gta(**kwargs):
    defaults = dict(
        window_frames=50, min_tracklet_len=1, sigma_feat=0.15,
        mu_gap_sec=2.0, sigma_gap_sec=20.0, link_threshold=-6.0,
        frame_rate=30.0,
    )
    defaults.update(kwargs)
    return GTA(**defaults)


def test_disjoint_similar_tracklets_converge_to_one_identity() -> bool:
    """Tracklet A (cam1, frames 1-30) closes; tracklet B (cam2, first seen at
    frame 90 - a 2.0s gap at 30fps, exactly at mu_gap_sec) has a near-identical
    feature. Once a tick fires they must converge into the same
    IdentityCluster - this is the whole point of tracklet-level GTA."""
    rng = np.random.RandomState(0)
    gta = _make_gta()
    alice = _unit(rng.randn(128))

    track_a = FakeTrack(101, alice)
    cam1 = FakeTracker(1, [track_a])
    gta.step(cam1, 1)
    gta.step(cam1, 30)
    gta.deactivate_track(101)   # closes the node, still unresolved

    track_b = FakeTrack(201, _unit(alice + rng.randn(128).astype(np.float32) * 0.01))
    cam2 = FakeTracker(2, [track_b])
    gta.step(cam2, 90)          # 90 - 0 >= window_frames(50) -> tick fires here

    ok = track_b.t_global_id != 0 and gta.size() == 1
    print(f"[disjoint_similar_tracklets_converge] gid={track_b.t_global_id} "
          f"clusters={gta.size()} (expect 1) -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_overlapping_tracklets_never_merge() -> bool:
    """Tracklet C (cam1, visible 1-60) and tracklet D (cam2, visible from 45)
    overlap in time - identical features must NOT be enough to merge them,
    since one body can't be in two places (two cameras) at once. The hard
    non-overlap veto in pairwise_log_likelihood must win over a perfect
    appearance score."""
    rng = np.random.RandomState(1)
    gta = _make_gta()
    alice = _unit(rng.randn(128))

    track_c = FakeTrack(102, alice)
    cam1 = FakeTracker(1, [track_c])
    gta.step(cam1, 1)
    gta.step(cam1, 40)

    track_d = FakeTrack(202, alice.copy())
    cam2 = FakeTracker(2, [track_d])
    gta.step(cam2, 45)          # opens D at frame 45, inside C's [1, 40+] span

    gta.step(cam1, 60)          # extends C to last_visible=60; 60 >= window_frames -> tick

    ok = (track_c.t_global_id != 0 and track_d.t_global_id != 0
          and track_c.t_global_id != track_d.t_global_id and gta.size() == 2)
    print(f"[overlapping_tracklets_never_merge] c={track_c.t_global_id} "
          f"d={track_d.t_global_id} clusters={gta.size()} (expect 2, distinct) -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def test_stale_identity_is_pruned_and_reappearance_mints_fresh() -> bool:
    """An identity cluster untouched for longer than prune_after_frames must be
    dropped, and a tracklet reappearing afterward (even with the same
    appearance) must mint a fresh identity rather than resurrect the pruned
    one - there is nothing left in the active gallery for it to match."""
    rng = np.random.RandomState(2)
    gta = _make_gta(prune_after_frames=100)
    alice = _unit(rng.randn(128))

    track_e = FakeTrack(103, alice)
    cam1 = FakeTracker(1, [track_e])
    gta.step(cam1, 1)
    gta.step(cam1, 10)
    gta.deactivate_track(103)

    gta.step(FakeTracker(1, []), 50)   # first tick: mints the identity (size 1)
    minted_ok = gta.size() == 1
    old_gid = next(iter(gta._clusters.keys()))

    gta.step(FakeTracker(1, []), 160)  # 160 - 10 = 150 > prune_after_frames(100)
    pruned_ok = gta.size() == 0

    track_f = FakeTrack(301, alice.copy())
    cam2 = FakeTracker(2, [track_f])
    gta.step(cam2, 165)
    gta.step(cam2, 210)                # 210 - 160 >= window_frames -> tick, no candidates left

    ok = (minted_ok and pruned_ok and old_gid not in gta._clusters
          and gta.size() == 1 and track_f.t_global_id != 0)
    print(f"[stale_identity_pruned_and_reappearance_mints_fresh] "
          f"minted={minted_ok} pruned={pruned_ok} old_gid={old_gid} "
          f"new_gid={track_f.t_global_id} clusters={gta.size()} (expect 1) -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def run_all() -> bool:
    results = [
        test_disjoint_similar_tracklets_converge_to_one_identity(),
        test_overlapping_tracklets_never_merge(),
        test_stale_identity_is_pruned_and_reappearance_mints_fresh(),
    ]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    return all(results)


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_all() else 1)
