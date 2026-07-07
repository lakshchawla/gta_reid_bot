"""
Standalone regression checks for the margin-gated commit + interval-
triggered Hungarian arbiter in botsort/global_registry.py.

Not pytest-based (matches this repo's standalone test style) - run:
`python3 -m botsort.test_interval_solver`.

Drives GlobalRegistry.step() directly with duck-typed fake tracks/trackers
(only the attributes step() actually reads) so every distance and timing is
exact and deterministic - no Kalman/IOU association noise in the way of
what's being tested, which is purely the registry's decision policy.
"""

from __future__ import annotations

import numpy as np

from .global_registry import GlobalRegistry, SpatioTemporalPrior


def _unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


class FakeTrack:
    def __init__(self, tid, feat, bbox=(0, 0, 50, 150)):
        self.track_id = tid
        self.smooth_feat = _unit(np.asarray(feat, dtype=np.float32))
        self._tlwh = np.asarray(bbox, dtype=np.float64)
        self.t_global_id = 0
        self.is_touching_edge = False
        self.tracklet_len = 100

    @property
    def tlwh(self):
        return self._tlwh.copy()

    @property
    def tlbr(self):
        ret = self._tlwh.copy()
        ret[2:] += ret[:2]
        return ret


class FakeTracker:
    def __init__(self, cam_source, tracks=()):
        self.cam_source = cam_source
        self.tracked_stracks = list(tracks)


def _make_registry(**kwargs):
    defaults = dict(match_threshold=0.35, min_frames=1, emb_dim=128,
                    use_gpu=False, min_margin=0.05,
                    solver_interval_frames=50)
    defaults.update(kwargs)
    return GlobalRegistry(**defaults)


def _seed_identity(registry, tid, feat, cam_source, frame_id, bbox=(0, 0, 50, 150)):
    """Mint one identity via the normal step() path, then deactivate its
    track so the identity sits unclaimed in the gallery."""
    track = FakeTrack(tid, feat, bbox=bbox)
    tracker = FakeTracker(cam_source, [track])
    registry.step(tracker, frame_id)
    gid = track.t_global_id
    assert gid != 0, "seeding failed to mint"
    registry.deactivate_track(tid)
    return gid


def test_confident_match_commits_immediately() -> bool:
    """A clean re-entry well under threshold with a fat top-2 margin must
    still get its global_id the same frame - deferral must only ever apply
    to thin-margin / claimed cases."""
    rng = np.random.RandomState(0)
    registry = _make_registry()
    alice = _unit(rng.randn(128))
    bob = _unit(rng.randn(128))          # random 128-d pair: nearly orthogonal
    gid_a = _seed_identity(registry, 101, alice, cam_source=1, frame_id=1)
    _seed_identity(registry, 102, bob, cam_source=1, frame_id=2)

    reentry = FakeTrack(201, _unit(alice + rng.randn(128).astype(np.float32) * 0.02),
                        bbox=(500, 0, 50, 150))
    registry.step(FakeTracker(1, [reentry]), 10)

    ok = reentry.t_global_id == gid_a and len(registry._pending) == 0
    print(f"[confident_match_commits_immediately] gid={reentry.t_global_id} "
          f"(expect {gid_a}) pending={len(registry._pending)} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_lookalike_collision_defers_and_solver_separates() -> bool:
    """The girl/guy case: two established identities with similar centroids
    (similar clothing), both people reappear in the same window. Per-frame
    margin is thin for both -> both deferred (no premature mint). The
    interval solve assigns each tracklet its own identity - exclusivity
    guarantees they can never share one."""
    rng = np.random.RandomState(1)
    registry = _make_registry(solver_interval_frames=50)

    # Centroids ~0.6 apart in cosine distance: distinct enough to seed as
    # two identities (> match_threshold), similar enough that a blended
    # observation lands in the ambiguity band.
    v = _unit(rng.randn(128))
    u = _unit(rng.randn(128))
    u = _unit(u - float(np.dot(u, v)) * v)   # orthogonalize for exact geometry
    girl = _unit(v + 0.6547 * u)
    guy = _unit(v - 0.6547 * u)
    gid_girl = _seed_identity(registry, 101, girl, cam_source=1, frame_id=1)
    gid_guy = _seed_identity(registry, 102, guy, cam_source=1, frame_id=2)

    # Both reappear, each slightly closer to their true identity but well
    # inside the ambiguity margin of the other - and above merge_threshold,
    # so the same-camera split-merge shortcut doesn't apply either.
    t_girl = FakeTrack(201, _unit(0.51 * girl + 0.49 * guy), bbox=(0, 0, 50, 150))
    t_guy = FakeTrack(202, _unit(0.49 * girl + 0.51 * guy), bbox=(800, 0, 50, 150))
    d_girl = 1.0 - float(np.dot(t_girl.smooth_feat, girl))
    d_cross = 1.0 - float(np.dot(t_girl.smooth_feat, guy))
    assert (registry.merge_threshold < d_girl < 0.35
            and (d_cross - d_girl) < registry.min_margin), (
        f"test construction broke: d1={d_girl:.3f} margin={d_cross - d_girl:.3f}")

    tracker = FakeTracker(1, [t_girl, t_guy])
    for f in range(10, 20):
        registry.step(tracker, f)

    deferred_ok = (t_girl.t_global_id == 0 and t_guy.t_global_id == 0
                   and len(registry._pending) == 2 and registry.size() == 2)

    registry.step(tracker, 60)  # crosses solver_interval_frames -> solve

    ok = (deferred_ok
          and t_girl.t_global_id == gid_girl
          and t_guy.t_global_id == gid_guy
          and t_girl.t_global_id != t_guy.t_global_id
          and len(registry._pending) == 0)
    print(f"[lookalike_collision_defers_and_solver_separates] "
          f"deferred_mid_window={deferred_ok} girl->{t_girl.t_global_id} (expect {gid_girl}) "
          f"guy->{t_guy.t_global_id} (expect {gid_guy}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_cannot_link_blocks_cooccurring_identity() -> bool:
    """Identity A is actively worn by a live track in cam 1 for the whole
    window. A lookalike appearing simultaneously in cam 2 (non-overlapping
    FOVs: one body, one place) must NOT be given A by the solver - it must
    come out with a fresh identity, even though its appearance matches A
    almost perfectly."""
    rng = np.random.RandomState(2)
    registry = _make_registry(solver_interval_frames=50)

    alice = _unit(rng.randn(128))
    wearer = FakeTrack(101, alice, bbox=(0, 0, 50, 150))
    cam1 = FakeTracker(1, [wearer])
    registry.step(cam1, 1)
    gid_a = wearer.t_global_id
    assert gid_a != 0

    lookalike = FakeTrack(2001, _unit(alice + rng.randn(128).astype(np.float32) * 0.02),
                          bbox=(0, 0, 50, 150))
    cam2 = FakeTracker(2, [lookalike])
    for f in range(2, 20):
        registry.step(cam1, f)   # keeps A claimed + stamps _gid_seen_at
        registry.step(cam2, f)   # MATCH but claimed -> deferred w/ cannot-link

    deferred_ok = lookalike.t_global_id == 0 and len(registry._pending) == 1

    registry.step(cam1, 60)      # triggers the solve

    ok = (deferred_ok
          and lookalike.t_global_id != 0
          and lookalike.t_global_id != gid_a
          and registry.size() == 2)
    print(f"[cannot_link_blocks_cooccurring_identity] deferred={deferred_ok} "
          f"lookalike_gid={lookalike.t_global_id} (must != {gid_a}, != 0) "
          f"gallery={registry.size()} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_st_prior_breaks_appearance_tie() -> bool:
    """Two candidate identities equidistant in appearance; the configured
    transit-time windows make one physically infeasible for the pending
    tracklet's camera/time -> the solver must pick the feasible one."""
    rng = np.random.RandomState(3)
    st_prior = SpatioTemporalPrior(windows={
        (1, 2): (0.0, 100.0),     # cam1 -> cam2 reachable within the test's elapsed time
        (3, 2): (500.0, None),    # cam3 -> cam2 needs >= 500s: infeasible here
    })
    registry = _make_registry(solver_interval_frames=50, st_prior=st_prior,
                              frame_rate=30.0, cost_alpha=0.7)

    v = _unit(rng.randn(128))
    u = _unit(rng.randn(128))
    u = _unit(u - float(np.dot(u, v)) * v)
    ida = _unit(v + 0.6547 * u)
    idb = _unit(v - 0.6547 * u)
    gid_feasible = _seed_identity(registry, 101, ida, cam_source=1, frame_id=1)
    gid_infeasible = _seed_identity(registry, 302, idb, cam_source=3, frame_id=2)

    # Exactly equidistant to both -> margin 0 -> AMBIGUOUS -> pending.
    query = FakeTrack(2001, _unit(ida + idb), bbox=(0, 0, 50, 150))
    cam2 = FakeTracker(2, [query])
    for f in range(10, 15):
        registry.step(cam2, f)

    deferred_ok = query.t_global_id == 0 and len(registry._pending) == 1

    registry.step(cam2, 60)

    ok = deferred_ok and query.t_global_id == gid_feasible
    print(f"[st_prior_breaks_appearance_tie] deferred={deferred_ok} "
          f"assigned={query.t_global_id} (expect feasible {gid_feasible}, "
          f"not {gid_infeasible}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def run_all() -> bool:
    results = [
        test_confident_match_commits_immediately(),
        test_lookalike_collision_defers_and_solver_separates(),
        test_cannot_link_blocks_cooccurring_identity(),
        test_st_prior_breaks_appearance_tie(),
    ]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    return all(results)


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_all() else 1)
