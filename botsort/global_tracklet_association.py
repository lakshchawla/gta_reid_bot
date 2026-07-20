"""
Global Tracklet Association (GTA) — tracklet-level, windowed cross-camera re-ID.

global_registry.py's GlobalRegistry decides identity per frame, per observation:
every unresolved live track gets a FAISS cosine query every tick, and outside the
narrow AMBIGUOUS/claimed cases (deferred to a periodic Hungarian solver) commits a
MATCH/NO_MATCH decision immediately off a single frame's embedding. Once
track.t_global_id is set it's never re-verified (short of the identity_revoke_streak
safety net) - a bad early observation (blur, occlusion, a lookalike) can lock in a
wrong global_id that then gets reinforced every subsequent frame via the EMA centroid
update.

GTA replaces that with a tracklet-level, windowed decision policy - and, unlike a
first pass at this design, identities are not a persistent gallery either. There is
no GalleryEntry-style object that lives across ticks accumulating an EMA centroid
that tracklets get matched against. Instead:

  - Every tracklet (STrack) becomes one TrackletNode - first_visible, last_visible,
    smoothed feature, camera - kept in one flat pool (`GTA._nodes`) for as long as
    it's live or recently closed.
  - Once every `window_frames` (~60s @ 30fps by default), GTA builds the FULL
    pairwise graph over every node still in that pool - not just ones that showed up
    since the last tick - and reconverges the whole thing into identity groups from
    scratch via a Gaussian log-likelihood edge score + greedy union-find. A node's
    previous identity label is only used as a tie-break for which label a merged
    group keeps (see `_run_tick`), not as ground truth to match new evidence
    against - nothing needs to "survive" a window on its own, because the whole
    graph is what's judged, every time.
  - `IdentityCluster` is a read-only summary computed fresh after each tick
    (first/last visible, cameras seen, an average feature for logging) - not
    mutable matching state.

Two tracklets are linked only if they're temporally disjoint (one fully ends before
the other starts - the same non-overlapping-FOV assumption multi_camera_tracker.py
already documents) and the log-likelihood combining an appearance term and a
time-gap term clears a threshold.

Signature-compatible with GlobalRegistry (`step(tracker, frame_id)`,
`deactivate_track(track_id)`) so botsort/multi_camera_tracker.py and bot_sort.py's
`self.registry.deactivate_track(...)` call site need zero changes - this is a
drop-in swap of the matching brain, `BoTSORT`/`STrack`/`MultiCameraTracker` are
reused as-is.
"""

from __future__ import annotations

import json
import math
import os
import queue
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

SQRT_2PI = math.sqrt(2.0 * math.pi)


class CameraTopologyPrior:
    def __init__(self, bonuses: Optional[dict] = None, default: float = 0.0):
        self._bonuses = {self._canon(a, b): v for (a, b), v in (bonuses or {}).items()}
        self._default = default

    @staticmethod
    def _canon(cam_a, cam_b):
        return tuple(sorted((cam_a, cam_b), key=str))

    def log_bonus(self, cam_a, cam_b) -> float:
        if cam_a == cam_b:
            return 0.0
        return self._bonuses.get(self._canon(cam_a, cam_b), self._default)


@dataclass
class GTAConfig:
    sigma_feat: float
    mu_gap_sec: float
    sigma_gap_sec: float
    frame_rate: float
    camera_prior: Optional[CameraTopologyPrior] = None


@dataclass
class TrackletNode:
    tid: int
    cam_source: object
    track_ref: object                       # STrack while open; cleared on close
    first_visible: int                      # DS shared frame_id, first qualifying obs
    last_visible: int                       # DS shared frame_id, most recent obs
    feat: np.ndarray
    closed: bool = False
    identity_id: Optional[int] = None       # label from the most recent tick's convergence


class IdentityCluster:
    def __init__(self, identity_id: int):
        self.identity_id = identity_id
        self.first_visible: Optional[int] = None
        self.last_visible: Optional[int] = None
        self.cams_seen: set = set()
        self.member_count = 0
        self._feat_sum: Optional[np.ndarray] = None

    def _add(self, node: TrackletNode) -> None:
        self.first_visible = node.first_visible if self.first_visible is None else min(self.first_visible, node.first_visible)
        self.last_visible = node.last_visible if self.last_visible is None else max(self.last_visible, node.last_visible)
        self.cams_seen.add(node.cam_source)
        self.member_count += 1
        self._feat_sum = node.feat.copy() if self._feat_sum is None else self._feat_sum + node.feat

    @property
    def centroid(self) -> Optional[np.ndarray]:
        if self._feat_sum is None:
            return None
        n = np.linalg.norm(self._feat_sum)
        return (self._feat_sum / n).astype(np.float32) if n > 1e-9 else self._feat_sum.astype(np.float32)

    def __repr__(self):
        return (f"IdentityCluster(id={self.identity_id}, members={self.member_count}, "
                f"cams={sorted(self.cams_seen, key=str)}, last_visible={self.last_visible})")


def pairwise_log_likelihood(a: TrackletNode, b: TrackletNode, cfg: GTAConfig) -> Optional[float]:
    """Log-likelihood that tracklet nodes `a` and `b` are the same real-world
    identity.

    Returns None if they're structurally impossible: overlapping visibility
    windows mean one body was in two places at once (non-overlapping-FOV
    assumption, same as multi_camera_tracker.py's docstring) - this is a hard
    veto, not a low score, so it can never be outvoted by a strong appearance
    match.

    Otherwise: log P(same identity) = log P_appearance(cos_dist) +
    log P_gap(elapsed seconds) [+ log-bonus from an optional camera prior,
    inactive by default]. Two independent Gaussian terms rather than a single
    multivariate Gaussian over the raw embedding - simpler to calibrate and
    only needs a scalar cosine-distance/time-gap sample per pair, not enough
    per-identity samples to estimate a full covariance matrix.
    """
    if a.last_visible >= b.first_visible and b.last_visible >= a.first_visible:
        return None

    early, late = (a, b) if a.last_visible < b.first_visible else (b, a)

    cos_dist = 1.0 - float(np.dot(a.feat, b.feat))
    feat_ll = -0.5 * (cos_dist / cfg.sigma_feat) ** 2 - math.log(cfg.sigma_feat * SQRT_2PI)

    gap_sec = (late.first_visible - early.last_visible) / cfg.frame_rate
    time_ll = (-0.5 * ((gap_sec - cfg.mu_gap_sec) / cfg.sigma_gap_sec) ** 2
               - math.log(cfg.sigma_gap_sec * SQRT_2PI))

    cam_ll = 0.0
    if cfg.camera_prior is not None and a.cam_source is not None and b.cam_source is not None:
        cam_ll = cfg.camera_prior.log_bonus(a.cam_source, b.cam_source)

    return feat_ll + time_ll + cam_ll


def _intervals_conflict(a_list, b_list) -> bool:
    """True if any (first_visible, last_visible) interval in a_list overlaps any
    interval in b_list - used to veto a graph merge that would put two
    temporally-overlapping tracklets in the same identity group, even when
    neither of them is the direct edge that triggered the merge (the usual
    correlation-clustering trap: A-B and B-C can each look fine locally while
    A-C overlap)."""
    for a0, a1 in a_list:
        for b0, b1 in b_list:
            if a1 >= b0 and b1 >= a0:
                return True
    return False


class GTA:
    """See module docstring. `window_frames`/`min_tracklet_len`/`sigma_feat`/
    `mu_gap_sec`/`sigma_gap_sec`/`link_threshold` all need empirical calibration
    against the CHIRLA harness (evaluate_tracking.py) - the defaults here are a
    starting point, not tuned values (same posture GlobalRegistry's docstring
    already takes for its own thresholds).

    window_frames    : tick interval - the tracklet graph is only ever
                        reconverged into identities once per this many frames
                        (~60s @ 30fps by default), never per frame.
    min_tracklet_len  : frames a tracklet must survive before it's trusted enough
                        to become a node (mirrors GlobalRegistry's min_frames).
    sigma_feat        : cosine-distance std for the appearance Gaussian term.
    mu_gap_sec/sigma_gap_sec : center/spread of the time-gap Gaussian term - a
                        loose prior in the absence of a per-camera-pair topology
                        (see CameraTopologyPrior).
    link_threshold    : minimum combined log-likelihood to keep a candidate edge
                        in the tick's graph. With the defaults, the best possible
                        score (identical features, gap exactly at mu_gap_sec) is
                        about -2.9 - the default threshold leaves headroom below
                        that for real-world jitter while still rejecting weak
                        pairs; recalibrate alongside sigma_feat/sigma_gap_sec.
    ema_alpha         : unused by matching (kept for interface parity / future
                        use); nothing in GTA maintains an EMA-updated centroid
                        anymore - see module docstring.
    prune_after_frames: tracklet nodes unseen for longer than this are dropped
                        from the graph entirely (default 5 * window_frames, ~5
                        minutes) - this is what keeps the full-graph rebuild
                        each tick bounded in size, and is the only sense in
                        which anything "expires": individual nodes age out,
                        there is no separate cluster-level TTL.
    camera_prior      : optional CameraTopologyPrior; None (inactive) by default.
    frame_rate        : frame <-> seconds conversion.
    """

    def __init__(
        self,
        window_frames: int = 1800,
        min_tracklet_len: int = 30,
        sigma_feat: float = 0.15,
        mu_gap_sec: float = 2.0,
        sigma_gap_sec: float = 20.0,
        link_threshold: float = -6.0,
        ema_alpha: float = 0.9,
        prune_after_frames: Optional[int] = None,
        camera_prior: Optional[CameraTopologyPrior] = None,
        frame_rate: float = 30.0,
        reid_log_path: Optional[str] = None,
        log_flush_interval: int = 500,
    ):
        self.window_frames = window_frames
        self.min_tracklet_len = min_tracklet_len
        self.link_threshold = link_threshold
        self.ema_alpha = ema_alpha
        self.prune_after_frames = (
            prune_after_frames if prune_after_frames is not None else 5 * window_frames
        )
        self.cfg = GTAConfig(
            sigma_feat=sigma_feat, mu_gap_sec=mu_gap_sec, sigma_gap_sec=sigma_gap_sec,
            frame_rate=frame_rate, camera_prior=camera_prior,
        )

        # The tracklet graph: every node GTA currently knows about, open or
        # closed, keyed by track_id (already globally unique - see
        # ID_Assigner(init_id=cam_source*1000) in bot_sort.py). Nothing here is
        # a persistent "gallery" - identity_id on a node is just last tick's
        # answer, recomputed from this same pool every tick.
        self._nodes: dict[int, TrackletNode] = {}
        self._clusters: dict[int, IdentityCluster] = {}   # last tick's summary
        self._next_identity_id = 1
        self._last_tick_frame = 0
        # cam_source -> tracklets whose identity_id CHANGED on the most recent
        # tick, consumed exactly once (per camera) by the next step() call for
        # that camera - see step()'s docstring for why this can't just be an
        # instance list reset on every step() call.
        self._pending_report: dict = {}

        # Optional non-blocking per-tick log of resolved-identity centroids.
        # Same queue+background-thread shape as GlobalRegistry's _log_worker
        # (global_registry.py:879-903) so the GStreamer thread never blocks on
        # disk I/O - but far lower volume here since GTA only writes once per
        # tick, not once per frame.
        self.reid_log_path = reid_log_path
        self.log_flush_interval = log_flush_interval
        self._log_queue: "queue.Queue" = queue.Queue()
        self._log_stop = threading.Event()
        self._log_thread: Optional[threading.Thread] = None
        if reid_log_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(reid_log_path)), exist_ok=True)
            self._log_thread = threading.Thread(
                target=self._log_worker, name="gta-log-writer", daemon=True
            )
            self._log_thread.start()

    # --- per-frame driver --------------------------------------------------

    def step(self, tracker, frame_id: int) -> list:
        """Run once per camera per batch tick (same call site/contract as
        GlobalRegistry.step - see multi_camera_tracker.py). Does only cheap
        per-frame bookkeeping (open/refresh this tracker's tracklet nodes); the
        actual identity decision only happens inside _run_tick, at most once
        every `window_frames`.

        Returns the STracks whose identity_id changed on the most recent tick,
        scoped to THIS tracker's camera and reported exactly once. The tick
        itself reconverges the whole tracklet graph across every camera sharing
        this GTA instance in one shot, so whichever camera's step() call happens
        to cross the window_frames threshold reports only its own
        newly-changed tracks; _pending_report holds the rest until that
        camera's own step() call runs later in the same batch tick
        (MultiCameraTracker.update_batch calls every camera once per tick, so
        this always resolves within the same tick).
        """
        cam_source = getattr(tracker, "cam_source", None)

        for track in tracker.tracked_stracks:
            if track.is_touching_edge or track.tracklet_len < self.min_tracklet_len or track.smooth_feat is None:
                continue
            node = self._nodes.get(track.track_id)
            if node is None:
                self._nodes[track.track_id] = TrackletNode(
                    tid=track.track_id, cam_source=cam_source, track_ref=track,
                    first_visible=frame_id, last_visible=frame_id,
                    feat=track.smooth_feat.astype(np.float32).copy(),
                )
            elif not node.closed:
                node.last_visible = frame_id
                node.feat = track.smooth_feat.astype(np.float32).copy()

        if frame_id - self._last_tick_frame >= self.window_frames:
            self._last_tick_frame = frame_id
            self._run_tick(frame_id)

        newly = self._pending_report.pop(cam_source, [])
        return [n.track_ref for n in newly if n.track_ref is not None]

    def deactivate_track(self, track_id: int) -> None:
        """Call when BoT-SORT permanently removes a track_id (bot_sort.py's
        max_time_lost path) - unchanged call site, mirrors
        GlobalRegistry.deactivate_track. Freezes the node's evidence (last
        real last_visible/feat) but keeps it in the graph - a closed tracklet
        is still fully eligible to be matched against future ones until it
        ages out via prune_after_frames."""
        node = self._nodes.get(track_id)
        if node is None:
            return
        node.closed = True
        node.track_ref = None

    # --- windowed tick: rebuild the whole graph, converge, prune -----------

    def _run_tick(self, frame_id: int) -> None:
        """Reconverge the ENTIRE tracklet graph (every node still in
        self._nodes, not just ones added since the last tick) into identity
        groups from scratch. A node's prior identity_id is used only to decide
        which label a merged group keeps (lowest existing id wins, for
        continuity/display stability) - it is not treated as ground truth that
        new evidence must be matched against, so nothing about a group's
        membership is "locked in" between ticks.
        """
        nodes = list(self._nodes.values())
        n = len(nodes)

        parent = list(range(n))
        group_intervals = [[(nd.first_visible, nd.last_visible)] for nd in nodes]
        group_identity = [nd.identity_id for nd in nodes]

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        if n > 1:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    score = pairwise_log_likelihood(nodes[i], nodes[j], self.cfg)
                    if score is not None and score >= self.link_threshold:
                        edges.append((score, i, j))
            edges.sort(key=lambda e: e[0], reverse=True)

            for score, i, j in edges:
                ri, rj = find(i), find(j)
                if ri == rj:
                    continue
                if (group_identity[ri] is not None and group_identity[rj] is not None
                        and group_identity[ri] != group_identity[rj]):
                    continue  # would merge two distinct existing identities - reject
                if _intervals_conflict(group_intervals[ri], group_intervals[rj]):
                    continue
                # Keep the lower-index root as parent so an existing identity
                # label already sitting at group_identity[rj] (with none at ri)
                # isn't silently dropped.
                if group_identity[ri] is None and group_identity[rj] is not None:
                    ri, rj = rj, ri
                parent[rj] = ri
                group_intervals[ri].extend(group_intervals[rj])
                if group_identity[ri] is None:
                    group_identity[ri] = group_identity[rj]

        groups: dict[int, list[int]] = {}
        for idx in range(n):
            groups.setdefault(find(idx), []).append(idx)

        resolved_nodes: list[TrackletNode] = []
        for root, idxs in groups.items():
            final_id = group_identity[root]
            if final_id is None:
                final_id = self._next_identity_id
                self._next_identity_id += 1
            since_frame = min(nodes[m].first_visible for m in idxs)
            for k in idxs:
                node = nodes[k]
                if node.identity_id != final_id:
                    node.identity_id = final_id
                    if node.track_ref is not None:
                        node.track_ref.t_global_id = final_id
                        node.track_ref.t_identity_since_frame = since_frame
                    resolved_nodes.append(node)

        by_cam: dict = {}
        for node in resolved_nodes:
            by_cam.setdefault(node.cam_source, []).append(node)
        self._pending_report = by_cam

        stale_tids = [tid for tid, nd in self._nodes.items()
                      if frame_id - nd.last_visible > self.prune_after_frames]
        for tid in stale_tids:
            del self._nodes[tid]

        clusters: dict[int, IdentityCluster] = {}
        for nd in self._nodes.values():
            if nd.identity_id is None:
                continue
            clusters.setdefault(nd.identity_id, IdentityCluster(nd.identity_id))._add(nd)
        self._clusters = clusters

        if self.reid_log_path is not None and self._clusters:
            snapshot = {cid: c.centroid for cid, c in self._clusters.items() if c.centroid is not None}
            if snapshot:
                self._log_queue.put((frame_id, snapshot))

    # --- logging (per-tick, non-blocking) -----------------------------------

    def _log_worker(self):
        buffer: list = []
        while not self._log_stop.is_set():
            try:
                item = self._log_queue.get(timeout=0.5)
                buffer.append(item)
                if len(buffer) >= self.log_flush_interval:
                    self._write_buffer(buffer)
                    buffer.clear()
            except queue.Empty:
                if buffer:
                    self._write_buffer(buffer)
                    buffer.clear()
        while True:
            try:
                buffer.append(self._log_queue.get_nowait())
            except queue.Empty:
                break
        if buffer:
            self._write_buffer(buffer)

    def _write_buffer(self, buffer: list):
        if not buffer:
            return
        with open(self.reid_log_path, "a") as f:
            for frame_id, gid_feat_map in buffer:
                line = {"frame": frame_id,
                        "gid": [[int(g), v.tolist()] for g, v in gid_feat_map.items()]}
                f.write(json.dumps(line) + "\n")

    def stop_logging(self):
        """Signal the background writer to finish and wait for it to exit. Call
        at pipeline shutdown to guarantee all queued ticks are written."""
        if self._log_thread is None:
            return
        self._log_stop.set()
        self._log_thread.join(timeout=15)
        self._log_thread = None

    # --- introspection -------------------------------------------------------

    def get_all_clusters(self) -> list:
        return list(self._clusters.values())

    def size(self) -> int:
        return len(self._clusters)

    def __repr__(self):
        return (f"GTA(clusters={len(self._clusters)}, nodes={len(self._nodes)}, "
                f"window_frames={self.window_frames})")
