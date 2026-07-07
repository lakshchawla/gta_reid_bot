from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import json
import math
import os
import queue
import threading

import numpy as np
import faiss
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN

from . import matching

MATCH = "match"
AMBIGUOUS = "ambiguous"
NO_MATCH = "no_match"


def cosine_similarity(unit_a: np.ndarray, unit_b: np.ndarray, transform: Optional[np.ndarray] = None) -> float:
    if transform is None:
        return float(np.dot(unit_a, unit_b))
    return float(unit_a @ transform @ unit_b)


class SpatioTemporalPrior:
    """P_st(dt, camera transition) for the interval solver's cost matrix.

    windows: {(cam_a, cam_b): (min_transit_sec, max_transit_sec)} - the
    physically plausible elapsed-time window for a person moving between
    that camera pair (non-overlapping FOVs). Pairs are canonicalized, so
    one entry covers both directions. Same-camera transitions and pairs
    with no configured window are uninformative (P=1, log_p=0) - fill in
    topology data incrementally, most-trafficked pairs first.

    Outside the window P drops to p_outside (small but NONZERO): -log P
    becomes a large finite cost that competes with the solver's dummy
    new-identity columns rather than an infinite mask - an implausible
    transition makes "new identity" win, it doesn't crash the assignment.
    """

    def __init__(self, windows: Optional[dict] = None, p_outside: float = 1e-6):
        self._windows = {self._canon(a, b): v for (a, b), v in (windows or {}).items()}
        self._log_p_outside = math.log(p_outside)

    @staticmethod
    def _canon(cam_a, cam_b):
        return tuple(sorted((cam_a, cam_b), key=str))

    def log_p(self, cam_a, t_a_sec: float, cam_b, t_b_sec: float) -> float:
        if cam_a == cam_b or cam_a is None or cam_b is None:
            return 0.0
        window = self._windows.get(self._canon(cam_a, cam_b))
        if window is None:
            return 0.0
        min_t, max_t = window
        elapsed = t_b_sec - t_a_sec
        if elapsed < min_t:
            return self._log_p_outside
        if max_t is not None and elapsed > max_t:
            return self._log_p_outside
        return 0.0


@dataclass
class PendingTracklet:
    tid: int
    cam_source: object
    track_ref: object                 # STrack reference 
    first_frame: int
    last_frame: int
    feat: np.ndarray                  # smooth_feat 
    bbox: np.ndarray = None           # tlwh - check taking from strack 
    cannot_link_gids: set = field(default_factory=set)
    hits: int = 1
    dead: bool = False


class GalleryEntry:
    def __init__(self, global_id: int, feat: np.ndarray, frame_id: int, bbox: np.ndarray, ema_alpha: float = 0.9, cam_source=None):
        self.global_id = global_id
        self.active_tid: Optional[int] = None
        self.last_frame = frame_id
        self.last_bbox = bbox.copy()
        self.ema_alpha = ema_alpha

        self.centroid = feat.copy()
        self.update_count = 1
        self.last_cam_source = cam_source
        self.mismatch_streak = 0

    def add_embedding(self, feat: np.ndarray, frame_id: int, bbox: np.ndarray,
                      outlier_reject_thresh: float = 1.0, occlusion_score: float = 0.0) -> bool:
        self.last_frame = frame_id
        self.last_bbox = bbox.copy()

        n = np.linalg.norm(feat)
        feat_n = (feat / n).astype(np.float32) if n > 1e-9 else feat.astype(np.float32)

        cos_dist = 1.0 - cosine_similarity(self.centroid, feat_n)
        if cos_dist >= outlier_reject_thresh:
            # Confidently inconsistent with this identity's established
            # appearance - don't let it perturb the centroid.
            self.mismatch_streak += 1
            return False

        effective_alpha = self.ema_alpha + (1.0 - self.ema_alpha) * float(occlusion_score)
        c = effective_alpha * self.centroid + (1.0 - effective_alpha) * feat_n
        cn = np.linalg.norm(c)
        self.centroid = (c / cn).astype(np.float32) if cn > 1e-9 else c.astype(np.float32)
        self.update_count += 1
        self.mismatch_streak = 0
        return True

    def __repr__(self):
        return (f"GalleryEntry(gid={self.global_id}, tid={self.active_tid}, "
                f"n_updates={self.update_count}, last_frame={self.last_frame})")


class GlobalRegistry:
    def __init__(
        self,
        match_threshold: float = 0.35,
        min_frames: int = 5,
        emb_dim: int = 256,
        use_gpu: bool = True,
        min_margin: float = 0.05,
        ema_alpha: float = 0.9,
        outlier_reject_thresh: Optional[float] = 0.3,
        identity_revoke_streak: int = 5,
        overlap_freeze_thresh: float = 0.3,
        overlap_cooldown_frames: int = 10,
        confusion_eps: float = 0.17,
        confusion_refresh_interval: int = 100,
        merge_threshold: float = 0.2,
        reid_log_path: Optional[str] = None,
        log_flush_interval: int = 500,
        log_format: str = "json",
        solver_interval_frames: int = 900,
        cost_alpha: float = 0.7,
        new_identity_cost: Optional[float] = None,
        st_prior: Optional[SpatioTemporalPrior] = None,
        frame_rate: float = 30.0,
    ):
        self.match_threshold = match_threshold
        self.min_frames = min_frames
        self.emb_dim = emb_dim
        self.min_margin = min_margin
        self.ema_alpha = ema_alpha
        self.identity_revoke_streak = identity_revoke_streak
        self.outlier_reject_thresh = (
            outlier_reject_thresh if outlier_reject_thresh is not None
            else min(1.0, match_threshold + 0.3)
        )

        self.overlap_freeze_thresh = overlap_freeze_thresh
        self.overlap_cooldown_frames = overlap_cooldown_frames
        # track_id → remaining cooldown frames after an overlap clears.
        # Prevents querying immediately after separation while smooth_feat
        # is still contaminated by the occluder.
        self._tid_cooldown: dict[int, int] = {}

        # Cosine-distance eps for confusion-zone DBSCAN and how often to
        # refresh. Gallery entries that cluster together at this distance
        # (i.e. two distinct identities whose centroids are suspiciously
        # similar — hard samples) get tighter match/margin thresholds in
        # query() to avoid merging two similar-clothing individuals.
        self.confusion_eps = confusion_eps
        self.confusion_refresh_interval = confusion_refresh_interval
        # Cosine distance below which a "claimed + new track" pair is treated
        # as a track split (same person) rather than a different person.
        # Must stay well below match_threshold so only near-certain splits
        # are merged — bypasses confusion-zone margin gating via _query_raw().
        self.merge_threshold = merge_threshold
        self._confusion_gids: set[int] = set()
        self._last_confusion_refresh: int = 0

        # Interval-triggered Hungarian arbiter over deferred (ambiguous)
        # tracklets. tau_high = match_threshold and m = min_margin above are
        # reused as-is for the per-frame confident-commit test; the params
        # below only govern the batch solve. new_identity_cost (tau_new) is
        # the cost of a dummy "mint a fresh identity" column - a real
        # gallery candidate must beat it to win a pending tracklet.
        self.solver_interval_frames = solver_interval_frames
        self.cost_alpha = cost_alpha
        self.new_identity_cost = (
            new_identity_cost if new_identity_cost is not None else match_threshold
        )
        self.st_prior = st_prior
        self.frame_rate = frame_rate
        self._pending: dict[int, PendingTracklet] = {}
        self._last_solve_frame: int = 0
        # gid -> frame_id at which an OWNED live track last refreshed that
        # identity. Read when refreshing a pending tracklet the same frame:
        # an identity actively worn by someone else while this tracklet is
        # on screen can never be this tracklet (non-overlapping FOVs - one
        # body, one place), so it becomes a cannot-link for the solver.
        self._gid_seen_at: dict[int, int] = {}

        # Per-frame feature log.
        # smooth_feat is the unit-normalised EMA embedding that query() uses.
        #
        # log_format="jsonl" (default) — one JSON line per frame, append-only.
        #   No load-merge-save; background thread just appends. Readable with
        #   jq / pandas. Each line: {"frame": N, "gid": [[gid, [f0..f255]], ...]}
        # log_format="npz"   — numpy compressed archive.
        #   frame_{id}_gids (int32) + frame_{id}_feats (float32 N×D) per frame.
        #   ~10x smaller than JSON, fastest for numpy analysis. Requires load-
        #   merge-save on flush — use a larger log_flush_interval (e.g. 1000).
        #
        # The pipeline thread never does file I/O. It puts (frame_id, dict) into
        # a queue.Queue and returns immediately. A daemon background thread drains
        # the queue and writes to disk so the GStreamer/DeepStream thread is never
        # blocked by disk latency.
        self.reid_log_path = reid_log_path
        self.log_flush_interval = log_flush_interval
        self.log_format = log_format
        self._log_queue: queue.Queue = queue.Queue()
        self._log_stop = threading.Event()
        self._log_thread: Optional[threading.Thread] = None
        if reid_log_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(reid_log_path)), exist_ok=True)
            self._log_thread = threading.Thread(
                target=self._log_worker, name="reid-log-writer", daemon=True
            )
            self._log_thread.start()

        self._next_gid = 1
        self._gid_to_entry: dict[int, GalleryEntry] = {}
        self._tid_to_gid: dict[int, int] = {}

        cpu_index = faiss.IndexFlatIP(emb_dim)
        self._index = cpu_index
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            except Exception as exc:
                print(f"[REGISTRY] GPU FAISS unavailable ({exc}); using CPU index")

        self._index_gids: list[int] = []  # row i in self._index -> global_id

    # --- gallery search ------------------------------------------------

    def query(self, feat: np.ndarray) -> tuple[str, Optional[GalleryEntry], float, float]:
        """Margin test on raw cosine over the FAISS top-2.

        Returns (status, best_entry, d1, d2):
          MATCH     - d1 < tau_high and the top-2 gap clears the margin:
                      confident, caller may commit immediately.
          AMBIGUOUS - d1 < tau_high but the runner-up is nearly as close
                      (the lookalike-collision case). best_entry/d1 are
                      still returned for bookkeeping, but the caller must
                      NOT commit - defer to the interval solver.
          NO_MATCH  - d1 >= tau_high (or empty gallery; d1=d2=1.0 then).

        The top-2 hits are guaranteed to be two DIFFERENT identities: the
        index holds exactly one centroid row per gid (_rebuild_index). If
        multi-prototype rows are ever added, the runner-up scan must skip
        rows sharing the top-1's gid before applying the margin.
        """
        if feat is None or self._index.ntotal == 0:
            return NO_MATCH, None, 1.0, 1.0

        n = np.linalg.norm(feat)
        if n < 1e-9:
            return NO_MATCH, None, 1.0, 1.0

        vec = np.ascontiguousarray((feat / n).astype(np.float32).reshape(1, -1))
        k = min(2, self._index.ntotal)
        sims, idxs = self._index.search(vec, k=k)
        pos = int(idxs[0, 0])
        if pos < 0:
            return NO_MATCH, None, 1.0, 1.0

        cos_dist = 1.0 - float(sims[0, 0])
        cos_dist_2 = 1.0
        if idxs.shape[1] > 1 and int(idxs[0, 1]) >= 0:
            cos_dist_2 = 1.0 - float(sims[0, 1])

        gid = self._index_gids[pos]
        entry = self._gid_to_entry.get(gid)

        # Tighten thresholds when this identity is in a confusion zone
        # (its centroid is suspiciously close to another gallery entry —
        # a known hard-sample pair). Require stronger evidence before
        # committing to a match rather than risk merging two similar-
        # clothing individuals. A tighter margin here simply routes more
        # cases to AMBIGUOUS (deferred), not to a hard reject.
        if gid in self._confusion_gids:
            effective_match_thresh = self.match_threshold * 0.75
            effective_min_margin   = self.min_margin * 2.0
        else:
            effective_match_thresh = self.match_threshold
            effective_min_margin   = self.min_margin

        if cos_dist >= effective_match_thresh:
            return NO_MATCH, entry, cos_dist, cos_dist_2

        if (cos_dist_2 - cos_dist) < effective_min_margin:
            # Runner-up nearly as close as the winner - exactly the
            # girl/guy collision. Don't decide now.
            return AMBIGUOUS, entry, cos_dist, cos_dist_2

        return MATCH, entry, cos_dist, cos_dist_2

    def _query_raw(self, feat: np.ndarray) -> tuple[Optional[GalleryEntry], float]:
        """Nearest gallery entry with NO threshold, margin, or confusion-zone
        gating. Used only by the track-split merge path in step() so that a
        confusion-zone margin rejection doesn't block a near-certain same-
        person merge (the feedback loop where splits keep minting new IDs
        that grow the confusion zone, making the next split's margin even
        tighter)."""
        if feat is None or self._index.ntotal == 0:
            return None, 1.0
        n = np.linalg.norm(feat)
        if n < 1e-9:
            return None, 1.0
        vec = np.ascontiguousarray((feat / n).astype(np.float32).reshape(1, -1))
        sims, idxs = self._index.search(vec, k=1)
        pos = int(idxs[0, 0])
        if pos < 0:
            return None, 1.0
        cos_dist = 1.0 - float(sims[0, 0])
        gid = self._index_gids[pos]
        return self._gid_to_entry.get(gid), cos_dist

    # --- gallery mutation ------------------------------------------------

    def _register(self, feat: np.ndarray, frame_id: int, bbox: np.ndarray, cam_source=None) -> int:
        gid = self._next_gid
        self._next_gid += 1
        self._gid_to_entry[gid] = GalleryEntry(gid, feat, frame_id, bbox, ema_alpha=self.ema_alpha, cam_source=cam_source)
        return gid

    def _rebuild_index(self):
        """FAISS IndexFlat has no cheap in-place update, so a centroid change
        means re-adding everything. Galleries here are small (one camera's
        worth of people), so this stays cheap."""
        self._index.reset()
        self._index_gids = []
        if not self._gid_to_entry:
            return
        gids = list(self._gid_to_entry.keys())
        centroids = np.ascontiguousarray(
            np.stack([self._gid_to_entry[g].centroid for g in gids], axis=0).astype(np.float32)
        )
        self._index.add(centroids)
        self._index_gids = gids

    def _refresh_confusion_zones(self, frame_id: int):
        """Periodically run DBSCAN on gallery centroids to detect identity
        pairs that are suspiciously close in embedding space (hard samples —
        typically similar-clothing individuals). Entries that cluster together
        are added to _confusion_gids so query() can apply tighter thresholds
        for them specifically."""
        if frame_id - self._last_confusion_refresh < self.confusion_refresh_interval:
            return
        self._last_confusion_refresh = frame_id
        if len(self._gid_to_entry) < 2:
            self._confusion_gids = set()
            return
        gids = list(self._gid_to_entry.keys())
        vecs = np.stack([self._gid_to_entry[g].centroid for g in gids])  # (N, D) unit vecs
        # Cosine distance matrix: 1 - dot(unit_vecs, unit_vecs.T)
        cos_dist = (1.0 - (vecs @ vecs.T)).clip(0.0, 2.0).astype(np.float64)
        labels = DBSCAN(eps=self.confusion_eps, min_samples=2, metric='precomputed').fit_predict(cos_dist)
        prev = self._confusion_gids
        self._confusion_gids = {gids[i] for i, lbl in enumerate(labels) if lbl >= 0}
        if self._confusion_gids != prev:
            print(f"[REGISTRY] confusion zones updated: {len(self._confusion_gids)} entries "
                  f"gids={sorted(self._confusion_gids)}")

    def deactivate_track(self, track_id: int):
        """Call when a track_id is permanently gone (removed, not merely
        lost) so its identity becomes available for re-ID again."""
        self._tid_cooldown.pop(track_id, None)
        # A pending (deferred) tracklet keeps its record - the accumulated
        # evidence stays valid for the next interval solve - but is marked
        # dead so the solver doesn't leave a stale active_tid claim behind.
        pending = self._pending.get(track_id)
        if pending is not None:
            pending.dead = True
        gid = self._tid_to_gid.pop(track_id, None)
        if gid is None:
            return
        entry = self._gid_to_entry.get(gid)
        # Only clear active_tid when this was the primary holder. After a
        # track-split merge, two track_ids share one global_id — deactivating
        # the old one must not evict the new one's primary claim.
        if entry is not None and entry.active_tid == track_id:
            entry.active_tid = None

    # --- deferred-decision pool + interval solver --------------------------

    def _pend_track(self, track, cam_source, frame_id: int, conflict_gid: Optional[int] = None):
        """Create or refresh the pending record for a track whose identity
        decision was deferred (AMBIGUOUS margin, or its best match is
        claimed by another live track). Also accumulates cannot-link
        evidence: any identity actively worn by a DIFFERENT track this same
        frame can never be this tracklet."""
        tid = track.track_id
        pending = self._pending.get(tid)
        feat_copy = track.smooth_feat.astype(np.float32).copy()
        bbox_copy = np.asarray(track.tlwh, dtype=np.float64).copy()
        if pending is None:
            pending = PendingTracklet(
                tid=tid, cam_source=cam_source, track_ref=track,
                first_frame=frame_id, last_frame=frame_id,
                feat=feat_copy, bbox=bbox_copy,
            )
            self._pending[tid] = pending
            print(f"[REGISTRY] deferred: track={tid} (cam={cam_source}) "
                  f"ambiguous/conflicted - queued for interval solver")
        else:
            pending.last_frame = frame_id
            pending.feat = feat_copy
            pending.bbox = bbox_copy
            pending.hits += 1
        if conflict_gid is not None:
            pending.cannot_link_gids.add(conflict_gid)
        # Cross-camera co-occurrence: _gid_seen_at is stamped by whichever
        # camera's step() refreshed an owned identity this tick. All cameras
        # share one frame_id per batch tick (MultiCameraTracker.update_batch),
        # so equality means "on screen at the same instant".
        for gid, seen_at in self._gid_seen_at.items():
            if seen_at == frame_id:
                entry = self._gid_to_entry.get(gid)
                if entry is not None and entry.active_tid not in (None, tid):
                    pending.cannot_link_gids.add(gid)

    def _solve_pending(self, frame_id: int):
        """Interval-triggered Hungarian arbiter over all deferred tracklets.

        Jointly assigns every pending tracklet to either a gallery identity
        or a dummy "new identity" column (cost tau_new). Exclusivity - each
        identity column is assignable to at most one tracklet - is what
        structurally prevents ID-stealing when two lookalikes are both in
        the window; the spatio-temporal prior breaks appearance ties.
        O(n^3) in the window's tracklet count, independent of gallery size.
        """
        if frame_id - self._last_solve_frame < self.solver_interval_frames:
            return
        self._last_solve_frame = frame_id
        if not self._pending:
            return

        pendings = list(self._pending.values())
        gids = list(self._gid_to_entry.keys())
        n, m = len(pendings), len(gids)
        MASKED = 1e6  # >> any real cost (cos_dist <= 2, -log p_outside ~ 13.8)

        cost = np.full((n, m + n), MASKED, dtype=np.float64)
        for i in range(n):
            cost[i, m + i] = self.new_identity_cost

        if m > 0:
            centroids = np.stack([self._gid_to_entry[g].centroid for g in gids])
            feats = np.stack([
                p.feat / max(float(np.linalg.norm(p.feat)), 1e-9) for p in pendings
            ])
            cos_dists = (1.0 - feats @ centroids.T).clip(0.0)  # (n, m)
            for i, pending in enumerate(pendings):
                for j, gid in enumerate(gids):
                    if gid in pending.cannot_link_gids:
                        continue
                    entry = self._gid_to_entry[gid]
                    # Claimed by someone else at any point since this
                    # tracklet appeared -> temporally overlapping pair,
                    # cannot-link (conservative: prefer a fresh identity
                    # over a possible steal).
                    seen_at = self._gid_seen_at.get(gid)
                    if (seen_at is not None and seen_at >= pending.first_frame
                            and entry.active_tid not in (None, pending.tid)):
                        continue
                    st_cost = 0.0
                    if self.st_prior is not None:
                        st_cost = -self.st_prior.log_p(
                            entry.last_cam_source, entry.last_frame / self.frame_rate,
                            pending.cam_source, pending.first_frame / self.frame_rate,
                        )
                    cost[i, j] = (self.cost_alpha * float(cos_dists[i, j])
                                  + (1.0 - self.cost_alpha) * st_cost)

        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            pending = pendings[i]
            if j < m and cost[i, j] < MASKED:
                entry = self._gid_to_entry[gids[j]]
                entry.last_cam_source = pending.cam_source
                entry.add_embedding(pending.feat, pending.last_frame, pending.bbox,
                                    outlier_reject_thresh=self.outlier_reject_thresh)
                if pending.track_ref is not None:
                    pending.track_ref.t_global_id = entry.global_id
                if not pending.dead:
                    entry.active_tid = pending.tid
                    self._tid_to_gid[pending.tid] = entry.global_id
                print(f"[REGISTRY] solver match: track={pending.tid} (cam={pending.cam_source}) "
                      f"-> global_id={entry.global_id} (cost={cost[i, j]:.3f}, "
                      f"hits={pending.hits}, dead={pending.dead})")
            else:
                new_gid = self._register(pending.feat, pending.last_frame, pending.bbox,
                                         cam_source=pending.cam_source)
                if pending.track_ref is not None:
                    pending.track_ref.t_global_id = new_gid
                if not pending.dead:
                    self._gid_to_entry[new_gid].active_tid = pending.tid
                    self._tid_to_gid[pending.tid] = new_gid
                print(f"[REGISTRY] solver new identity: track={pending.tid} "
                      f"(cam={pending.cam_source}) -> global_id={new_gid} "
                      f"(hits={pending.hits}, dead={pending.dead})")

        self._pending.clear()
        self._rebuild_index()


    def step(self, tracker, frame_id: int) -> list:
        index_dirty = False
        assigned_this_step: list = []
        cam_source = getattr(tracker, "cam_source", None)

        self._refresh_confusion_zones(frame_id)
        self._solve_pending(frame_id)
        live_tids = {t.track_id for t in tracker.tracked_stracks}

        
        live = tracker.tracked_stracks
        if len(live) >= 2:
            iou_cost = matching.iou_distance(live, live) 
            iou_mat = 1.0 - iou_cost
            np.fill_diagonal(iou_mat, 0.0)
            peer_iou_vals = iou_mat.max(axis=1)
        else:
            peer_iou_vals = np.zeros(len(live))
        track_overlap = {t.track_id: float(peer_iou_vals[i]) for i, t in enumerate(live)}

        for track in tracker.tracked_stracks:
            feat = track.smooth_feat
            bbox = track.tlwh
            tid = track.track_id
            occlusion_score = track_overlap.get(tid, 0.0)

            if track.t_global_id != 0:
                entry = self._gid_to_entry.get(track.t_global_id)
                if entry is not None:
                    entry.active_tid = tid
                    entry.last_cam_source = cam_source
                    self._tid_to_gid[tid] = track.t_global_id
                    # Stamp "this identity is actively worn right now" -
                    # read by _pend_track/_solve_pending for cannot-link
                    # (an identity on screen under another track can never
                    # be a co-occurring pending tracklet).
                    self._gid_seen_at[track.t_global_id] = frame_id
                    if feat is not None and not track.is_touching_edge:
                        index_dirty = True
                        if not entry.add_embedding(feat, frame_id, bbox,
                                                   outlier_reject_thresh=self.outlier_reject_thresh,
                                                   occlusion_score=occlusion_score):
                            if entry.mismatch_streak >= self.identity_revoke_streak:
                                print(f"[REGISTRY] identity revoked: track={tid} global_id={entry.global_id} "
                                      f"(appearance mismatched {entry.mismatch_streak} frames straight)")
                                entry.active_tid = None
                                entry.mismatch_streak = 0
                                self._tid_to_gid.pop(tid, None)
                                track.t_global_id = 0
                continue

            if track.is_touching_edge or track.tracklet_len < self.min_frames or feat is None:
                continue

            # --- Overlap freeze / cooldown for unidentified tracks ---------
            # While two people overlap, don't query: the embedding is
            # contaminated by the occluder. After they separate, wait
            # overlap_cooldown_frames for smooth_feat to recover before
            # committing to a gallery query.
            if occlusion_score > self.overlap_freeze_thresh:
                self._tid_cooldown[tid] = self.overlap_cooldown_frames
                continue
            if self._tid_cooldown.get(tid, 0) > 0:
                self._tid_cooldown[tid] -= 1
                continue
            # ---------------------------------------------------------------

            status, entry, dist, dist2 = self.query(feat)
            # "Claimed" = the best entry is actively worn by a DIFFERENT
            # track right now - can't just hand it over. live_tids only
            # covers THIS tracker, so also consult _gid_seen_at (stamped by
            # every camera's owned-track refresh) to catch a claim held by
            # another camera in this same tick; >= frame_id - 1 tolerates
            # either per-tick step() ordering between cameras.
            worn_elsewhere = (entry is not None
                              and self._gid_seen_at.get(entry.global_id, -1) >= frame_id - 1)
            claimed = (entry is not None
                       and entry.active_tid not in (None, tid)
                       and (entry.active_tid in live_tids or worn_elsewhere))

            if status == MATCH and entry is not None and not claimed:
                # Confident: d1 under threshold AND a fat top-2 margin.
                prev_cam = entry.last_cam_source
                cross_camera = prev_cam is not None and cam_source is not None and prev_cam != cam_source
                entry.active_tid = tid
                entry.last_cam_source = cam_source
                entry.add_embedding(feat, frame_id, bbox,
                                    outlier_reject_thresh=self.outlier_reject_thresh,
                                    occlusion_score=occlusion_score)
                self._tid_to_gid[tid] = entry.global_id
                track.t_global_id = entry.global_id
                self._pending.pop(tid, None)
                index_dirty = True
                assigned_this_step.append(track)
                print(f"[REGISTRY] re-id match: track={tid} (cam={cam_source}) -> global_id={entry.global_id} "
                      f"(dist={dist:.3f}, margin={dist2 - dist:.3f}, prev_cam={prev_cam}, "
                      f"cross_camera={cross_camera})")
            else:
                # Not committable right now: claimed conflict, thin margin
                # (AMBIGUOUS), or no match. Check for a BoT-SORT track split
                # first - a raw near-duplicate (dist < merge_threshold) is
                # the same person under a fragmented track_id, not an
                # ambiguity, and _query_raw() bypasses all policy gating.
                # A track split is a SAME-camera phenomenon (BoT-SORT
                # fragmenting one person's track); a near-duplicate whose
                # identity was last seen on a different camera is either a
                # genuine cross-camera re-id (handled by the MATCH path) or
                # a co-occurring lookalike (must defer, not merge - with
                # non-overlapping FOVs one body can't be in two cameras).
                raw_entry, raw_dist = self._query_raw(feat) if dist < self.merge_threshold else (None, 1.0)
                if (raw_entry is not None and raw_dist < self.merge_threshold
                        and raw_entry.last_cam_source == cam_source):
                    old_tid = raw_entry.active_tid
                    raw_entry.active_tid = tid
                    raw_entry.last_cam_source = cam_source
                    raw_entry.add_embedding(feat, frame_id, bbox,
                                            outlier_reject_thresh=self.outlier_reject_thresh,
                                            occlusion_score=occlusion_score)
                    self._tid_to_gid[tid] = raw_entry.global_id
                    track.t_global_id = raw_entry.global_id
                    self._pending.pop(tid, None)
                    # Keep old track's global_id mapping alive while BoT-SORT
                    # still reports it as tracked.
                    if old_tid is not None and old_tid in live_tids:
                        self._tid_to_gid[old_tid] = raw_entry.global_id
                        for t in tracker.tracked_stracks:
                            if t.track_id == old_tid:
                                t.t_global_id = raw_entry.global_id
                                break
                    index_dirty = True
                    assigned_this_step.append(track)
                    print(f"[REGISTRY] track split resolved: track={tid} -> global_id={raw_entry.global_id} "
                          f"(dist={raw_dist:.3f}, old_track={old_tid}, cam={cam_source})")
                elif status in (MATCH, AMBIGUOUS):
                    # Defer instead of minting: a MATCH whose identity is
                    # claimed by a live track (the ID-stealing setup - the
                    # thief may get revoked before the window closes), or a
                    # thin-margin lookalike collision. The interval solver
                    # arbitrates the whole pool jointly with exclusivity +
                    # cannot-link + the spatio-temporal prior. If a later
                    # frame's margin test turns confident, the track commits
                    # immediately via the MATCH path above and leaves the
                    # pool.
                    self._pend_track(track, cam_source, frame_id,
                                     conflict_gid=entry.global_id if claimed else None)
                else:
                    # NO_MATCH: confidently novel appearance - mint now,
                    # don't make a genuinely new person wait for the solver.
                    new_gid = self._register(feat, frame_id, bbox, cam_source=cam_source)
                    self._gid_to_entry[new_gid].active_tid = tid
                    self._tid_to_gid[tid] = new_gid
                    track.t_global_id = new_gid
                    self._pending.pop(tid, None)
                    index_dirty = True
                    assigned_this_step.append(track)
                    print(f"[REGISTRY] new identity: track={tid} (cam={cam_source}) -> global_id={new_gid} "
                          f"(closest_dist={dist:.3f})")

        if index_dirty:
            self._rebuild_index()

        # --- Per-frame feature vector log (non-blocking) -------------------
        # Copy smooth_feat for every identified track and hand it off to the
        # background writer thread via a queue. The pipeline thread does zero
        # file I/O here — just builds a small dict and puts it in the queue.
        if self.reid_log_path is not None:
            frame_entry = {}
            for t in tracker.tracked_stracks:
                if t.t_global_id != 0 and t.smooth_feat is not None:
                    frame_entry[t.t_global_id] = t.smooth_feat.astype(np.float32).copy()
            if frame_entry:
                self._log_queue.put((frame_id, frame_entry))

        return assigned_this_step

    def _log_worker(self):
        """Background daemon thread: drains the log queue and writes to disk.
        Accumulates log_flush_interval frames in memory before each write so
        disk I/O happens in batches rather than per-frame."""
        buffer: list = []   # [(frame_id, {gid: np.ndarray})]
        while not self._log_stop.is_set():
            try:
                item = self._log_queue.get(timeout=0.5)
                buffer.append(item)
                if len(buffer) >= self.log_flush_interval:
                    self._write_buffer(buffer)
                    buffer.clear()
            except queue.Empty:
                # Timeout — flush whatever we have so data isn't held too long
                if buffer:
                    self._write_buffer(buffer)
                    buffer.clear()
        # Drain any items queued after stop was signalled
        while True:
            try:
                buffer.append(self._log_queue.get_nowait())
            except queue.Empty:
                break
        if buffer:
            self._write_buffer(buffer)

    def _write_buffer(self, buffer: list):
        """Write a batch of (frame_id, {gid: feat}) pairs to disk."""
        if not buffer:
            return
        if self.log_format == "npz":
            flat: dict = {}
            if os.path.exists(self.reid_log_path):
                try:
                    existing_npz = np.load(self.reid_log_path)
                    flat = {k: existing_npz[k] for k in existing_npz.files}
                    existing_npz.close()
                except Exception:
                    pass
            for frame_id, gid_feat_map in buffer:
                gids  = np.array(list(gid_feat_map.keys()), dtype=np.int32)
                feats = np.stack(list(gid_feat_map.values()), axis=0)
                flat[f"frame_{frame_id}_gids"]  = gids
                flat[f"frame_{frame_id}_feats"] = feats
            np.savez_compressed(self.reid_log_path, **flat)
        else:
            # jsonl — append one line per frame, no load needed
            with open(self.reid_log_path, "a") as f:
                for frame_id, gid_feat_map in buffer:
                    line = {"frame": frame_id,
                            "gid": [[int(g), v.tolist()] for g, v in gid_feat_map.items()]}
                    f.write(json.dumps(line) + "\n")

    def stop_logging(self):
        """Signal the background writer to finish and wait for it to exit.
        Call at pipeline shutdown to guarantee all queued frames are written."""
        if self._log_thread is None:
            return
        self._log_stop.set()
        self._log_thread.join(timeout=15)
        self._log_thread = None

    def get_all_entries(self) -> list[GalleryEntry]:
        return list(self._gid_to_entry.values())

    def size(self) -> int:
        return len(self._gid_to_entry)

    def __repr__(self):
        active = sum(1 for e in self._gid_to_entry.values() if e.active_tid is not None)
        return f"GlobalRegistry(total={len(self._gid_to_entry)}, active={active}, threshold={self.match_threshold})"
