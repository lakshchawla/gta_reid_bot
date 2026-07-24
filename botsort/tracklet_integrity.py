"""
Intra-tracklet contamination detector + split surgery.

Every fix discussed for cross-camera association (GlobalRegistry, GTA) assumes a
single-camera tracklet is already a clean, single-identity unit. It isn't always:
under heavy occlusion, BoT-SORT's own Hungarian association can hand one track_id to
a different person mid-track (two people cross paths, the box continues, the
identity underneath silently swaps). Feeding a two-identity node into even a perfect
cross-camera solver still produces a wrong answer - nothing downstream can tell one
"node" secretly contains two people.

TrackletIntegrityChecker watches each live track's raw per-frame embeddings (not the
EMA-smoothed `smooth_feat`, which is exactly what would blur a swap away) and
periodically fits a genuine 2-component vs 1-component Gaussian mixture. If it comes
back clearly bimodal - well-separated, well-supported components, not noise - that's
evidence of a swap: the track is split at the changepoint into two track_ids before
either cross-camera layer (GlobalRegistry or GTA) ever sees it as one node.

Deliberately independent of botsort/global_registry.py and
botsort/global_tracklet_association.py - it operates directly on BoTSORT/STrack
objects and only calls the one method both of those already expose,
`deactivate_track(track_id)`, to close out a contaminated node's evidence at the
moment of the split. The one accepted approximation: whichever cross-camera layer is
in use may have already refreshed the old node's evidence with a few
already-contaminated post-changepoint frames before this module catches up (detecting
a changepoint needs some trailing samples to confirm it isn't noise) - that residual
lag is not corrected retroactively, since doing so would require reaching into
GlobalRegistry/GTA's own state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from .bot_sort import STrack


@dataclass
class SplitEvent:
    track_id: int
    changepoint_frame: int
    pre_mean_feat: np.ndarray
    post_mean_feat: np.ndarray
    post_feats: np.ndarray        # raw post-changepoint samples, to reseed the new track_id's history
    post_frames: np.ndarray       # their frame ids
    pre_count: int
    post_count: int
    bic_gain: float
    separation: float


class _TrackHistory:
    """Bounded per-track_id ring buffer of raw (frame_id, feat) samples."""

    __slots__ = ("frames", "feats", "last_checked_frame")

    def __init__(self, maxlen: int):
        self.frames: deque = deque(maxlen=maxlen)
        self.feats: deque = deque(maxlen=maxlen)
        self.last_checked_frame: int = 0

    def append(self, frame_id: int, feat: np.ndarray) -> None:
        self.frames.append(frame_id)
        self.feats.append(feat)

    def __len__(self) -> int:
        return len(self.frames)


class TrackletIntegrityChecker:
    """
    history_maxlen        : max buffered samples per track (bounds memory for
                             long-lived tracks; oldest samples drop first).
    check_interval_frames : how often (in frames) an already-buffered-enough
                             track gets re-examined for bimodality.
    min_samples_for_gmm   : minimum buffered samples before a fit is attempted -
                             fitting 2 Gaussians on too few points is noise.
    bic_margin            : required BIC(1) - BIC(2) improvement to prefer the
                             2-component model over the 1-component baseline.
    min_component_weight  : each component must carry at least this fraction of
                             the samples - guards against a few outlier frames
                             masquerading as a second identity.
    min_separation        : minimum cosine distance between the two components'
                             responsibility-weighted centroids (computed back in
                             the original embedding space, not PCA units) to
                             call the split genuine.
    min_run_length        : the changepoint must be sustained for at least this
                             many consecutive samples on both sides - a single
                             flipped frame is noise, not a swap. If the
                             2-component labels flip more than once, decline to
                             split (this only handles the clean, one-swap case;
                             messier multi-flip signals are left alone rather
                             than guessed at).
    """

    def __init__(
        self,
        history_maxlen: int = 1800,
        check_interval_frames: int = 750,
        min_samples_for_gmm: int = 60,
        bic_margin: float = 10.0,
        min_component_weight: float = 0.2,
        min_separation: float = 0.3,
        min_run_length: int = 15,
    ):
        self.history_maxlen = history_maxlen
        self.check_interval_frames = check_interval_frames
        self.min_samples_for_gmm = min_samples_for_gmm
        self.bic_margin = bic_margin
        self.min_component_weight = min_component_weight
        self.min_separation = min_separation
        self.min_run_length = min_run_length

        self._history: dict[int, _TrackHistory] = {}

    # --- per-frame driver ----------------------------------------------------

    def observe(self, tracker, frame_id: int, registry) -> list:
        """Call once per camera per frame (not gated to a tick - a bimodal swap
        needs raw per-frame resolution to detect and localize). Buffers this
        frame's raw embeddings, periodically checks buffered-enough tracks for
        contamination, and immediately performs the split surgery on any that
        fire. Returns the SplitEvents applied this call, for logging/testing.

        `registry` is whichever cross-camera layer is in use (GTA or
        GlobalRegistry, both expose `deactivate_track`) - passed straight
        through to close out a contaminated node at the moment of the split.
        """
        applied: list[SplitEvent] = []
        for track in list(tracker.tracked_stracks):
            feat = track.curr_feat
            if feat is None:
                continue
            hist = self._history.get(track.track_id)
            if hist is None:
                hist = self._history[track.track_id] = _TrackHistory(self.history_maxlen)
            hist.append(frame_id, feat.astype(np.float32).copy())

            if (len(hist) < self.min_samples_for_gmm
                    or frame_id - hist.last_checked_frame < self.check_interval_frames):
                continue
            hist.last_checked_frame = frame_id

            event = self._check_bimodal(track.track_id, hist)
            if event is not None:
                self._apply_split(tracker, track, event, registry, frame_id)
                applied.append(event)
        return applied

    # --- detection -------------------------------------------------------------

    def _check_bimodal(self, track_id: int, hist: "_TrackHistory") -> Optional[SplitEvent]:
        feats = np.stack(hist.feats, axis=0)            # (N, D) unit vectors
        frames = np.asarray(hist.frames)                 # (N,)

        # Fit on the dominant axis of variation, not the raw high-dim vectors -
        # a handful-to-hundred samples against a 128-2048 dim covariance is
        # numerically fragile (near-singular); a real identity swap shows up
        # as separation along this axis regardless.
        proj = PCA(n_components=1).fit_transform(feats)

        gmm1 = GaussianMixture(n_components=1, random_state=0, n_init=1).fit(proj)
        gmm2 = GaussianMixture(n_components=2, random_state=0, n_init=3).fit(proj)
        bic_gain = gmm1.bic(proj) - gmm2.bic(proj)
        if bic_gain <= self.bic_margin:
            return None

        if min(gmm2.weights_) < self.min_component_weight:
            return None

        labels = gmm2.predict(proj)
        centroid_0 = feats[labels == 0].mean(axis=0)
        centroid_1 = feats[labels == 1].mean(axis=0)
        n0, n1 = np.linalg.norm(centroid_0), np.linalg.norm(centroid_1)
        if n0 < 1e-9 or n1 < 1e-9:
            return None
        separation = 1.0 - float(np.dot(centroid_0, centroid_1) / (n0 * n1))
        if separation < self.min_separation:
            return None

        changepoint_idx = self._find_single_changepoint(labels)
        if changepoint_idx is None:
            return None

        pre_feats = feats[:changepoint_idx]
        post_feats = feats[changepoint_idx:]
        post_frames = frames[changepoint_idx:]

        return SplitEvent(
            track_id=track_id,
            changepoint_frame=int(frames[changepoint_idx]),
            pre_mean_feat=self._unit_mean(pre_feats),
            post_mean_feat=self._unit_mean(post_feats),
            post_feats=post_feats.copy(),
            post_frames=post_frames.copy(),
            pre_count=len(pre_feats),
            post_count=len(post_feats),
            bic_gain=float(bic_gain),
            separation=float(separation),
        )

    def _find_single_changepoint(self, labels: np.ndarray) -> Optional[int]:
        """Index where the dominant label switches and stays switched for at
        least min_run_length samples on both sides. None if there's no clean
        single transition (never switches, or switches more than once - a
        genuine swap is one switch; repeated flips are noise/occlusion
        flicker, not the case this module handles)."""
        switches = np.where(np.diff(labels) != 0)[0] + 1
        if len(switches) != 1:
            return None
        idx = int(switches[0])
        if idx < self.min_run_length or (len(labels) - idx) < self.min_run_length:
            return None
        return idx

    @staticmethod
    def _unit_mean(feats: np.ndarray) -> np.ndarray:
        m = feats.mean(axis=0)
        n = np.linalg.norm(m)
        return (m / n).astype(np.float32) if n > 1e-9 else m.astype(np.float32)

    # --- surgery ---------------------------------------------------------------

    def _apply_split(self, tracker, old_track, event: SplitEvent, registry, frame_id: int) -> None:
        old_id = old_track.track_id

        if registry is not None:
            registry.deactivate_track(old_id)

        old_track.mark_removed()
        tracker.tracked_stracks = [t for t in tracker.tracked_stracks if t.track_id != old_id]
        tracker.removed_stracks.append(old_track)

        new_track = STrack(tlwh=old_track.tlwh, score=old_track.score,
                            feat=event.post_mean_feat.copy())
        new_track.activate(tracker.kalman_filter, frame_id, id_assigner=tracker.id_assigner)
        # activate() only sets is_activated=True when frame_id==1 (a fresh
        # video start); a split happens well into a run, so force it here -
        # otherwise BoT-SORT's own next update() would route this track into
        # the unconfirmed bucket and remove it on the first frame that
        # doesn't luckily IOU-match a detection.
        new_track.is_activated = True
        tracker.tracked_stracks.append(new_track)

        del self._history[old_id]
        new_hist = self._history[new_track.track_id] = _TrackHistory(self.history_maxlen)
        for f, feat in zip(event.post_frames, event.post_feats):
            new_hist.append(int(f), feat)
        new_hist.last_checked_frame = frame_id

        print(f"[TRACKLET-INTEGRITY] bimodal split: track={old_id} -> "
              f"new_track={new_track.track_id} changepoint_frame={event.changepoint_frame} "
              f"bic_gain={event.bic_gain:.2f} separation={event.separation:.3f} "
              f"(pre={event.pre_count} post={event.post_count})")
