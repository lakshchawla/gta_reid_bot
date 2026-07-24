"""
Per-camera person-dwelling (loitering) detection, layered on top of the
cross-camera GTA identity engine (botsort/global_tracklet_association.py).
Ported from DSMetroPerception_py/src/perception/dwelling.py, adapted for
this project's frame_id convention and generalized to configurable,
multi-level alert severities.

Unlike a station-wide "how long has this person been anywhere" metric, the
dwell clock is scoped PER CAMERA on purpose - a person can legitimately
dwell in one camera's zone while just passing through another's, and a
global aggregate would conflate the two. Every camera in this pipeline gets
its own independent dwell clock per global_id, keyed by (cam_id, global_id).

Driven from each tick's own track results (the {cam_id: [STrack, ...]} dict
MultiCameraTracker.update_batch() already returns), NOT from any GTA
introspection method: the gallery-based GTA keeps identities alive forever
once registered, with no per-camera visibility history to query - a
GalleryEntry only remembers the CURRENT active_tid/last_cam_source, not
"when did this identity first appear on camera X". Recomputing that here
from raw per-tick observations needs nothing from GTA beyond t_global_id -
and since an identity is never retroactively relabeled once assigned in this
GTA design, first-seen bookkeeping here can't be invalidated by a later GTA
decision.

DwellingMonitor.update() should be called once per batch tick (same cadence
as MultiCameraTracker.update_batch()) with the SAME frame_id passed to it.
frame_rate must match whatever units that frame_id is actually in (this
project drives GTA off frame_meta.frame_num - real decoder frame counts at
the source's native fps - not a wall-clock tick counter, so frame_rate here
should be that same fps, e.g. 30.0 for the CHIRLA dataset).

State cleanup: a (camera, identity) pair not re-observed within
recency_window_sec is treated as "already left this camera's view" and
pruned, so this never accumulates more state than "currently present +
recently departed", per camera, regardless of daily foot traffic. The grace
period matters specifically because BoT-SORT's own lost-track buffer means a
track can vanish from results for a while (brief occlusion, missed
detection) and reappear later still carrying the same global_id - without
this window, a routine gap would wrongly reset the dwell clock.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class SeverityLevel:
    """One escalating alert level, expressed as a multiple of
    DwellingMonitor.threshold_sec - e.g. multiplier=2.0 means "twice the
    base dwell threshold". Kept as data (not hardcoded WARNING/CRITICAL)
    so message/logging thresholds are tunable from the config file alone."""
    name: str
    multiplier: float


DEFAULT_SEVERITY_LEVELS = [SeverityLevel("WARNING", 1.0), SeverityLevel("CRITICAL", 2.0)]


class DwellingMonitor:
    """
    threshold_sec         : seconds a global_id must be continuously present
                             on one camera before it counts as dwelling - the
                             base/first severity level (multiplier=1.0).
    recency_window_sec    : grace period - a (camera, global_id) pair not
                             re-observed within this many seconds is treated
                             as departed and its dwell clock is reset.
    re_alert_sec          : minimum seconds between repeat alerts for the
                             SAME still-dwelling pair AT THE SAME severity
                             level. A NEW, higher severity level always
                             alerts immediately regardless of this cooldown -
                             an escalating situation shouldn't wait out a
                             cooldown meant for "still the same old news".
    frame_rate            : ticks-per-second of whatever frame_id update()
                             is driven with (see module docstring).
    severity_levels       : ascending list[SeverityLevel]; first entry should
                             have multiplier=1.0 (the base threshold IS that
                             level). Defaults to WARNING(1x)/CRITICAL(2x).
    log_console/log_path  : where alerts go - see module docstring / the
                             `logging:` block in ds_include/dwelling_config.yml.
    """

    def __init__(
        self,
        threshold_sec: float = 600.0,
        recency_window_sec: float = 90.0,
        re_alert_sec: float = 300.0,
        frame_rate: float = 30.0,
        severity_levels: Optional[list] = None,
        log_console: bool = True,
        log_path: Optional[str] = None,
    ):
        self.threshold_sec = threshold_sec
        self.recency_window_sec = recency_window_sec
        self.re_alert_sec = re_alert_sec
        self.frame_rate = frame_rate
        self.severity_levels = sorted(severity_levels or DEFAULT_SEVERITY_LEVELS,
                                       key=lambda s: s.multiplier)
        self.log_console = log_console
        self.log_path = log_path
        if log_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)

        self._first_seen: dict[tuple, int] = {}     # (cam_id, gid) -> frame_id first observed
        self._last_seen: dict[tuple, int] = {}       # (cam_id, gid) -> frame_id last observed
        self._dwell_sec: dict[tuple, float] = {}
        self._dwelling_keys: set[tuple] = set()
        self._last_alert_monotonic: dict[tuple, float] = {}
        self._last_alert_severity: dict[tuple, str] = {}   # highest severity NAME already alerted

    def update(self, results: dict, current_frame_id: int) -> None:
        """results: {cam_id: [STrack, ...]} - exactly what
        MultiCameraTracker.update_batch() returns. Every currently-tracked
        (cam_id, t_global_id) pair with a resolved (nonzero) global_id
        counts as "seen now" on that camera."""
        for cam_id, tracks in results.items():
            for t in tracks:
                gid = t.t_global_id
                if gid <= 0:
                    continue
                key = (cam_id, gid)
                self._last_seen[key] = current_frame_id
                self._first_seen.setdefault(key, current_frame_id)

        stale = []
        for key, first in self._first_seen.items():
            last = self._last_seen.get(key, first)
            recency_sec = (current_frame_id - last) / self.frame_rate
            if recency_sec > self.recency_window_sec:
                stale.append(key)
                continue

            cam_id, gid = key
            dwell_sec = (current_frame_id - first) / self.frame_rate
            self._dwell_sec[key] = dwell_sec
            if dwell_sec >= self.threshold_sec:
                self._dwelling_keys.add(key)
                self._maybe_alert(cam_id, gid, dwell_sec)
            else:
                self._dwelling_keys.discard(key)

        for key in stale:
            self._first_seen.pop(key, None)
            self._last_seen.pop(key, None)
            self._dwell_sec.pop(key, None)
            self._dwelling_keys.discard(key)
            self._last_alert_monotonic.pop(key, None)
            self._last_alert_severity.pop(key, None)

    def _severity_rank(self, name: str) -> int:
        for i, level in enumerate(self.severity_levels):
            if level.name == name:
                return i
        return -1

    def _current_severity(self, dwell_sec: float) -> Optional[SeverityLevel]:
        """Highest severity level whose threshold (multiplier * threshold_sec)
        this dwell time has cleared. None only if it somehow hasn't cleared
        even the first level - update() only calls this once
        dwell_sec >= threshold_sec, so that shouldn't happen in practice."""
        hit = None
        for level in self.severity_levels:
            if dwell_sec >= level.multiplier * self.threshold_sec:
                hit = level
        return hit

    def _maybe_alert(self, cam_id, identity_id: int, dwell_sec: float) -> None:
        level = self._current_severity(dwell_sec)
        if level is None:
            return
        key = (cam_id, identity_id)
        now = time.monotonic()
        prev_severity = self._last_alert_severity.get(key)
        escalated = (prev_severity is not None
                     and self._severity_rank(level.name) > self._severity_rank(prev_severity))
        last_alert = self._last_alert_monotonic.get(key)
        if not escalated and last_alert is not None and now - last_alert < self.re_alert_sec:
            return

        self._last_alert_monotonic[key] = now
        self._last_alert_severity[key] = level.name
        self._emit(cam_id, identity_id, dwell_sec, level.name)

    def _emit(self, cam_id, identity_id: int, dwell_sec: float, severity: str) -> None:
        if self.log_console:
            sys.stdout.write(
                f"[DWELL-ALERT] {severity} camera {cam_id} global_id {identity_id}: "
                f"{dwell_sec / 60.0:.1f} min dwelling\n"
            )
        if self.log_path is not None:
            record = {
                "ts": time.time(),
                "camera": cam_id,
                "global_id": identity_id,
                "severity": severity,
                "dwell_sec": round(dwell_sec, 1),
            }
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    def is_dwelling(self, cam_id, identity_id: int) -> bool:
        return (cam_id, identity_id) in self._dwelling_keys

    def dwell_seconds(self, cam_id, identity_id: int) -> float:
        return self._dwell_sec.get((cam_id, identity_id), 0.0)

    def severity(self, cam_id, identity_id: int) -> Optional[str]:
        """Last-alerted severity name for this (camera, identity), or None
        if it isn't currently dwelling / hasn't been alerted on yet."""
        return self._last_alert_severity.get((cam_id, identity_id))


def load_dwelling_config(path: str, frame_rate: float = 30.0) -> Optional[DwellingMonitor]:
    """Build a DwellingMonitor from a YAML config file (see
    ds_include/dwelling_config.yml for the shape). Returns None if the file
    is missing or `dwelling.enabled` is false - callers should treat that as
    "dwelling monitoring disabled for this run", not an error."""
    if not os.path.exists(path):
        sys.stderr.write(f"[dwelling] config {path} not found - dwelling monitoring disabled\n")
        return None

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    dwell_cfg = raw.get("dwelling", {}) or {}
    if not bool(dwell_cfg.get("enabled", True)):
        sys.stdout.write("[dwelling] disabled (dwelling.enabled: false in config)\n")
        return None

    log_cfg = raw.get("logging", {}) or {}
    severity_raw = dwell_cfg.get("severity-levels") or []
    severity_levels = [
        SeverityLevel(name=str(lvl["name"]), multiplier=float(lvl["multiplier"]))
        for lvl in severity_raw
    ] or None

    return DwellingMonitor(
        threshold_sec=float(dwell_cfg.get("threshold-sec", 600.0)),
        recency_window_sec=float(dwell_cfg.get("recency-window-sec", 90.0)),
        re_alert_sec=float(dwell_cfg.get("re-alert-interval-sec", 300.0)),
        frame_rate=frame_rate,
        severity_levels=severity_levels,
        log_console=bool(log_cfg.get("console", True)),
        log_path=(log_cfg.get("log-path") or None),
    )
