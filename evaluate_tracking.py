from __future__ import annotations

import glob
import json
import os
import pickle
import re
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

_CAM_RE = re.compile(r"cam_(\d+)")
_SEQ_RE = re.compile(r"seq_\d+")


class EvaluateTracking:
    """Accumulates CHIRLA ground truth + predictions and scores re-id /
    person identity accuracy only - MOTA and HOTA are deliberately out of
    scope here and will be added later once reid scoring is trusted.

    Standalone by design: no import from chirla_tracking.py, so this file
    can evolve (or replace it) independently.

    unassigned_id: sentinel meaning "no global identity assigned yet" (0 -
    GlobalRegistry/STrack.t_global_id convention). Never scored right-or-
    wrong; a detection sitting on it is simply skipped everywhere below.
    """

    REID_IOU_THRESH = 0.5

    def __init__(self, unassigned_id: int = 0):
        self.unassigned_id = unassigned_id
        # gt[cam_source][frame_id] = [(gt_id, bbox_xyxy), ...]
        self._gt: dict = defaultdict(lambda: defaultdict(list))
        # pred[cam_source][frame_id] = [(track_id, global_id, bbox_xyxy), ...]
        self._pred: dict = defaultdict(lambda: defaultdict(list))

    # --- source/sequence resolution ---------------------------------------

    @staticmethod
    def _parse_cam_source(path: str):
        m = _CAM_RE.search(os.path.basename(path))
        return int(m.group(1)) if m else os.path.basename(path)

    @classmethod
    def resolve_seq_from_sources(cls, sources, dataset_root: str):
        """Match a DeepStream app_config.yml source-list (file:// URIs)
        against the CHIRLA dataset layout (<dataset_root>/annotations/
        seq_<NNN>/cam_<N>.json) so the caller never hand-picks a
        ground-truth directory or camera numbering - both are recovered
        from the video paths the pipeline is already reading.

        Matches by the seq_<NNN> path component shared between a video's
        path (.../videos/seq_<NNN>/cam_<N>.avi) and its annotation
        (.../annotations/seq_<NNN>/cam_<N>.json): CHIRLA reuses the same
        cam_<N> filename in every sequence, so the sequence folder name
        is the only thing that disambiguates which annotation a video
        belongs to.

        Returns (seq_gt_dir, {source_index: cam_source}); seq_gt_dir is
        None and the mapping is empty if no source matches any CHIRLA
        annotation.
        """
        ann_root = os.path.join(dataset_root, "annotations")
        source_to_cam: dict = {}
        gt_dir = None
        for i, uri in enumerate(sources):
            path = uri[len("file://"):] if uri.startswith("file://") else uri
            cam = cls._parse_cam_source(path)
            if not isinstance(cam, int):
                continue
            seq_match = _SEQ_RE.search(path)
            if seq_match is None:
                continue
            json_path = os.path.join(ann_root, seq_match.group(0), f"cam_{cam}.json")
            if os.path.exists(json_path):
                source_to_cam[i] = cam
                gt_dir = os.path.dirname(json_path)
        return gt_dir, source_to_cam

    @classmethod
    def from_sources(cls, sources, dataset_root: str, unassigned_id: int = 0):
        """The "fetch from the sources only, which sequence to compare to"
        entry point: resolve which CHIRLA sequence + cameras a pipeline's
        source-list corresponds to, load ground truth for only those
        cameras (a CHIRLA seq dir can hold up to 7 camera_*.json files;
        loading ones this run never produced a single prediction for would
        manufacture meaningless FNs), and return (evaluator, source_to_cam)
        so the caller can map its own source_id -> cam_source when calling
        add_prediction(). Returns (None, {}) if no source matched.
        """
        gt_dir, source_to_cam = cls.resolve_seq_from_sources(sources, dataset_root)
        if gt_dir is None:
            return None, {}
        ev = cls(unassigned_id=unassigned_id)
        for cam in sorted(set(source_to_cam.values())):
            for json_path in glob.glob(os.path.join(gt_dir, f"cam_{cam}.json")):
                ev.load_ground_truth_json(json_path, cam_source=cam)
        return ev, source_to_cam

    # --- loading -----------------------------------------------------------

    def load_ground_truth_json(self, path: str, cam_source=None) -> None:
        """Load one CHIRLA `cam_<N>.json` annotation file.
        cam_source defaults to the <N> parsed out of the filename."""
        if cam_source is None:
            cam_source = self._parse_cam_source(path)
        with open(path, "r") as f:
            raw = json.load(f)
        for frame_str, dets in raw.items():
            frame_id = int(frame_str)
            for det in dets:
                bbox = np.asarray(det["BboxP"], dtype=np.float64)
                self._gt[cam_source][frame_id].append((int(det["id"]), bbox))

    def load_ground_truth_seq(self, seq_dir: str) -> None:
        """Load every `camera_*.json` in a CHIRLA sequence directory."""
        for path in sorted(glob.glob(os.path.join(seq_dir, "cam_*.json"))):
            self.load_ground_truth_json(path)

    def add_prediction(self, cam_source, frame_id: int, track_id: int,
                       global_id: int, bbox_xyxy) -> None:
        """Record one predicted detection. Call once per live track per
        frame regardless of whether its identity just changed - shift
        detection (see _compute_shifts) walks this per-frame history."""
        self._pred[cam_source][frame_id].append(
            (track_id, global_id, np.asarray(bbox_xyxy, dtype=np.float64))
        )

    def add_predictions_frame(self, cam_source, frame_id: int, dets) -> None:
        """dets: iterable of (track_id, global_id, bbox_xyxy)."""
        for track_id, global_id, bbox in dets:
            self.add_prediction(cam_source, frame_id, track_id, global_id, bbox)

    def dump_raw(self, path: str) -> None:
        """Pickle the accumulated (ground truth, predictions) to disk - lets
        you re-run/trace evaluate()'s internals offline without re-running
        the pipeline that produced them."""
        gt = {cam: dict(frames) for cam, frames in self._gt.items()}
        pred = {cam: dict(frames) for cam, frames in self._pred.items()}
        with open(path, "wb") as f:
            pickle.dump({"gt": gt, "pred": pred, "unassigned_id": self.unassigned_id}, f)

    @classmethod
    def load_raw(cls, path: str) -> "EvaluateTracking":
        """Inverse of dump_raw()."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        ev = cls(unassigned_id=data["unassigned_id"])
        for cam, frames in data["gt"].items():
            for frame_id, dets in frames.items():
                ev._gt[cam][frame_id] = dets
        for cam, frames in data["pred"].items():
            for frame_id, dets in frames.items():
                ev._pred[cam][frame_id] = dets
        return ev

    # --- matching ------------------------------------------------------------

    @staticmethod
    def _iou_matrix(boxes_a, boxes_b) -> np.ndarray:
        """Vectorized IoU, boxes in xyxy."""
        a = np.asarray(boxes_a, dtype=np.float64)
        b = np.asarray(boxes_b, dtype=np.float64)
        if len(a) == 0 or len(b) == 0:
            return np.zeros((len(a), len(b)), dtype=np.float64)
        area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
        area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
        x1 = np.maximum(a[:, None, 0], b[None, :, 0])
        y1 = np.maximum(a[:, None, 1], b[None, :, 1])
        x2 = np.minimum(a[:, None, 2], b[None, :, 2])
        y2 = np.minimum(a[:, None, 3], b[None, :, 3])
        inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
        union = area_a[:, None] + area_b[None, :] - inter
        return np.where(union > 1e-9, inter / union, 0.0)

    def _hungarian_match(self, gt_dets, pred_dets, iou_thresh: float):
        """One frame: Hungarian-match GT boxes ("tracklets") to predicted
        boxes gated at iou_thresh - the "tracklets with similar IOUs" step.
        Returns (matched_pairs, gt_boxes, pred_boxes) where matched_pairs
        is a list of (gt_row, pred_col, iou)."""
        gt_boxes = [g[1] for g in gt_dets]
        pred_boxes = [p[2] for p in pred_dets]
        if not gt_boxes or not pred_boxes:
            return [], gt_boxes, pred_boxes
        iou = self._iou_matrix(gt_boxes, pred_boxes)
        cost = 1.0 - iou
        cost[iou < iou_thresh] = 1e6
        rows, cols = linear_sum_assignment(cost)
        matched = [(r, c, float(iou[r, c])) for r, c in zip(rows, cols) if iou[r, c] >= iou_thresh]
        return matched, gt_boxes, pred_boxes

    def _gt_frames(self, cam_source=None):
        """Yield (cam, frame_id, gt_dets, pred_dets) for every frame present
        in the CHIRLA annotation json - i.e. driven off ground truth, not
        off whatever frames the tracker happened to predict on. A frame
        with no annotation has nothing to score reid against."""
        cams = [cam_source] if cam_source is not None else sorted(self._gt, key=str)
        for cam in cams:
            for frame_id in sorted(self._gt.get(cam, {})):
                yield cam, frame_id, self._gt[cam][frame_id], self._pred.get(cam, {}).get(frame_id, [])

    def _majority_gt_mapping(self, iou_thresh: float, cam_source=None) -> dict:
        """global_id -> the GT id it overlapped with most often (majority
        vote over every frame it was matched). CHIRLA's GT ids and this
        system's global_ids are different numbering schemes with no fixed
        correspondence, so this is the reconciliation step between them."""
        votes: dict = defaultdict(Counter)
        for cam, frame_id, gt_dets, pred_dets in self._gt_frames(cam_source):
            real_preds = [(t, g, b) for t, g, b in pred_dets if g != self.unassigned_id]
            matched, _, _ = self._hungarian_match(gt_dets, real_preds, iou_thresh)
            gt_ids = [g[0] for g in gt_dets]
            for r, c, _ in matched:
                _, gid, _ = real_preds[c]
                votes[gid][gt_ids[r]] += 1
        return {gid: counter.most_common(1)[0][0] for gid, counter in votes.items()}

    # --- reid / person accuracy ---------------------------------------------

    def build_match_records(self, iou_thresh: Optional[float] = None, cam_source=None) -> dict:
        """The unordered map: one entry per (frame, matched GT tracklet)
        that has a real (non-unassigned) global_id sitting on it, keyed by
        (cam_source, frame_id, gt_tracking_id) ->
            {one_hot_label, gt_tracking_id, my_tracking_id}

        one_hot_label is 1 if this global_id's majority-vote identity (see
        _majority_gt_mapping) equals the GT identity actually matched here
        this frame, else 0 - i.e. did the re-id system attach the right
        person to my_tracking_id. Detections whose global_id never matched
        anywhere else (no majority entry) are skipped - nothing to grade
        against yet.
        """
        iou_thresh = self.REID_IOU_THRESH if iou_thresh is None else iou_thresh
        majority = self._majority_gt_mapping(iou_thresh, cam_source)

        records: dict = {}
        for cam, frame_id, gt_dets, pred_dets in self._gt_frames(cam_source):
            real_preds = [(t, g, b) for t, g, b in pred_dets if g != self.unassigned_id]
            matched, _, _ = self._hungarian_match(gt_dets, real_preds, iou_thresh)
            gt_ids = [g[0] for g in gt_dets]
            for r, c, _ in matched:
                _, global_id, _ = real_preds[c]
                gt_tracking_id = gt_ids[r]
                mapped_gt_id = majority.get(global_id)
                if mapped_gt_id is None:
                    continue
                one_hot_label = 1 if mapped_gt_id == gt_tracking_id else 0
                records[(cam, frame_id, gt_tracking_id)] = dict(
                    one_hot_label=one_hot_label,
                    gt_tracking_id=gt_tracking_id,
                    my_tracking_id=global_id,
                )
        return records

    def _gid_history(self, cam_source=None) -> dict:
        """(cam, track_id) -> [(frame_id, global_id), ...] sorted by frame,
        replayed from every add_prediction() call so far. Shared by
        _compute_shifts and _compute_query_events, which each walk this
        same timeline looking for a different kind of assignment event."""
        history: dict = defaultdict(list)
        cams = [cam_source] if cam_source is not None else sorted(self._pred, key=str)
        for cam in cams:
            for frame_id in sorted(self._pred.get(cam, {})):
                for track_id, global_id, _ in self._pred[cam][frame_id]:
                    history[(cam, track_id)].append((frame_id, global_id))
        for seq in history.values():
            seq.sort(key=lambda x: x[0])
        return history

    def _grade_gid_events(self, events, majority: dict, iou_thresh: float) -> tuple:
        """Score a list of {cam, frame_id, track_id, new_gid} decision
        events against the majority mapping at the frame each landed on:
        was the new global_id's majority identity the GT identity actually
        matched there? Mutates each event with an 'outcome' key
        ('correct' / 'wrong' / 'unresolved') and returns (n_correct,
        n_wrong, n_unscored)."""
        n_correct = n_wrong = n_unscored = 0
        for e in events:
            gt_dets = self._gt.get(e["cam"], {}).get(e["frame_id"], [])
            preds_here = self._pred.get(e["cam"], {}).get(e["frame_id"], [])
            bbox = next((b for t, g, b in preds_here if t == e["track_id"] and g == e["new_gid"]), None)
            mapped_gt_id = majority.get(e["new_gid"])
            if bbox is None or mapped_gt_id is None or not gt_dets:
                n_unscored += 1
                e["outcome"] = "unresolved"
                continue
            matched, _, _ = self._hungarian_match(gt_dets, [(0, e["new_gid"], bbox)], iou_thresh)
            if not matched:
                n_unscored += 1
                e["outcome"] = "unresolved"
                continue
            actual_gt_id = gt_dets[matched[0][0]][0]
            if actual_gt_id == mapped_gt_id:
                n_correct += 1
                e["outcome"] = "correct"
            else:
                n_wrong += 1
                e["outcome"] = "wrong"
        return n_correct, n_wrong, n_unscored

    def _compute_shifts(self, iou_thresh: float, cam_source=None) -> dict:
        """Walk each (camera, track_id)'s global_id over time and flag
        every frame where it CHANGES from one real identity to a different
        one (a re-id revoke+remint or ambiguity resolution). Each shift is
        graded against the majority mapping at the frame it landed on:
        was the new global_id's majority identity the correct GT identity
        right then? A shift landing on unassigned (revoke with no
        immediate re-mint) is recorded but not scored.
        """
        majority = self._majority_gt_mapping(iou_thresh, cam_source)
        history = self._gid_history(cam_source)

        shifts = []
        for (cam, track_id), seq in history.items():
            prev_gid = None
            for frame_id, gid in seq:
                if prev_gid is not None and gid != prev_gid and prev_gid != self.unassigned_id:
                    shifts.append(dict(cam=cam, track_id=track_id, frame_id=frame_id,
                                       old_gid=prev_gid, new_gid=gid))
                prev_gid = gid

        for s in shifts:
            if s["new_gid"] == self.unassigned_id:
                s["outcome"] = "revoked_pending"
        scorable = [s for s in shifts if s["new_gid"] != self.unassigned_id]
        n_correct, n_wrong, n_unscored = self._grade_gid_events(scorable, majority, iou_thresh)
        n_unscored += len(shifts) - len(scorable)

        return dict(shifts=shifts, n_correct=n_correct, n_wrong=n_wrong, n_unscored=n_unscored)

    def _compute_query_events(self, iou_thresh: float, cam_source=None) -> dict:
        """Every identity commitment traceable to one GlobalRegistry.query()
        call (botsort/global_registry.py step() calls query() exactly once
        per unassigned track per frame): a track's first real global_id -
        the MATCH / track-split / NO_MATCH-mint outcome of that call, or of
        the interval solver later resolving a deferred AMBIGUOUS verdict -
        plus every subsequent reassignment. This is what query_accuracy in
        compute_reid_accuracy() reports: was the identity query() (or the
        solver acting on its behalf) committed to actually correct? That's
        a different question from reid_accuracy, which re-scores every
        single frame a track is already carrying an identity and so is
        dominated by repeat confirmations rather than real decisions.
        """
        majority = self._majority_gt_mapping(iou_thresh, cam_source)
        history = self._gid_history(cam_source)

        events = []
        for (cam, track_id), seq in history.items():
            prev_gid = None
            for frame_id, gid in seq:
                if gid != self.unassigned_id and gid != prev_gid:
                    events.append(dict(cam=cam, track_id=track_id, frame_id=frame_id,
                                       old_gid=prev_gid, new_gid=gid))
                prev_gid = gid

        n_correct, n_wrong, n_unscored = self._grade_gid_events(events, majority, iou_thresh)
        return dict(events=events, n_correct=n_correct, n_wrong=n_wrong, n_unscored=n_unscored)

    def compute_reid_accuracy(self, iou_thresh: Optional[float] = None, cam_source=None) -> dict:
        """Person re-id accuracy: fraction of matched, identity-assigned
        detections whose global_id correctly points at the GT person
        present that frame (built from build_match_records), plus how
        often the re-id system's identity shifts were themselves correct,
        plus query_accuracy - the same correctness question asked only at
        the moments a GlobalRegistry.query() call (or the solver resolving
        one of its AMBIGUOUS deferrals) actually committed an identity;
        see _compute_query_events.
        """
        iou_thresh = self.REID_IOU_THRESH if iou_thresh is None else iou_thresh
        records = self.build_match_records(iou_thresh, cam_source)

        n_correct = sum(1 for v in records.values() if v["one_hot_label"] == 1)
        n_wrong = sum(1 for v in records.values() if v["one_hot_label"] == 0)
        scored = n_correct + n_wrong
        reid_accuracy = n_correct / scored if scored > 0 else float("nan")

        shift_info = self._compute_shifts(iou_thresh, cam_source)
        query_info = self._compute_query_events(iou_thresh, cam_source)
        query_scored = query_info["n_correct"] + query_info["n_wrong"]
        query_accuracy = query_info["n_correct"] / query_scored if query_scored > 0 else float("nan")

        return dict(
            reid_accuracy=reid_accuracy, n_correct=n_correct, n_wrong=n_wrong,
            num_shifts=len(shift_info["shifts"]),
            shift_n_correct=shift_info["n_correct"],
            shift_n_wrong=shift_info["n_wrong"],
            shift_n_unscored=shift_info["n_unscored"],
            shifts=shift_info["shifts"],
            query_accuracy=query_accuracy,
            query_n_correct=query_info["n_correct"],
            query_n_wrong=query_info["n_wrong"],
            query_n_unscored=query_info["n_unscored"],
            query_events=query_info["events"],
            records=records,
        )

    def evaluate(self, cam_source=None) -> dict:
        """Full report - reid/person accuracy only for now. MOTA and HOTA
        will be added here once reid scoring is validated."""
        return dict(reid=self.compute_reid_accuracy(cam_source=cam_source))

    @staticmethod
    def print_report(report: dict) -> None:
        r = report["reid"]
        print(f"reid_accuracy={r['reid_accuracy']:.4f}  "
              f"(n_correct={r['n_correct']} n_wrong={r['n_wrong']})")
        print(f"shifts={r['num_shifts']}  "
              f"(n_correct={r['shift_n_correct']} n_wrong={r['shift_n_wrong']} "
              f"unscored={r['shift_n_unscored']})")
        print(f"query_accuracy={r['query_accuracy']:.4f}  "
              f"(n_correct={r['query_n_correct']} n_wrong={r['query_n_wrong']} "
              f"unscored={r['query_n_unscored']})")


# --- standalone smoke test against real CHIRLA ground truth -----------------
# run: `python3 evaluate_tracking.py [path/to/CHIRLA_dataset/annotations/seq_NNN]`
# Uses real GT boxes as "predictions" too (perfect tracker) as a sanity
# check, then deliberately swaps two identities halfway through to confirm
# reid_accuracy and shifts degrade in the expected direction.

def _self_test(seq_dir: str) -> bool:
    ev = EvaluateTracking()
    ev.load_ground_truth_seq(seq_dir)
    if not ev._gt:
        print(f"[evaluate_tracking self-test] no ground truth found under {seq_dir}, skipping")
        return True

    for cam, frames in ev._gt.items():
        for frame_id, dets in frames.items():
            for gt_id, bbox in dets:
                ev.add_prediction(cam, frame_id, gt_id, gt_id, bbox)
    perfect = ev.evaluate()["reid"]
    ok_perfect = (abs(perfect["reid_accuracy"] - 1.0) < 1e-9 and perfect["num_shifts"] == 0)
    print(f"[perfect predictions] reid_accuracy={perfect['reid_accuracy']:.4f} "
          f"shifts={perfect['num_shifts']} -> {'PASS' if ok_perfect else 'FAIL'}")

    ev2 = EvaluateTracking()
    ev2._gt = ev._gt
    counts = Counter()
    for cam, frames in ev._gt.items():
        for dets in frames.values():
            for gt_id, _ in dets:
                counts[gt_id] += 1
    if len(counts) < 2:
        print("[evaluate_tracking self-test] fewer than 2 identities in GT, skipping swap test")
        return ok_perfect
    (id_a, _), (id_b, _) = counts.most_common(2)
    cam0 = next(iter(ev._gt))
    frame_ids = sorted(ev._gt[cam0])
    swap_from = frame_ids[len(frame_ids) // 2]

    for cam, frames in ev._gt.items():
        for frame_id, dets in frames.items():
            for gt_id, bbox in dets:
                gid = gt_id
                if frame_id >= swap_from and gt_id in (id_a, id_b):
                    gid = id_b if gt_id == id_a else id_a
                ev2.add_prediction(cam, frame_id, gt_id, gid, bbox)
    corrupted = ev2.evaluate()["reid"]
    print(f"[id_a<->id_b swapped after frame {swap_from}] "
          f"reid_accuracy={corrupted['reid_accuracy']:.4f} shifts={corrupted['num_shifts']} "
          f"(shift n_correct={corrupted['shift_n_correct']} n_wrong={corrupted['shift_n_wrong']})")
    ok_corrupt = corrupted["reid_accuracy"] < 1.0 and corrupted["num_shifts"] > 0
    print(f"[evaluate_tracking self-test] degradation-direction check -> {'PASS' if ok_corrupt else 'FAIL'}")
    return ok_perfect and ok_corrupt


if __name__ == "__main__":
    import sys
    default_dir = ("/home/lakshh/workspace/reid/datasets/"
                   "2247f442a9784b5c959e7bead89c0313_V2/CHIRLA_dataset/annotations/seq_001")
    seq_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    sys.exit(0 if _self_test(seq_dir) else 1)
