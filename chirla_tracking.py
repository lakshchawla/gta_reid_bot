from __future__ import annotations

import glob
import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

_CAM_RE = re.compile(r"camera_(\d+)_")


class ChirlaTracking:
    """Accumulates ground truth + predictions across one or more cameras of
    one CHIRLA sequence, then computes HOTA, MOTA, and identity-consistency.

    unassigned_id: the sentinel value meaning "no global identity assigned
    yet" (0 throughout this codebase - GlobalRegistry/STrack.t_global_id
    convention). Never scored as right-or-wrong; see module docstring.
    """

    HOTA_ALPHAS = tuple(round(a, 2) for a in np.arange(0.05, 0.96, 0.05))  # 19 thresholds, standard sweep
    MOTA_IOU_THRESH = 0.5

    def __init__(self, unassigned_id: int = 0):
        self.unassigned_id = unassigned_id
        # gt[cam_source][frame_id] = [(gt_id, bbox_xyxy), ...]
        self._gt: dict = defaultdict(lambda: defaultdict(list))
        # pred[cam_source][frame_id] = [(track_id, global_id, bbox_xyxy), ...]
        self._pred: dict = defaultdict(lambda: defaultdict(list))

    # --- loading ---------------------------------------------------------

    @staticmethod
    def _parse_cam_source(path: str):
        m = _CAM_RE.search(os.path.basename(path))
        return int(m.group(1)) if m else os.path.basename(path)

    def load_ground_truth_json(self, path: str, cam_source=None) -> None:
        """Load one CHIRLA `camera_<N>_<timestamp>.json` annotation file.
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
        """Load every `camera_*.json` in a CHIRLA sequence directory (e.g.
        `.../CHIRLA_dataset/annotations/seq_001`), one cam_source per file."""
        for path in sorted(glob.glob(os.path.join(seq_dir, "camera_*.json"))):
            self.load_ground_truth_json(path)

    def add_prediction(self, cam_source, frame_id: int, track_id: int,
                       global_id: int, bbox_xyxy) -> None:
        """Record one predicted detection. Call once per live track per
        frame, regardless of whether its identity just changed - the
        per-frame history is what shift detection (see
        compute_id_consistency) walks."""
        self._pred[cam_source][frame_id].append(
            (track_id, global_id, np.asarray(bbox_xyxy, dtype=np.float64))
        )

    def add_predictions_frame(self, cam_source, frame_id: int, dets) -> None:
        """dets: iterable of (track_id, global_id, bbox_xyxy)."""
        for track_id, global_id, bbox in dets:
            self.add_prediction(cam_source, frame_id, track_id, global_id, bbox)

    def dump_raw(self, path: str) -> None:
        """Pickle the accumulated (ground truth, predictions) to disk - lets
        you re-run/trace evaluate()'s internals (e.g. why a specific shift
        went unscored) offline, without re-running the pipeline that
        produced them. Converts out of the nested defaultdict(lambda: ...)
        this class uses internally, which isn't picklable as-is."""
        gt = {cam: dict(frames) for cam, frames in self._gt.items()}
        pred = {cam: dict(frames) for cam, frames in self._pred.items()}
        with open(path, "wb") as f:
            pickle.dump({"gt": gt, "pred": pred, "unassigned_id": self.unassigned_id}, f)

    @classmethod
    def load_raw(cls, path: str) -> "ChirlaTracking":
        """Inverse of dump_raw() - rebuild a ChirlaTracking instance from a
        dumped (ground truth, predictions) snapshot."""
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

    # --- shared utilities --------------------------------------------------

    def _identity_key(self, cam_source, track_id, global_id):
        """See module docstring: an unassigned prediction gets a key unique
        to its own (camera, local track) so it stays self-consistent across
        frames without ever being comparable to a different real identity."""
        if global_id == self.unassigned_id:
            return ("unassigned", cam_source, track_id)
        return ("gid", global_id)

    @staticmethod
    def _iou_matrix(boxes_a, boxes_b) -> np.ndarray:
        """Vectorized IoU, boxes in xyxy. Pure numpy - no cython_bbox
        dependency, so this module stays usable standalone against any
        tracker's output, not just this repo's."""
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

    def _frames(self, cam_source=None):
        cams = [cam_source] if cam_source is not None else sorted(
            set(self._gt) | set(self._pred), key=str)
        for cam in cams:
            frame_ids = sorted(set(self._gt.get(cam, {})) | set(self._pred.get(cam, {})))
            for f in frame_ids:
                yield cam, f, self._gt[cam].get(f, []), self._pred[cam].get(f, [])

    def _hungarian_match(self, gt_dets, pred_dets, iou_thresh: float):
        """One frame: Hungarian-match GT boxes to predicted boxes gated at
        iou_thresh. Returns (matched_pairs, gt_boxes, pred_boxes) where
        matched_pairs is a list of (gt_row, pred_col, iou)."""
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

    # --- MOTA --------------------------------------------------------------

    def compute_mota(self, iou_thresh: Optional[float] = None, cam_source=None) -> dict:
        """CLEARMOT-style MOTA/MOTP, aggregated across all cameras (or one,
        via cam_source) since gt_id/global_id are shared namespaces across
        cameras in this dataset/system - an ID switch between cam_source=1
        and cam_source=3 for the same physical person is exactly the
        cross-camera re-id failure this whole system exists to prevent, and
        must show up here, not be hidden by evaluating cameras separately.

        Matching preserves the previous frame's correspondence whenever
        it's still valid (standard CLEARMOT tie-break), so Hungarian
        doesn't manufacture spurious switches among near-equal-cost
        alternatives. A transition into/out of the unassigned identity key
        is never counted as a switch (see _identity_key / module
        docstring) - the tracker declining to commit isn't a wrong
        association.
        """
        iou_thresh = self.MOTA_IOU_THRESH if iou_thresh is None else iou_thresh
        tp = fp = fn = idsw = 0
        sum_iou = 0.0
        prev_match: dict = {}  # gt_id -> identity_key of last match

        for cam, frame_id, gt_dets, pred_dets in self._frames(cam_source):
            gt_ids = [g[0] for g in gt_dets]
            gt_boxes = [g[1] for g in gt_dets]
            pred_keys = [self._identity_key(cam, t, g) for t, g, _ in pred_dets]
            pred_boxes = [b for _, _, b in pred_dets]

            matched_gt_rows, matched_pred_cols = set(), set()
            if gt_boxes and pred_boxes:
                iou = self._iou_matrix(gt_boxes, pred_boxes)

                # Pass 1: keep prior correspondences that are still valid.
                pinned = []
                for r, gid in enumerate(gt_ids):
                    prior_key = prev_match.get(gid)
                    if prior_key is None:
                        continue
                    for c, key in enumerate(pred_keys):
                        if key == prior_key and iou[r, c] >= iou_thresh:
                            pinned.append((r, c))
                            break
                for r, c in pinned:
                    matched_gt_rows.add(r)
                    matched_pred_cols.add(c)

                # Pass 2: Hungarian over whatever's left.
                free_rows = [r for r in range(len(gt_ids)) if r not in matched_gt_rows]
                free_cols = [c for c in range(len(pred_keys)) if c not in matched_pred_cols]
                extra = []
                if free_rows and free_cols:
                    sub_cost = 1.0 - iou[np.ix_(free_rows, free_cols)]
                    sub_cost[sub_cost > 1.0 - iou_thresh] = 1e6
                    rr, cc = linear_sum_assignment(sub_cost)
                    for ri, ci in zip(rr, cc):
                        r, c = free_rows[ri], free_cols[ci]
                        if iou[r, c] >= iou_thresh:
                            extra.append((r, c))
                matched = pinned + extra
                for r, c in matched:
                    matched_gt_rows.add(r)
                    matched_pred_cols.add(c)
                    sum_iou += float(iou[r, c])

                    gid, key = gt_ids[r], pred_keys[c]
                    prior_key = prev_match.get(gid)
                    is_unassigned = (
                        (isinstance(prior_key, tuple) and prior_key[0] == "unassigned")
                        or (isinstance(key, tuple) and key[0] == "unassigned")
                    )
                    if prior_key is not None and prior_key != key and not is_unassigned:
                        idsw += 1
                    prev_match[gid] = key
                tp += len(matched)

            fn += len(gt_ids) - len(matched_gt_rows)
            fp += len(pred_boxes) - len(matched_pred_cols)
            # GT ids absent this frame drop out of prev_match so a much-later
            # reappearance isn't compared to a stale correspondence.
            for gid in list(prev_match):
                if gid not in gt_ids:
                    prev_match.pop(gid, None)

        total_gt = tp + fn
        mota = 1.0 - (fn + fp + idsw) / total_gt if total_gt > 0 else float("nan")
        motp = sum_iou / tp if tp > 0 else float("nan")
        return dict(MOTA=mota, MOTP=motp, TP=tp, FP=fp, FN=fn, IDSW=idsw, total_gt=total_gt)

    # --- HOTA ----------------------------------------------------------------

    def compute_hota(self, alphas=None, cam_source=None) -> dict:
        """Standard HOTA (Luiten et al., 2020): for each IoU threshold alpha,
        DetA = TP/(TP+FN+FP) from independent per-frame matching, AssA =
        the TP-match-weighted average of TPA/(TPA+FNA+FPA) per matched
        (gt_id, identity_key) pair accumulated over the WHOLE sequence
        (across all cameras - see compute_mota's docstring for why),
        HOTA_alpha = sqrt(DetA * AssA). Final scores are the mean over the
        alpha sweep (0.05..0.95 step 0.05, 19 thresholds).
        """
        alphas = alphas if alphas is not None else self.HOTA_ALPHAS
        frames = list(self._frames(cam_source))

        per_alpha = []
        for alpha in alphas:
            tp = fp = fn = 0
            pair_tp = Counter()     # TPA(c) for c=(gt_id, identity_key)
            gt_total = Counter()    # frames this gt_id appears at all
            pred_total = Counter()  # frames this identity_key appears at all
            sum_iou = 0.0

            for cam, frame_id, gt_dets, pred_dets in frames:
                gt_ids = [g[0] for g in gt_dets]
                pred_keys = [self._identity_key(cam, t, g) for t, g, _ in pred_dets]
                for k in gt_ids:
                    gt_total[k] += 1
                for k in pred_keys:
                    pred_total[k] += 1

                matched, gt_boxes, pred_boxes = self._hungarian_match(gt_dets, pred_dets, alpha)
                matched_rows = {r for r, _, _ in matched}
                matched_cols = {c for _, c, _ in matched}
                for r, c, iou_val in matched:
                    tp += 1
                    sum_iou += iou_val
                    pair_tp[(gt_ids[r], pred_keys[c])] += 1
                fn += len(gt_boxes) - len(matched_rows)
                fp += len(pred_boxes) - len(matched_cols)

            det_a = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else float("nan")

            ass_scores, weights = [], []
            for (gid, key), tpa in pair_tp.items():
                fna = gt_total[gid] - tpa
                fpa = pred_total[key] - tpa
                ass_scores.append(tpa / (tpa + fna + fpa))
                weights.append(tpa)
            ass_a = float(np.average(ass_scores, weights=weights)) if ass_scores else float("nan")

            hota_alpha = math.sqrt(det_a * ass_a) if tp > 0 else 0.0
            loc_a = sum_iou / tp if tp > 0 else float("nan")
            per_alpha.append(dict(alpha=alpha, TP=tp, FP=fp, FN=fn,
                                  DetA=det_a, AssA=ass_a, HOTA=hota_alpha, LocA=loc_a))

        def _mean(key):
            vals = [p[key] for p in per_alpha if not math.isnan(p[key])]
            return float(np.mean(vals)) if vals else float("nan")

        return dict(HOTA=_mean("HOTA"), DetA=_mean("DetA"), AssA=_mean("AssA"),
                   LocA=_mean("LocA"), per_alpha=per_alpha)

    # --- custom identity-consistency metric --------------------------------

    def _majority_gt_mapping(self, iou_thresh: float, cam_source=None) -> dict:
        """global_id -> the GT id it overlapped with most often (majority
        vote over every frame it was matched). This is the "ground truth
        IDs could be different from what I am assigning" reconciliation -
        no fixed numbering is assumed between the two ID spaces."""
        votes: dict = defaultdict(Counter)
        for cam, frame_id, gt_dets, pred_dets in self._frames(cam_source):
            real_preds = [(t, g, b) for t, g, b in pred_dets if g != self.unassigned_id]
            matched, _, _ = self._hungarian_match(gt_dets, real_preds, iou_thresh)
            gt_ids = [g[0] for g in gt_dets]
            for r, c, _ in matched:
                _, gid, _ = real_preds[c]
                votes[gid][gt_ids[r]] += 1
        return {gid: counter.most_common(1)[0][0] for gid, counter in votes.items()}

    def compute_id_consistency(self, iou_thresh: Optional[float] = None, cam_source=None) -> dict:
        """The metric you actually asked for, built on top of the majority
        mapping above:

          - Detections with global_id == unassigned_id are skipped entirely
            (never right, never wrong - "no issues if the id is not
            assigned").
          - Every OTHER detection that has a GT match this frame is scored:
            does its global_id's majority-mapped GT identity equal the GT
            id actually matched here? consistency_ratio is the fraction
            that do.
          - Separately: walks each (camera, track_id)'s global_id over time
            and flags every frame where it CHANGES from one real identity
            to a different one (an ambiguity resolution / interval-solver
            reassignment / revoke-then-remint - a "shift"). Each such shift
            is itself graded: was the new global_id's majority identity the
            correct GT identity at that frame? shift_accuracy is the
            fraction of shifts that were. A shift landing on unassigned
            (a revoke with no immediate re-mint) is recorded but not
            scored - same "no issues if 0" rule.
        """
        iou_thresh = self.MOTA_IOU_THRESH if iou_thresh is None else iou_thresh
        majority = self._majority_gt_mapping(iou_thresh, cam_source)

        consistent = inconsistent = unresolved = 0
        history: dict = defaultdict(list)  # (cam, track_id) -> [(frame_id, global_id)]

        for cam, frame_id, gt_dets, pred_dets in self._frames(cam_source):
            gt_ids = [g[0] for g in gt_dets]
            for track_id, global_id, bbox in pred_dets:
                history[(cam, track_id)].append((frame_id, global_id))
                if global_id == self.unassigned_id:
                    continue
                matched, _, _ = self._hungarian_match(gt_dets, [(0, global_id, bbox)], iou_thresh)
                if not matched:
                    continue  # detection-level FP, not an identity question
                actual_gt_id = gt_ids[matched[0][0]]
                mapped_gt_id = majority.get(global_id)
                if mapped_gt_id is None:
                    unresolved += 1
                elif mapped_gt_id == actual_gt_id:
                    consistent += 1
                else:
                    inconsistent += 1

        scored = consistent + inconsistent
        consistency_ratio = consistent / scored if scored > 0 else float("nan")

        shifts = []
        for (cam, track_id), seq in history.items():
            seq.sort(key=lambda x: x[0])
            prev_gid = None
            for frame_id, gid in seq:
                if prev_gid is not None and gid != prev_gid and prev_gid != self.unassigned_id:
                    shifts.append(dict(cam=cam, track_id=track_id, frame_id=frame_id,
                                       old_gid=prev_gid, new_gid=gid))
                prev_gid = gid

        shift_correct = shift_wrong = shift_unscored = 0
        for s in shifts:
            if s["new_gid"] == self.unassigned_id:
                shift_unscored += 1
                s["outcome"] = "revoked_pending"
                continue
            gt_dets = self._gt.get(s["cam"], {}).get(s["frame_id"], [])
            preds_here = self._pred.get(s["cam"], {}).get(s["frame_id"], [])
            bbox = next((b for t, g, b in preds_here if t == s["track_id"] and g == s["new_gid"]), None)
            mapped_gt_id = majority.get(s["new_gid"])
            if bbox is None or mapped_gt_id is None or not gt_dets:
                shift_unscored += 1
                s["outcome"] = "unresolved"
                continue
            matched, _, _ = self._hungarian_match(gt_dets, [(0, s["new_gid"], bbox)], iou_thresh)
            if not matched:
                shift_unscored += 1
                s["outcome"] = "unresolved"
                continue
            actual_gt_id = gt_dets[matched[0][0]][0]
            if actual_gt_id == mapped_gt_id:
                shift_correct += 1
                s["outcome"] = "correct"
            else:
                shift_wrong += 1
                s["outcome"] = "wrong"

        shift_scored = shift_correct + shift_wrong
        shift_accuracy = shift_correct / shift_scored if shift_scored > 0 else float("nan")

        return dict(
            consistency_ratio=consistency_ratio, consistent=consistent,
            inconsistent=inconsistent, unresolved=unresolved,
            num_shifts=len(shifts), shift_accuracy=shift_accuracy,
            shift_correct=shift_correct, shift_wrong=shift_wrong,
            shift_unscored=shift_unscored, shifts=shifts, gid_to_gt_id=majority,
        )

    # --- combined report -----------------------------------------------------

    def evaluate(self, cam_source=None) -> dict:
        return dict(
            mota=self.compute_mota(cam_source=cam_source),
            hota=self.compute_hota(cam_source=cam_source),
            id_consistency=self.compute_id_consistency(cam_source=cam_source),
        )

    @staticmethod
    def print_report(report: dict) -> None:
        m, h, c = report["mota"], report["hota"], report["id_consistency"]
        print(f"MOTA={m['MOTA']:.4f}  MOTP={m['MOTP']:.4f}  "
              f"TP={m['TP']} FP={m['FP']} FN={m['FN']} IDSW={m['IDSW']}")
        print(f"HOTA={h['HOTA']:.4f}  DetA={h['DetA']:.4f}  AssA={h['AssA']:.4f}  LocA={h['LocA']:.4f}")
        print(f"ID consistency={c['consistency_ratio']:.4f}  "
              f"(consistent={c['consistent']} inconsistent={c['inconsistent']} unresolved={c['unresolved']})")
        print(f"Shift accuracy={c['shift_accuracy']:.4f}  "
              f"(shifts={c['num_shifts']} correct={c['shift_correct']} "
              f"wrong={c['shift_wrong']} unscored={c['shift_unscored']})")


def resolve_seq_from_sources(sources, dataset_root: str):
    """Match a DeepStream app_config.yml source-list (file:// URIs) against
    the CHIRLA dataset layout (<dataset_root>/annotations/seq_*/camera_<N>_
    <timestamp>.json) so callers don't need to hand-configure a ground-truth
    directory or camera numbering - both are recovered from the video
    filenames the pipeline is already reading.

    Matches by (camera number, timestamp stem), normalizing ':' to '_' since
    ad-hoc copies of CHIRLA videos (e.g. dragged into ~/Downloads) commonly
    have their timestamp's colons sanitized to underscores by the
    filesystem/tool that copied them, while the dataset's own copy keeps the
    colons - same recording, cosmetically different filename.

    Returns (seq_gt_dir, {source_index: cam_source}); seq_gt_dir is None and
    the mapping is empty if no source matches any CHIRLA annotation.
    """
    def _stem(p):
        return os.path.splitext(os.path.basename(p))[0].replace(":", "_")

    ann_root = os.path.join(dataset_root, "annotations")
    source_to_cam: dict = {}
    gt_dir = None
    for i, uri in enumerate(sources):
        path = uri[len("file://"):] if uri.startswith("file://") else uri
        cam = ChirlaTracking._parse_cam_source(path)
        if not isinstance(cam, int):
            continue
        target = _stem(path)
        for json_path in glob.glob(os.path.join(ann_root, "seq_*", f"camera_{cam}_*.json")):
            if _stem(json_path) == target:
                source_to_cam[i] = cam
                gt_dir = os.path.dirname(json_path)
                break
    return gt_dir, source_to_cam


# --- standalone smoke test against real CHIRLA ground truth -----------------
# Not pytest-based (matches this repo's standalone *_test.py convention) -
# run: `python3 chirla_tracking.py [path/to/CHIRLA_dataset]`. Uses real GT
# boxes as "predictions" too (perfect tracker) as a sanity check, then
# deliberately corrupts a slice of frames to confirm the metrics degrade in
# the expected direction - this validates the implementation, not a tracker.

def _self_test(seq_dir: str) -> bool:
    import random

    ev = ChirlaTracking()
    ev.load_ground_truth_seq(seq_dir)
    if not ev._gt:
        print(f"[chirla_tracking self-test] no ground truth found under {seq_dir}, skipping")
        return True

    # Perfect predictions: feed GT straight back as track_id=global_id=gt_id.
    for cam, frames in ev._gt.items():
        for frame_id, dets in frames.items():
            for gt_id, bbox in dets:
                ev.add_prediction(cam, frame_id, gt_id, gt_id, bbox)
    perfect = ev.evaluate()
    ok_perfect = (abs(perfect["mota"]["MOTA"] - 1.0) < 1e-9
                  and abs(perfect["hota"]["HOTA"] - 1.0) < 1e-9
                  and abs(perfect["id_consistency"]["consistency_ratio"] - 1.0) < 1e-9)
    print(f"[perfect predictions] MOTA={perfect['mota']['MOTA']:.4f} "
          f"HOTA={perfect['hota']['HOTA']:.4f} "
          f"consistency={perfect['id_consistency']['consistency_ratio']:.4f} "
          f"-> {'PASS' if ok_perfect else 'FAIL'}")

    # Corrupt: swap the global_id of the two most frequent ids in the
    # second half of the sequence, on whichever camera has the most total
    # detections - simulates exactly the girl/guy identity-swap failure.
    ev2 = ChirlaTracking()
    ev2._gt = ev._gt
    counts = Counter()
    for cam, frames in ev._gt.items():
        for dets in frames.values():
            for gt_id, _ in dets:
                counts[gt_id] += 1
    if len(counts) < 2:
        print("[chirla_tracking self-test] fewer than 2 identities in GT, skipping swap test")
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
    corrupted = ev2.evaluate()
    print(f"[id_a<->id_b swapped after frame {swap_from}] "
          f"MOTA={corrupted['mota']['MOTA']:.4f} HOTA={corrupted['hota']['HOTA']:.4f} "
          f"consistency={corrupted['id_consistency']['consistency_ratio']:.4f} "
          f"shifts={corrupted['id_consistency']['num_shifts']} "
          f"shift_accuracy={corrupted['id_consistency']['shift_accuracy']:.4f}")
    ok_corrupt = (corrupted["mota"]["MOTA"] < perfect["mota"]["MOTA"]
                  and corrupted["hota"]["HOTA"] < perfect["hota"]["HOTA"]
                  and corrupted["id_consistency"]["consistency_ratio"] < 1.0
                  and corrupted["id_consistency"]["num_shifts"] > 0)
    print(f"[chirla_tracking self-test] degradation-direction check -> {'PASS' if ok_corrupt else 'FAIL'}")
    return ok_perfect and ok_corrupt


if __name__ == "__main__":
    import sys
    default_dir = ("/home/lakshh/workspace/reid/datasets/"
                   "2247f442a9784b5c959e7bead89c0313_V2/CHIRLA_dataset/annotations/seq_001")
    seq_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    sys.exit(0 if _self_test(seq_dir) else 1)
