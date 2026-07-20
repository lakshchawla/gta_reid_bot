"""
Compute-load benchmark for GlobalRegistry's association logic at
emb_dim=2048 - the part of the system the ReID-net trtexec benchmark
(benchmark_reid_variants.py) does NOT cover: FAISS gallery search,
centroid EMA update, and the O(gallery_size) index rebuild that runs
every frame a track's embedding changes.

Measures three things in isolation, at a sweep of gallery sizes (distinct
identities seen so far) since that's what this logic scales with, not
batch size:

  1. query()          - FAISS top-2 search for one fresh track's embedding
                         against the gallery. This is the per-unidentified-
                         track cost.
  2. _rebuild_index()  - full FAISS re-add of every gallery centroid.
                         Runs once per frame if ANY track's embedding
                         updated (the common case in steady state).
  3. step()            - the full per-frame driver: N live tracks each
                         refreshing an already-owned identity (add_embedding
                         + one dirty rebuild at the end), the realistic
                         steady-state hot path.

Usage: python3 benchmark_association_2048.py
"""
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from botsort.global_registry import GlobalRegistry  # noqa: E402

EMB_DIM = 2048
GALLERY_SIZES = [10, 50, 100, 300, 500, 1000]
LIVE_TRACKS = [5, 10, 20]
N_ITERS = 300


class FakeTrack:
    def __init__(self, tid, feat, gid=0, x=0.0):
        self.track_id = tid
        self.smooth_feat = feat
        # Spread tracks out on x so peer IoU is 0 (no synthetic occlusion
        # contaminating the steady-state benchmark).
        self.tlwh = np.array([x, 0, 10, 20], dtype=np.float64)
        self.t_global_id = gid
        self.tracklet_len = 999
        self.is_touching_edge = False

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret


class FakeTracker:
    def __init__(self, tracks, cam_source=1):
        self.tracked_stracks = tracks
        self.cam_source = cam_source


def unit_vec(dim):
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def build_gallery(n_identities):
    reg = GlobalRegistry(emb_dim=EMB_DIM, use_gpu=True, min_frames=0)
    frame_id = 0
    for i in range(n_identities):
        gid = reg._register(unit_vec(EMB_DIM), frame_id, np.array([0, 0, 10, 20], dtype=np.float64))
        reg._gid_to_entry[gid].active_tid = None
    reg._rebuild_index()
    return reg


def bench_query(reg, n_iters=N_ITERS):
    times = []
    for _ in range(n_iters):
        feat = unit_vec(EMB_DIM)
        t0 = time.perf_counter()
        reg.query(feat)
        times.append((time.perf_counter() - t0) * 1000.0)
    return np.array(times)


def bench_rebuild(reg, n_iters=N_ITERS):
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        reg._rebuild_index()
        times.append((time.perf_counter() - t0) * 1000.0)
    return np.array(times)


def bench_step(reg, n_live_tracks, n_iters=N_ITERS):
    gids = list(reg._gid_to_entry.keys())[:n_live_tracks]
    if len(gids) < n_live_tracks:
        raise ValueError("gallery smaller than requested live-track count")
    times = []
    for it in range(n_iters):
        tracks = [
            FakeTrack(tid=1000 + i, feat=unit_vec(EMB_DIM), gid=gid, x=i * 100.0)
            for i, gid in enumerate(gids)
        ]
        tracker = FakeTracker(tracks)
        t0 = time.perf_counter()
        reg.step(tracker, frame_id=it)
        times.append((time.perf_counter() - t0) * 1000.0)
    return np.array(times)


def summarize(label, arr_ms):
    print(
        f"  {label:<28} mean={arr_ms.mean():7.3f} ms  "
        f"p50={np.median(arr_ms):7.3f} ms  p95={np.percentile(arr_ms, 95):7.3f} ms  "
        f"max={arr_ms.max():7.3f} ms"
    )


def main():
    print(f"=== GlobalRegistry association benchmark @ emb_dim={EMB_DIM} ===")
    print(f"(FAISS GPU, {N_ITERS} iterations per measurement)\n")

    print("--- query() latency vs gallery size (single unidentified track) ---")
    for n in GALLERY_SIZES:
        reg = build_gallery(n)
        arr = bench_query(reg)
        summarize(f"gallery={n}", arr)

    print("\n--- _rebuild_index() latency vs gallery size (full FAISS re-add) ---")
    for n in GALLERY_SIZES:
        reg = build_gallery(n)
        arr = bench_rebuild(reg)
        summarize(f"gallery={n}", arr)

    print("\n--- step() latency: steady-state frame (N live, already-owned tracks) ---")
    for n in GALLERY_SIZES:
        for k in LIVE_TRACKS:
            if k > n:
                continue
            reg = build_gallery(n)
            arr = bench_step(reg, k)
            summarize(f"gallery={n:<5} live_tracks={k}", arr)


if __name__ == "__main__":
    main()
