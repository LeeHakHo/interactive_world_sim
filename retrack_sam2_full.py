"""全量 SAM2 object-flow re-track(2026-07-27, 用户批准)。
把验证过的 SAM2 recipe(原生分辨率+点prompt+手负点+clean_mask+网格+CoTracker+per-frame约束)跑全部 clip 两视角。
输出干净 tracks 替换 clips 的 tracks/tracks_low(+vis)。★seek解码(跳到目标帧, 免迭代1万帧); ★网格固定48点。
env: DOMAIN(human|robot)/VIEW(high|low)/SMOKE/CLIPS(逗号,调试)。
输出: outputs/flow_render_dataset_can_dual/retrack/{domain}_{view}.npz (tracks(N,L,48,2), vis(N,L,48))。
iws env(SAM2+CoTracker). 大job走sbatch(每 domain×view 一个)。
"""
import os, sys, numpy as np, cv2, av, torch
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, ".")
from footprint_warp import load_sam2
from viz_sam2_objflow import grid_in_mask, clean_mask
from PIL import Image
GDINO = "IDEA-Research/grounding-dino-base"
_GD = {}


def load_gdino():
    if "m" not in _GD:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        _GD["pr"] = AutoProcessor.from_pretrained(GDINO)
        _GD["m"] = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO).to(dev).eval()
    return _GD["pr"], _GD["m"]


def gdino_bbox(img):
    """检测 'a can' → 最高分 bbox [x0,y0,x1,y1](像素)。无检测返 None。"""
    pr, m = load_gdino()
    inp = pr(images=Image.fromarray(img), text="a can.", return_tensors="pt").to(dev)
    with torch.no_grad():
        out = m(**inp)
    res = pr.post_process_grounded_object_detection(out, inp["input_ids"], threshold=0.2, text_threshold=0.2, target_sizes=[img.shape[:2]])[0]
    if len(res["boxes"]) == 0:
        return None
    return res["boxes"][int(res["scores"].argmax())].cpu().numpy()


def sam2_from_bbox(p, img, box):
    """SAM2 box prompt(紧bbox约束到罐)+ bbox中心正点 → clean mask。"""
    p.set_image(img)
    c = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]], np.float32)
    masks, scores, _ = p.predict(box=box[None], point_coords=c, point_labels=np.ones(1, np.int32), multimask_output=False)
    return clean_mask(masks[0])
DOMAIN = os.environ.get("DOMAIN", "human"); VIEW = os.environ.get("VIEW", "high")
SMOKE = os.environ.get("SMOKE", "0") == "1"
DIRS = {"human": [f"human_play_eef_data/play_human_can_eef_{i}" for i in range(1, 13)],
        "robot": [f"human_play_data/play_robot_can_{i}_eef" for i in range(1, 19)]}
CROPS = {"high": (60, 60, 390, 390), "low": (0, 0, 640, 480)}
CAM = {"high": "observation.images.cam_high", "low": "observation.images.cam_low"}
VIDBASE = {"human": 0, "robot": 100}
BASE = os.environ.get("BASE", "outputs/flow_render_dataset_can_dual")   # 数据集根(L48重建走 ..._L48)
HCLIP = os.environ.get("HCLIP", "human_L24")                            # human clip basename(L48重建=human_L48)
P = 48
dev = "cuda" if torch.cuda.is_available() else "cpu"
OUT = f"{BASE}/retrack"; os.makedirs(OUT, exist_ok=True)


def decode_seek(path, fids, crop):
    need = sorted(set(int(f) for f in fids)); nset = set(need)
    cont = av.open(path); vs = cont.streams.video[0]
    fps = float(vs.average_rate); tb = float(vs.time_base); st0 = vs.start_time or 0
    try:
        cont.seek(int(need[0] / fps / tb) + st0, stream=vs)
    except Exception:
        pass
    got = {}; x, y, w, h = crop
    for frame in cont.decode(video=0):
        if frame.pts is None:
            continue
        fn = int(round((frame.pts - st0) * tb * fps))
        if fn in nset:
            got[fn] = frame.to_ndarray(format="rgb24")[y:y+h, x:x+w].copy()
        if fn > need[-1]:
            break
    cont.close()
    if not got:
        return None
    out = []
    for f in fids:
        f = int(f)
        out.append(got[f] if f in got else got[min(got, key=lambda k: abs(k - f))])
    return out


def grid48(mask):
    g = grid_in_mask(mask, P)
    if len(g) >= P:
        return g[np.linspace(0, len(g) - 1, P).astype(int)]
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    pick = np.random.default_rng(0).choice(len(xs), P - len(g))
    pad = np.stack([xs[pick], ys[pick]], 1).astype(np.float32)
    return np.concatenate([g, pad]) if len(g) else pad


def per_frame_vis(trk):
    T = trk.shape[0]; vis = np.zeros((T, P), np.float32)
    for t in range(T):
        d = np.linalg.norm(trk[t] - np.median(trk[t], 0), axis=-1)
        vis[t] = (d <= np.median(d) * 2.5 + 4).astype(np.float32)
    return vis


def main():
    z = np.load(f"{BASE}/clips_{HCLIP if DOMAIN=='human' else 'robot'}.npz")
    tk = "tracks" if VIEW == "high" else "tracks_low"; ek = "eef" if VIEW == "high" else "eef_low"
    vid, fidx = z["vid"], z["fidx"]; N, L = fidx.shape
    old_tr = np.nan_to_num(z[tk]).astype(np.float32); ef = np.nan_to_num(z[ek]).astype(np.float32)
    _wscp = f"{BASE}/wrist_sidecar_human.npz"
    wsc = np.load(_wscp) if (DOMAIN == "human" and VIEW == "high" and os.path.exists(_wscp)) else None
    clips = [int(x) for x in os.environ["CLIPS"].split(",")] if os.environ.get("CLIPS") else list(range(3 if SMOKE else N))
    p = load_sam2(); ct = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(dev).eval()
    new_tr = old_tr.copy(); new_vis = z["vis" if VIEW == "high" else "vis_low"].astype(np.float32).copy()
    okmask = np.zeros(N, bool)                                          # ★每clip是否成功re-track(False=回退老track, 下游排除)
    ok = 0; fail = 0; failreasons = {}
    for ci in clips:
        reason = None
        try:
            frames = decode_seek(f"{DIRS[DOMAIN][int(vid[ci])-VIDBASE[DOMAIN]]}/videos/chunk-000/{CAM[VIEW]}/episode_000000.mp4", fidx[ci], CROPS[VIEW])
            if frames is None or frames[0] is None:
                reason = "decode"; raise ValueError("decode")
            H, W = frames[0].shape[:2]
            bbox = gdino_bbox(frames[0])                        # ★GDINO检测can bbox(替启发式种子)
            if bbox is None:
                reason = "no_detect"; raise ValueError("no_detect")
            mask = sam2_from_bbox(p, frames[0], bbox)           # SAM2 box prompt约束到罐
            grid = grid48(mask)
            if grid is None:
                reason = "empty_mask"; raise ValueError("mask")
            vidt = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].float().to(dev)
            q = np.concatenate([np.zeros((P, 1), np.float32), grid], 1)
            with torch.no_grad():
                trk, _ = ct(vidt, queries=torch.from_numpy(q)[None].to(dev))
            trk = trk[0].cpu().numpy()                                  # (L,48,2)px
            new_tr[ci] = (trk / [W, H]).astype(np.float32)
            new_vis[ci] = per_frame_vis(trk)
            okmask[ci] = True; ok += 1
            if ok % 50 == 0:
                print(f"[{DOMAIN}/{VIEW}] {ok} ok / {fail} fail", flush=True)
        except Exception as e:
            fail += 1; reason = reason or f"exc:{type(e).__name__}"
            failreasons[reason] = failreasons.get(reason, 0) + 1
            if fail <= 5:
                print(f"  ci{ci} FAIL({reason}) {e}", flush=True)
    op = f"{OUT}/{DOMAIN}_{VIEW}.npz"
    np.savez(op, tracks=new_tr, vis=new_vis, retrack_ok=okmask, attempted=np.array(clips))
    tot = len(clips)
    print(f"=== {DOMAIN}/{VIEW} DONE: {ok} ok, {fail} fail ({100*fail/max(tot,1):.2f}% 回退老track) reasons={failreasons} -> {op} ===", flush=True)
    print(f"★★[{DOMAIN}/{VIEW}] 失败回退比例 = {fail}/{tot} = {100*fail/max(tot,1):.2f}% (下游应用 retrack_ok 排除)", flush=True)


if __name__ == "__main__":
    main()
