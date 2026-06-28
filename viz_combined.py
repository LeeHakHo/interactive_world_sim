"""Shared pretty compositor for the V3 three-regime gifs: stacks the RENDERED row (③ output) on top
of the FLOW-ON-FRAME row (② flow points over the GT frame), with clean column/row titles, a per-seq
error sub-label, a step counter, and a legend. Used by exp_v3_full_vs_scarce_human.py (live) and
remake_v3_combined_gifs.py (re-skin from cache, no retrain). One drawing implementation, two callers.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

CELL = 128; COLS = 4; CG = 8; RG = 10; ML = 96; MT = 44; MB = 34; MR = 12
def _canvas_w(ncol): return ML + ncol * CELL + (ncol - 1) * CG + MR   # width adapts to #columns
H_ = MT + 2 * CELL + RG + MB
BG = (248, 248, 250); DARK = (32, 34, 40); GREY = (120, 124, 132)
GOOD = (20, 150, 60); BAD = (205, 45, 45); MID = (40, 44, 52)
GT_C = (0, 210, 0); PRED_C = (240, 55, 55); EEF_C = (245, 215, 0)     # eef = single given action


def overlay_flow(bg, gt_obj_h, eef_h, pred_obj_h, show_pred):
    """flow points on a 128px frame. cube: GT=green / pred=red. eef=yellow (the GIVEN action,
    NOT predicted — so a single color, drawn on every column)."""
    im = Image.fromarray((bg.astype(np.float32) * 0.55).astype(np.uint8)).convert("RGB")
    dr = ImageDraw.Draw(im); n = bg.shape[0]

    def dot(xy, col, r):
        x, y = float(xy[0]) * n, float(xy[1]) * n
        dr.ellipse([x - r, y - r, x + r, y + r], fill=col)
    for p in gt_obj_h: dot(p, GT_C, 1)                            # GT cube (truth) green
    if show_pred:
        for p in pred_obj_h: dot(p, PRED_C, 1)                   # predicted cube red
    for p in eef_h: dot(p, EEF_C, 3)                             # eef = action input (given)
    return np.array(im)


def build_flow_cols(bg_seq, gt_obj, col_preds, eef_seq):
    """Generic: one flow column per entry in col_preds. Entry=None -> GT-only (green, no red);
    entry=(H,48,2) -> that column's predicted cube (red) over GT (green). eef (yellow) on all.
    bg_seq (H,128,128,3); gt_obj (H,48,2); eef_seq (H,3,2). -> (len(col_preds),H,128,128,3)."""
    H = len(bg_seq); nc = len(col_preds)
    f = np.zeros((nc, H, bg_seq.shape[1], bg_seq.shape[2], 3), np.uint8)
    for h in range(H):
        for ci, pr in enumerate(col_preds):
            f[ci, h] = overlay_flow(bg_seq[h], gt_obj[h], eef_seq[h], None if pr is None else pr[h], pr is not None)
    return f


def build_flow4(bg_seq, gt_obj, preds, eef_seq):
    """Back-compat 4-col: col0 = GT-flow (no pred); col1..3 = the three ② regimes."""
    return build_flow_cols(bg_seq, gt_obj, [None] + list(preds), eef_seq)


def _font(sz, bold=False):
    try:
        return ImageFont.truetype(fm.findfont("DejaVu Sans:bold" if bold else "DejaVu Sans"), sz)
    except Exception:
        return ImageFont.load_default()


_F = {"main": _font(13, True), "sub": _font(11), "row": _font(12, True),
      "rowsub": _font(9), "step": _font(11, True), "leg": _font(10)}


def _err_color(errs, i):
    vals = [e for e in errs if e is not None]
    if errs[i] is None or not vals: return GREY
    if errs[i] == min(vals): return GOOD
    if errs[i] == max(vals): return BAD
    return MID


def compose_frame(render4, flow4, col_titles, errs, step, caption=None):
    """render4/flow4: lists of N uint8 (128,128,3) cells (col order = col_titles). -> PIL Image.
    caption: optional 指令字符串, 显示在顶部居中 (有则列标题下移)."""
    ncol = len(col_titles); W = _canvas_w(ncol)
    im = Image.new("RGB", (W, H_), BG); dr = ImageDraw.Draw(im)
    xc = [ML + c * (CELL + CG) for c in range(ncol)]
    yr = [MT, MT + CELL + RG]
    title_y, sub_y = (24, 37) if caption else (12, 30)
    if caption:
        dr.text((W // 2, 8), caption, font=_F["main"], fill=(180, 40, 40), anchor="mm")
    # cells
    for c in range(ncol):
        im.paste(Image.fromarray(np.ascontiguousarray(render4[c])), (xc[c], yr[0]))
        im.paste(Image.fromarray(np.ascontiguousarray(flow4[c])), (xc[c], yr[1]))
        dr.rectangle([xc[c], yr[0], xc[c] + CELL, yr[1] + CELL], outline=(210, 212, 218))
    # column titles + per-seq error sub-label
    for c in range(ncol):
        cx = xc[c] + CELL // 2
        dr.text((cx, title_y), col_titles[c], font=_F["main"], fill=DARK, anchor="mm")
        sub = "reference" if errs[c] is None else f"{errs[c]:.1f}px"
        dr.text((cx, sub_y), sub, font=_F["sub"], fill=_err_color(errs, c), anchor="mm")
    # row titles (left margin)
    for r, (t, s) in enumerate([("Rendered", "③ output"), ("Flow", "pts on GT")]):
        cy = yr[r] + CELL // 2
        dr.text((ML // 2, cy - 8), t, font=_F["row"], fill=DARK, anchor="mm")
        dr.text((ML // 2, cy + 9), s, font=_F["rowsub"], fill=GREY, anchor="mm")
    # legend (bottom-left) + step counter (bottom-right, clear of the column titles)
    ly = H_ - MB // 2; lx = ML
    for col, lab in [(GT_C, "GT cube"), (PRED_C, "pred cube"), (EEF_C, "eef (action)")]:
        dr.ellipse([lx, ly - 4, lx + 8, ly + 4], fill=col); lx += 13
        dr.text((lx, ly), lab, font=_F["leg"], fill=DARK, anchor="lm")
        lx += int(dr.textlength(lab, font=_F["leg"])) + 18
    dr.text((_canvas_w(ncol) - MR, ly), f"t = {step:02d}", font=_F["step"], fill=DARK, anchor="rm")
    return im


def save_combined_gif(out_path, render4_seq, flow4_seq, col_titles, errs, k0, duration=180, caption=None):
    """render4_seq/flow4_seq: (N, H, 128,128,3) uint8. errs: [None, ...] len N. N inferred from data.
    caption: optional 指令字符串显示在顶部 (keyboard 合成用)."""
    ncol = render4_seq.shape[0]; Hh = render4_seq.shape[1]
    frames = [compose_frame([render4_seq[c, h] for c in range(ncol)],
                            [flow4_seq[c, h] for c in range(ncol)],
                            col_titles, errs, k0 + h, caption=caption) for h in range(Hh)]
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
