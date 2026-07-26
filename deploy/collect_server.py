# deploy/collect_server.py
"""Interactive world-model teleop + demo collection server (BC-ready recording).

Same browser client (deploy/index.html) as the demo server, but this variant lets you
(1) pick the world-model BACKEND at launch, (2) pick the TASK from the UI, and (3) record
the rollout + action labels as an HDF5 demo in the training/BC (eef) format.

Launch (backend is chosen by which env you run it in + the BACKEND var):
    BACKEND=iws    uvicorn deploy.collect_server:app --host 0.0.0.0 --port 8000   # conda iws
    BACKEND=weaver uvicorn deploy.collect_server:app --host 0.0.0.0 --port 8001   # weaver_venv (WIP)

The redcube (single-arm eef) step follows scripts/inference/play_single_eef_inference.py
exactly: keep the raw 7-dim eef action, add a fixed step per key, clip to the per-dim data
bounds, normalize fresh each step, and feed cat([past_actions, future]) to dynamics_forward.
"""
import asyncio
import datetime as _dt
import os
import queue
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from deploy.server import (
    encode_frame_jpeg,
    hist_context,
    kybd_action_to_rob_action,
    load_task_config,
    parse_action,
)

BACKEND = os.environ.get("BACKEND", "iws").lower()
OUTPUT_DIR = Path(os.environ.get("COLLECT_OUT", "data/wm_demo"))
FRAME_DT = 0.1  # 10 Hz render/control

# eef step sizes (from play_single_eef_inference.py: STEP=0.01, gripper=GRIPPER_MAX/10).
EEF_STEP = 0.01
EEF_GRIPPER_STEP = 0.04

TASK_OBS_KEY = {
    "pusht": "camera_1_color",
    "bimanual_rope_cam_0": "camera_0_color",
    "single_grasp_cam_0": "camera_0_color",
    "bimanual_sweep_cam_0": "camera_0_color",
    "redcube": "camera_0_color",  # cam_high (third-person / agent view)
}
EEF_TASKS = {"redcube"}

# redcube (single-arm eef) loads exactly like scripts/inference/play_single_eef_inference.py.
REDCUBE_CKPT = "/home/hyeonhoo/code/interactive_world_sim/outputs/2026-06-13/22-11-27/checkpoints/epoch=3-step=250000.ckpt"
REDCUBE_DATASET = "data/play_robot_v3_hdf5"
REDCUBE_START = 0
REDCUBE_RES = 128
REDCUBE_DEC_INFER_STEPS = int(os.environ.get("DEC_INFER_STEPS", 2))  # lower = faster refresh
REDCUBE_DEVICE = "cuda:0"

# --------------------------------------------------------------------------- #
#  Terminal keyboard control. The browser is DISPLAY-ONLY (streams frames); you
#  type the actions in the server's SSH terminal (one keypress = one step, like
#  play_single_eef_inference.py). A background thread reads raw stdin into a queue
#  the WebSocket loop drains. Move keys map to a 7-dim eef delta index.
# --------------------------------------------------------------------------- #
KEY_QUEUE: "queue.Queue[str]" = queue.Queue()
TERM_KEYMAP = {  # char -> (eef index, sign)
    "w": (0, +1), "s": (0, -1),
    "a": (1, +1), "d": (1, -1),
    "q": (2, +1), "e": (2, -1),
    "u": (3, +1), "o": (3, -1),
    "i": (4, +1), "k": (4, -1),
    "j": (5, +1), "l": (5, -1),
    "g": (6, +1), "h": (6, -1),
}


def _stdin_reader() -> None:
    import termios
    import tty

    try:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
    except Exception:
        print("[terminal] stdin is not a TTY; terminal control disabled "
              "(run the server in a foreground terminal).", flush=True)
        return
    import signal
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in ("\x03", "\x04"):  # Ctrl-C / Ctrl-D: restore terminal and shut down
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
                os.kill(os.getpid(), signal.SIGINT)
                return
            if ch:
                KEY_QUEUE.put(ch)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


threading.Thread(target=_stdin_reader, daemon=True).start()

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
_curr_dir = Path(__file__).parent.resolve()


@app.get("/")
def root() -> HTMLResponse:
    return HTMLResponse((_curr_dir / "index.html").read_text())


def parse_delta(msg: Dict[str, Any], task: str) -> np.ndarray:
    """Keyboard -> action delta. eef tasks use a 7-dim per-index delta (matching
    play_single_eef_inference.py's key_map); other tasks delegate to server.parse_action."""
    if task in EEF_TASKS:
        d = np.zeros(7, dtype=np.float32)
        g = lambda k: 1 if msg.get(k) == 1 else 0
        d[0] += g("w") - g("s")   # x
        d[1] += g("a") - g("d")   # y
        d[2] += g("q") - g("e")   # z (pinned by clip: constant in data)
        d[3] += g("u") - g("o")   # roll (pinned)
        d[4] += g("i") - g("k")   # pitch (pinned)
        d[5] += g("j") - g("l")   # yaw (pinned)
        d[6] += g("g") - g("h")   # gripper
        return d
    return parse_action(msg, task)


# --------------------------------------------------------------------------- #
def scale_keyboard_delta(delta_action: np.ndarray, scene: str, model) -> np.ndarray:
    """Per-task keyboard-delta scaling for the joint-based tasks (from deploy/server.py)."""
    action_max = model.normalizer["action"].state_dict()["params_dict.input_stats.max"]
    action_min = model.normalizer["action"].state_dict()["params_dict.input_stats.min"]
    action_range = action_max - action_min
    if scene in ["bimanual_rope_cam_0", "bimanual_rope_cam_1"]:
        action_range = torch.cat([action_range[:3], action_range[4:7]]).to(model.device)
    action_range_scale = (action_range / action_range.max()).detach().cpu().numpy()

    if scene in ["pusht", "pusht_cam_0"]:
        delta_action = delta_action / (50.0 * action_range_scale)
    elif scene in ["bimanual_rope_cam_0", "bimanual_rope_cam_1"]:
        delta_action = delta_action / (30.0 * action_range_scale)
    elif scene in ["bimanual_sweep_cam_0", "bimanual_sweep_cam_1"]:
        delta_action = delta_action / 20.0
    elif scene in ["single_grasp_cam_0", "single_grasp_cam_1"]:
        delta_action = delta_action.copy()
        delta_action[:3] = delta_action[:3] * action_range_scale[:3].max() / (50.0 * action_range_scale[:3])
        delta_action[3] = delta_action[3] / 10.0
    else:
        raise NotImplementedError(f"scene '{scene}' not recognized")
    return delta_action


# --------------------------------------------------------------------------- #
class Backend:
    name = "base"

    def load_task(self, task: str) -> Dict[str, Any]:
        raise NotImplementedError

    def render(self, st: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

    def step(self, st: Dict[str, Any], delta: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError


class IWSBackend(Backend):
    name = "iws"

    def load_task(self, task: str) -> Dict[str, Any]:
        from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm  # noqa

        if task in EEF_TASKS:
            return self._load_redcube(render_img_cm)

        resolution, model, normalizer, latent, curr_action = load_task_config(scene=task)
        curr_action = normalizer["action"].normalize(
            torch.from_numpy(np.asarray(curr_action)).to(model.device).float())
        return {
            "task": task, "resolution": resolution, "model": model,
            "normalizer": normalizer, "latent": latent, "render_img_cm": render_img_cm,
            "obs_key": TASK_OBS_KEY.get(task, "camera_0_color"), "eef": False,
            "curr_action": curr_action, "action_hist": [curr_action.clone()],
        }

    def _load_redcube(self, render_img_cm) -> Dict[str, Any]:
        """Mirror scripts/inference/play_single_eef_inference.py exactly (load_model +
        encode_frame + action bounds), so the deploy rollout matches the working script."""
        import glob
        import h5py
        from omegaconf import OmegaConf
        from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
            LatentWorldModel,
        )

        device = REDCUBE_DEVICE
        cfg_path = Path(REDCUBE_CKPT).parent.parent / ".hydra" / "config.yaml"
        cfg = OmegaConf.load(cfg_path)
        cfg.algorithm.load_ae = None
        cfg.algorithm.dec_infer_steps = REDCUBE_DEC_INFER_STEPS
        model = LatentWorldModel.load_from_checkpoint(
            REDCUBE_CKPT, cfg=cfg.algorithm, map_location=device,
            strict=False, weights_only=False,
        ).to(device)
        model.eval()
        normalizer = model.normalizer
        dtype = model.dtype

        # episode 0 (val or train) + per-dim action bounds from all train episodes
        ep = None
        for sub in ("val", "train", ""):
            p = os.path.join(REDCUBE_DATASET, sub, f"episode_{REDCUBE_START}.hdf5")
            if os.path.exists(p):
                ep = p
                break
        with h5py.File(ep, "r") as f:
            cam0 = f["obs"]["images"]["camera_0_color"][0]
            cam1 = f["obs"]["images"]["camera_1_color"][0]
            eef_state = f["action"][0].astype(np.float32)
        all_a = []
        for fp in sorted(glob.glob(os.path.join(REDCUBE_DATASET, "train", "episode_*.hdf5"))):
            with h5py.File(fp, "r") as f:
                all_a.append(f["action"][:])
        all_a = np.concatenate(all_a, axis=0) if all_a else eef_state[None]
        a_min, a_max = all_a.min(axis=0).astype(np.float32), all_a.max(axis=0).astype(np.float32)

        def to_tensor(img):
            return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)

        with torch.no_grad():
            img0 = normalizer["camera_0_color"].normalize(to_tensor(cam0))
            img1 = normalizer["camera_1_color"].normalize(to_tensor(cam1))
            z0 = model.encoder_forward(torch.cat([img0, img1], dim=1))
        curr_latent = z0.unsqueeze(1)  # (1,1,C,H,W)
        action_norm = normalizer["action"].normalize(
            torch.from_numpy(eef_state[None]).to(device=device, dtype=dtype))  # (1,7)

        return {
            "task": "redcube", "resolution": REDCUBE_RES, "model": model,
            "normalizer": normalizer, "latent": curr_latent, "render_img_cm": render_img_cm,
            "obs_key": "camera_0_color", "eef": True,
            "eef_state": eef_state, "action_min": a_min, "action_max": a_max,
            "past_actions": action_norm.unsqueeze(0),  # (1,1,7)
            "n_tokens": int(model.n_tokens),
        }

    def _render(self, st: Dict[str, Any], z: torch.Tensor) -> np.ndarray:
        model = st["model"]
        xs = st["render_img_cm"](
            model, z, st["resolution"], normalizer=st["normalizer"],
            num_views=len(model.obs_keys),
        )
        xs = xs[:, :3]  # first view (obs_keys[0] = camera_0 = agent third-person view)
        frame = xs.permute(0, 2, 3, 1).detach().cpu().numpy()[0]
        return np.clip(frame * 255, 0, 255).astype(np.uint8)

    def render(self, st: Dict[str, Any]) -> np.ndarray:
        return self._render(st, st["latent"][:, -1])

    def step(self, st: Dict[str, Any], delta: np.ndarray) -> Dict[str, Any]:
        model = st["model"]
        if st["eef"]:
            device, dtype = model.device, model.dtype
            inc = delta.astype(np.float32) * EEF_STEP
            inc[6] = delta[6] * EEF_GRIPPER_STEP
            st["eef_state"] = np.clip(st["eef_state"] + inc, st["action_min"], st["action_max"])

            action_norm = st["normalizer"]["action"].normalize(
                torch.from_numpy(st["eef_state"][None]).to(device=device, dtype=dtype))  # (1,7)
            future = action_norm.unsqueeze(0)  # (1,1,7)
            action_input = torch.cat([st["past_actions"], future], dim=1)  # (1,T+1,7)

            z_pred = model.dynamics_forward(st["latent"], action_input)
            n_tok = st["n_tokens"]
            st["latent"] = torch.cat([st["latent"], z_pred[:, -1:]], dim=1)[:, -n_tok:]
            st["past_actions"] = torch.cat([st["past_actions"], future], dim=1)[:, -n_tok:]

            frame = self._render(st, z_pred[:, -1])
            return {"frame": frame, "action": st["eef_state"].copy()}  # raw 7-dim = BC label

        # joint-based tasks
        scaled = scale_keyboard_delta(delta, st["task"], model)
        delta_rob = kybd_action_to_rob_action(scaled, scene=st["task"])
        st["curr_action"] = torch.clamp(
            st["curr_action"] + torch.from_numpy(delta_rob).to(model.device), -1.0, 1.0)
        st["action_hist"].append(st["curr_action"].clone())
        action = torch.cat(st["action_hist"], dim=0)[-(hist_context + 1):].float()
        latent_pred = model.dynamics_forward(st["latent"], action[None])
        st["latent"] = torch.cat([st["latent"], latent_pred], axis=1)[:, -hist_context:]
        frame = self._render(st, st["latent"][:, -1])
        act = st["normalizer"]["action"].unnormalize(st["curr_action"]).detach().cpu().numpy()[0]
        return {"frame": frame, "action": act.astype(np.float32)}


class WeaverBackend(Backend):
    name = "weaver"

    def load_task(self, task: str) -> Dict[str, Any]:
        raise NotImplementedError(
            "WeaverBackend not implemented yet: WEAVER's interactive single-step rollout "
            "(forward_cached + history/memory management) still needs to be built. The "
            "recording/WebSocket layers are backend-agnostic. Run BACKEND=iws for now."
        )

    def render(self, st):
        raise NotImplementedError

    def step(self, st, delta):
        raise NotImplementedError


BACKENDS = {"iws": IWSBackend, "weaver": WeaverBackend}


# --------------------------------------------------------------------------- #
class Recorder:
    """Records (obs_t, action_t) pairs for BC (eef): obs before the action + the action."""

    def __init__(self) -> None:
        self.active = False
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.task: Optional[str] = None
        self.backend: Optional[str] = None
        self.obs_key: str = "camera_0_color"

    def start(self, task: str, backend: str, obs_key: str) -> None:
        self.active = True
        self.obs, self.actions = [], []
        self.task, self.backend, self.obs_key = task, backend, obs_key

    def add(self, obs_frame: np.ndarray, action: np.ndarray) -> None:
        if self.active:
            self.obs.append(obs_frame)
            self.actions.append(action)

    def discard(self) -> None:
        self.active = False
        self.obs, self.actions = [], []

    def stop_and_save(self) -> Optional[str]:
        if not self.active or not self.obs:
            self.discard()
            return None
        import h5py

        out_dir = OUTPUT_DIR / (self.task or "unknown")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"episode_{ts}.hdf5"

        obs = np.stack(self.obs)                             # (T, H, W, 3) uint8
        actions = np.stack(self.actions).astype(np.float32)  # (T, A) eef action label
        mask = np.ones(len(actions), dtype=np.float32)
        with h5py.File(path, "w") as f:
            f.attrs["task"] = self.task or ""
            f.attrs["backend"] = self.backend or ""
            f.attrs["fps"] = int(round(1.0 / FRAME_DT))
            f.attrs["ctrl_mode"] = "eef"
            f.create_dataset("action", data=actions, compression="gzip")
            f.create_dataset("action_mask", data=mask, compression="gzip")
            g = f.create_group("obs")
            g.create_group("images").create_dataset(self.obs_key, data=obs, compression="gzip")
        try:
            import imageio
            imageio.mimwrite(str(path).replace(".hdf5", ".mp4"), obs,
                             fps=int(round(1.0 / FRAME_DT)))
        except Exception as e:
            print(f"  (preview mp4 skipped: {e})", flush=True)
        self.discard()
        return str(path)


# --------------------------------------------------------------------------- #
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    """Browser = display only (streams frames). Control comes from the SERVER TERMINAL
    (KEY_QUEUE via _stdin_reader): one keypress = one step, matching play_single_eef."""
    await ws.accept()
    backend: Backend = BACKENDS[BACKEND]()
    task = os.environ.get("INIT_TASK", "redcube")
    st = backend.load_task(task)
    recorder = Recorder()
    pending_obs: Optional[np.ndarray] = None

    async def send_frame(frame: np.ndarray) -> None:
        await ws.send_bytes(struct.pack(">d", 0.0) + encode_frame_jpeg(frame))

    await send_frame(backend.render(st))  # show the starting state

    async def recv_loop() -> None:
        # Browser is display-only; only honor task switches + record buttons (optional).
        nonlocal task, st, pending_obs
        try:
            while True:
                data = await ws.receive_json()
                mtype = data.get("type", "action")
                if mtype in ("init", "set_task"):
                    new = data.get("task")
                    if new and new != task:
                        task = new
                        st = backend.load_task(task)
                        recorder.discard()
                        pending_obs = None
                        await send_frame(backend.render(st))
                elif mtype == "record_start":
                    recorder.start(task, backend.name, st["obs_key"])
                    pending_obs = backend.render(st)
                elif mtype == "record_stop":
                    saved = recorder.stop_and_save()
                    await ws.send_json({"type": "record_saved", "path": saved})
                elif mtype == "record_discard":
                    recorder.discard()
        except Exception:
            return

    recv_task = asyncio.create_task(recv_loop())
    print("[terminal] control ready. Focus THIS terminal and press keys: "
          "W/S x  A/D y  Q/E z  U/O/I/K/J/L rot  G/H grip | c=rec  v=save  r=reset", flush=True)

    try:
        with torch.no_grad():
            while True:
                did = False
                while True:
                    try:
                        ch = KEY_QUEUE.get_nowait()
                    except queue.Empty:
                        break
                    did = True
                    if ch == "c":
                        recorder.start(task, backend.name, st["obs_key"])
                        pending_obs = backend.render(st)
                        print("[REC] started", flush=True)
                    elif ch == "v":
                        saved = recorder.stop_and_save()
                        await ws.send_json({"type": "record_saved", "path": saved})
                        print(f"[REC] saved -> {saved}", flush=True)
                    elif ch == "r":
                        st = backend.load_task(task)
                        pending_obs = None
                        await send_frame(backend.render(st))
                    elif ch in TERM_KEYMAP and st.get("eef"):
                        idx, sign = TERM_KEYMAP[ch]
                        delta = np.zeros(7, dtype=np.float32)
                        delta[idx] = sign
                        out = backend.step(st, delta)
                        if recorder.active:
                            recorder.add(pending_obs, out["action"])
                            pending_obs = out["frame"]
                        await send_frame(out["frame"])
                    # other chars ignored
                if not did:
                    await asyncio.sleep(0.02)
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        recv_task.cancel()
