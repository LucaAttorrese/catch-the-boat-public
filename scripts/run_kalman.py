#!/usr/bin/env python3
"""Test runner for agent_kalman with live GUI — medium and hard scenarios.

Layout:
    +-------------------------------+------------------+
    |                               | drone camera     |
    |  third-person chase view      | (what ArUco sees)|
    |  (boat + drone)               +------------------+
    |                               | HUD + Kalman     |
    |                               | filter state     |
    +-------------------------------+------------------+

Usage:
    # Both scenarios back-to-back (default)
    python scripts/run_kalman.py

    # Single scenario
    python scripts/run_kalman.py --scenario medium
    python scripts/run_kalman.py --scenario hard

    # Slow down to real time for easier watching
    python scripts/run_kalman.py --scenario hard --realtime

    # Extra-fast headless run (no pygame window) — just prints scores
    python scripts/run_kalman.py --headless

    # Open the PyBullet OpenGL window too (faster camera rendering)
    python scripts/run_kalman.py --pybullet-gui

Keys while the window is open:
    Q / Escape  — skip the current scenario and move to the next
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.agent_kalman import KalmanAgent, make_agent          # noqa: E402
from agents.drone_sim_baseline import BaselineDroneSimulator     # noqa: E402
from boat_landing.env import BoatLandingEnv                      # noqa: E402
from evaluation.evaluate import resolve_drone_spec, resolve_scenario  # noqa: E402
from evaluation.scorer import compute_score                      # noqa: E402


# ── colour palette ──────────────────────────────────────────────────────────
C_GREEN  = (100, 220, 120)
C_YELLOW = (240, 200,  60)
C_RED    = (230,  80,  80)
C_CYAN   = ( 80, 210, 210)
C_WHITE  = (220, 220, 230)
C_DIM    = (140, 140, 155)
C_HEADER = (255, 255, 255)


# ── HUD content builder ─────────────────────────────────────────────────────
def build_hud_lines(agent: KalmanAgent, obs: dict, scenario_id: str) -> List[tuple]:
    """Return a list of (text, colour) pairs for the HUD panel."""
    pos     = np.asarray(obs["state"]["position"])
    vel     = np.asarray(obs["state"]["velocity"])
    speed   = float(np.linalg.norm(vel))
    kf      = agent.kf
    est     = agent._last_estimate or {}
    phase   = agent.phase
    fresh   = est.get("fresh", False)
    stale   = est.get("stale_steps", 0)
    cmd     = agent.last_cmd

    phase_col = {
        "SEARCH":   C_YELLOW,
        "APPROACH": C_CYAN,
        "DESCEND":  C_GREEN,
        "LAND":     C_RED,
    }.get(phase, C_WHITE)

    lines: List[tuple] = [
        ("── KALMAN + MPC AGENT ────", C_DIM),
        (f"scenario  : {scenario_id}", C_WHITE),
        (f"phase     : {phase}",       phase_col),
        (f"time      : {obs['time']:6.2f} s",  C_WHITE),
        (f"battery   : {obs['battery']*100:5.1f} %",
         C_GREEN if obs["battery"] > 0.4 else C_YELLOW if obs["battery"] > 0.2 else C_RED),
        ("",                           C_DIM),
        ("── DRONE ───────────────",   C_DIM),
        (f"alt       : {pos[2]:+7.2f} m",  C_WHITE),
        (f"speed     : {speed:6.2f} m/s",  C_WHITE),
        (f"pos x     : {pos[0]:+7.2f} m",  C_DIM),
        (f"pos y     : {pos[1]:+7.2f} m",  C_DIM),
        ("",                           C_DIM),
        ("── MPC COMMANDS ────────",   C_DIM),
        (f"pitch     : {cmd['pitch']:+6.3f}  (fwd/back)", C_CYAN),
        (f"roll      : {cmd['roll']:+6.3f}  (left/right)", C_CYAN),
        (f"thrust    : {cmd['thrust']:+6.3f}  (up/down)",
         C_GREEN if cmd["thrust"] > 0.05 else C_RED if cmd["thrust"] < -0.05 else C_WHITE),
        ("",                           C_DIM),
        ("── KALMAN FILTER ───────",   C_DIM),
        (f"KF ready  : {'YES' if kf.initialized else 'NO  (searching...)'}",
         C_GREEN if kf.initialized else C_YELLOW),
        (f"detection : {'FRESH' if fresh else f'stale  {stale:3d} fr'}",
         C_GREEN if fresh else C_YELLOW if stale < 25 else C_RED),
    ]

    if kf.initialized:
        kp        = kf.position
        kv        = kf.velocity
        horiz     = float(np.linalg.norm(pos[:2] - kp[:2]))
        boat_spd  = float(np.linalg.norm(kv[:2]))
        pos_sigma = float(np.sqrt(np.mean(np.diag(kf.P)[:3])))

        # MPC target z for current phase
        from agents.agent_kalman import DroneMPC
        target_z = float(kp[2]) + DroneMPC.PHASE_ALT.get(phase, 3.0)

        lines += [
            ("",                               C_DIM),
            ("── BOAT ESTIMATE ────────",      C_DIM),
            (f"est x     : {kp[0]:+7.2f} m",   C_WHITE),
            (f"est y     : {kp[1]:+7.2f} m",   C_WHITE),
            (f"vel x     : {kv[0]:+6.2f} m/s", C_CYAN),
            (f"vel y     : {kv[1]:+6.2f} m/s", C_CYAN),
            (f"boat spd  : {boat_spd:6.2f} m/s", C_CYAN),
            (f"horiz dist: {horiz:6.2f} m",
             C_GREEN if horiz < 0.5 else C_YELLOW if horiz < 2.0 else C_RED),
            (f"target z  : {target_z:+7.2f} m  (MPC ref)", C_DIM),
            (f"pos σ     : {pos_sigma:6.3f} m",
             C_GREEN if pos_sigma < 0.3 else C_YELLOW if pos_sigma < 1.0 else C_RED),
        ]
    return lines


# ── extended Viewer with coloured HUD ───────────────────────────────────────
class KalmanViewer:
    """Pygame window: chase view (left) + drone cam (top-right) + coloured HUD."""

    SCENE_W_FRAC   = 0.68
    DRONE_CAM_H_FRAC = 0.52

    CHASE_EYE_OFFSET = np.array([6.0, 6.0, 4.0])
    CHASE_FOV_DEG    = 50.0

    def __init__(self, env: BoatLandingEnv, width: int = 1400, height: int = 800):
        import os
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        self._pg  = pygame
        self.env  = env
        self.W, self.H = width, height

        pygame.init()
        pygame.display.set_caption("Kalman Agent — Catch the Boat")
        self.screen = pygame.display.set_mode((self.W, self.H))
        self.font_sm  = pygame.font.SysFont("monospace", 15)
        self.font_md  = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 22, bold=True)

        sx = int(self.W * self.SCENE_W_FRAC)
        sw = self.W - sx
        ch = int(self.H * self.DRONE_CAM_H_FRAC)

        self._rect_scene  = pygame.Rect(0,   0,   sx,       self.H)
        self._rect_cam    = pygame.Rect(sx,  0,   sw,       ch)
        self._rect_hud    = pygame.Rect(sx,  ch,  sw,       self.H - ch)

    # -- events --------------------------------------------------------------
    def poll_events(self) -> bool:
        for ev in self._pg.event.get():
            if ev.type == self._pg.QUIT:
                return False
            if ev.type == self._pg.KEYDOWN and ev.key in (
                self._pg.K_ESCAPE, self._pg.K_q
            ):
                return False
        return True

    # -- render --------------------------------------------------------------
    def render(self, camera_image: Optional[np.ndarray], hud_lines: List[tuple]) -> None:
        self.screen.fill((8, 8, 12))
        self._draw_scene()
        self._draw_cam(camera_image)
        self._draw_hud(hud_lines)
        self._pg.display.flip()

    def close(self) -> None:
        try:
            self._pg.quit()
        except Exception:
            pass

    # -- helpers -------------------------------------------------------------
    def _draw_scene(self) -> None:
        import pybullet as p
        rect = self._rect_scene
        try:
            drone_pos = self.env._get_position()
        except Exception:
            drone_pos = np.zeros(3)
        try:
            boat_pos = np.asarray(self.env.boat.position)
        except Exception:
            boat_pos = drone_pos.copy()

        # Look midway between drone and boat so both stay in frame.
        target = 0.5 * (drone_pos + boat_pos)
        eye    = drone_pos + self.CHASE_EYE_OFFSET
        view = p.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=[0, 0, 1],
        )
        proj = p.computeProjectionMatrixFOV(
            fov=self.CHASE_FOV_DEG,
            aspect=rect.w / max(rect.h, 1),
            nearVal=0.05,
            farVal=200.0,
        )
        renderer = (
            p.ER_BULLET_HARDWARE_OPENGL
            if getattr(self.env, "gui", False)
            else p.ER_TINY_RENDERER
        )
        try:
            _, _, rgba, _, _ = p.getCameraImage(
                rect.w, rect.h,
                viewMatrix=view,
                projectionMatrix=proj,
                renderer=renderer,
                flags=p.ER_NO_SEGMENTATION_MASK,
                physicsClientId=self.env.client,
            )
            img = np.asarray(rgba, dtype=np.uint8).reshape(rect.h, rect.w, 4)[:, :, :3]
        except Exception:
            img = np.full((rect.h, rect.w, 3), 25, dtype=np.uint8)

        surf = self._arr2surf(img)
        self.screen.blit(surf, rect.topleft)

        # Label
        lbl = self.font_sm.render("side view", True, (180, 180, 180))
        self.screen.blit(lbl, (rect.x + 8, rect.y + 8))

    def _draw_cam(self, image: Optional[np.ndarray]) -> None:
        rect = self._rect_cam
        self._pg.draw.rect(self.screen, (15, 15, 20), rect)
        self._pg.draw.rect(self.screen, (50, 50, 65), rect, 1)
        if image is not None:
            surf   = self._arr2surf(image)
            scaled = self._pg.transform.scale(surf, (rect.w, rect.h))
            self.screen.blit(scaled, rect.topleft)
        lbl = self.font_sm.render("drone camera  (ArUco view)", True, (200, 200, 200))
        self.screen.blit(lbl, (rect.x + 8, rect.y + 8))

    def _draw_hud(self, hud_lines: List[tuple]) -> None:
        rect = self._rect_hud
        self._pg.draw.rect(self.screen, (14, 14, 18), rect)
        self._pg.draw.rect(self.screen, (55, 55, 70), rect, 1)
        x = rect.x + 10
        y = rect.y + 10
        for text, colour in hud_lines:
            if text == "":
                y += 6
                continue
            surf = self.font_sm.render(text, True, colour)
            self.screen.blit(surf, (x, y))
            y += 20
            if y > rect.bottom - 20:
                break

    def _arr2surf(self, img: np.ndarray):
        return self._pg.surfarray.make_surface(np.ascontiguousarray(img.swapaxes(0, 1)))


# ── single episode runner ───────────────────────────────────────────────────
def run_episode(
    scenario_name: str,
    use_gui: bool,
    realtime: bool,
    seed: Optional[int],
    max_steps: Optional[int],
    headless: bool,
) -> dict:
    """Run one episode and return a result dict (same schema as evaluate.py)."""
    scenario_path  = resolve_scenario(scenario_name)
    drone_spec_path = resolve_drone_spec("vtol")
    drone_sim = BaselineDroneSimulator(str(drone_spec_path))
    env   = BoatLandingEnv(str(scenario_path), drone_sim=drone_sim, gui=use_gui)
    agent = make_agent(drone_spec=drone_sim.spec)

    viewer: Optional[KalmanViewer] = None
    if not headless:
        viewer = KalmanViewer(env)

    obs, info = env.reset(seed=seed)
    last_info  = info
    step       = 0
    estimation_log = []

    try:
        while True:
            action = agent.act(obs)

            # collect estimation log for RMSE bonus
            est = agent.get_last_estimate()
            if est is not None:
                estimation_log.append(
                    (info["boat_position"].copy(), np.asarray(est["position"]))
                )

            obs, _, terminated, truncated, info = env.step(action)
            last_info = info
            step += 1

            if viewer is not None:
                if not viewer.poll_events():
                    print(f"  [viewer closed — skipping {scenario_name}]")
                    break
                hud = build_hud_lines(agent, obs, scenario_name)
                viewer.render(obs["camera"], hud)
                if realtime:
                    time.sleep(BoatLandingEnv.DT)

            if terminated or truncated:
                break
            if max_steps is not None and step >= max_steps:
                break

    finally:
        if viewer is not None:
            viewer.close()
        env.close()

    outcome  = last_info.get("outcome") or "TIMEOUT"
    traj     = env._traj
    fin_drone = traj[-1]["drone_pos"] if traj else np.zeros(3)
    fin_boat  = traj[-1]["boat_pos"]  if traj else np.zeros(3)
    land_err  = float(np.linalg.norm(fin_drone[:2] - fin_boat[:2]))

    from evaluation.scorer import compute_estimation_rmse
    rmse = compute_estimation_rmse(estimation_log) if estimation_log else None

    score, breakdown = compute_score(
        outcome=outcome,
        landing_position_error=land_err if outcome == "LANDED" else None,
        time_to_land=env.t,
        duration_max=float(env.scenario["duration_max"]),
        battery_remaining=obs["battery"],
        max_descent_velocity=last_info.get("max_descent_velocity", 0.0),
        estimation_rmse=rmse,
        hw_readiness=None,
    )

    return {
        "scenario":            scenario_name,
        "outcome":             outcome,
        "score":               score,
        "steps":               step,
        "time_s":              env.t,
        "battery_pct":         obs["battery"] * 100.0,
        "land_err_m":          land_err,
        "max_descent_mps":     last_info.get("max_descent_velocity", 0.0),
        "estimation_rmse_m":   rmse,
        "breakdown":           breakdown,
    }


# ── result pretty-printer ───────────────────────────────────────────────────
def print_result(r: dict) -> None:
    ok   = r["outcome"] == "LANDED"
    sep  = "═" * 54
    print(f"\n{sep}")
    print(f"  SCENARIO  :  {r['scenario'].upper()}")
    print(f"  OUTCOME   :  {r['outcome']}")
    print(f"  SCORE     :  {r['score']:.2f}")
    print(sep)
    bd = r["breakdown"].get("components", {})
    if ok:
        print(f"  landing error    : {r['land_err_m']:.3f} m")
        print(f"  time to land     : {r['time_s']:.2f} s")
        print(f"  battery left     : {r['battery_pct']:.1f} %")
        print(f"  max descent      : {r['max_descent_mps']:.2f} m/s")
        if r["estimation_rmse_m"] is not None:
            print(f"  estimation RMSE  : {r['estimation_rmse_m']:.3f} m")
        print()
        print(f"  precision factor : {bd.get('precision_factor', 0):.3f}")
        print(f"  time factor      : {bd.get('time_factor', 0):.3f}")
        print(f"  battery factor   : {bd.get('battery_factor', 0):.3f}")
        print(f"  soft-landing +5  : {'YES' if bd.get('soft_landing_bonus',0)>0 else 'NO'}")
        print(f"  estimation bonus : {bd.get('estimation_bonus', 0):.1f} / 15")
    else:
        print(f"  ran for          : {r['time_s']:.2f} s  ({r['steps']} steps)")
        print(f"  battery left     : {r['battery_pct']:.1f} %")
    print(sep)


# ── main ────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--scenario",
        default="both",
        choices=["medium", "hard", "both"],
        help="Which scenario(s) to run (default: both)",
    )
    ap.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep between steps so the sim runs at wall-clock speed.",
    )
    ap.add_argument(
        "--headless",
        action="store_true",
        help="Skip the pygame window — just print scores.",
    )
    ap.add_argument(
        "--pybullet-gui",
        action="store_true",
        dest="pybullet_gui",
        help="Also open the PyBullet OpenGL window (faster camera rendering).",
    )
    ap.add_argument("--seed", type=int, default=None, help="RNG seed.")
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Hard cap on env steps per episode.",
    )
    args = ap.parse_args()

    scenarios = (
        ["medium", "hard"] if args.scenario == "both" else [args.scenario]
    )

    all_results = []
    for sc in scenarios:
        print(f"\nRunning scenario: {sc.upper()}  (press Q/Esc to skip)")
        result = run_episode(
            scenario_name=sc,
            use_gui=args.pybullet_gui,
            realtime=args.realtime,
            seed=args.seed,
            max_steps=args.max_steps,
            headless=args.headless,
        )
        all_results.append(result)
        print_result(result)

    if len(all_results) > 1:
        total = sum(r["score"] for r in all_results)
        landed = sum(1 for r in all_results if r["outcome"] == "LANDED")
        print(f"\n{'═'*54}")
        print(f"  TOTAL SCORE  :  {total:.2f}  ({landed}/{len(all_results)} landed)")
        print(f"{'═'*54}\n")

    return 0 if all(r["outcome"] == "LANDED" for r in all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
