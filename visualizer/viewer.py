"""Live pygame visualizer for demos.

Layout:
    +----------------------------------+--------------+
    |                                  |  Drone cam   |
    |                                  |   (top-right |
    |        Third-person view         |   30%x50%)   |
    |        of the scene              +--------------+
    |        (left 70%)                |    HUD       |
    |                                  |              |
    |                                  | + SkyEU logo |
    +----------------------------------+--------------+

The third-person view is rendered by PyBullet's TINY (CPU) renderer from a
fixed-offset chase camera, so it works whether the env is in GUI or DIRECT
mode. The drone cam panel echoes the agent's camera (`obs['camera']`).

Used for the on-stage demo. For batch evaluation, just don't construct it.
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pybullet as p


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_PATH = ASSETS_DIR / "skyeu_logo.png"


class Viewer:
    SCENE_W_FRAC = 0.70  # left panel width fraction
    DRONE_CAM_H_FRAC = 0.50

    # Chase camera placement (drone-relative). Tighter than a wide overview
    # so the action with the boat reads on a big screen.
    CHASE_EYE_OFFSET = np.array([5.0, 5.0, 3.0])
    CHASE_FOV_DEG = 50.0

    def __init__(self, env, width: int = 1280, height: int = 720, title: str = "Catch the Boat"):
        # Defer pygame import so headless installs don't pay the cost.
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame  # noqa: WPS433

        self._pygame = pygame
        self.env = env
        self.width = int(width)
        self.height = int(height)

        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.SysFont("monospace", 16)
        self.font_big = pygame.font.SysFont("monospace", 22, bold=True)

        # Cached panel rects
        self._scene_rect = pygame.Rect(
            0, 0, int(self.width * self.SCENE_W_FRAC), self.height
        )
        side_x = int(self.width * self.SCENE_W_FRAC)
        side_w = self.width - side_x
        self._cam_rect = pygame.Rect(
            side_x, 0, side_w, int(self.height * self.DRONE_CAM_H_FRAC)
        )
        self._hud_rect = pygame.Rect(
            side_x,
            int(self.height * self.DRONE_CAM_H_FRAC),
            side_w,
            self.height - int(self.height * self.DRONE_CAM_H_FRAC),
        )

        # Pre-load and pre-scale the SkyEU logo so render() is cheap. The
        # logo PNG is white-background-with-black-text; we display it on
        # the dark HUD panel, so we composite directly with alpha (the
        # PNG already carries it).
        self._logo: Optional[object] = None
        if LOGO_PATH.is_file():
            try:
                logo = pygame.image.load(str(LOGO_PATH)).convert_alpha()
                # Scale to fit ~80% of HUD width, preserve aspect.
                target_w = int(self._hud_rect.w * 0.8)
                aspect = logo.get_height() / max(logo.get_width(), 1)
                target_h = int(target_w * aspect)
                self._logo = pygame.transform.smoothscale(logo, (target_w, target_h))
            except Exception as exc:  # pragma: no cover — best-effort
                print(f"[Viewer] could not load logo: {exc}")

    # ------------------------------------------------------------------ events
    def poll_events(self) -> bool:
        """Pump pygame events. Returns False if the user closed the window."""
        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                return False
            if event.type == self._pygame.KEYDOWN and event.key in (
                self._pygame.K_ESCAPE,
                self._pygame.K_q,
            ):
                return False
        return True

    # ------------------------------------------------------------------ render
    def render(
        self,
        agent_camera_image: Optional[np.ndarray],
        hud_lines: List[str],
    ) -> None:
        self.screen.fill((10, 10, 12))
        self._render_scene()
        self._render_drone_cam(agent_camera_image)
        self._render_hud(hud_lines)
        self._pygame.display.flip()

    def close(self) -> None:
        try:
            self._pygame.quit()
        except Exception:
            pass

    # ------------------------------------------------------------------ helpers
    def _render_scene(self) -> None:
        # Chase camera offset from the drone, looking at the drone. Targeted
        # framing: drone fills ~10–15% of the frame, boat is comfortably
        # visible during the descent.
        try:
            drone_pos = self.env._get_position()
        except Exception:
            drone_pos = np.zeros(3)
        # Aim the camera midway between the drone and the boat's position
        # so the boat doesn't slide out of frame during the final approach.
        try:
            boat_pos = self.env.boat.position
        except Exception:
            boat_pos = drone_pos
        target = 0.5 * (drone_pos + np.asarray(boat_pos))
        eye = drone_pos + self.CHASE_EYE_OFFSET
        view = p.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=[0, 0, 1],
        )
        proj = p.computeProjectionMatrixFOV(
            fov=self.CHASE_FOV_DEG,
            aspect=self._scene_rect.w / max(self._scene_rect.h, 1),
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
                self._scene_rect.w,
                self._scene_rect.h,
                viewMatrix=view,
                projectionMatrix=proj,
                renderer=renderer,
                flags=p.ER_NO_SEGMENTATION_MASK,
                physicsClientId=self.env.client,
            )
            img = np.asarray(rgba, dtype=np.uint8).reshape(
                self._scene_rect.h, self._scene_rect.w, 4
            )[:, :, :3]
        except Exception:
            img = np.full(
                (self._scene_rect.h, self._scene_rect.w, 3), 30, dtype=np.uint8
            )
        surf = self._np_to_surface(img)
        self.screen.blit(surf, self._scene_rect.topleft)

    def _render_drone_cam(self, image: Optional[np.ndarray]) -> None:
        rect = self._cam_rect
        self._pygame.draw.rect(self.screen, (20, 20, 25), rect)
        if image is None:
            return
        surf = self._np_to_surface(image)
        scaled = self._pygame.transform.scale(surf, (rect.w, rect.h))
        self.screen.blit(scaled, rect.topleft)
        # Label
        label = self.font.render("drone camera", True, (220, 220, 220))
        self.screen.blit(label, (rect.x + 6, rect.y + 6))

    def _render_hud(self, hud_lines: List[str]) -> None:
        rect = self._hud_rect
        self._pygame.draw.rect(self.screen, (18, 18, 22), rect)
        self._pygame.draw.rect(self.screen, (60, 60, 70), rect, 1)
        x = rect.x + 12
        y = rect.y + 12
        title = self.font_big.render("HUD", True, (240, 240, 240))
        self.screen.blit(title, (x, y))
        y += 32
        for line in hud_lines:
            text = self.font.render(line, True, (210, 210, 220))
            self.screen.blit(text, (x, y))
            y += 22

        # SkyEU logo centred at the bottom of the HUD panel. The PNG is
        # black-on-transparent; on the dark panel we lighten it once at
        # load time. Here we just blit.
        if self._logo is not None:
            logo_w = self._logo.get_width()
            logo_h = self._logo.get_height()
            lx = rect.x + (rect.w - logo_w) // 2
            ly = rect.y + rect.h - logo_h - 16
            # Backdrop strip so the logo reads cleanly on the dark panel.
            backdrop = self._pygame.Rect(
                rect.x + 6, ly - 8, rect.w - 12, logo_h + 16
            )
            self._pygame.draw.rect(self.screen, (240, 240, 240), backdrop, border_radius=6)
            self.screen.blit(self._logo, (lx, ly))

    def _np_to_surface(self, img: np.ndarray):
        # pygame.surfarray expects (W, H, 3); our images are (H, W, 3).
        return self._pygame.surfarray.make_surface(np.ascontiguousarray(img.swapaxes(0, 1)))
