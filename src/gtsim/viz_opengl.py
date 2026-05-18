"""OpenGL replay renderer + ffmpeg-piped MP4 export.

Same visual approach as omnidrone_defense_sim's renderer.py + video3d.py:
pygame + PyOpenGL (legacy fixed pipeline) with lit primitives, drawn from a
recorded run.  This module does NOT re-simulate; it replays a finished
`World` (its `MetricsLog` + `plan_debug_history`) frame-by-frame, opens a
hidden pygame OpenGL window, paints each frame, reads it with
`glReadPixels`, and pipes the raw RGB stream to ffmpeg.

Three contrasts with the matplotlib video module:
* The defender attitude comes straight from the logged R_D (rotation
  matrix) -- no velocity-derived hacks, so no apparent flipping.
* The intruder attitude is derived from its *actual acceleration*
  (thrust direction = a + g e_3), which gives a stable, physically
  meaningful orientation even when lateral velocity passes through zero
  during a juke -- this fixes the "drone flipping" artefact of the
  matplotlib renderer.
* The camera is a fixed orbit around the scene midpoint; there is no
  terminal zoom-in.  (Per user request: "no need to zoom in when
  collision".)

Public entry point: `render_video_opengl(metrics, cfg, out_path, ...)`.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import E3, SimConfig, cos_theta_F
from .metrics import MetricsLog


# pygame + GL deferred import (allow headless test envs to skip).
try:  # pragma: no cover - env-dependent
    import pygame
    from OpenGL.GL import *  # noqa: F401,F403
    from OpenGL.GLU import *  # noqa: F401,F403
    _HAS_GL = True
    _IMPORT_EXC = None
except Exception as _exc:  # pragma: no cover
    _HAS_GL = False
    _IMPORT_EXC = _exc


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #


@dataclass
class Camera:
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    distance: float = 38.0
    yaw_deg: float = 38.0
    pitch_deg: float = 22.0
    fov_deg: float = 55.0
    aspect: float = 16.0 / 9.0

    def eye(self) -> np.ndarray:
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
        x = self.distance * math.cos(pitch) * math.cos(yaw)
        y = self.distance * math.cos(pitch) * math.sin(yaw)
        z = self.distance * math.sin(pitch)
        return self.target + np.array([x, y, z])

    def up(self) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------- #
# Helpers shared with the rest of the codebase but inlined for self-containment
# --------------------------------------------------------------------------- #


def _stable_intruder_R(a_true: np.ndarray, v: np.ndarray, g: float,
                        speed_to_tilt: float = 0.055,
                        max_tilt_rad: float = 1.35,
                        bank_per_lat_accel: float = 0.018,
                        max_bank_rad: float = 0.65,
                        ) -> np.ndarray:
    """Visualisation-only racing-drone attitude for the kinematic intruder.

    Real quadrotors tilt forward to overcome drag when cruising; this
    simulator has no drag in the intruder model, so the actual thrust
    direction is mostly vertical and the rocket would appear upright.
    For the cinematic, we derive a visual attitude that LEANS the body
    forward into the velocity vector by an angle proportional to speed,
    and banks the body away from the lateral acceleration vector to
    emulate the cornering of a real racing quad.  Physics is unaffected
    -- this only controls how the body is rendered.

    Construction:
        1. Horizontal forward-tilt: body-z = cos(tilt) * world_up +
           sin(tilt) * v_hat_horizontal, with tilt = min(speed *
           speed_to_tilt, max_tilt_rad).
        2. Lateral bank: rotate body-z away from the horizontal
           component of acceleration that's perpendicular to v_hat (the
           juke's lateral push), so the rocket leans INTO the turn.
        3. Orthonormalise body-x along v_hat (gives a stable forward
           indicator), then body-y = body-z x body-x.
    """
    a = np.asarray(a_true, dtype=float)
    v = np.asarray(v, dtype=float)
    vh = np.array([v[0], v[1], 0.0])
    speed = float(np.linalg.norm(vh))
    if speed < 0.5:
        # Slow / stationary intruder: rocket sits upright.
        z = E3.copy()
        x_hint = np.array([1.0, 0.0, 0.0])
    else:
        v_dir = vh / speed
        tilt = min(max_tilt_rad, speed * speed_to_tilt)
        # Step 1: forward tilt into velocity.
        z = math.cos(tilt) * E3 + math.sin(tilt) * v_dir
        # Step 2: bank into lateral acceleration (juke turn).
        a_h = np.array([a[0], a[1], 0.0])
        a_lat = a_h - float(np.dot(a_h, v_dir)) * v_dir
        a_lat_mag = float(np.linalg.norm(a_lat))
        if a_lat_mag > 1e-3:
            a_lat_dir = a_lat / a_lat_mag
            bank = min(max_bank_rad, a_lat_mag * bank_per_lat_accel)
            # Lean body-z in the +a_lat direction (so the bottom of the
            # craft swings away from the turn, like a banking aircraft).
            z = z + math.tan(bank) * a_lat_dir
        z = z / max(float(np.linalg.norm(z)), 1e-9)
        x_hint = v_dir
    # Orthonormalise: pick body-x from x_hint perpendicular to body-z.
    x = x_hint - float(np.dot(x_hint, z)) * z
    nx = float(np.linalg.norm(x))
    if nx < 1e-3:
        x = np.cross(z, np.array([0.0, 1.0, 0.0]))
        nx = float(np.linalg.norm(x))
        if nx < 1e-3:
            x = np.cross(z, np.array([1.0, 0.0, 0.0]))
            nx = float(np.linalg.norm(x))
    x = x / max(nx, 1e-9)
    y = np.cross(z, x)
    y = y / max(float(np.linalg.norm(y)), 1e-9)
    x = np.cross(y, z)
    x = x / max(float(np.linalg.norm(x)), 1e-9)
    return np.column_stack([x, y, z])


def _rotation_to_gl_matrix(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation to a column-major 4x4 float32 matrix."""
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = np.asarray(R, dtype=np.float32)
    return M.T  # GL is column-major


def _smooth_attitude_sequence(R_seq: np.ndarray, dt: float,
                              tau: float = 0.18) -> np.ndarray:
    """Cosmetic first-order LPF on the body-z axis of an attitude sequence.

    The smart attacker re-aims at every sim tick; the visualisation
    derives its attitude instantaneously from `(a, v)`, so a 5 Hz juke
    flip whips the body-z and the rendered drone looks like it's
    wobbling.  This filter blends each frame's body-z with the previous
    smoothed value using `alpha = 1 - exp(-dt/tau)` and rebuilds R
    from the smoothed body-z plus a body-x hint inherited from the
    previous frame.  It only affects how the drone is drawn -- physics
    and metrics are unchanged.
    """
    if len(R_seq) == 0:
        return R_seq
    alpha = 1.0 - float(np.exp(-dt / max(tau, 1e-6)))
    out = [R_seq[0].copy()]
    z_filt = R_seq[0] @ np.array([0.0, 0.0, 1.0])
    for i in range(1, len(R_seq)):
        z_inst = R_seq[i] @ np.array([0.0, 0.0, 1.0])
        z_filt = (1.0 - alpha) * z_filt + alpha * z_inst
        z_filt = z_filt / max(float(np.linalg.norm(z_filt)), 1e-9)
        # Inherit body-x from the previous frame for continuity.
        x_hint = out[-1] @ np.array([1.0, 0.0, 0.0])
        x = x_hint - float(np.dot(x_hint, z_filt)) * z_filt
        nx = float(np.linalg.norm(x))
        if nx < 1e-3:
            x = np.cross(z_filt, np.array([0.0, 1.0, 0.0]))
            nx = float(np.linalg.norm(x))
            if nx < 1e-3:
                x = np.cross(z_filt, np.array([1.0, 0.0, 0.0]))
                nx = float(np.linalg.norm(x))
        x = x / max(nx, 1e-9)
        y = np.cross(z_filt, x)
        y = y / max(float(np.linalg.norm(y)), 1e-9)
        x = np.cross(y, z_filt)
        x = x / max(float(np.linalg.norm(x)), 1e-9)
        out.append(np.column_stack([x, y, z_filt]))
    return np.array(out)


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #


PALETTE = {
    "bg_top": (0.04, 0.05, 0.09),
    "bg_bottom": (0.07, 0.09, 0.14),
    "grid": (0.22, 0.28, 0.40, 0.55),
    "asset_glow": (0.18, 0.85, 0.40, 0.22),
    "asset_core": (0.20, 0.95, 0.45, 1.0),
    "defender_body": (0.20, 0.65, 1.0, 1.0),
    "defender_fin": (1.00, 0.85, 0.20, 1.0),
    "defender_arm": (0.72, 0.78, 0.86, 1.0),
    "defender_rotor": (0.92, 0.92, 0.95, 0.7),
    "defender_motor": (0.18, 0.18, 0.22, 1.0),
    "defender_trail": (0.30, 0.70, 1.0, 1.0),
    "intruder_body": (1.00, 0.30, 0.25, 1.0),
    "intruder_arm": (0.95, 0.70, 0.65, 1.0),
    "intruder_rotor": (1.00, 0.85, 0.85, 0.7),
    "intruder_motor": (0.22, 0.15, 0.15, 1.0),
    "intruder_trail": (1.0, 0.30, 0.30, 1.0),
    "fov_cone": (0.30, 0.80, 1.0, 0.18),
    "los_line": (0.7, 0.7, 0.8, 0.55),
    "tube_line": (0.55, 0.55, 0.65, 0.40),
    "vfx_hot": (1.0, 0.75, 0.20, 1.0),
    "vfx_outer": (1.0, 0.35, 0.15, 1.0),
    "breach_red": (1.0, 0.18, 0.12, 1.0),
    "breach_hot": (1.0, 0.55, 0.30, 1.0),   # bright red-orange core
    "vloss_gold": (0.98, 0.85, 0.18, 1.0),
}


class OpenGLReplayRenderer:
    def __init__(self, width: int = 1280, height: int = 720) -> None:
        if not _HAS_GL:
            raise RuntimeError(
                f"pygame+PyOpenGL not available: {_IMPORT_EXC!r}.  "
                "Install with `pip install pygame PyOpenGL`."
            )
        self.width = width
        self.height = height
        self.cam = Camera(aspect=width / height)
        self._screen = None
        self._font = None
        self._small_font = None
        self._mono_font = None

    # ------------------------------------------------------------------ #
    def init(self, hidden: bool = True) -> None:
        pygame.init()
        pygame.display.set_caption("gtsim replay")
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if hidden and hasattr(pygame, "HIDDEN"):
            try:
                self._screen = pygame.display.set_mode(
                    (self.width, self.height), flags | pygame.HIDDEN
                )
            except Exception:
                self._screen = pygame.display.set_mode(
                    (self.width, self.height), flags
                )
        else:
            self._screen = pygame.display.set_mode(
                (self.width, self.height), flags
            )
        self._font = pygame.font.SysFont("Consolas", 22, bold=True)
        self._small_font = pygame.font.SysFont("Consolas", 16)
        self._mono_font = pygame.font.SysFont("Consolas", 14)
        self._init_gl()

    def _init_gl(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        c = PALETTE["bg_bottom"]
        glClearColor(c[0], c[1], c[2], 1.0)
        # Lighting.
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glLightfv(GL_LIGHT0, GL_POSITION, [12.0, 18.0, 28.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.30, 0.32, 0.36, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.90, 0.90, 0.95, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.45, 0.45, 0.50, 1.0])
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.20, 0.22, 0.26, 1.0])

    def shutdown(self) -> None:
        if self._screen is not None:
            pygame.display.quit()
            pygame.quit()
            self._screen = None

    # ------------------------------------------------------------------ #
    def _set_perspective(self) -> None:
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.cam.fov_deg, self.cam.aspect, 0.5, 500.0)
        eye = self.cam.eye()
        up = self.cam.up()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(eye[0], eye[1], eye[2],
                  self.cam.target[0], self.cam.target[1], self.cam.target[2],
                  up[0], up[1], up[2])

    def _set_pov_perspective(self, p_D: np.ndarray, R_D: np.ndarray,
                              b_c: np.ndarray, fov_full_deg: float,
                              near: float = 0.25, far: float = 500.0) -> None:
        """Set up projection as if looking through the defender's camera.

        Eye = p_D, forward = R_D @ b_c (world-frame boresight),
        up = R_D @ [0, 1, 0] (body-y).  FoV here is the *vertical* FoV
        used by `gluPerspective`; pass `2 * theta_F_deg + margin` if you
        want the visibility cone boundary to sit inside the viewport.
        """
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(fov_full_deg, self.width / max(self.height, 1),
                       near, far)
        eye = np.asarray(p_D, dtype=float)
        forward = R_D @ np.asarray(b_c, dtype=float)
        up = R_D @ np.array([0.0, 1.0, 0.0])
        center = eye + forward
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(eye[0], eye[1], eye[2],
                  center[0], center[1], center[2],
                  up[0], up[1], up[2])

    # ------------------------------------------------------------------ #
    # Primitive drawing
    # ------------------------------------------------------------------ #
    def _grid(self, size: float = 80.0, step: float = 5.0,
              centre: np.ndarray | None = None) -> None:
        if centre is None:
            centre = np.zeros(3)
        glDisable(GL_LIGHTING)
        c = PALETTE["grid"]
        glColor4f(*c)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        n = int(size / step)
        cx, cy = float(centre[0]), float(centre[1])
        for i in range(-n, n + 1):
            x = cx + i * step
            glVertex3f(x, cy - size, 0.0)
            glVertex3f(x, cy + size, 0.0)
            y = cy + i * step
            glVertex3f(cx - size, y, 0.0)
            glVertex3f(cx + size, y, 0.0)
        glEnd()

    def _sphere(self, centre, radius, color, wire=False) -> None:
        glPushMatrix()
        glTranslatef(*centre)
        glColor4f(*color)
        q = gluNewQuadric()
        if wire:
            glDisable(GL_LIGHTING)
            gluQuadricDrawStyle(q, GLU_LINE)
        else:
            glEnable(GL_LIGHTING)
            gluQuadricDrawStyle(q, GLU_FILL)
            gluQuadricNormals(q, GLU_SMOOTH)
        gluSphere(q, radius, 24, 18)
        gluDeleteQuadric(q)
        glPopMatrix()

    def _cylinder(self, p0, p1, radius, color, slices=12) -> None:
        d = np.asarray(p1) - np.asarray(p0)
        h = float(np.linalg.norm(d))
        if h < 1e-6:
            return
        glEnable(GL_LIGHTING)
        glPushMatrix()
        glTranslatef(*p0)
        z = np.array([0.0, 0.0, 1.0])
        axis = np.cross(z, d / h)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(z, d / h))))))
        if float(np.linalg.norm(axis)) > 1e-6:
            glRotatef(ang, axis[0], axis[1], axis[2])
        elif float(np.dot(z, d / h)) < 0:
            glRotatef(180.0, 1.0, 0.0, 0.0)
        glColor4f(*color)
        q = gluNewQuadric()
        gluQuadricNormals(q, GLU_SMOOTH)
        gluCylinder(q, radius, radius, h, slices, 1)
        gluDeleteQuadric(q)
        glPopMatrix()

    def _cone(self, p0, p1, radius, color, slices=12) -> None:
        d = np.asarray(p1) - np.asarray(p0)
        h = float(np.linalg.norm(d))
        if h < 1e-6:
            return
        glEnable(GL_LIGHTING)
        glPushMatrix()
        glTranslatef(*p0)
        z = np.array([0.0, 0.0, 1.0])
        axis = np.cross(z, d / h)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(z, d / h))))))
        if float(np.linalg.norm(axis)) > 1e-6:
            glRotatef(ang, axis[0], axis[1], axis[2])
        elif float(np.dot(z, d / h)) < 0:
            glRotatef(180.0, 1.0, 0.0, 0.0)
        glColor4f(*color)
        q = gluNewQuadric()
        gluQuadricNormals(q, GLU_SMOOTH)
        gluCylinder(q, radius, 0.0, h, slices, 1)
        gluDeleteQuadric(q)
        glPopMatrix()

    def _disk(self, centre, axis, radius, color) -> None:
        """Flat disk perpendicular to `axis`."""
        glEnable(GL_LIGHTING)
        glPushMatrix()
        glTranslatef(*centre)
        z = np.array([0.0, 0.0, 1.0])
        d = np.asarray(axis, dtype=float)
        nd = float(np.linalg.norm(d))
        if nd < 1e-9:
            d = z
        else:
            d = d / nd
        axisrot = np.cross(z, d)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(z, d))))))
        if float(np.linalg.norm(axisrot)) > 1e-6:
            glRotatef(ang, axisrot[0], axisrot[1], axisrot[2])
        elif float(np.dot(z, d)) < 0:
            glRotatef(180.0, 1.0, 0.0, 0.0)
        glColor4f(*color)
        q = gluNewQuadric()
        gluQuadricNormals(q, GLU_SMOOTH)
        gluDisk(q, 0.0, radius, 18, 1)
        gluDeleteQuadric(q)
        glPopMatrix()

    def _line(self, a, b, color, width=1.5) -> None:
        glDisable(GL_LIGHTING)
        glColor4f(*color)
        glLineWidth(width)
        glBegin(GL_LINES)
        glVertex3f(*a); glVertex3f(*b)
        glEnd()

    def _polyline(self, pts, color, width=1.5, dashed=False) -> None:
        if len(pts) < 2:
            return
        glDisable(GL_LIGHTING)
        glColor4f(*color)
        glLineWidth(width)
        if dashed:
            glLineStipple(2, 0x00FF)
            glEnable(GL_LINE_STIPPLE)
        glBegin(GL_LINE_STRIP)
        for p in pts:
            glVertex3f(float(p[0]), float(p[1]), float(p[2]))
        glEnd()
        if dashed:
            glDisable(GL_LINE_STIPPLE)

    def _arrow(self, origin, vec, color, max_len=5.0, scale=0.18, width=2.5) -> None:
        n = float(np.linalg.norm(vec))
        if n < 1e-3:
            return
        L = min(max_len, n * scale)
        d = np.asarray(vec) / n
        tip = origin + d * L
        shaft_end = origin + d * (L * 0.85)
        self._line(origin, shaft_end, color, width=width)
        self._cone(shaft_end, tip, L * 0.06, color=color, slices=10)

    # ------------------------------------------------------------------ #
    # Rocket-style quadrotor body (after omnidrone_defense_sim's renderer)
    #
    # Fuselage is a cylinder along +body_z with a nose cone at the top.
    # Four arms sit in the body-xy plane (X-frame).  A small yellow fin
    # protrudes along +body_x to give the viewer a visible "forward" cue
    # so yaw rotations read on screen.  When the quadrotor tilts to
    # translate, the whole rocket leans together -- the unmistakable
    # visual signature of a racing quad diving for a target.
    #
    # When `draw_camera_cone` is True (defender only), a small cone
    # protrudes along the camera bore-sight direction (+body_z, since
    # we set b_c = +body_z) to make target lock visually obvious.
    # ------------------------------------------------------------------ #
    def _draw_quadrotor(self, centre: np.ndarray, R: np.ndarray,
                        arm_length: float, palette_prefix: str,
                        draw_fin: bool = True,
                        draw_camera_cone: bool = False) -> None:
        body_color = PALETTE[f"{palette_prefix}_body"]
        arm_color = PALETTE[f"{palette_prefix}_arm"]
        rotor_color = PALETTE[f"{palette_prefix}_rotor"]
        motor_color = PALETTE[f"{palette_prefix}_motor"]
        bx = R @ np.array([1.0, 0.0, 0.0])
        by = R @ np.array([0.0, 1.0, 0.0])
        bz = R @ np.array([0.0, 0.0, 1.0])
        L = float(arm_length)

        # ---------------- Fuselage along +body_z ----------------
        body_bottom = centre - bz * (L * 0.20)
        body_top    = centre + bz * (L * 0.55)
        nose_tip    = centre + bz * (L * 0.85)
        self._cylinder(body_bottom, body_top, L * 0.13, body_color)
        self._cone(body_top, nose_tip, L * 0.13, body_color)
        # Bottom cap (small inverted cone -- gives a finished look).
        cap_color = (body_color[0] * 0.55, body_color[1] * 0.35,
                     body_color[2] * 0.35, 1.0)
        self._cone(body_bottom, body_bottom - bz * (L * 0.07),
                   L * 0.13, cap_color)

        # ---------------- Forward indicator fin ----------------
        if draw_fin:
            fin_a = centre + bx * (L * 0.16)
            fin_b = centre + bx * (L * 0.32)
            self._cylinder(fin_a, fin_b, L * 0.04,
                           PALETTE["defender_fin"], slices=8)

        # ---------------- Four X-frame arms ----------------
        arm_len = L * 0.55
        arm_r = L * 0.045
        diag1 = (bx + by) / math.sqrt(2.0)
        diag2 = (bx - by) / math.sqrt(2.0)
        offsets = [diag1, diag2, -diag1, -diag2]
        for o in offsets:
            arm_tip = centre + o * arm_len
            self._cylinder(centre, arm_tip, arm_r, arm_color, slices=8)
            # Rotor disc: thin cylinder centred slightly above the arm tip,
            # oriented along body_z (so the disc face is perpendicular to
            # the thrust axis -- exactly where a real rotor sits).
            disc_base = arm_tip + bz * (L * 0.02)
            disc_top  = arm_tip + bz * (L * 0.06)
            self._cylinder(disc_base, disc_top, L * 0.18,
                           rotor_color, slices=14)
            # Tiny motor block under the rotor.
            motor_bot = arm_tip - bz * (L * 0.02)
            self._cylinder(motor_bot, disc_base, L * 0.06,
                           motor_color, slices=10)

        # ---------------- Camera cone on the nose ----------------
        # The camera sits at the rocket nose looking forward along body_z.
        # Drawn as a small bright cone so the target-lock direction is
        # visually unambiguous.
        if draw_camera_cone:
            cam_base = nose_tip
            cam_tip = cam_base + bz * (L * 0.32)
            self._cone(cam_base, cam_tip, L * 0.11,
                       (0.98, 0.95, 0.30, 1.0))

    # ------------------------------------------------------------------ #
    # FoV cone (translucent hollow cone along boresight)
    # ------------------------------------------------------------------ #
    def _draw_fov_cone(self, apex: np.ndarray, axis: np.ndarray,
                       half_angle: float, length: float) -> None:
        d = np.asarray(axis, dtype=float)
        nd = float(np.linalg.norm(d))
        if nd < 1e-6:
            return
        d = d / nd
        base_centre = apex + d * length
        base_radius = length * math.tan(half_angle)
        # Outline circle at the far end + a few slant lines for cone shape.
        glDisable(GL_LIGHTING)
        c = PALETTE["fov_cone"]
        glColor4f(c[0], c[1], c[2], 0.85)
        glLineWidth(1.4)
        # base ring
        if abs(d @ np.array([0.0, 0.0, 1.0])) < 0.95:
            e1 = np.cross(d, np.array([0.0, 0.0, 1.0]))
        else:
            e1 = np.cross(d, np.array([1.0, 0.0, 0.0]))
        e1 = e1 / max(float(np.linalg.norm(e1)), 1e-9)
        e2 = np.cross(d, e1)
        n_ring = 28
        ring_pts = []
        glBegin(GL_LINE_LOOP)
        for i in range(n_ring):
            t = 2 * math.pi * i / n_ring
            p = base_centre + base_radius * (math.cos(t) * e1
                                             + math.sin(t) * e2)
            ring_pts.append(p)
            glVertex3f(*p)
        glEnd()
        # 8 slant lines (apex -> ring)
        glBegin(GL_LINES)
        for i in range(0, n_ring, n_ring // 8):
            glVertex3f(*apex)
            glVertex3f(*ring_pts[i])
        glEnd()
        # filled translucent cone surface
        glColor4f(c[0], c[1], c[2], c[3])
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(*apex)
        for i in range(n_ring + 1):
            p = ring_pts[i % n_ring]
            glVertex3f(*p)
        glEnd()

    # ------------------------------------------------------------------ #
    # VFX
    # ------------------------------------------------------------------ #
    def _vfx_intercept(self, centre: np.ndarray, phase: float,
                       r_c: float) -> None:
        # Concentric expanding lit spheres, alpha decays with phase.
        # Sized so the entire VFX fits within ~2.2 r_c at maximum -- the
        # explosion is a localized accent, not an overlay that fills the
        # frame.
        rng = np.random.default_rng(42)
        for k, (rmul, abase) in enumerate(
            [(0.25, 0.95), (0.50, 0.80), (0.85, 0.55), (1.25, 0.30),
             (1.70, 0.18)]):
            r = r_c * (0.35 + rmul * (0.15 + 0.55 * phase))
            a = abase * (1.0 - 0.70 * phase)
            col = PALETTE["vfx_hot"] if k < 2 else PALETTE["vfx_outer"]
            self._sphere(centre, r, (col[0], col[1], col[2], max(0.05, a)))
        # 18 fragment streaks, shorter
        glDisable(GL_LIGHTING)
        glColor4f(1.0, 1.0, 1.0, max(0.0, 0.85 - phase))
        glLineWidth(1.6)
        L = r_c * (0.6 + 2.2 * phase)
        dirs = rng.normal(size=(18, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        glBegin(GL_LINES)
        for d in dirs:
            glVertex3f(*centre)
            glVertex3f(centre[0] + d[0] * L,
                       centre[1] + d[1] * L,
                       centre[2] + d[2] * L)
        glEnd()

    def _vfx_breach(self, centre: np.ndarray, phase: float, r_P: float) -> None:
        # Concentric expanding red spheres (same structure as the
        # intercept VFX but in red), plus 18 red fragment streaks --
        # a clearly visible "asset destroyed" event sized comparable
        # to the intercept flash.
        rng = np.random.default_rng(91)
        for k, (rmul, abase) in enumerate(
            [(0.25, 0.95), (0.50, 0.80), (0.85, 0.55), (1.25, 0.30),
             (1.70, 0.18)]):
            r = r_P * (0.35 + rmul * (0.15 + 0.55 * phase))
            a = abase * (1.0 - 0.70 * phase)
            col = PALETTE["breach_hot"] if k < 2 else PALETTE["breach_red"]
            self._sphere(centre, r, (col[0], col[1], col[2], max(0.05, a)))
        # Red fragment streaks
        glDisable(GL_LIGHTING)
        c = PALETTE["breach_red"]
        glColor4f(c[0], c[1], c[2], max(0.0, 0.85 - phase))
        glLineWidth(1.6)
        L = r_P * (0.6 + 2.2 * phase)
        dirs = rng.normal(size=(18, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        glBegin(GL_LINES)
        for d in dirs:
            glVertex3f(*centre)
            glVertex3f(centre[0] + d[0] * L,
                       centre[1] + d[1] * L,
                       centre[2] + d[2] * L)
        glEnd()

    def _vfx_vloss(self, centre: np.ndarray, phase: float) -> None:
        for rmul in (0.4, 0.9, 1.5):
            r = 1.2 * (1.0 + rmul * phase)
            a = max(0.05, 0.55 - 0.35 * phase)
            c = PALETTE["vloss_gold"]
            self._sphere(centre, r, (c[0], c[1], c[2], a))

    # ------------------------------------------------------------------ #
    # HUD via pygame fonts -> RGBA texture -> glDrawPixels
    # ------------------------------------------------------------------ #
    def _hud_begin(self) -> None:
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

    def _hud_end(self) -> None:
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW); glPopMatrix()
        glEnable(GL_DEPTH_TEST)

    def _blit_text(self, x: int, y: int, text: str,
                   color=(235, 235, 245), font=None) -> None:
        font = font or self._font
        surf = font.render(text, True, color)
        w, h = surf.get_size()
        data = pygame.image.tostring(surf, "RGBA", True)
        # glRasterPos sets origin at bottom-left of the bitmap; we flipped
        # the projection so origin is top-left.  Position the raster so
        # the text top lands on `y`.
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glRasterPos2i(int(x), int(y) + h)
        glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, data)

    # ------------------------------------------------------------------ #
    # Scene draw
    # ------------------------------------------------------------------ #
    def draw_frame(self, frame_state: dict, cfg: SimConfig) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._set_perspective()

        # ----- World ground grid (centered under the engagement) -----
        self._grid(size=60.0, step=5.0,
                   centre=np.array([cfg.geom.p_P[0], cfg.geom.p_P[1], 0.0]))

        # ----- Protected asset: glow shell + solid core -----
        p_P = np.asarray(cfg.geom.p_P, dtype=float)
        self._sphere(p_P, cfg.geom.r_P, PALETTE["asset_glow"], wire=False)
        self._sphere(p_P, cfg.geom.r_P, (0.20, 0.95, 0.45, 0.45), wire=True)
        self._sphere(p_P, cfg.geom.r_P * 0.22, PALETTE["asset_core"])

        # ----- Intruder tube scenarios (faint dashed) -----
        if frame_state.get("intruder_tube") is not None:
            tube = frame_state["intruder_tube"]   # (M, N+1, 3)
            for m in range(tube.shape[0]):
                self._polyline(tube[m], PALETTE["tube_line"],
                               width=1.0, dashed=True)

        # ----- MPPI defender candidate fan -----
        if frame_state.get("mppi_samples") is not None:
            for color, pts in frame_state["mppi_samples"]:
                self._polyline(pts, color, width=1.4)

        # ----- Trails -----
        trail_def = frame_state.get("trail_def", [])
        trail_int = frame_state.get("trail_int", [])
        if len(trail_def) >= 2:
            self._polyline(trail_def, PALETTE["defender_trail"], width=2.4)
        if len(trail_int) >= 2:
            self._polyline(trail_int, PALETTE["intruder_trail"], width=2.4)

        # ----- LOS line -----
        if not frame_state.get("in_vfx", False):
            self._line(frame_state["p_D"], frame_state["p_A"],
                       PALETTE["los_line"], width=1.0)

        # ----- Vehicles -----
        if not frame_state.get("in_vfx", False):
            self._draw_quadrotor(frame_state["p_D"], frame_state["R_D"],
                                 cfg.defender.visual_arm_length,
                                 palette_prefix="defender",
                                 draw_fin=True,
                                 draw_camera_cone=True)
            self._draw_quadrotor(frame_state["p_A"], frame_state["R_A"],
                                 cfg.intruder.visual_arm_length,
                                 palette_prefix="intruder",
                                 draw_fin=True,
                                 draw_camera_cone=False)
            # FoV cone along the actual defender boresight.
            b_c = np.asarray(cfg.defender.b_c, dtype=float)
            b_world = frame_state["R_D"] @ b_c
            theta_F = float(np.deg2rad(cfg.visibility.theta_F_deg))
            self._draw_fov_cone(frame_state["p_D"], b_world, theta_F,
                                length=4.0)
            # Velocity arrows.
            self._arrow(frame_state["p_D"], frame_state["v_D"],
                        PALETTE["defender_body"], max_len=4.0, scale=0.18)
            self._arrow(frame_state["p_A"], frame_state["v_A"],
                        PALETTE["intruder_body"], max_len=4.0, scale=0.16)

        # ----- VFX -----
        if frame_state.get("in_vfx", False):
            outcome = frame_state["outcome"]
            phase = frame_state["vfx_phase"]
            if outcome == "intercept":
                self._vfx_intercept(frame_state["vfx_centre"], phase,
                                    cfg.geom.r_c)
            elif outcome == "breach":
                self._vfx_breach(frame_state["vfx_centre"], phase,
                                 cfg.geom.r_P)
            elif outcome == "visual_loss":
                self._vfx_vloss(frame_state["vfx_centre"], phase)

        # ----- HUD -----
        self._hud_begin()
        self._draw_hud(frame_state, cfg)
        self._hud_end()

    # ------------------------------------------------------------------ #
    # First-person POV scene draw (eye attached to defender)
    # ------------------------------------------------------------------ #
    def draw_pov_frame(self, frame_state: dict, cfg: SimConfig) -> None:
        """Render the scene from the defender's onboard camera.

        Eye attached to p_D, looking along R_D @ b_c.  Skips the defender
        body itself (we're sitting inside it), keeps the intruder,
        protected asset, MPPI candidate fan, intruder trail, intruder
        prediction tube, and VFX so the user can see exactly what the
        on-board pipeline is seeing.  A reticle, FoV ring (at the actual
        visibility-cone half-angle), and a small status block are drawn
        on the HUD.
        """
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        b_c = np.asarray(cfg.defender.b_c, dtype=float)
        theta_F_deg = float(cfg.visibility.theta_F_deg)
        # Use a slightly wider FoV than 2*theta_F so the cone boundary
        # sits visibly inside the viewport (rather than at the edge).
        margin_deg = 12.0
        fov_full_deg = 2.0 * theta_F_deg + margin_deg
        self._set_pov_perspective(frame_state["p_D"], frame_state["R_D"],
                                   b_c, fov_full_deg)

        # ----- World ground grid (so the horizon is readable) -----
        self._grid(size=80.0, step=5.0,
                   centre=np.array([cfg.geom.p_P[0], cfg.geom.p_P[1], 0.0]))

        # ----- Protected asset -----
        p_P = np.asarray(cfg.geom.p_P, dtype=float)
        self._sphere(p_P, cfg.geom.r_P, PALETTE["asset_glow"], wire=False)
        self._sphere(p_P, cfg.geom.r_P, (0.20, 0.95, 0.45, 0.45), wire=True)
        self._sphere(p_P, cfg.geom.r_P * 0.22, PALETTE["asset_core"])

        # ----- Intruder tube (faint dashed) -----
        if frame_state.get("intruder_tube") is not None:
            tube = frame_state["intruder_tube"]
            for m in range(tube.shape[0]):
                self._polyline(tube[m], PALETTE["tube_line"],
                               width=1.0, dashed=True)

        # ----- MPPI candidate fan -----
        # Each candidate starts at p_D (the eye), so its first vertex sits
        # exactly at the camera origin and the leading segment projects
        # degenerately through the near plane.  Drop any vertices that
        # are not safely ahead of the eye along the boresight.
        if frame_state.get("mppi_samples") is not None:
            eye = np.asarray(frame_state["p_D"], dtype=float)
            fwd = np.asarray(frame_state["R_D"], dtype=float) @ b_c
            near_keep = 0.30
            for color, pts in frame_state["mppi_samples"]:
                arr = np.asarray(pts, dtype=float)
                depth = (arr - eye) @ fwd
                mask = depth > near_keep
                if int(mask.sum()) >= 2:
                    self._polyline(arr[mask], color, width=1.4)

        # ----- Intruder trail only (defender trail is invisible from
        # inside its own cockpit). -----
        trail_int = frame_state.get("trail_int", [])
        if len(trail_int) >= 2:
            self._polyline(trail_int, PALETTE["intruder_trail"], width=2.4)

        # ----- LOS line to the intruder -----
        if not frame_state.get("in_vfx", False):
            self._line(frame_state["p_D"], frame_state["p_A"],
                       PALETTE["los_line"], width=1.0)

        # ----- Intruder vehicle (this is the target the defender sees) -----
        if not frame_state.get("in_vfx", False):
            self._draw_quadrotor(frame_state["p_A"], frame_state["R_A"],
                                 cfg.intruder.visual_arm_length,
                                 palette_prefix="intruder",
                                 draw_fin=True,
                                 draw_camera_cone=False)

        # ----- VFX -----
        if frame_state.get("in_vfx", False):
            outcome = frame_state["outcome"]
            phase = frame_state["vfx_phase"]
            if outcome == "intercept":
                self._vfx_intercept(frame_state["vfx_centre"], phase,
                                    cfg.geom.r_c)
            elif outcome == "breach":
                self._vfx_breach(frame_state["vfx_centre"], phase,
                                 cfg.geom.r_P)
            elif outcome == "visual_loss":
                self._vfx_vloss(frame_state["vfx_centre"], phase)

        # ----- HUD: reticle, FoV ring, status -----
        self._hud_begin()
        self._draw_pov_hud(frame_state, cfg, theta_F_deg, fov_full_deg)
        self._hud_end()

    def _circle_hud(self, cx: float, cy: float, radius: float,
                    color: tuple, width: float = 2.0,
                    segments: int = 96) -> None:
        glDisable(GL_LIGHTING)
        glLineWidth(width)
        glColor4f(*color)
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            th = 2.0 * math.pi * i / segments
            glVertex2f(cx + radius * math.cos(th),
                       cy + radius * math.sin(th))
        glEnd()

    def _rect_hud(self, x: float, y: float, w: float, h: float,
                  color: tuple, filled: bool = False,
                  line_width: float = 2.0) -> None:
        glDisable(GL_LIGHTING)
        glColor4f(*color)
        if filled:
            glBegin(GL_QUADS)
            glVertex2f(x, y); glVertex2f(x + w, y)
            glVertex2f(x + w, y + h); glVertex2f(x, y + h)
            glEnd()
        else:
            glLineWidth(line_width)
            glBegin(GL_LINE_LOOP)
            glVertex2f(x, y); glVertex2f(x + w, y)
            glVertex2f(x + w, y + h); glVertex2f(x, y + h)
            glEnd()

    def _crosshair_hud(self, cx: float, cy: float, size: float = 18.0,
                       gap: float = 4.0, color: tuple = (1, 1, 1, 0.7),
                       width: float = 1.5) -> None:
        glDisable(GL_LIGHTING)
        glLineWidth(width)
        glColor4f(*color)
        glBegin(GL_LINES)
        glVertex2f(cx - size, cy); glVertex2f(cx - gap, cy)
        glVertex2f(cx + gap, cy); glVertex2f(cx + size, cy)
        glVertex2f(cx, cy - size); glVertex2f(cx, cy - gap)
        glVertex2f(cx, cy + gap); glVertex2f(cx, cy + size)
        glEnd()

    def _draw_pov_hud(self, fs: dict, cfg: SimConfig,
                      theta_F_deg: float, fov_full_deg: float) -> None:
        cx = self.width * 0.5
        cy = self.height * 0.5
        # Vertical FoV spans the height; an angle theta off-axis maps to
        # screen radius theta / (fov_full / 2) * (H / 2).
        radius_px = theta_F_deg / (fov_full_deg * 0.5) * (self.height * 0.5)

        h_V = float(fs.get("h_V", 0.0))
        lock_on = h_V > 0.0
        if lock_on:
            ring_color = (0.30, 0.90, 1.00, 0.85)
        else:
            ring_color = (1.00, 0.32, 0.28, 0.90)

        # Subtle inner ring at half-angle (gimballed read), then the
        # primary FoV boundary ring at theta_F.
        self._circle_hud(cx, cy, radius_px * 0.5,
                         (ring_color[0], ring_color[1], ring_color[2], 0.30),
                         width=1.2)
        self._circle_hud(cx, cy, radius_px, ring_color, width=2.4)

        # Center reticle.
        self._crosshair_hud(cx, cy, size=22, gap=5,
                            color=(0.85, 0.92, 1.00, 0.75), width=1.6)

        # Corner brackets to give it that "targeting display" feel.
        bracket = min(self.width, self.height) * 0.06
        for (bx, by, sx, sy) in [
            (12, 12, +1, +1),
            (self.width - 12, 12, -1, +1),
            (12, self.height - 12, +1, -1),
            (self.width - 12, self.height - 12, -1, -1),
        ]:
            glDisable(GL_LIGHTING)
            glLineWidth(2.0)
            glColor4f(0.55, 0.85, 1.00, 0.85)
            glBegin(GL_LINES)
            glVertex2f(bx, by); glVertex2f(bx + sx * bracket, by)
            glVertex2f(bx, by); glVertex2f(bx, by + sy * bracket)
            glEnd()

        # Title strip (translucent black band).
        self._rect_hud(0, 0, self.width, 38, (0.02, 0.04, 0.08, 0.55),
                       filled=True)
        self._blit_text(14, 8, "DEFENDER CAMERA POV",
                        color=(235, 240, 255))
        self._blit_text(self.width - 230, 10,
                        f"FoV ±{theta_F_deg:.0f}°",
                        color=(180, 220, 255), font=self._small_font)

        # Status block (bottom-left): lock / h_V / rho.
        self._rect_hud(0, self.height - 58, self.width, 58,
                       (0.02, 0.04, 0.08, 0.55), filled=True)
        lock_txt = "LOCK ON" if lock_on else "LOCK LOST"
        lock_color = (140, 240, 170) if lock_on else (255, 130, 120)
        self._blit_text(14, self.height - 50, lock_txt,
                        color=lock_color)
        line2 = (f"t = {fs['t']:5.2f} s   rho = {fs['rho']:5.2f} m   "
                 f"h_V = {fs['h_V']:+.3f}")
        self._blit_text(14, self.height - 24, line2,
                        color=(220, 225, 235), font=self._small_font)
        line3 = f"|Omega| = {fs['Omega_inf']:5.2f}"
        self._blit_text(self.width - 130, self.height - 24, line3,
                        color=(220, 225, 235), font=self._small_font)

    def _draw_hud(self, fs: dict, cfg: SimConfig) -> None:
        oc = fs.get("outcome", "in_progress")
        outcome_color = {
            "intercept": (124, 252, 139),
            "breach": (255, 110, 99),
            "visual_loss": (249, 212, 35),
            "timeout": (200, 200, 200),
            "in_progress": (200, 200, 200),
        }.get(oc, (220, 220, 220))
        title = cfg.scenario.name.upper().replace("_", " ")
        self._blit_text(18, 14, title, color=(235, 235, 245))
        self._blit_text(18, 44, f"outcome: {oc}", color=outcome_color)
        stress_tag = " STRESSED" if fs.get("stressed", 0.0) > 0.5 else ""
        line2 = (f"t = {fs['t']:5.2f} s    rho = {fs['rho']:5.2f} m    "
                 f"h_V = {fs['h_V']:+.3f}    mu_V = {fs['mu_V']:+.2f}    "
                 f"eta_V = {fs['eta_V']:+.2f}    "
                 f"|Omega| = {fs['Omega_inf']:5.2f}{stress_tag}")
        self._blit_text(18, self.height - 32, line2,
                        color=(220, 220, 230), font=self._small_font)
        # Bottom-right legend.
        legend_lines = [
            "blue: defender",
            "red:  intruder",
            "cyan-mag: MPPI candidates",
            "gray-dash: intruder tube",
            "cyan cone: camera FoV",
        ]
        for i, ln in enumerate(legend_lines):
            self._blit_text(self.width - 280, 18 + 18 * i, ln,
                            color=(180, 180, 195), font=self._mono_font)

    # ------------------------------------------------------------------ #
    def read_rgb(self) -> bytes:
        glReadBuffer(GL_BACK)
        pixels = glReadPixels(0, 0, self.width, self.height, GL_RGB,
                              GL_UNSIGNED_BYTE)
        # OpenGL origin is bottom-left, ffmpeg wants top-left.  Flip rows.
        arr = np.frombuffer(pixels, dtype=np.uint8).reshape(
            self.height, self.width, 3)
        return arr[::-1].tobytes()


# --------------------------------------------------------------------------- #
# Frame builders
# --------------------------------------------------------------------------- #


def _ffmpeg_writer(out_path: Path, width: int, height: int, fps: int
                   ) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "20",
        "-loglevel", "warning",
        str(out_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _color_for_cost(weight: float, norm_cost: float) -> tuple:
    """Cyan -> magenta -> orange gradient by cost, alpha by softmin weight."""
    # Cool to hot.  norm_cost in [0,1]: 0 = cyan (low cost), 1 = magenta.
    r = 0.20 + 0.80 * norm_cost
    g = 0.85 - 0.55 * norm_cost
    b = 1.00 - 0.30 * norm_cost
    a = float(min(1.0, 0.22 + weight * 9.0))
    return (r, g, b, a)


def render_video_opengl(
    metrics: MetricsLog,
    cfg: SimConfig,
    out_path: str | Path,
    plan_debug_history: list | None = None,
    fps: int = 30,
    stride: int = 3,
    width: int = 1280,
    height: int = 720,
    cam_yaw_deg: float = 38.0,
    cam_pitch_deg: float = 22.0,
    cam_distance: float = 40.0,
    cam_orbit_speed_deg_s: float = 6.0,
    trail_seconds: float = 1.5,
    extra_seconds_after_outcome: float = 1.4,
    show_mppi_samples: bool = True,
    n_samples_drawn: int = 64,
    show_intruder_tube: bool = True,
    hidden: bool = True,
    view_mode: str = "thirdperson",
    method_tag: str | None = None,
) -> Path:
    """Render a recorded run as a cinematic MP4 via pygame+PyOpenGL+ffmpeg.

    Returns the actual output path (may swap .mp4 for .gif fallback if
    ffmpeg is missing -- but the GIF path is not implemented here; we
    raise instead, since the OpenGL pipeline assumes ffmpeg).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not str(out_path).endswith(".mp4"):
        out_path = out_path.with_suffix(".mp4")

    recs = metrics.records
    if not recs:
        return out_path
    dt = (recs[1].t - recs[0].t) if len(recs) > 1 else cfg.sim.dt_sim
    n_pad = int(round(extra_seconds_after_outcome / max(dt, 1e-6)))
    n_real = len(recs)
    full_records = recs + [recs[-1]] * n_pad
    full_records = full_records[::stride]
    n_real_strided = (n_real + stride - 1) // stride

    # Pre-pack arrays.
    p_D = np.array([r.p_D for r in full_records], dtype=float)
    p_A = np.array([r.p_A for r in full_records], dtype=float)
    v_D = np.array([r.v_D for r in full_records], dtype=float)
    v_A = np.array([r.v_A for r in full_records], dtype=float)
    a_A_true = np.array([r.a_A_true for r in full_records], dtype=float)
    R_D = np.array([np.asarray(r.R_D_flat, dtype=float).reshape(3, 3)
                    for r in full_records])
    ts = np.array([r.t for r in full_records], dtype=float)
    rho_log = np.array([r.rho for r in full_records], dtype=float)
    hv_log = np.array([r.h_V for r in full_records], dtype=float)
    mv_log = np.array([r.mu_V for r in full_records], dtype=float)
    Om_log = np.array([r.Omega_applied for r in full_records], dtype=float)
    eta_log = np.array([r.eta_V for r in full_records], dtype=float)
    stressed_log = np.array([1.0 if r.planner_stressed else 0.0
                             for r in full_records], dtype=float)
    plan_ids = [r.plan_debug_idx for r in full_records]
    g = cfg.sim.g
    R_A_seq = np.array([_stable_intruder_R(a_A_true[i], v_A[i], g)
                        for i in range(len(full_records))])
    # Cinematic LPF on the intruder's body-z so juke flips translate
    # into smooth banking instead of frame-to-frame wobble.
    frame_dt = max(dt * stride, 1e-6)
    R_A_seq = _smooth_attitude_sequence(R_A_seq, frame_dt, tau=0.18)
    final_outcome = metrics.summary.outcome

    # Camera target = centroid of the asset, defender, intruder paths.
    # Fixed throughout (no terminal zoom per user request).
    asset = np.asarray(cfg.geom.p_P, dtype=float)
    target_xyz = (asset + p_D.mean(axis=0) + p_A.mean(axis=0)) / 3.0
    # Pick a distance that frames the full engagement but stays close
    # enough that the rocket bodies + camera FoV cone read clearly.
    spans = np.ptp(np.vstack([p_D, p_A, [asset]]), axis=0)
    cam_distance = max(cam_distance, float(spans.max()) * 1.05)

    renderer = OpenGLReplayRenderer(width=width, height=height)
    renderer.cam.target = target_xyz
    renderer.cam.distance = cam_distance
    renderer.cam.yaw_deg = cam_yaw_deg
    renderer.cam.pitch_deg = cam_pitch_deg
    renderer.init(hidden=hidden)

    trail_n = max(2, int(trail_seconds / max(dt * stride, 1e-6)))

    proc = _ffmpeg_writer(out_path, width, height, fps)
    try:
        for i in range(len(full_records)):
            # Orbit camera (slow rotation, fixed framing).
            renderer.cam.yaw_deg = cam_yaw_deg + cam_orbit_speed_deg_s * ts[i]
            # Build MPPI overlay if we have a plan debug for this tick.
            mppi_samples = None
            intruder_tube = None
            if (show_mppi_samples and plan_debug_history
                    and plan_ids[i] >= 0):
                idx = min(plan_ids[i], len(plan_debug_history) - 1)
                _t, dbg = plan_debug_history[idx]
                weights = dbg.weights
                risk = dbg.risk_scores
                vmin = float(risk.min()); vmax = float(risk.max())
                if vmax - vmin < 1e-6:
                    vmax = vmin + 1.0
                order = np.argsort(-weights)
                order = order[:min(n_samples_drawn, len(order))]
                mppi_samples = []
                for k in order:
                    norm_cost = float((risk[k] - vmin) / (vmax - vmin))
                    color = _color_for_cost(weights[k], norm_cost)
                    mppi_samples.append((color, dbg.p_D[k]))
                if show_intruder_tube:
                    intruder_tube = dbg.p_A   # (M, N+1, 3)
            # Trails.
            s = max(0, i - trail_n)
            trail_def = p_D[s:i + 1].tolist()
            trail_int = p_A[s:i + 1].tolist()

            in_vfx = i >= n_real_strided
            if in_vfx:
                if final_outcome == "intercept":
                    vfx_centre = 0.5 * (p_D[i] + p_A[i])
                elif final_outcome == "breach":
                    vfx_centre = np.asarray(cfg.geom.p_P, dtype=float)
                else:
                    vfx_centre = p_D[i]
                vfx_phase = (i - n_real_strided) / max(
                    1, len(full_records) - n_real_strided)
            else:
                vfx_centre = np.zeros(3)
                vfx_phase = 0.0

            frame_state = {
                "t": float(ts[i]),
                "rho": float(rho_log[min(i, len(rho_log) - 1)]),
                "h_V": float(hv_log[min(i, len(hv_log) - 1)]),
                "mu_V": float(mv_log[min(i, len(mv_log) - 1)]),
                "Omega_inf": float(np.max(np.abs(Om_log[min(i, len(Om_log) - 1)]))),
                "eta_V": float(eta_log[min(i, len(eta_log) - 1)]),
                "stressed": float(stressed_log[min(i, len(stressed_log) - 1)]),
                "p_D": p_D[i],
                "v_D": v_D[i],
                "R_D": R_D[i],
                "p_A": p_A[i],
                "v_A": v_A[i],
                "R_A": R_A_seq[i],
                "trail_def": trail_def,
                "trail_int": trail_int,
                "mppi_samples": mppi_samples,
                "intruder_tube": intruder_tube,
                "outcome": final_outcome,
                "in_vfx": in_vfx,
                "vfx_centre": vfx_centre,
                "vfx_phase": vfx_phase,
                "method_tag": method_tag,
            }
            if view_mode == "pov":
                renderer.draw_pov_frame(frame_state, cfg)
            else:
                renderer.draw_frame(frame_state, cfg)
            # Read the back buffer (where we just drew) BEFORE flipping it
            # to the front -- otherwise we read the *previous* frame and
            # the first frame is blank.
            glFinish()
            rgb = renderer.read_rgb()
            pygame.display.flip()
            proc.stdin.write(rgb)
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()
        renderer.shutdown()
    return out_path


# --------------------------------------------------------------------------- #
# PiP composition (third-person base + POV inset)
# --------------------------------------------------------------------------- #


def composite_pip_video(
    base_path: str | Path,
    inset_path: str | Path,
    out_path: str | Path,
    inset_scale: float = 0.28,
    margin_px: int = 24,
    corner: str = "top_right",
    border_color: str = "white",
    border_thickness: int = 3,
    label: str | None = None,
) -> Path:
    """Composite a POV MP4 as a labeled PiP inset over a third-person MP4.

    Uses ffmpeg's filter_complex.  Both inputs must share an fps but may
    differ in size and duration -- the shorter input is held on its last
    frame to match (`tpad=stop_mode=clone`).
    """
    base = Path(base_path)
    inset = Path(inset_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if corner == "top_right":
        pos = f"x=W-w-{margin_px}:y={margin_px}"
    elif corner == "top_left":
        pos = f"x={margin_px}:y={margin_px}"
    elif corner == "bottom_right":
        pos = f"x=W-w-{margin_px}:y=H-h-{margin_px}"
    elif corner == "bottom_left":
        pos = f"x={margin_px}:y=H-h-{margin_px}"
    else:
        pos = f"x=W-w-{margin_px}:y={margin_px}"

    bt = max(1, int(border_thickness))
    inset_chain = (
        f"[1:v]tpad=stop_mode=clone:stop_duration=10,"
        f"scale=iw*{inset_scale:.3f}:ih*{inset_scale:.3f},"
        f"pad=iw+{2*bt}:ih+{2*bt}:{bt}:{bt}:color={border_color}"
    )
    if label:
        safe_label = label.replace(":", r"\:").replace("'", r"\'")
        inset_chain += (
            f",drawtext=text='{safe_label}':"
            f"x=8:y=h-th-6:fontcolor=white:fontsize=18:"
            f"box=1:boxcolor=black@0.55:boxborderw=4"
        )
    inset_chain += "[pip]"

    overlay = f"[0:v][pip]overlay={pos}:shortest=0"
    filter_complex = inset_chain + ";" + overlay

    cmd = [
        "ffmpeg", "-y",
        "-i", str(base),
        "-i", str(inset),
        "-filter_complex", filter_complex,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "20",
        "-loglevel", "warning",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg PiP composite failed:\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {result.stderr}"
        )
    return out
