import numpy as np
from pyrr import Matrix44

IMAGE_SIZE = 128
NUM_SAMPLES = 3000
NUM_TEST = 600
NUM_VAL = 300

SPHERE_RADIUS = 1.0

POSITION_RANGE = 20.0
SHININESS_RANGE = (3.0, 20.0)

CAMERA_POSITION = [5.0, 5.0, 15.0]
CAMERA_TARGET = (0.0, 0.0, 0.0)
CAMERA_UP = (0.0, 1.0, 0.0)
FOV = 45.0
NEAR = 0.1
FAR = 1000.0

BACKGROUND_COLOR = (0.0, 0.0, 0.0)

MIN_LIGHT_TO_OBJECT = SPHERE_RADIUS + 0.5
MIN_CAMERA_TO_OBJECT = SPHERE_RADIUS + 1.0

PARAM_NAMES = [
    "object_x",
    "object_y",
    "object_z",  # położenie obiektu / 20
    "light_rel_x",
    "light_rel_y",
    "light_rel_z",  # (światło - obiekt) / 40
    "diffuse_r",
    "diffuse_g",
    "diffuse_b",  # kolor rozproszenia / 255 -> [-1, 1]
    "shininess",  # (s - 11.5) / 8.5
    "screen_x",
    "screen_y",
    "apparent_radius",  # rzut 2D środka kuli i jej rozmiar na ekranie
]
PARAM_DIM = len(PARAM_NAMES)


def sample_params(
    rng: np.random.Generator, position_range: float = POSITION_RANGE
) -> dict:
    while True:
        object_position = rng.uniform(-position_range, position_range, size=3)
        if (
            np.linalg.norm(object_position - np.asarray(CAMERA_POSITION))
            < MIN_CAMERA_TO_OBJECT
        ):
            continue

        diffuse_color = rng.uniform(0.0, 255.0, size=3)
        shininess = rng.uniform(*SHININESS_RANGE)  # [3, 20]

        for _ in range(1000):
            light_position = rng.uniform(-position_range, position_range, size=3)
            if np.linalg.norm(light_position - object_position) >= MIN_LIGHT_TO_OBJECT:
                break
        else:
            continue

        return {
            "object_position": object_position.astype(np.float32),
            "diffuse_color": diffuse_color.astype(np.float32),
            "shininess": np.float32(shininess),
            "light_position": light_position.astype(np.float32),
        }


def params_to_vector(params: dict) -> np.ndarray:
    obj = np.asarray(params["object_position"], dtype=np.float32)
    light = np.asarray(params["light_position"], dtype=np.float32)
    diffuse = np.asarray(params["diffuse_color"], dtype=np.float32)
    shininess = float(params["shininess"])

    obj_norm = obj / POSITION_RANGE
    light_rel = np.clip((light - obj) / (2.0 * POSITION_RANGE), -1.0, 1.0)
    diffuse_norm = diffuse / 255.0 * 2.0 - 1.0
    shininess_norm = (shininess - 11.5) / 8.5

    # Rzut środka kuli (0, 0, 0, 1) przez macierz MVP na współrzędne ekranowe
    _, mvp = build_matrices(params, aspect_ratio=1.0)
    clip = np.asarray(mvp, dtype=np.float32)[3]  # [clip_x, clip_y, clip_z, clip_w]
    w = float(clip[3])

    if w > NEAR:
        screen_x = np.clip(clip[0] / w, -2.0, 2.0)
        screen_y = np.clip(-clip[1] / w, -2.0, 2.0)
        tan_half_fov = np.tan(np.deg2rad(FOV / 2.0))
        r_ndc = SPHERE_RADIUS / (w * tan_half_fov)
        apparent_radius = np.clip(r_ndc, 0.0, 1.0) * 2.0 - 1.0
    else:
        screen_x, screen_y, apparent_radius = 2.0, 2.0, -1.0

    return np.concatenate(
        [
            obj_norm,
            light_rel,
            diffuse_norm,
            [shininess_norm, screen_x, screen_y, apparent_radius],
        ]
    ).astype(np.float32)


def build_matrices(params: dict, aspect_ratio: float = 1.0):
    """Macierze jak PhongWindow.on_render."""
    model_matrix = Matrix44.from_translation(params["object_position"])
    proj = Matrix44.perspective_projection(FOV, aspect_ratio, NEAR, FAR)
    lookat = Matrix44.look_at(CAMERA_POSITION, CAMERA_TARGET, CAMERA_UP)
    model_view_projection = proj * lookat * model_matrix
    return model_matrix, model_view_projection
