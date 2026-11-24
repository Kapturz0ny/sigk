"""Generowanie zbioru danych renderera Phonga.
- Obiekt: położenie <-20, 20>, diffuse <0, 255>, shininess [3, 20],
  ambient [76, 76, 76], specular [255, 255, 255], brak skali i rotacji.
- Światło: położenie <-20, 20>, kolory stałe:
  ambient [25, 25, 25], diffuse [255, 255, 255], specular [255, 255, 255].
- Kamera: stałe parametry z programu (CAMERA_POSITION i macierze z scene_params).
- Obrazy: 128x128 RGB, 3000 próbek.

Uruchomienie:
    python -m data.generate_dataset --output_dir ../dataset --seed 42
"""

import argparse
import json
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

from common.scene_params import (
    CAMERA_POSITION,
    NUM_TEST,
    NUM_VAL,
    PARAM_DIM,
    PARAM_NAMES,
    build_matrices,
    params_to_vector,
    sample_params,
)

ROOT = Path(__file__).resolve().parents[2]
SHADERS_DIR = ROOT / "resources" / "shaders" / "phong"
MODEL_PATH = ROOT / "resources" / "models" / "sphere.obj"


REQUIRED_IMAGE_SIZE = 128
REQUIRED_NUM_SAMPLES = 3000
REQUIRED_POS_MIN, REQUIRED_POS_MAX = -20.0, 20.0
REQUIRED_DIFFUSE_MIN, REQUIRED_DIFFUSE_MAX = 0.0, 255.0
REQUIRED_SHININESS_MIN, REQUIRED_SHININESS_MAX = 3.0, 20.0

MATERIAL_AMBIENT = np.array([76.0, 76.0, 76.0], dtype="f4") / 255.0
MATERIAL_SPECULAR = np.array([255.0, 255.0, 255.0], dtype="f4") / 255.0

LIGHT_AMBIENT = np.array([25.0, 25.0, 25.0], dtype="f4") / 255.0
LIGHT_DIFFUSE = np.array([255.0, 255.0, 255.0], dtype="f4") / 255.0
LIGHT_SPECULAR = np.array([255.0, 255.0, 255.0], dtype="f4") / 255.0


def create_context(prefer_backend: str | None = None) -> moderngl.Context:
    candidates = [prefer_backend] if prefer_backend else ["egl", None]
    errors = []
    for backend in candidates:
        try:
            if backend:
                return moderngl.create_standalone_context(backend=backend)
            return moderngl.create_standalone_context()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{backend or 'domyślny'}: {exc}")
    raise RuntimeError(
        "Nie udało się utworzyć kontekstu GL bez okna:\n  "
        + "\n  ".join(errors)
        + "\nMożesz wygenerować zbiór w oknie: python -m renderer.main (patrz README)."
    )


def load_mesh(path: Path):
    positions, normals, indices = [], [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                positions.append([float(p) for p in parts[1:4]])
            elif parts[0] == "vn":
                normals.append([float(p) for p in parts[1:4]])
            elif parts[0] == "f":
                face = [
                    (int(t.split("/")[0]) - 1, int(t.split("/")[2]) - 1)
                    for t in parts[1:]
                ]
                for i in range(1, len(face) - 1):
                    indices.extend(face[0])
                    indices.extend(face[i])
                    indices.extend(face[i + 1])

    positions = np.asarray(positions, dtype="f4")
    normals = np.asarray(normals, dtype="f4")

    pairs = np.asarray(indices, dtype=int).reshape(-1, 3, 2)
    pos = positions[pairs[:, :, 0]]
    nrm = normals[pairs[:, :, 1]]

    verts = np.concatenate([pos, nrm], axis=2).reshape(-1, 6)
    return np.ascontiguousarray(verts, dtype="f4")


def validate_sample(params: dict, model_matrix: np.ndarray) -> None:
    obj_pos = np.asarray(params["object_position"], dtype="f4")
    light_pos = np.asarray(params["light_position"], dtype="f4")
    diffuse = np.asarray(params["diffuse_color"], dtype="f4")
    shininess = float(params["shininess"])

    assert np.all((obj_pos >= REQUIRED_POS_MIN) & (obj_pos <= REQUIRED_POS_MAX)), (
        f"object_position poza zakresem [-20, 20]: {obj_pos}"
    )
    assert np.all((light_pos >= REQUIRED_POS_MIN) & (light_pos <= REQUIRED_POS_MAX)), (
        f"light_position poza zakresem [-20, 20]: {light_pos}"
    )
    assert np.all(
        (diffuse >= REQUIRED_DIFFUSE_MIN) & (diffuse <= REQUIRED_DIFFUSE_MAX)
    ), f"diffuse_color poza zakresem [0, 255]: {diffuse}"
    assert REQUIRED_SHININESS_MIN <= shininess <= REQUIRED_SHININESS_MAX, (
        f"shininess poza zakresem [3, 20]: {shininess}"
    )

    m = np.asarray(model_matrix, dtype="f4")
    assert np.allclose(m[:3, :3], np.eye(3, dtype="f4"), atol=1e-5), (
        f"model_matrix zawiera skalowanie lub rotację:\n{m[:3, :3]}"
    )


class OffscreenPhongRenderer:
    def __init__(self, size: int = REQUIRED_IMAGE_SIZE, gl_backend: str | None = None):
        self.ctx = create_context(gl_backend)
        vert = (SHADERS_DIR / "phong.vert").read_text()
        frag = (SHADERS_DIR / "phong.frag").read_text()
        self.program = self.ctx.program(vertex_shader=vert, fragment_shader=frag)

        verts = load_mesh(MODEL_PATH)
        vbo = self.ctx.buffer(verts.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program, [(vbo, "3f 3f", "in_position", "in_normal")]
        )

        self.color = self.ctx.texture((size, size), 4)
        self.depth = self.ctx.depth_renderbuffer((size, size))
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.color], depth_attachment=self.depth
        )
        self.fbo.use()
        self.size = size

        u = self.program
        self._uniforms = (
            u["model_view_projection"],
            u["model_matrix"],
            u["material_diffuse"],
            u["material_shininess"],
            u["light_position"],
            u["camera_position"],
        )

        optional_constants = {
            "material_ambient": MATERIAL_AMBIENT,
            "material_specular": MATERIAL_SPECULAR,
            "light_ambient": LIGHT_AMBIENT,
            "light_diffuse": LIGHT_DIFFUSE,
            "light_specular": LIGHT_SPECULAR,
        }
        for name, value in optional_constants.items():
            if name in u:
                u[name].write(value.tobytes())

    def render(self, params: dict) -> Image.Image:
        model_matrix, mvp = build_matrices(params, aspect_ratio=1.0)
        validate_sample(params, model_matrix)

        mvp_u, model_u, diff_u, shin_u, light_u, cam_u = self._uniforms
        mvp_u.write(mvp.astype("f4").tobytes())
        model_u.write(model_matrix.astype("f4").tobytes())
        diff_u.write(
            (np.asarray(params["diffuse_color"], dtype="f4") / 255.0).tobytes()
        )
        shin_u.write(np.array([params["shininess"]], dtype="f4").tobytes())
        light_u.write(np.asarray(params["light_position"], dtype="f4").tobytes())
        cam_u.write(np.asarray(CAMERA_POSITION, dtype="f4").tobytes())

        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.vao.render()

        raw = self.fbo.read(components=3)
        img = Image.frombytes("RGB", (self.size, self.size), raw)
        return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def make_preview(image_paths, out_path, grid=10):
    thumbs = [Image.open(p) for p in image_paths[: grid * grid]]
    sheet = Image.new("RGB", (grid * REQUIRED_IMAGE_SIZE, grid * REQUIRED_IMAGE_SIZE))
    for i, th in enumerate(thumbs):
        sheet.paste(
            th, ((i % grid) * REQUIRED_IMAGE_SIZE, (i // grid) * REQUIRED_IMAGE_SIZE)
        )
    sheet.save(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="../dataset")
    parser.add_argument("--num_samples", type=int, default=REQUIRED_NUM_SAMPLES)
    parser.add_argument("--num_test", type=int, default=NUM_TEST)
    parser.add_argument(
        "--num_val",
        type=int,
        default=NUM_VAL,
        help="walidacja wydzielona ze zbioru treningowego (dla Lightning)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--visible_ratio",
        type=float,
        default=0.8,
        help="udział klatek z widoczną kulą (0.0-1.0); "
        "np. 0.8 = 80%% widocznych, 20%% czarnych",
    )
    parser.add_argument(
        "--gl_backend",
        default=None,
        choices=[None, "egl", "glx"],
        help="Wymuś backend OpenGL (domyślnie: próbuj EGL, potem domyślny)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    renderer = OffscreenPhongRenderer(REQUIRED_IMAGE_SIZE, args.gl_backend)
    print(
        f"Kontekst GL: {renderer.ctx.info['GL_VERSION']} "
        f"({renderer.ctx.info['GL_RENDERER']})"
    )

    n_visible = round(args.num_samples * args.visible_ratio)
    n_black = args.num_samples - n_visible
    want_visible = np.array([True] * n_visible + [False] * n_black, dtype=bool)
    rng.shuffle(want_visible)

    records = {
        "object_position": [],
        "diffuse_color": [],
        "shininess": [],
        "light_position": [],
        "vector": [],
        "is_visible": [],
    }
    image_paths = []
    attempts = 0

    for i, need_visible in enumerate(want_visible):
        while True:
            attempts += 1
            params = sample_params(rng, REQUIRED_POS_MAX)
            img = renderer.render(params)
            is_visible = bool(np.asarray(img).max() > 0)
            if is_visible == need_visible:
                break

        path = images_dir / f"image_{i:05}.png"
        img.save(path)
        image_paths.append(path)
        records["object_position"].append(params["object_position"])
        records["diffuse_color"].append(params["diffuse_color"])
        records["shininess"].append(params["shininess"])
        records["light_position"].append(params["light_position"])
        records["vector"].append(params_to_vector(params))
        records["is_visible"].append(is_visible)
        if (i + 1) % 250 == 0:
            print(f"  wyrenderowano {i + 1}/{args.num_samples} (prób: {attempts})")

    np.savez_compressed(
        out_dir / "params.npz",
        object_position=np.stack(records["object_position"]),
        diffuse_color=np.stack(records["diffuse_color"]),
        shininess=np.stack(records["shininess"]),
        light_position=np.stack(records["light_position"]),
        vector=np.stack(records["vector"]),
        is_visible=np.array(records["is_visible"], dtype=bool),
        param_names=np.array(PARAM_NAMES),
    )

    all_idx = np.arange(args.num_samples)
    test_idx = np.sort(rng.choice(all_idx, size=args.num_test, replace=False))
    rest = np.setdiff1d(all_idx, test_idx)
    val_idx = np.sort(rng.choice(rest, size=args.num_val, replace=False))
    train_idx = np.setdiff1d(rest, val_idx)
    with open(out_dir / "split.json", "w") as f:
        json.dump(
            {
                "train": train_idx.tolist(),
                "val": val_idx.tolist(),
                "test": test_idx.tolist(),
            },
            f,
        )

    make_preview(image_paths, out_dir / "preview.png")
    print(
        f"Zapisano {args.num_samples} obrazów ({REQUIRED_IMAGE_SIZE}x{REQUIRED_IMAGE_SIZE}), "
        f"wektor cech: {PARAM_DIM}-wymiarowy."
    )
    print(
        f"  trening: {len(train_idx)}, walidacja: {len(val_idx)}, test: {len(test_idx)}"
    )
    print(f"Zbiór w: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
