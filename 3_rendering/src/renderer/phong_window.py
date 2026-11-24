import json
import os.path

import moderngl
import numpy as np
from PIL import Image

from common.scene_params import (
    CAMERA_POSITION,
    PARAM_NAMES,
    build_matrices,
    params_to_vector,
    sample_params,
)
from renderer.base_window import BaseWindow


class PhongWindow(BaseWindow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame = 0
        self.num_frames = self.argv.num_frames
        self.rng = np.random.default_rng(self.argv.seed)
        self._records = []

    def init_shaders_variables(self):
        self.model_view_projection = self.program["model_view_projection"]
        self.model_matrix = self.program["model_matrix"]
        self.material_diffuse = self.program["material_diffuse"]
        self.material_shininess = self.program["material_shininess"]
        self.light_position = self.program["light_position"]
        self.camera_position = self.program["camera_position"]

    @classmethod
    def add_arguments(cls, parser):
        super().add_arguments(parser)
        parser.add_argument(
            "--num_frames",
            type=int,
            required=False,
            default=None,
            help="Zamknij okno po wyrenderowaniu tylu klatek (generowanie zbioru)",
        )
        parser.add_argument(
            "--seed",
            type=int,
            required=False,
            default=42,
            help="Ziarno generatora liczb losowych",
        )

    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        # Losowanie parametrów sceny (patrz scene_params.sample_params)
        params = sample_params(self.rng)
        model_matrix, model_view_projection = build_matrices(params, self.aspect_ratio)

        self.model_view_projection.write(model_view_projection.astype("f4").tobytes())
        self.model_matrix.write(model_matrix.astype("f4").tobytes())
        self.material_diffuse.write(
            (params["diffuse_color"] / 255.0).astype("f4").tobytes()
        )
        self.material_shininess.write(
            np.array([params["shininess"]], dtype="f4").tobytes()
        )
        self.light_position.write(params["light_position"].astype("f4").tobytes())
        self.camera_position.write(np.array(CAMERA_POSITION, dtype="f4").tobytes())

        self.vao.render()

        if self.output_path:
            img = (
                Image.frombuffer("RGBA", self.wnd.size, self.wnd.fbo.read(components=4))
                .transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                .convert("RGB")
            )
            img.save(os.path.join(self.output_path, f"image_{self.frame:05}.png"))

            # Zapis parametrów sceny razem z obrazem - będą etykietami dla sieci
            self._records.append(
                {
                    "object_position": params["object_position"].tolist(),
                    "diffuse_color": params["diffuse_color"].tolist(),
                    "shininess": float(params["shininess"]),
                    "light_position": params["light_position"].tolist(),
                    "vector": params_to_vector(params).tolist(),
                }
            )
            if self.frame % 100 == 0 or (
                self.num_frames and self.frame + 1 >= self.num_frames
            ):
                self._save_params()

        self.frame += 1
        if self.num_frames is not None and self.frame >= self.num_frames:
            self._save_params()
            self.wnd.close()

    def _save_params(self):
        if not self.output_path or not self._records:
            return
        path = os.path.join(self.output_path, "params.json")
        with open(path, "w") as f:
            json.dump({"param_names": PARAM_NAMES, "records": self._records}, f)

    def on_close(self):
        self._save_params()
