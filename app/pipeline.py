"""Image -> 3D mesh pipeline.

`run()` is the single entry point the job worker calls. It dispatches to
either the real TRELLIS model or a placeholder mesh, so the rest of the
service (upload, job queue, GLB serving, three.js viewer) can be built and
tested without a GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from app.config import settings


def run(image_path: Path, output_path: Path) -> None:
    if settings.use_real_trellis:
        _run_trellis(image_path, output_path)
    else:
        _run_placeholder(image_path, output_path)


def _run_placeholder(image_path: Path, output_path: Path) -> None:
    """Turns any input image into a simple colored cube GLB.

    Lets you verify the full upload -> job -> render pipeline end to end
    without a GPU. Replace calls to this with `_run_trellis` (by setting
    TRELLIS_USE_REAL_TRELLIS=true) once this service runs on a GPU box.
    """
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    # Give it a visible color so it's obviously a placeholder, not a bug.
    mesh.visual.vertex_colors = np.tile([255, 120, 0, 255], (mesh.vertices.shape[0], 1))
    mesh.export(output_path, file_type="glb")


def _run_trellis(image_path: Path, output_path: Path) -> None:
    """Real TRELLIS inference. Requires a GPU + requirements-trellis.txt.

    TODO: implement once this service is deployed on a GPU machine. Rough
    shape, following microsoft/TRELLIS's example scripts:

        from PIL import Image
        from trellis.pipelines import TrellisImageTo3DPipeline
        from trellis.utils import postprocessing_utils

        pipeline = TrellisImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS-image-large"
        )
        pipeline.cuda()

        image = Image.open(image_path)
        outputs = pipeline.run(image)  # -> gaussians, radiance field, mesh

        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0], outputs["mesh"][0]
        )
        glb.export(output_path)

    Load the pipeline once at process startup (module-level singleton)
    rather than per-request — it's expensive to construct.
    """
    raise NotImplementedError(
        "Real TRELLIS inference isn't wired up yet. Set "
        "TRELLIS_USE_REAL_TRELLIS=false (default) to use the placeholder "
        "pipeline, or implement _run_trellis on a GPU machine."
    )
