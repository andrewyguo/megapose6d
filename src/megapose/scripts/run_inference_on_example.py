# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

# Third Party
import numpy as np
from bokeh.io import export_png
from bokeh.plotting import gridplot
from PIL import Image
import torch 
import cv2 
import logging 

import glob 

# MegaPose
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level
from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay

logger = get_logger(__name__)

rot = [
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
]

rot_z_90 = [
    [0, -1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
]

rot_y_90 = [
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 0, 1],
]

rot_x_90 = [
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
]

def load_observation(
    example_dir: Path,
    load_depth: bool = False,
    use_fp: bool = False,
) -> Tuple[np.ndarray, Union[None, np.ndarray], CameraData]:
    camera_data = CameraData.from_json((example_dir / "camera_data.json").read_text())

    img_path = example_dir / "image_rgb.png" 
    if not img_path.exists():
        img_path = example_dir / "image_rgb.jpg"

    rgb = np.array(Image.open(img_path), dtype=np.uint8)
    assert rgb.shape[:2] == camera_data.resolution

    depth = None
    if load_depth:
        depth = np.array(Image.open(example_dir / "image_depth.png"), dtype=np.float32) / 1000
        assert depth.shape[:2] == camera_data.resolution

    return rgb, depth, camera_data

def load_observation_fp(data_dir, load_depth: bool = False, index: str=None):
    img_path = data_dir / "img" / f"{index}.rgb.png" 
    if not img_path.exists():
        img_path = data_dir / "img" / f"{index}.rgb.jpg"

    rgb = np.array(Image.open(img_path), dtype=np.uint8)

    camera_data = CameraData.from_json((data_dir / "img" / f"{index}.camera_data.json").read_text())

    depth = None 
    if load_depth:
        depth = np.array(Image.open(data_dir / "img" / f"{index}.depth.png"), dtype=np.float32) / 1000
        assert depth.shape[:2] == camera_data.resolution

    return rgb, depth, camera_data

def load_observation_batch(image_path: Path, load_depth: bool = False, file_extension: str = None):
    rgb = np.array(Image.open(image_path), dtype=np.uint8)

    camera_data = CameraData.from_json(Path(str(image_path).replace(f".rgb.{file_extension}", ".camera_data.json")).read_text())

    depth = None 
    if load_depth:
        depth = np.array(Image.open(str(image_path).replace(".rgb.", ".depth.")), dtype=np.float32) / 1000
        assert depth.shape[:2] == camera_data.resolution

    observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K)
    return observation 

def load_observation_tensor(
    example_dir: Path,
    load_depth: bool = False,
) -> ObservationTensor:
    rgb, depth, camera_data = load_observation(example_dir, load_depth)
    observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K)
    return observation


def load_object_data(data_path: Path) -> List[ObjectData]:
    object_data = json.loads(data_path.read_text())
    object_data = [ObjectData.from_json(d) for d in object_data]
    return object_data


def load_detections(
    example_dir: Path = None,
    detections_fp: Path = None,
) -> DetectionsType:
    if detections_fp:
        input_object_data = load_object_data(detections_fp)
    else:
        input_object_data = load_object_data(example_dir / "inputs/object_data.json")
    detections = make_detections_from_object_data(input_object_data).cuda()
    return detections


def make_object_dataset(example_dir: Path, mesh_units="m") -> RigidObjectDataset:
    rigid_objects = []
    # mesh_units = "m"
    object_dirs = (example_dir / "meshes").iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"there multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
        # TODO: fix mesh units
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


def make_detections_visualization(
    example_dir: Path,
) -> None:
    rgb, _, _ = load_observation(example_dir, load_depth=False)
    detections = load_detections(example_dir)
    plotter = BokehPlotter()
    fig_rgb = plotter.plot_image(rgb)
    fig_det = plotter.plot_detections(fig_rgb, detections=detections)
    output_fn = example_dir / "visualizations" / "detections.png"
    output_fn.parent.mkdir(exist_ok=True)
    export_png(fig_det, filename=output_fn)
    logger.info(f"Wrote detections visualization: {output_fn}")
    return


def save_predictions(
    example_dir: Path,
    pose_estimates: PoseEstimatesType,
) -> None:
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    object_data = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]
    object_data_json = json.dumps([x.to_json() for x in object_data])
    output_fn = example_dir / "outputs" / "object_data.json"
    output_fn.parent.mkdir(exist_ok=True)
    output_fn.write_text(object_data_json)
    logger.info(f"Wrote predictions: {output_fn}")
    return

def save_predictions_batch(output_fp, pose_estimates):
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    object_data = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]
    object_data_json = json.dumps([x.to_json() for x in object_data])
    output_fp.parent.mkdir(exist_ok=True)
    output_fp.write_text(object_data_json)
    logger.info(f"Wrote predictions: {output_fp}")
    return

def run_inference(
    example_dir: Path,
    model_name: str,
    use_ngp_renderer: bool = False
) -> None:

    model_info = NAMED_MODELS[model_name]

    observation = load_observation_tensor(
        example_dir, load_depth=model_info["requires_depth"]
    ).cuda()
    detections = load_detections(example_dir).cuda()
    object_dataset = make_object_dataset(example_dir, mesh_units="mm")

    logger.info(f"Loading model {model_name}.")
    pose_estimator = load_named_model(model_name, object_dataset, use_ngp_renderer=use_ngp_renderer).cuda()

    logger.info(f"Running inference.")
    output, _ = pose_estimator.run_inference_pipeline(
        observation, detections=detections, **model_info["inference_parameters"], use_coarse_estimates=False, 
    )

    save_predictions(example_dir, output)
    return


def make_output_visualization(
    example_dir: Path,
) -> None:

    rgb, _, camera_data = load_observation(example_dir, load_depth=False)
    camera_data.TWC = Transform(np.eye(4))
    object_datas = load_object_data(example_dir / "outputs" / "object_data.json")
    object_dataset = make_object_dataset(example_dir)

    renderer = Panda3dSceneRenderer(object_dataset)

    for object_data in object_datas:
        print("TWO", object_data.TWO.toHomogeneousMatrix())
        homo = object_data.TWO.toHomogeneousMatrix()
        object_data.TWO = Transform( homo @ rot_x_90 @ rot_x_90 @ rot_y_90 @ rot ) 
        print("TWO", object_data.TWO.toHomogeneousMatrix())

    camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
    light_datas = [
        Panda3dLightData(
            light_type="ambient",
            color=((1.0, 1.0, 1.0, 1)),
        ),
    ]
    renderings = renderer.render_scene(
        object_datas,
        [camera_data],
        light_datas,
        render_depth=False,
        render_binary_mask=False,
        render_normals=False,
        copy_arrays=True,
    )[0]

    plotter = BokehPlotter()

    fig_rgb = plotter.plot_image(rgb)
    fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
    contour_overlay = make_contour_overlay(
        rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
    )["img"]
    fig_contour_overlay = plotter.plot_image(contour_overlay)
    fig_all = gridplot([[fig_rgb, fig_contour_overlay, fig_mesh_overlay]], toolbar_location=None)
    vis_dir = example_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    export_png(fig_mesh_overlay, filename=vis_dir / "mesh_overlay.png")
    export_png(fig_contour_overlay, filename=vis_dir / "contour_overlay.png")
    export_png(fig_all, filename=vis_dir / "all_results.png")
    logger.info(f"Wrote visualizations to {vis_dir}.")
    return

def make_output_visualization_batch(data_dir: Path):
    object_dataset = make_object_dataset(data_dir)
    renderer = Panda3dSceneRenderer(object_dataset, use_ngp_renderer=False)

    detections_fp = sorted(glob.glob(str(data_dir) + "/img/*.out.object_data.json"))
    # print("detection_fp", detections_fp)

    for detection_fp in detections_fp:
        index = os.path.basename(detection_fp).split(".")[0]
   
        rgb, _, camera_data = load_observation_fp(data_dir, load_depth=False, index=index)
        camera_data.TWC = Transform(np.eye(4))
        object_data = load_object_data(Path(detection_fp))

        camera_data, object_data = convert_scene_observation_to_panda3d(camera_data, object_data)
        light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=((1.0, 1.0, 1.0, 1)),
            ),
        ]

        print("camera_data.shape", camera_data.shape)

        renderings = renderer.render_scene(
            object_data,
            [camera_data],
            light_datas,
            render_depth=False,
            render_binary_mask=False,
            render_normals=False,
            copy_arrays=True,
        )[0]

        plotter = BokehPlotter()

        fig_rgb = plotter.plot_image(rgb) 
        print("rgb.shape", rgb.shape, "renderings.rgb.shape", renderings.rgb.shape)
        fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
        contour_overlay = make_contour_overlay(
            rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
        )["img"]
        fig_contour_overlay = plotter.plot_image(contour_overlay)
        fig_all = gridplot([[fig_rgb, fig_contour_overlay, fig_mesh_overlay]], toolbar_location=None)

        vis_dir = data_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        fig_all_fp = vis_dir / f"{index}.all_results.png"
        export_png(fig_all, filename=fig_all_fp)
        logger.info(f"Wrote visualizations to {fig_all_fp}.")  

        # input("Press Enter to continue...")
    
    return 

def run_inference_on_batch(
    data_dir: Path,
    model_name: str,
    use_ngp_renderer: bool = True,
) -> None:
    logging.getLogger().setLevel(logging.INFO)

    model_info = NAMED_MODELS[model_name]

    img_fps = sorted(glob.glob(str(data_dir) + "/img/*.rgb.png") + glob.glob(str(data_dir) + "/img/*.rgb.jpg"))

    object_dataset = make_object_dataset(data_dir)

    logger.info(f"Loading model {model_name}.")
    pose_estimator = load_named_model(model_name, object_dataset, use_ngp_renderer=use_ngp_renderer).cuda()

    for i, img_path_str in enumerate(img_fps):
        # if not img_path_str.endswith("000610.rgb.jpg"):
        #     print("skipping", img_path_str)
        #     continue 
        # else:
        #     print("img_path_str", img_path_str)

        file_extension = img_path_str.split(".")[-1]
        img_path = Path(img_path_str)
        logger.info(f"Running inference on {img_path_str}.")

        observation = load_observation_batch(
            img_path, load_depth=model_info["requires_depth"], file_extension=file_extension,
        ).cuda()
        detections = load_detections(detections_fp=Path(img_path_str.replace(f".rgb.{file_extension}", ".object_datas.json"))).cuda()

        # this is torch.Size([1, 3, 1440, 1920]), observation just contains the image 
        print("observation.images.shape", observation.images.shape)

        coarse_estimates_fp = Path(img_path_str.replace(f".rgb.{file_extension}", ".coarse_estimates.pkl"))
        # coarse_estimates_fp = None 
        print("model_info['inference_parameters']", model_info["inference_parameters"])
        output, extra_data = pose_estimator.run_inference_pipeline(
            observation, detections=detections, **model_info["inference_parameters"], coarse_estimates_fp=coarse_estimates_fp, use_coarse_estimates=True
        )

        save_predictions_batch(Path(img_path_str.replace(f".rgb.{file_extension}", ".out.object_data.json")), output)
        
        if i == 0:
            break 
            input("Press Enter to continue...")


if __name__ == "__main__":
    set_logging_level("info")
    parser = argparse.ArgumentParser()
    parser.add_argument("example_name")
    parser.add_argument("--model", type=str, default="megapose-1.0-RGB-multi-hypothesis")
    parser.add_argument("--vis-detections", action="store_true")
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--vis-outputs", action="store_true")
    parser.add_argument("--batch-inference", action="store_true")
    parser.add_argument("--batch-vis", action="store_true")
    parser.add_argument("--pandas-renderer", "--no-ngp", action="store_true")
    args = parser.parse_args()

    data_dir = os.getenv("MEGAPOSE_DATA_DIR")
    assert data_dir
    example_dir = Path(data_dir) / "examples" / args.example_name

    torch.multiprocessing.set_start_method("spawn")
    print(f"Mode: args.batch_inference: {args.batch_inference}, args.run_inference: {args.run_inference}")

    if args.vis_detections:
        make_detections_visualization(example_dir)

    if args.run_inference:
        run_inference(example_dir, args.model)

    if args.vis_outputs:
        make_output_visualization(example_dir)

    if args.batch_inference:
        run_inference_on_batch(example_dir, args.model, not args.pandas_renderer)

    if args.batch_vis:
        make_output_visualization_batch(example_dir)
