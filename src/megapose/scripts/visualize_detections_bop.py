
import nvisii as visii 
import argparse
import numpy as np

import simplejson as json
import pyrender 
import cv2 
import numpy as np
import os 
import glob
import time 
import shutil
import pickle 
import pyrr


conversion_rotation = [
    [-1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
]

parser = argparse.ArgumentParser()

parser.add_argument(
    '--spp', 
    default=50,
    type=int,
    help = "number of sample per pixel, higher the more costly"
)

parser.add_argument(
    '--contour',
    action='store_true',
    default=False,
    help = "Only draw the contour instead of the 3d model overlay"
)
parser.add_argument(
    '--debug',
    action='store_true',
    help = "debug mode" 
)

parser.add_argument(
    '--coarse',
    action='store_true',
    help = "visualize coarse detections (if available) " 
)
parser.add_argument(
    '--overlay',
    action='store_true',
    default=False,
    help = "add the overlay"
)

parser.add_argument(
    '--gray',
    action='store_true',
    default=False,
    help = "draw the 3d model in gray"
)

parser.add_argument(
    '--path_json',
    required=True,
    help = "path to the json files you want loaded,\
            it assumes that there is a png accompanying."
)

parser.add_argument(
    '--bop_gt',
    type=str,
)

parser.add_argument(
    '--gt_translation',
    action='store_true',
    help="Use the GT translation value. "
)
opt = parser.parse_args()

# # # # # # # # # # # # # # # # # # # # # # # # #

def create_obj(
    name = 'name',
    path_obj = "",
    path_tex = None,
    scale = 1, 
    rot_base = None, #visii quat
    pos_base = (-10,-10,-10), # visii vec3
    base_color = (255, 255, 255),
    ):

    
    # This is for YCB like dataset
    if path_obj in create_obj.meshes:
        obj_mesh = create_obj.meshes[path_obj]
    else:
        obj_mesh = visii.mesh.create_from_obj(name, path_obj)
        create_obj.meshes[path_obj] = obj_mesh

    try: 
        visii.transform.remove(name)
        visii.material.remove(name)
        visii.entity.remove(name)
    except: 
        pass 
    
    obj_entity = visii.entity.create(
        name = name,
        # mesh = visii.mesh.create_sphere("mesh1", 1, 128, 128),
        mesh = obj_mesh,
        transform = visii.transform.create(name),
        material = visii.material.create(name)
    )

    # should randomize
    obj_entity.get_material().set_metallic(0)  # should 0 or 1      
    obj_entity.get_material().set_transmission(0)  # should 0 or 1      
    obj_entity.get_material().set_roughness(1) # default is 1  

    if not path_tex is None:

        if path_tex in create_obj.textures:
            obj_texture = create_obj.textures[path_tex]
        else:
            obj_texture = visii.texture.create_from_image(name,path_tex)
            create_obj.textures[path_tex] = obj_texture


        obj_entity.get_material().set_base_color_texture(obj_texture)
    else:
        obj_entity.get_material().set_base_color(visii.vec3(*base_color))

    obj_entity.get_transform().set_scale(visii.vec3(scale))

    if not rot_base is None:
        obj_entity.get_transform().set_rotation(rot_base)
    if not pos_base is None:
        obj_entity.get_transform().set_position(pos_base)
    print(f' created: {obj_entity.get_name()}')
    return obj_entity

create_obj.meshes = {}
create_obj.textures = {}

# # # # # # # # # # # # # # # # # # # # # # # # #

# with open(os.path.join(os.path.dirname(opt.path_json), "scene_camera.json")) as f:
#     camera_data_json = json.load(f)

# # visii.initialize_headless()
# visii.enable_denoiser()

# camera = visii.entity.create(
#     name = "camera",
#     transform = visii.transform.create("camera"),
#     # camera = visii.camera.create(
#     #     name = "camera", 
#     # )
#     camera = visii.camera.create_from_intrinsics(
#         name="camera",
#         fx=camera_data_json[list(camera_data_json.keys())[0]]["cam_K"][0],
#         fy=camera_data_json[list(camera_data_json.keys())[0]]["cam_K"][4],
#         cx=camera_data_json[list(camera_data_json.keys())[0]]["cam_K"][2],
#         cy=camera_data_json[list(camera_data_json.keys())[0]]["cam_K"][5],
#     )
# )

# camera.get_transform().look_at(
#     visii.vec3(0,0,-1), # look at (world coordinate)
#     visii.vec3(0,1,0), # up vector
#     visii.vec3(0,0,0), # camera_origin    
# )

# visii.set_camera_entity(camera)

# visii.set_dome_light_intensity(1)

# try:
#     visii.set_dome_light_color(visii.vec3(1, 1, 1), 0)
# except TypeError:
#     # Support for alpha transparent backgrounds was added in nvisii ef1880aa,
#     # but as of 2022-11-03, the latest released version (1.1) does not include
#     # that change yet.
#     print("WARNING! Your version of NVISII does not support alpha transparent backgrounds yet; --contour will not work properly.")
#     visii.set_dome_light_color(visii.vec3(1, 1, 1))

# # # # # # # # # # # # # # # # # # # # # # # # #

# LOAD THE SCENE 

objects_added = []
dope_trans = []
gt = None


import glob 

detections_fp = sorted(glob.glob(opt.path_json + "/*out.object_data.json"))

print("len(detections_fp)", len(detections_fp))


first_run = True 

for detection_fp in detections_fp:

    frame = os.path.basename(detection_fp).split(".")[0]

    with open(detection_fp) as f:
        detection_data = json.load(f)

    # ONLY VISUALIZES FIRST OJBECT FOR NOW, WILL CHANGE LATER 
    # detection_data = detection_data[0]

    print("detection_data", detection_data)

    camera_data_fp = detection_fp.replace("out.object_data.json", "camera_data.json")

    coarse_detections_fp = detection_fp.replace("out.object_data.json", "coarse_estimates.pkl")

    visualize_coarse_detections = (opt.coarse and os.path.exists(coarse_detections_fp))
    
    with open(camera_data_fp) as f:
        camera_data = json.load(f)

    load_bop_gt = opt.bop_gt is not None and os.path.isfile(opt.bop_gt)
    if load_bop_gt:
        with open(opt.bop_gt) as f:
            bop_gt = json.load(f)

    st = time.time() 

    camera_data['intrinsics'] = {}
    camera_data['intrinsics']['fx'] = camera_data["K"][0][0]
    camera_data['intrinsics']['fy'] = camera_data["K"][1][1]
    camera_data['intrinsics']['cx'] = camera_data["K"][0][2]
    camera_data['intrinsics']['cy'] = camera_data["K"][1][2]


    img_path = detection_fp.replace("out.object_data.json", "rgb.jpg")

    if not os.path.exists(img_path):
        img_path = detection_fp.replace("out.object_data.json", "rgb.png")
    
    
    # print(f"Reading image from {img_path}")
    opt.im_path = img_path

    img = cv2.imread(img_path)
    camera_data['height'] = img.shape[0]
    camera_data['width'] = img.shape[1]

    # set the camera
    intrinsics = camera_data['intrinsics']
    im_height = camera_data['height']
    im_width = camera_data['width']
    
    if first_run:
        ############################################## 
        visii.initialize_headless()
        visii.enable_denoiser()

        camera = visii.entity.create(
            name = "camera",
            transform = visii.transform.create("camera"),

            camera = visii.camera.create_from_intrinsics(
                name="camera",
                fx=camera_data['intrinsics']['fx'],
                fy=camera_data['intrinsics']['fy'],
                cx=camera_data['intrinsics']['cx'],
                cy=camera_data['intrinsics']['cy'],
                height=camera_data["height"],
                width=camera_data["width"],
                near=0.00,
                far=10000.0,            
            )
        )

        camera.get_transform().look_at(
            visii.vec3(0,0,-1), # look at (world coordinate)
            visii.vec3(0,1,0), # up vector
            visii.vec3(0,0,0), # camera_origin    
        )

        visii.set_camera_entity(camera)

        visii.set_dome_light_intensity(1)

        try:
            visii.set_dome_light_color(visii.vec3(1, 1, 1), 0)
        except TypeError:
            # Support for alpha transparent backgrounds was added in nvisii ef1880aa,
            # but as of 2022-11-03, the latest released version (1.1) does not include
            # that change yet.
            print("WARNING! Your version of NVISII does not support alpha transparent backgrounds yet; --contour will not work properly.")
            visii.set_dome_light_color(visii.vec3(1, 1, 1))
        ############################################## 

        cam = pyrender.IntrinsicsCamera(intrinsics['fx'],intrinsics['fy'],intrinsics['cx'],intrinsics['cy'])

        proj_matrix = cam.get_projection_matrix(im_width, im_height)
        
        # print("--\nprojeciton matrix: ")
        # for row in proj_matrix:
        #     print(row)
        proj_matrix = visii.mat4(
                proj_matrix.flatten()[0],
                proj_matrix.flatten()[1],
                proj_matrix.flatten()[2],
                proj_matrix.flatten()[3],
                proj_matrix.flatten()[4],
                proj_matrix.flatten()[5],
                proj_matrix.flatten()[6],
                proj_matrix.flatten()[7],
                proj_matrix.flatten()[8],
                proj_matrix.flatten()[9],
                proj_matrix.flatten()[10],
                proj_matrix.flatten()[11],
                proj_matrix.flatten()[12],
                proj_matrix.flatten()[13],
                proj_matrix.flatten()[14],
                proj_matrix.flatten()[15],
        )


        proj_matrix = visii.transpose(proj_matrix)

        camera.get_camera().set_projection(proj_matrix)

    all_objs = [] 

    # get the objects to load. 
    for obj in detection_data:
        to_add = {}
        to_add['label'] = str(obj['label'])
        to_add['location'] = obj['TWO'][1]

        rotation_quat = obj['TWO'][0]   

        # # NEED TO CHANGE 
        # rot = np.array(obj['cam_R_m2c']).reshape(-1)
        # # rot = obj["cam_R_m2c"]
        # m = pyrr.Matrix33(
        #     [
        #         [rot[0],rot[1],rot[2]],
        #         [rot[3],rot[4],rot[5]],
        #         [rot[6],rot[7],rot[8]],              
        #     ])
        # quat = m.quaternion 

        ## **** CHANGE IF ROTATION IS NOT XYZW **** ##
        # Assume rotation_quat is wxyz 
        # to_add['quaternion_xyzw'] = [rotation_quat[1],rotation_quat[2],rotation_quat[3],rotation_quat[0]]
        to_add['quaternion_xyzw'] = rotation_quat
        all_objs.append(to_add)

    for i_obj, obj in enumerate(all_objs):
        label = obj['label']

        # mesh_path = os.path.join(os.path.dirname(opt.path_json), f"meshes/{label}/{label}.ply")
        mesh_path = "/backup/handal_dataset_raw/handal_dataset_hammers/models/obj_000003.ply"

        if first_run:
            print(f"\nloading {label} from {mesh_path}\n")
            entity_visii = create_obj(
                name = f"{label}_{str(i_obj).zfill(2)}",
                path_obj = mesh_path,
                path_tex = None, # f"{opt.objs_folder}/obj_{str(name).zfill(6)}.png",
                scale = 0.001 * 6,
                rot_base = None
            )
            if visualize_coarse_detections:
                coarse_entity_visii = create_obj(
                    name = f"{label}_{str(i_obj).zfill(2)}_coarse",
                    path_obj = mesh_path,
                    path_tex = None, # f"{opt.objs_folder}/obj_{str(name).zfill(6)}.png",
                    scale = 0.001,
                    rot_base = None,
                    base_color=(251, 206, 177), # orange 
                )
            
            if opt.gt_translation:
                gt_translation_entity = create_obj(
                    name = f"{label}_{str(i_obj).zfill(2)}_gt_translation",
                    path_obj = mesh_path,
                    path_tex = None, # f"{opt.objs_folder}/obj_{str(name).zfill(6)}.png",
                    scale = 0.001 * 6,
                    rot_base = None,
                    base_color=(0, 0, 255), # blue  
                )
        else:
            entity_visii = visii.entity.get(name=f"{label}_{str(i_obj).zfill(2)}") 

            if visualize_coarse_detections:
                coarse_entity_visii = visii.entity.get(name=f"{label}_{str(i_obj).zfill(2)}_coarse")
            
            if opt.gt_translation:
                gt_translation_entity = visii.entity.get(name=f"{label}_{str(i_obj).zfill(2)}_gt_translation")

        pos = obj['location']
        rot = obj['quaternion_xyzw']

        if not opt.debug: 
            entity_visii.get_transform().set_rotation(
                visii.quat(
                    rot[3],
                    rot[0],
                    rot[1],
                    rot[2],
                )
            )

            entity_visii.get_transform().set_position(
                visii.vec3(
                    pos[0] / 1000 * 6,
                    pos[1] / 1000 * 6,
                    pos[2] / 1000 * 6,
                )
            )
        else:
            entity_visii.get_transform().set_position(
                visii.vec3(
                    0,0,0
                )
            )

        entity_visii.get_transform().rotate_around(visii.vec3(0,0,0),visii.angleAxis(visii.pi(), visii.vec3(1,0,0)))

        if visualize_coarse_detections:
            with open(coarse_detections_fp, 'rb') as f:
                coarse_detections = pickle.load(f)

            print("coarse_detections", coarse_detections)

            matrix4 = coarse_detections.poses[0].detach().cpu().numpy()

            conversion_rotation = np.array(conversion_rotation)
            print("matrix4.shape", matrix4.shape)
            print("conversion_rotation.shape", conversion_rotation.shape)
            matrix4 =  matrix4 #  np.linalg.inv(conversion_rotation)

            if first_run:
                transform = visii.transform.create(f"{label}_{str(i_obj).zfill(2)}_coarse_tf")
            else:
                transform = visii.transform.get(f"{label}_{str(i_obj).zfill(2)}_coarse_tf")

            transform.set_transform(matrix4.flatten(order="F"))

            coarse_rot = transform.get_rotation()
            coarse_pos = transform.get_position()

            print('-')
            print("coarse_pos", [coarse_pos[0], coarse_pos[1], coarse_pos[2]])
            print("det    pos", pos)

            print("coarse_rot", [coarse_rot.x, coarse_rot.y, coarse_rot.z, coarse_rot.w])
            print("det    rot", rot)

            coarse_entity_visii.get_transform().set_rotation(
                visii.quat(
                    coarse_rot.w,
                    coarse_rot.x,
                    coarse_rot.y,
                    coarse_rot.z,
                )
            )

            coarse_entity_visii.get_transform().set_position(
                visii.vec3(
                    coarse_pos[0],
                    coarse_pos[1],
                    coarse_pos[2],
                )
            )

            coarse_entity_visii.get_transform().rotate_around(visii.vec3(0,0,0),visii.angleAxis(visii.pi(), visii.vec3(1,0,0)))
        if load_bop_gt:
            frame_bop_gt = bop_gt[str(int(frame))]
            gt_location = frame_bop_gt[0]['cam_t_m2c']
            gt_rotation = np.array(frame_bop_gt[0]['cam_R_m2c']).reshape(-1)

            m = pyrr.Matrix33(
                [
                    [gt_rotation[0],gt_rotation[1],gt_rotation[2]],
                    [gt_rotation[3],gt_rotation[4],gt_rotation[5]],
                    [gt_rotation[6],gt_rotation[7],gt_rotation[8]],              
                ])
            
            gt_quat = m.quaternion

            # print('-\nBOP GT')
            # print("gt_location", gt_location)
            # print("quat", quat)
            # print("m", m)

            if opt.gt_translation:
                gt_translation_entity.get_transform().set_rotation(
                    visii.quat(
                        rot[3],
                        rot[0],
                        rot[1],
                        rot[2],
                    )
                )

                gt_translation_entity.get_transform().set_position(
                    visii.vec3(
                        gt_location[0] / 1000 * 6,
                        gt_location[1] / 1000 * 6,
                        gt_location[2] / 1000 * 6,
                    )
                )

                gt_translation_entity.get_transform().rotate_around(visii.vec3(0,0,0),visii.angleAxis(visii.pi(), visii.vec3(1,0,0)))

                print("gt_quat", [gt_quat[0], gt_quat[1], gt_quat[2], gt_quat[3]])
                print("rot", [rot[0], rot[1], rot[2], rot[3]])
                print("gt_location", [gt_location[0], gt_location[1], gt_location[2]])
                print("pos", [pos[0], pos[1], pos[2]])


    # # # # # # # # # # # # # # # # # # # # # # # # #
    output_path = os.path.join(os.path.dirname(opt.path_json), "visualizations", f"out.{frame.zfill(6)}.png")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pt = time.time() 
    visii.render_to_file(
        width=im_width, 
        height=im_height, 
        samples_per_pixel=opt.spp,
        file_path=output_path
    )
    # print(f"time taken to render_to_file: {time.time() - pt}")

    # raise()

    # # # # # # # # # # # # # # # # # # # # # # # # #

    # # create overlay

    im_path = opt.im_path

    print("im_path", im_path)
    # if os.path.exists(opt.path_json.replace("json",'png')):
    #     im_path = opt.path_json.replace("json",'png')
        
    # elif os.path.exists(opt.path_json.replace("json",'jpg')):
    #     im_path = opt.path_json.replace("json",'jpg')

    im = cv2.imread(im_path)

    print("im.shape", im.shape)

    if im is not None: 
        im_pred = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)

        alpha = im_pred[:,:,-1]/255.0 * 0.75
        alpha = np.stack((alpha,)*3, axis=-1)
        im_pred = im_pred.astype(float)
        
        im = im.astype(float)

        foreground = cv2.multiply(alpha,im_pred[:,:,:3])
        

        if opt.gray:
            im_gray = cv2.imread(im_path)
            im_gray = cv2.cvtColor(im_gray, cv2.COLOR_BGR2GRAY)
            im_gray = im_gray.astype(float)
            im_gray = np.stack((im_gray,)*3, axis=-1)
            background = cv2.multiply(1-alpha,im_gray)
        elif opt.overlay:
            background = cv2.multiply(1-alpha,im[:,:,:3])
        else:
            # background = im[:,:,:3]
            background = cv2.multiply(alpha,im[:,:,:3])

            foreground = background

        outrgb = cv2.add(foreground,background)
        
        if opt.contour:     
            gray = cv2.cvtColor((alpha*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            for c in cnts:
                cv2.drawContours(outrgb, [c], -1, (36, 255, 12), thickness=2,lineType=cv2.LINE_AA)
        print(f'outputting {output_path}')
        if opt.gt_translation:
            cv2.putText(outrgb, fontScale=1, org=(10, 50), text="Blue: GT translation and predicted rotation", fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255,0, 0), thickness=2)
        
            cv2.putText(outrgb, fontScale=1, org=(10, 100), text="White: Predicted translation and predicted rotation", fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255,255,255), thickness=2)

        cv2.imwrite(output_path, outrgb)
    
    first_run = False 
    print(f"time taken for frame: {time.time() - st}")

    if opt.debug: 
        break 
    
    input("Press Enter to continue...")

# let's clean up the GPU
visii.deinitialize()
