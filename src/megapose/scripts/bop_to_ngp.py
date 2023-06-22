import os
import argparse
import subprocess
import glob 
import simplejson as json 
import cv2 
import pyrr
import trimesh

import meshcat
import meshcat.geometry as g
import meshcat.transformations as mtf
import numpy as np 
import transforms3d

# These functions were provided by Lucas Manuelli 
def create_visualizer(clear=True, zmq_url='tcp://127.0.0.1:6000'):
    """
    If you set zmq_url=None it will start a server
    """

    print('Waiting for meshcat server... have you started a server? Run `meshcat-server` to start a server')
    vis = meshcat.Visualizer(zmq_url=zmq_url)
    if clear:
        vis.delete()
    return vis

def make_frame(vis, name, T=None, h=0.15, radius=0.001, o=1.0):
    """Add a red-green-blue triad to the Meschat visualizer.
    Args:
      vis (MeshCat Visualizer): the visualizer
      name (string): name for this frame (should be unique)
      h (float): height of frame visualization
      radius (float): radius of frame visualization
      o (float): opacity
    """
    vis[name]['x'].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0xff0000, reflectivity=0.8, opacity=o))
    rotate_x = mtf.rotation_matrix(np.pi / 2.0, [0, 0, 1])
    rotate_x[0, 3] = h / 2
    vis[name]['x'].set_transform(rotate_x)

    vis[name]['y'].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0x00ff00, reflectivity=0.8, opacity=o))
    rotate_y = mtf.rotation_matrix(np.pi / 2.0, [0, 1, 0])
    rotate_y[1, 3] = h / 2
    vis[name]['y'].set_transform(rotate_y)

    vis[name]['z'].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0x0000ff, reflectivity=0.8, opacity=o))
    rotate_z = mtf.rotation_matrix(np.pi / 2.0, [1, 0, 0])
    rotate_z[2, 3] = h / 2
    vis[name]['z'].set_transform(rotate_z)

    if T is not None:
        print(T)
        vis[name].set_transform(T)

def visii_camera_frame_to_rdf(T_world_Cv):
    """Rotates the camera frame to "right-down-forward" frame
    Returns:
        T_world_camera: 4x4 numpy array in "right-down-forward" coordinates
    """
    # C = camera frame (right-down-forward)
    # Cv = visii camera frame (right-up-back)
    T_Cv_C = np.eye(4)
    T_Cv_C[:3, :3] = transforms3d.euler.euler2mat(np.pi, 0, 0)
    T_world_C = T_world_Cv @ T_Cv_C
    return T_world_C



#################################### MAIN ########################################

parser = argparse.ArgumentParser(description='Renders glbs')

parser.add_argument(
    '--path', 
    type=str, 
    default='/media/jtremblay/bf64b840-723c-4e19-9dbc-f6a092b66406/home/jtremblay/Downloads/andrew_BOP_power_drills_p01/p01/train/',
    help='path where the data is, add the transform.json in the digit folder'
    )

parser.add_argument(
    '--make_alpha', 
    action="store_true",
    help='make a folder to make new images with alpha mask as png for ngp'
    )

parser.add_argument(
    '--only_alpha_trans', 
    action="store_true",
    help='change the name of the file'
    )

parser.add_argument(
    '--debug', 
    action="store_true",
    help='add poses to meshcat server'
    )


opt = parser.parse_args()

if opt.debug:
    vis = create_visualizer()

folders_to_compute = sorted(glob.glob(os.path.join(opt.path, "*/")))

for folder in folders_to_compute:
    
    #to export
    out = {}

    # camera intrinsics, assumes that they are all the same for the scene 
    with open(os.path.join(folder,"scene_camera.json"), 'r') as f:
        camera_data_json = json.load(f) 
    
    key_0 = list(camera_data_json.keys())[0]
    out['fl_x']=camera_data_json[key_0]['cam_K'][0]
    out['fl_y']=camera_data_json[key_0]['cam_K'][4]
    out['cx']=camera_data_json[key_0]['cam_K'][2]
    out['cy']=camera_data_json[key_0]['cam_K'][5]

    # check a single image for w,h
    img_folder = os.path.join(folder,"rgb/")
    mask_folder = os.path.join(folder,"mask/")

    img0_path = glob.glob(os.path.join(img_folder,"*.png"))[0]
    img = cv2.imread(img0_path)

    out['w']=img.shape[1]
    out['h']=img.shape[0]

    # make the final transforms

    with open(os.path.join(folder,"scene_gt.json"), 'r') as f:
        scene_data = json.load(f) 
    out['frames'] = []

    for trans_id in scene_data.keys():
        name_file = trans_id.zfill(7)
        
        rot = scene_data[trans_id][0]['cam_R_m2c']
        pos = scene_data[trans_id][0]['cam_t_m2c']
        m = pyrr.Matrix44(
            [
                [rot[0],rot[1],rot[2],pos[0]],
                [rot[3],rot[4],rot[5],pos[1]],
                [rot[6],rot[7],rot[8],pos[2]],
                [0,0,0,1]
            ])
        
        # to the world space
        m = pyrr.matrix44.inverse(m)
        # to opengl coordinate
        m = visii_camera_frame_to_rdf(m)

        # rotate to have gravity down 
        m = m* pyrr.Matrix44.from_y_rotation(-np.pi/2) 
        if opt.debug:
            make_frame(vis,trans_id,m)

        # make alpha add the alpha to the image and create a new folder
        if opt.make_alpha:
            os.makedirs(os.path.join(folder,'rgb_with_mask'),exist_ok=True)
            # load rgb
            rgb = cv2.imread(os.path.join(folder,"rgb",name_file+".png"))
            alpha = cv2.imread(os.path.join(folder,"mask",trans_id.zfill(6)+"_"+"0".zfill(6)+".png"))
            final = np.zeros([rgb.shape[0],rgb.shape[1],4])
            final[:,:,:3] = rgb
            final[:,:,-1] = alpha[:,:,0]
            cv2.imwrite(os.path.join(folder,'rgb_with_mask',name_file+".png"), final)
            # raise()
        # store the data 
        frame = {}

        # TODO update the path with alpha masks when generating them
        if opt.make_alpha or opt.only_alpha_trans:
            frame["file_path"] = os.path.join('rgb_with_mask',name_file+".png")
        else:
            frame["file_path"] = os.path.join("rgb",name_file+".png")
        frame["transform_matrix"] = [
            [m.m11,m.m12,m.m13,m.m14],
            [m.m21,m.m22,m.m23,m.m24],
            [m.m31,m.m32,m.m33,m.m34],
            [m.m41,m.m42,m.m43,m.m44]
        ]
        out['frames'].append(frame)

    # update this to usin the mesh
    path_2_mesh = os.path.join(opt.path,'../models/obj_000001.ply')
    print(path_2_mesh)
    mesh = trimesh.load(path_2_mesh)
    corners=trimesh.bounds.corners(mesh.bounding_box_oriented.bounds)
    print(corners)
    out['aabb'] = [
        [min(corners[:,0]),min(corners[:,1]),min(corners[:,2])],
        [max(corners[:,0]),max(corners[:,1]),max(corners[:,2])],
    ]
    with open(os.path.join(folder,'transforms.json'), 'w') as outfile:
        json.dump(out, outfile, indent=2)
    raise()