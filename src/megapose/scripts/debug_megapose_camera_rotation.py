import numpy as np 
import cv2 

import pyrr 
import transforms3d


def post_process_ngp_render(ngp_render):
    def linear_to_srgb(img):
        limit = 0.0031308
        return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)
    
    ngp_render = np.copy(ngp_render)
    ngp_render[...,0:3] = np.divide(ngp_render[...,0:3], ngp_render[...,3:4], out=np.zeros_like(ngp_render[...,0:3]), where=ngp_render[...,3:4] != 0)
    ngp_render[...,0:3] = linear_to_srgb(ngp_render[...,0:3])

    ngp_render = (np.clip(ngp_render, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    ngp_render = cv2.cvtColor(ngp_render, cv2.COLOR_RGBA2RGB)

    return ngp_render

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



cam_R_m2c = np.array(
[
        -0.9913483522260499,
        -0.11113187056247648,
        0.06984688761413468,
        0.009587059267189653,
        0.4694051608624985,
        0.8829310953158577,
        -0.13090824518991917,
        0.8759617218385388,
        -0.46427850374571417
      ]
    )

cam_t_m2c = np.array([
        0.03813781238218269,
        0.010862865881845154,
        1.059550502266836
      ])

# cam_R_m2c = cam_R_m2c.reshape(3,3).T

# inv_cam_R_m2c = np.linalg.inv(cam_R_m2c)

# print("inv_cam_R_m2c\n", inv_cam_R_m2c)

# print(cam_R_m2c)

cam_m2c_col_major  = pyrr.Matrix44(
    [
        [cam_R_m2c[0],cam_R_m2c[1],cam_R_m2c[2],cam_t_m2c[0]],
        [cam_R_m2c[3],cam_R_m2c[4],cam_R_m2c[5],cam_t_m2c[1]],
        [cam_R_m2c[6],cam_R_m2c[7],cam_R_m2c[8],cam_t_m2c[2]],
        [0,0,0,1],
    ])

# cam_m2c_col_major = np.eye(4)

# cam_m2c_col_major[:3,:3] = cam_R_m2c
# cam_m2c_col_major[:3,3] = cam_t_m2c

print("cam_m2c_col_major\n", cam_m2c_col_major)
# for i in range(3):
#     print(np.linalg.norm(cam_m2c_col_major[:,i]))


import sys 
sys.path.append("/instant-ngp/build")
sys.path.append("/home/andrewg/instant-ngp/build")
try:
    import pyngp as ngp  # noqa
except Exception as e:
    print(e)

inv_cam_m2c_col_major = np.linalg.inv(cam_m2c_col_major)


print("inv_cam_m2c_col_major:\n", inv_cam_m2c_col_major)
# print("visii_camera_frame_to_rdf(inv_cam_m2c_col_major):\n", visii_camera_frame_to_rdf(inv_cam_m2c_col_major))
inv_cam_m2c_col_major = visii_camera_frame_to_rdf(inv_cam_m2c_col_major) 
print("inv_cam_m2c_col_major: after visii_camera_frame_to_rdf \n", inv_cam_m2c_col_major)

inv_cam_m2c_col_major = inv_cam_m2c_col_major * pyrr.Matrix44.from_y_rotation(-np.pi/2)

print("pyrr.Matrix44.from_y_rotation(-np.pi/2) \n", pyrr.Matrix44.from_y_rotation(-np.pi/2))

print("inv_cam_m2c_col_major: after from_y_rotation \n", inv_cam_m2c_col_major)


print("setup_testbed in App()")
testbed = ngp.Testbed()
testbed.load_snapshot("data/hammers_models_nerf/handal_dataset_raw/handal_dataset_hammers/models_nerf/obj_000003.latents.ingp")

# testbed.exposure = ...
testbed.background_color = [0.0, 0.0, 0.0, 1.0]

testbed.fov_axis = 0
testbed.shall_train = False 
testbed.exposure = 0.0
testbed.nerf.sharpen = float(0)
testbed.nerf.render_with_lens_distortion = True
testbed.fov_axis = 0
testbed.fov = 1.0819319066613973 * 180 / np.pi

testbed.render_mode = ngp.RenderMode.Shade 
testbed.set_nerf_camera_matrix(inv_cam_m2c_col_major[:-1, :])
rgb = testbed.render(1920, 1440, 16, True)

cv2.imwrite(f"data/test_out/debug_render.png", cv2.cvtColor(post_process_ngp_render(rgb), cv2.COLOR_RGB2BGR))

# negate_z = [
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0,0,-1,0],
#     [0,0,0,1]
# ]
# rot = [
#     [-1, 0, 0, 0],
#     [0, 0, -1, 0],
#     [0, -1, 0, 0],
#     [0, 0, 0, 1],
# ]

# print("inv_cam_m2c_col_major[:3,:3]\n", inv_cam_m2c_col_major[:3,:3])

fixed_transform =   inv_cam_m2c_col_major

# print("fixed_transform[:-1, :]\n", fixed_transform[:3,:3])

extrinsics = testbed.nerf.training.get_camera_extrinsics(0)

print("extrinsics\n", extrinsics)

extrinsics_4x4 = np.eye(4)
extrinsics_4x4[:3, :] = extrinsics
# print("extrinsics_4x4\n", extrinsics_4x4)

extrinsics_inv = np.linalg.inv(extrinsics_4x4)
# print("extrinsics_inv\n", extrinsics_inv)

fixed_transform_swap = np.eye(4)
fixed_transform_swap[0,0] = fixed_transform[2,0]
fixed_transform_swap[1,0] = fixed_transform[2,1]
testbed.set_nerf_camera_matrix(fixed_transform[:-1, :])
rgb = testbed.render(1920, 1440, 16, True)

cv2.imwrite(f"data/test_out/debug_render_1.png", cv2.cvtColor(post_process_ngp_render(rgb), cv2.COLOR_RGB2BGR))


print("output to ", f"data/test_out/debug_render.png")


# print(inv_cam_m2c_col_major.tolist())


# A = np.array([
#     [11, 12, 13, 14],
#     [21, 22, 23, 24],
#     [31, 32, 33, 34],
#     [41, 42, 43, 44],
# ])

# B = np.array([
#     [31, 21, -11],
#     [-32, -22, 12],
#     [-33, -23, 13],
# ])

# C = np.array([
#     [0, 0, 1, 0],
#     [0, 1, 0, 0],
#     [-1, 0, 0, 0],
#     [0, 0, 0, 1],
# ])

# C_neg = np.array([
#     [1, 0, 0, 0],
#     [0, -1, 0, 0],
#     [0, 0, -1, 0],
#     [0, 0, 0, 1],
# ])

# # C = C_neg @ C

# res = C @ A  
# res = C_neg @ res.T 
# # print(res)


# print(res.T)

