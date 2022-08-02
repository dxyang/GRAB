
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import os, glob
import smplx
import argparse
from tqdm import tqdm
import pickle
import h5py
from pytorch3d import transforms
from meshcat_viewer import get_visualizer, draw_point_cloud, draw_transform
from smplx.vertex_ids import vertex_ids
from smplx.joint_names import JOINT_NAMES
import time


from tools.objectmodel import ObjectModel
from tools.meshviewer import Mesh, MeshViewer, points2sphere, colors
from tools.utils import parse_npz
from tools.utils import params2torch
from tools.utils import makepath
from tools.utils import to_cpu
from tools.utils import euler
from tools.cfg_parser import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_smpl_joint_names():
    # these are similar to the first 24 joints available in JOINT_NAMES from joint_names.py as well
    return [
        'hips',            # 0 (this is global orientation)
        'leftUpLeg',       # 1
        'rightUpLeg',      # 2
        'spine',           # 3
        'leftLeg',         # 4
        'rightLeg',        # 5
        'spine1',          # 6
        'leftFoot',        # 7
        'rightFoot',       # 8
        'spine2',          # 9
        'leftToeBase',     # 10
        'rightToeBase',    # 11
        'neck',            # 12
        'leftShoulder',    # 13
        'rightShoulder',   # 14
        'head',            # 15
        'leftArm',         # 16
        'rightArm',        # 17
        'leftForeArm',     # 18
        'rightForeArm',    # 19
        'leftHand',        # 20
        'rightHand',       # 21
        'leftHandIndex1',  # 22 (dropped for smplx)
        'rightHandIndex1', # 23 (dropped for smplx)
    ]


def get_smpl_skeleton():
    return np.array(
        [
            [ 0, 1 ],
            [ 0, 2 ],
            [ 0, 3 ],
            [ 1, 4 ],
            [ 2, 5 ],
            [ 3, 6 ],
            [ 4, 7 ],
            [ 5, 8 ],
            [ 6, 9 ],
            [ 7, 10],
            [ 8, 11],
            [ 9, 12],
            [ 9, 13],
            [ 9, 14],
            [12, 15],
            [13, 16],
            [14, 17],
            [16, 18],
            [17, 19],
            [18, 20],
            [19, 21],
            [20, 22],
            [21, 23],
        ]
    )

def calculate_global_rotations(global_orient: torch.Tensor, body_pose: torch.Tensor):
    NUM_JOINTS = 22 # 0 = global hip, # 1 - 21 in the get_smpl_joint_names
    smpl_skeleton = get_smpl_skeleton()


    global_rot_mat = transforms.axis_angle_to_matrix(global_orient)
    joint_aas = [body_pose[i * 3: i * 3 + 3] for i in range(NUM_JOINTS - 1)]
    joint_rot_mats = [transforms.axis_angle_to_matrix(aa) for aa in joint_aas]
    joint_rot_mats.insert(0, global_rot_mat)

    absolute_rots = {}

    for joint_idx in range(NUM_JOINTS):
        # find chain to root
        chain = [joint_idx]
        curr_joint = joint_idx
        while curr_joint != 0:
            for (a, b) in smpl_skeleton:
                if b == curr_joint:
                    curr_joint = a
                    chain.append(a)

        # multiply rotation of child relative to parent by whatever the parent is relative to its parents
        curr_absolute = torch.eye(3)
        for joint_chain_idx in chain:
            curr_absolute = torch.matmul(joint_rot_mats[joint_chain_idx], curr_absolute)
        absolute_rots[joint_idx] = curr_absolute

    return absolute_rots

def generate_global_skeleton_transforms(joint_translations: torch.Tensor, joint_rotations: torch.Tensor):
    # joint translations is a 3xN points cloud
    # joint rotations is a list of torch tensors
    Ts_world_joint = []
    NUM_JOINTS = 22 # 0 = global hip, # 1 - 21 in the get_smpl_joint_names

    for joint_idx in range(NUM_JOINTS):
        T_world_joint = np.eye(4)
        T_world_joint[:3, :3] = joint_rotations[joint_idx]
        T_world_joint[:3, 3] = joint_translations[:, joint_idx].squeeze()
        Ts_world_joint.append(T_world_joint)

    return Ts_world_joint


def check_num_points_in_camera_view(T_world_camera: np.ndarray, y_fov: float, x_fov: float, pointcloud: np.ndarray):
    # T_world_camera: 4x4
    # pointcloud: 3xN
    pointcloud_Nx3 = pointcloud.T

    # get the camera view direction towards the scene
    pos_x_axis = np.expand_dims(T_world_camera[:3, 0], axis=1)
    pos_y_axis = np.expand_dims(T_world_camera[:3, 1], axis=1)
    pos_z_axis = np.expand_dims(T_world_camera[:3, 2], axis=1)
    neg_z_axis = -1.0 * pos_z_axis

   # get the vector going from the camera to the point
    camera_xyz = T_world_camera[:3, 3]
    cam_to_pointcloud = pointcloud_Nx3 - camera_xyz

    # project all vectors onto the plane defined by the YZ plane. check if within Y fov limits
    scale_x = np.dot(cam_to_pointcloud, pos_x_axis).squeeze()
    cam_to_pc_on_yz = cam_to_pointcloud - (scale_x * pos_x_axis).T
    normalized_cam_to_pc_on_yz = cam_to_pc_on_yz.T / np.linalg.norm(cam_to_pc_on_yz, axis=1)
    y_angle = np.arccos(np.dot(normalized_cam_to_pc_on_yz.T, neg_z_axis))

    # project all vectors onto the plane defined by the XZ plane. check if within X fov limits
    scale_y = np.dot(cam_to_pointcloud, pos_y_axis).squeeze()
    cam_to_pc_on_xz = cam_to_pointcloud - (scale_y * pos_y_axis).T
    normalized_cam_to_pc_on_xz = cam_to_pc_on_xz.T / np.linalg.norm(cam_to_pc_on_xz, axis=1)
    x_angle =np.arccos(np.dot(normalized_cam_to_pc_on_xz.T, neg_z_axis))

    # a point is visible if it is both within the x and y fov
    in_fov = np.logical_and(y_angle < y_fov / 2, x_angle < x_fov / 2)
    num_in_fov = np.sum(in_fov)

    return num_in_fov

def render_sequences(cfg, subject: int = None):
    random_sequence = True

    grab_path = cfg.grab_path
    if subject is not None:
        all_seqs = sorted(glob.glob(grab_path + f'/s{subject}/*.npz'))
    else:
        all_seqs = sorted(glob.glob(grab_path + '/*/*.npz'))

    mv = MeshViewer(width=600, height=600,offscreen=True)

    # set the camera pose
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = euler([80, -15, 0], 'xzx')
    camera_pose[:3, 3] = np.array([-.5, -1.4, 1.5])
    mv.update_camera_pose(camera_pose)

    if random_sequence:
        choice = np.random.choice(len(all_seqs), 10, replace=False)
    else:
        choice = range(len(all_seqs))
    for i in tqdm(choice):
        # vis_sequence(cfg, "/home/dxyang/localdata/mpi_grab/grab/s1/airplane_fly_1.npz", mv)
        vis_sequence(cfg,all_seqs[i], mv)
    mv.close_viewer()


def vis_sequence(cfg,sequence, mv):
        draw_meshcat = True
        render_meshes = False
        gen_dset_mode = False
        full_scene = False
        if gen_dset_mode:
            assert render_meshes and not full_scene and not draw_meshcat

        '''
        required setup - load data, load models, create base template models
        '''
        seq_data = parse_npz(sequence)
        n_comps = seq_data['n_comps']
        gender = seq_data['gender']

        T = seq_data.n_frames

        # what is in sequence data?
        # for k,v in seq_data.items():
        #     print(f"{k}: {v}")

        sbj_mesh = os.path.join(grab_path, '..', seq_data.body.vtemp)
        sbj_vtemp = np.array(Mesh(filename=sbj_mesh).vertices)

        sbj_m = smplx.create(model_path=cfg.model_path,
                             model_type='smplx',
                             gender=gender,
                             num_pca_comps=n_comps,
                             v_template=sbj_vtemp,
                             batch_size=T)

        left_mano_default_model = smplx.create(
            model_path=os.path.expanduser("~/localdata/mpi_mano/models"),
            model_type='mano',
            is_rhand= False,
        )
        right_mano_default_model = smplx.create(
            model_path=os.path.expanduser("~/localdata/mpi_mano/models"),
            model_type='mano',
            is_rhand= True,
        )

        mano_smplx_correspondence_file_path = os.path.expanduser("~/localdata/mpi_smplx/MANO_SMPLX_vertex_ids.pkl")
        with open(mano_smplx_correspondence_file_path, 'rb') as f:
            idxs_data = pickle.load(f)
            hand_idxs = np.concatenate(
                [idxs_data['left_hand'], idxs_data['right_hand']]
            )
            left_hand_idxs = idxs_data['left_hand']
            right_hand_idxs = idxs_data['right_hand']

        '''
        global_orient: (T, 3)
        body_pose: (T, 63)
        jaw_pose: (T, 3)
        leye_pose: (T, 3)
        reye_pose: (T, 3)
        left_hand_pose: (T, 24)
        right_hand_pose: (T, 24)
        fullpose: (T, 165)
        expression: (T, 10)
        '''
        # for k, v in seq_data.body.params.items():
        #     print(f"{k}: {v.shape}")
        #     if k != 'global_orient':
        #         seq_data.body.params[k] = np.zeros_like(v)
        sbj_parms = params2torch(seq_data.body.params)

        '''
        ['vertices', 'joints', 'full_pose', 'global_orient', 'transl', 'v_shaped', 'betas', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'expression', 'jaw_pose']
        body pose is 21 joints in axis-angle format
        https://files.is.tue.mpg.de/black/talks/SMPL-made-simple-FAQs.pdf
        full pose in axis-angle: (N, 165)
        global_orient = pose[:, :3]
        body_pose = pose[:, 3:66]
        jaw_pose = pose[:, 66:69]
        leye_pose = pose[:, 69:72]
        reye_pose = pose[:, 72:75]
        left_hand_pose = pose[:, 75:120]
        right_hand_pose = pose[:, 120:]
        '''
        out_sbj = sbj_m(**sbj_parms)

        verts_sbj = to_cpu(out_sbj.vertices)

        if draw_meshcat:
            viz = get_visualizer()
            viz.delete()
            viz1 = get_visualizer(zmq_url="tcp://127.0.0.1:6001")
            viz1.delete()

        obj_mesh = os.path.join(grab_path, '..', seq_data.object.object_mesh)
        obj_mesh = Mesh(filename=obj_mesh)
        obj_vtemp = np.array(obj_mesh.vertices)
        obj_m = ObjectModel(v_template=obj_vtemp,
                            batch_size=T)
        obj_parms = params2torch(seq_data.object.params)
        verts_obj = to_cpu(obj_m(**obj_parms).vertices)

        table_mesh = os.path.join(grab_path, '..', seq_data.table.table_mesh)
        table_mesh = Mesh(filename=table_mesh)
        table_vtemp = np.array(table_mesh.vertices)
        table_m = ObjectModel(v_template=table_vtemp,
                            batch_size=T)
        table_parms = params2torch(seq_data.table.params)
        verts_table = to_cpu(table_m(**table_parms).vertices)

        '''
        output setup depending on what we're doing
        '''
        if gen_dset_mode:
            egoseq_render_path = makepath(sequence.replace('.npz','').replace(cfg.grab_path, cfg.render_path))
            skip_frame = 1

            # h5py setup
            f = h5py.File(f"{egoseq_render_path}/trajectory_data.hdf5", "w")
            f.create_dataset("T_world_camera", shape=(T, 4, 4), dtype=np.float32)
            f.create_dataset("T_camera_lhand", shape=(T, 4, 4), dtype=np.float32)
            f.create_dataset("T_camera_rhand", shape=(T, 4, 4), dtype=np.float32)
            f.create_dataset("T_world_spine1", shape=(T, 4, 4), dtype=np.float32)
            f.create_dataset("T_spine1_lhand", shape=(T, 4, 4), dtype=np.float32)
            f.create_dataset("T_spine1_rhand", shape=(T, 4, 4), dtype=np.float32)
            f.create_dataset("T_alignedSpine1_spine1", shape=(T, 4, 4), dtype=np.float32)
            f.create_dataset("num_pts_visible_lhand", shape=(T), dtype=np.int32)
            f.create_dataset("num_pts_visible_rhand", shape=(T), dtype=np.int32)
            trajectory_dict = {
                "T_world_camera": [],
                "T_camera_lhand": [],
                "T_camera_rhand": [],
                "T_world_spine1": [],
                "T_spine1_lhand": [],
                "T_spine1_rhand": [],
                "T_alignedSpine1_spine1": [],
                "num_pts_visible_lhand": [],
                "num_pts_visible_rhand": [],
            }
        else:
            seq_render_path = makepath(sequence.replace('.npz','').replace(cfg.grab_path, cfg.render_path))
            egoseq_render_path = makepath(f"{seq_render_path}/egocam")
            skip_frame = 1

        '''
        yay let's finally do the frame by frame rendering thing!
        '''
        for frame in tqdm(range(0, T, skip_frame)):
            # take pose of object and subject at frame and set meshes to those poses
            o_mesh = Mesh(vertices=verts_obj[frame], faces=obj_mesh.faces, vc=colors['yellow'])
            o_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['object'][frame] > 0)

            s_mesh = Mesh(vertices=verts_sbj[frame], faces=sbj_m.faces, vc=colors['pink'], smooth=True)
            s_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=seq_data['contact']['body'][frame] > 0)

            s_mesh_wf = Mesh(vertices=verts_sbj[frame], faces=sbj_m.faces, vc=colors['grey'], wireframe=True)
            t_mesh = Mesh(vertices=verts_table[frame], faces=table_mesh.faces, vc=colors['white'])

            '''
            extract things i actually need
            '''
            # get all the joint rotations
            world_frame_rotations = calculate_global_rotations(out_sbj.global_orient[frame], out_sbj.body_pose[frame])
            jnts = to_cpu(out_sbj.joints[frame]).T
            T_world_joints = generate_global_skeleton_transforms(jnts, world_frame_rotations)
            joint_names = get_smpl_joint_names()
            T_world_headJoint = T_world_joints[joint_names.index('head')]

            # get translation pose of eyes
            jnts = to_cpu(out_sbj.joints[frame]).T
            vertices = to_cpu(out_sbj.vertices[frame]).T
            l_eye_idx = JOINT_NAMES.index('left_eye')
            r_eye_idx = JOINT_NAMES.index('right_eye')
            l_eye = np.expand_dims(jnts[:, l_eye_idx], axis=1)
            r_eye = np.expand_dims(jnts[:, r_eye_idx], axis=1)

            # generate pose of egocentric camera
            rotation_joint_egocamera = transforms.euler_angles_to_matrix(torch.Tensor([0, np.pi, 0]), "XYZ")
            T_joint_betweenEyes = np.eye(4)
            T_joint_betweenEyes[:3, :3] = rotation_joint_egocamera
            T_world_betweenEyes = np.matmul(T_world_headJoint, T_joint_betweenEyes)
            T_world_betweenEyes[:3, 3] = ((l_eye + r_eye) / 2.0).squeeze()

            # let's say your wearing a gopro strapped to a headband pointing slightly down. make it happen
            xaxis_rot = np.pi / 6
            T_betweenEyes_mountedCam = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, np.cos(xaxis_rot), np.sin(xaxis_rot), 0.03],
                [0.0, -np.sin(xaxis_rot), np.cos(xaxis_rot), -0.03],
                [0.0, 0.0, 0.0, 1.0],
            ])

            T_world_egoCam = np.matmul(T_world_betweenEyes, T_betweenEyes_mountedCam)


            # generate hand vertices / meshes
            left_hand_vertices = vertices[:, left_hand_idxs]
            right_hand_vertices = vertices[:, right_hand_idxs]
            lh_mesh = Mesh(vertices=left_hand_vertices.T, faces=left_mano_default_model.faces, smooth=True)#vc=colors['purple'], smooth=True)
            rh_mesh = Mesh(vertices=right_hand_vertices.T, faces=right_mano_default_model.faces, smooth=True) #vc=colors['green'], smooth=True)

            # see if the hands will be rendered within the camera fov
            CAMERA_FOV = np.pi / 2

            num_in_fov_lh = check_num_points_in_camera_view(T_world_egoCam, CAMERA_FOV, CAMERA_FOV, left_hand_vertices)
            num_in_fov_rh = check_num_points_in_camera_view(T_world_egoCam, CAMERA_FOV, CAMERA_FOV, right_hand_vertices)

            pct_lh = num_in_fov_lh * 1.0 / left_hand_vertices.shape[0]
            pct_rh = num_in_fov_rh * 1.0 / right_hand_vertices.shape[0]

            # update meshes and generate snapshot
            if render_meshes:
                if full_scene:
                    mv.set_static_meshes([o_mesh, s_mesh, s_mesh_wf, t_mesh, lh_mesh, rh_mesh])
                else:
                    mv.set_static_meshes([lh_mesh, rh_mesh])

                if not gen_dset_mode:
                    camera_pose = np.eye(4)
                    camera_pose[:3, :3] = euler([80, -15, 0], 'xzx')
                    camera_pose[:3, 3] = np.array([-.5, -1.4, 1.5])
                    mv.update_camera_pose(camera_pose)
                    mv.save_snapshot(seq_render_path+'/%04d.png'%frame)

                mv.update_camera_pose(T_world_egoCam)
                mv.save_snapshot(egoseq_render_path+'/%04d.png'%frame)


            # extract wrist wrt spine1 pose (but also realign spine1 such that it follows world frame better)
            T_world_spine1 = T_world_joints[joint_names.index('spine1')]
            T_world_lhand = T_world_joints[joint_names.index('leftHand')]
            T_world_rhand = T_world_joints[joint_names.index('rightHand')]
            T_spine1_world = np.linalg.inv(T_world_spine1)
            T_spine1_lhand = np.matmul(T_spine1_world, T_world_lhand)
            T_spine1_rhand = np.matmul(T_spine1_world, T_world_rhand)

            rot_world_spine1 = T_world_spine1[:3, :3]
            T_alignedSpine1_spine1 = np.eye(4)
            T_alignedSpine1_spine1[:3, :3] = rot_world_spine1

            T_alignedSpine_lhand = np.matmul(T_alignedSpine1_spine1, T_spine1_lhand)
            T_alignedSpine_rhand = np.matmul(T_alignedSpine1_spine1, T_spine1_rhand)

            # extract wrist wrt egocam pose
            T_egoCam_world = np.linalg.inv(T_world_egoCam)
            T_egoCam_lhand = np.matmul(T_egoCam_world, T_world_lhand)
            T_egoCam_rhand = np.matmul(T_egoCam_world, T_world_rhand)

            # save the things!
            if gen_dset_mode:
                trajectory_dict["T_world_camera"].append(np.expand_dims(T_world_egoCam, axis=0))
                trajectory_dict["T_camera_lhand"].append(np.expand_dims(T_egoCam_lhand, axis=0))
                trajectory_dict["T_camera_rhand"].append(np.expand_dims(T_egoCam_rhand, axis=0))
                trajectory_dict["T_world_spine1"].append(np.expand_dims(T_world_spine1, axis=0))
                trajectory_dict["T_spine1_lhand"].append(np.expand_dims(T_spine1_lhand, axis=0))
                trajectory_dict["T_spine1_rhand"].append(np.expand_dims(T_spine1_rhand, axis=0))
                trajectory_dict["T_alignedSpine1_spine1"].append(np.expand_dims(T_alignedSpine1_spine1, axis=0))
                trajectory_dict["num_pts_visible_lhand"].append(num_in_fov_lh)
                trajectory_dict["num_pts_visible_rhand"].append(num_in_fov_rh)

            '''
            for debug
            '''
            if draw_meshcat:
                for idx, T_world_joint in enumerate(T_world_joints):
                    draw_transform(viz, joint_names[idx], T_world_joint, length=0.1)
                draw_transform(viz, "mountedcam", T_world_egoCam, length=0.1, linewidth=10)
                draw_transform(viz, "T_world_betweenEyes", T_world_betweenEyes, length=0.05, linewidth=5)
                red = np.zeros_like(left_hand_vertices)
                red[0] = 1
                blue = np.zeros_like(left_hand_vertices)
                blue[2] = 1
                draw_point_cloud(viz, 'vertices', vertices, np.ones_like(vertices), size=0.001)
                draw_point_cloud(viz, 'l_hand', left_hand_vertices, red, size=0.005)
                draw_point_cloud(viz, 'r_hand', right_hand_vertices, blue, size=0.005)
                draw_point_cloud(viz, "skeleton", jnts[:, :22], np.ones_like(jnts[:, :22]))
                draw_point_cloud(viz, "l_eye", l_eye, np.array([1, 0, 0], dtype=np.float32))
                draw_point_cloud(viz, "r_eye", r_eye, np.array([0, 1, 0], dtype=np.float32))
                camera_pose = np.eye(4)
                camera_pose[:3, :3] = euler([80, -15, 0], 'xzx')
                camera_pose[:3, 3] = np.array([-.5, -1.4, 1.5])
                draw_transform(viz, 'og_camera', camera_pose)

                # draw_transform(viz1, "T_alignedSpine_lhand", T_alignedSpine_lhand, length=0.1, linewidth=10)
                # draw_transform(viz1, "T_alignedSpine_rhand", T_alignedSpine_rhand, length=0.1, linewidth=10)
                draw_transform(viz1, "T_egoCam_lhand", T_egoCam_lhand, length=0.1, linewidth=10)
                draw_transform(viz1, "T_egoCam_rhand", T_egoCam_rhand, length=0.1, linewidth=10)
                draw_transform(viz1, "T_world_lhand", np.matmul(T_world_egoCam, T_egoCam_lhand), length=0.1, linewidth=10)
                draw_transform(viz1, "T_world_rhand", np.matmul(T_world_egoCam, T_egoCam_rhand), length=0.1, linewidth=10)

            # res = input(f"\nframe num: {frame}")
            # if res == 'pdb':
            #     import pdb; pdb.set_trace()

        # end of a trajectory - egress all the data
        if gen_dset_mode:
            # numpy arrays
            trajectory_dict["T_world_camera"] = np.concatenate(trajectory_dict["T_world_camera"])
            trajectory_dict["T_camera_lhand"] = np.concatenate(trajectory_dict["T_camera_lhand"])
            trajectory_dict["T_camera_rhand"] = np.concatenate(trajectory_dict["T_camera_rhand"])
            trajectory_dict["T_world_spine1"] = np.concatenate(trajectory_dict["T_world_spine1"])
            trajectory_dict["T_spine1_lhand"] = np.concatenate(trajectory_dict["T_spine1_lhand"])
            trajectory_dict["T_spine1_rhand"] = np.concatenate(trajectory_dict["T_spine1_rhand"])
            trajectory_dict["T_alignedSpine1_spine1"] = np.concatenate(trajectory_dict["T_alignedSpine1_spine1"])
            trajectory_dict["num_pts_visible_lhand"] = np.array(trajectory_dict["num_pts_visible_lhand"])
            trajectory_dict["num_pts_visible_rhand"] = np.array(trajectory_dict["num_pts_visible_rhand"])

            # egress data
            f["T_world_camera"][:] = trajectory_dict["T_world_camera"]
            f["T_camera_lhand"][:] = trajectory_dict["T_camera_lhand"]
            f["T_camera_rhand"][:] = trajectory_dict["T_camera_rhand"]
            f["T_world_spine1"][:] = trajectory_dict["T_world_spine1"]
            f["T_spine1_lhand"][:] = trajectory_dict["T_spine1_lhand"]
            f["T_spine1_rhand"][:] = trajectory_dict["T_spine1_rhand"]
            f["T_alignedSpine1_spine1"][:] = trajectory_dict["T_alignedSpine1_spine1"]
            f["num_pts_visible_lhand"][:] = trajectory_dict["num_pts_visible_lhand"]
            f["num_pts_visible_rhand"][:] = trajectory_dict["num_pts_visible_rhand"]
            f.flush()
            f.close()
        else:
            input()

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='GRAB-render')

    parser.add_argument('--grab-path', required=True, type=str,
                        help='The path to the downloaded grab data')
    parser.add_argument('--render-path', required=True, type=str,
                        help='The path to the folder to save the renderings')
    parser.add_argument('--model-path', required=True, type=str,
                        help='The path to the folder containing smplx models')
    parser.add_argument('--subject', required=False, type=int,
                        help='Which subject (for generating data in parallel faster')

    args = parser.parse_args()

    grab_path = args.grab_path
    render_path = args.render_path
    model_path = args.model_path

    # grab_path = 'PATH_TO_DOWNLOADED_GRAB_DATA/grab'
    # render_path = 'PATH_TO_THE LOCATION_TO_SAVE_RENDERS'
    # model_path = 'PATH_TO_DOWNLOADED_MODELS_FROM_SMPLX_WEBSITE/'

    cfg = {
        'grab_path': grab_path,
        'model_path': model_path,
        'render_path':render_path
    }

    cfg = Config(**cfg)
    render_sequences(cfg)

