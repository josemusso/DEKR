# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on deep-high-resolution-net.pytorch.
# (https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import csv
import os
import shutil
import time
import sys
sys.path.append("../lib")

import boto3
import cv2
import numpy as np
from PIL import Image
import ffmpeg
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision

import _init_paths
import models

from config import cfg
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.nms import pose_nms
from core.match import match_pose_to_heatmap
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
from utils.transforms import up_interpolate

# angles and exercise eval imports
import modules.dataScienceMediapipe as dataScience
import modules.score as score
import pandas as pd
pd.options.mode.chained_assignment = None
import json

if torch.cuda.is_available():
    print('Using GPU: ' + torch.cuda.get_device_name(0))
    CTX = torch.device('cuda')
else:
    print('Using CPU')
    torch.device('cpu')

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}


CROWDPOSE_KEYPOINT_INDEXES = {
    0: 'left_shoulder',
    1: 'right_shoulder',
    2: 'left_elbow',
    3: 'right_elbow',
    4: 'left_wrist',
    5: 'right_wrist',
    6: 'left_hip',
    7: 'right_hip',
    8: 'left_knee',
    9: 'right_knee',
    10: 'left_ankle',
    11: 'right_ankle',
    12: 'head',
    13: 'neck'
}

def draw_skeleton(image, points):
    skeleton = [
                # # [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                # # [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                # [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                # [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
                [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],  # [3, 5], [4, 6]
                [0, 5], [0, 6]
    ]

    purple = (255,0,255)
    yellow = (0,255,255)

    for i, joint in enumerate(skeleton):
            pt1, pt2 = points[joint]
            image = cv2.line(
                image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])),
                purple, 2)

    return image


def get_pose_estimation_prediction(cfg, model, image, vis_thre, transforms):
    # size at scale 1.0
    base_size, center, scale = get_multi_scale_size(
        image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
    )

    with torch.no_grad():
        heatmap_sum = 0
        poses = []

        for scale in sorted(cfg.TEST.SCALE_FACTOR, reverse=True):
            image_resized, center, scale_resized = resize_align_multi_scale(
                image, cfg.DATASET.INPUT_SIZE, scale, 1.0
            )

            image_resized = transforms(image_resized)
            image_resized = image_resized.unsqueeze(0).cuda()

            heatmap, posemap = get_multi_stage_outputs(
                cfg, model, image_resized, cfg.TEST.FLIP_TEST
            )
            heatmap_sum, poses = aggregate_results(
                cfg, heatmap_sum, poses, heatmap, posemap, scale
            )

        # get heatmap of every frame, select slice depending on keypoint
        selected_keypoint = 14
        heatmap_slice = []
        heatmap_slice = heatmap_sum.cpu().numpy()[0,selected_keypoint]

        heatmap_avg = heatmap_sum/len(cfg.TEST.SCALE_FACTOR)
        poses, scores = pose_nms(cfg, heatmap_avg, poses)

        if len(scores) == 0:
            return []
        else:
            if cfg.TEST.MATCH_HMP:
                poses = match_pose_to_heatmap(cfg, poses, heatmap_avg)

            final_poses = get_final_preds(
                poses, center, scale_resized, base_size
            )

        final_results = []
        for i in range(len(scores)):
            if scores[i] > vis_thre:
                final_results.append(final_poses[i])

        if len(final_results) == 0:
            return [],[]

    return final_results, heatmap_slice


def prepare_output_dirs(prefix='/output/'):
    pose_dir = os.path.join(prefix, "pose")
    if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    os.makedirs(pose_dir, exist_ok=True)
    return pose_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--videoFile', type=str, required=True)
    parser.add_argument('--outputDir', type=str, default='/output/')
    parser.add_argument('--inferenceFps', type=int, default=1)
    parser.add_argument('--visthre', type=float, default=0)
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def main():
    # transformation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)
    pose_dir = prepare_output_dirs(args.outputDir)
    csv_output_rows = []

    # import model architecture
    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    # import weights
    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(
            cfg.TEST.MODEL_FILE), strict=False)
    else:
        raise ValueError('expected model defined in config at TEST.MODEL_FILE')

    pose_model.to(CTX)
    pose_model.eval()

    # Loading an video
    vidcap = cv2.VideoCapture(args.videoFile)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps < args.inferenceFps:
        raise ValueError('Video file not found!')
    skip_frame_cnt = round(fps / args.inferenceFps)
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # adjust dimensions if rotation is needed
    rotate = False
    if rotate:    
        frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # define writers to save videos
    video_dets_name = '{}/{}_pose.mp4'.format(args.outputDir, os.path.splitext(os.path.basename(args.videoFile))[0])
    video_heatmaps_name = '{}/{}_pose_heatmap.mp4'.format(args.outputDir, os.path.splitext(os.path.basename(args.videoFile))[0])
    outcap = cv2.VideoWriter(video_dets_name,
                             cv2.VideoWriter_fourcc(*'MP4V'), int(skip_frame_cnt), (frame_width, frame_height))
    outcap_heatmap = cv2.VideoWriter(video_heatmaps_name,
                             cv2.VideoWriter_fourcc(*'MP4V'), int(skip_frame_cnt), (frame_width, frame_height))

    count = 0
    now_full= time.time()
    data = []
    while vidcap.isOpened():
        total_now = time.time()
        ret, image_bgr = vidcap.read()
        count += 1

        if rotate:    
            image_bgr = cv2.rotate(image_bgr, cv2.cv2.ROTATE_90_CLOCKWISE)
            # image_bgr = cv2.rotate(image_bgr, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # image_bgr = cv2.flip(image_bgr, 1)


        if not ret:
            break

        # if count % skip_frame_cnt != 0:
        #     continue
        print('Processing frame {} out of {}'.format(str(count),str(length)))

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        image_pose = image_rgb.copy()

        # Clone 1 image for debugging purpose
        image_debug = image_bgr.copy()

        now = time.time()

        # added return heatmap_slice
        pose_preds, heatmap_slice = get_pose_estimation_prediction(
            cfg, pose_model, image_pose, args.visthre, transforms=pose_transform)
        then = time.time()

        # save heatmap_slice as image over original image
        # print(heatmap_slice.shape)
        # print(np.max(heatmap_slice))
        # print(np.min(heatmap_slice))
        # plt.imshow(heatmap_slice, cmap='hot', interpolation='nearest')
        # plt.show()
        # plt.savefig(os.path.join(pose_dir, 'heatmap_{:08d}.jpg'.format(count)))

        # generate 3 chann Gray image
        image_gray = np.asarray(cv2.cvtColor(image_debug, cv2.COLOR_BGR2GRAY), np.float32)
        image_gray_3chan=cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

        # case where person is detected
        if pose_preds:
            heatmap_slice_image = (heatmap_slice/np.max(heatmap_slice))*255.0
            heatmap_slice_image = cv2.resize(heatmap_slice_image,(frame_width,frame_height))

            heatmap_slice_image_3chan=np.zeros((frame_height,frame_width,3), np.float32)
            heatmap_slice_image_3chan[:, :, 2] = heatmap_slice_image

            image_w_heatmap = cv2.addWeighted(image_gray_3chan,0.5,heatmap_slice_image_3chan,0.5,0)

            # write heatmap image
            cv2.imwrite(os.path.join(pose_dir, 'heatmap_{:08d}.jpg'.format(count)), image_w_heatmap)
        
            print("Found person pose at {:03.2f} fps".format(1/(then - now)))

            new_csv_row = []
            for coords in pose_preds:
                # Draw each point on image
                for coord in coords:
                    x_coord, y_coord = int(coord[0]), int(coord[1])
                    cv2.circle(image_debug, (x_coord, y_coord), 4, (255, 0, 0), 2)
                    new_csv_row.extend([x_coord, y_coord])
                # draw skeleton
                draw_skeleton(image_debug, coords)
            csv_output_rows.append(new_csv_row)

            #################
            # format detections as Aictive server mediapipe_test for ex. eval.
            #################
            # pose_pred[persona][punto][x:0 o y:1]
            # ver si estan normalizados

            array_x = [
                abs((pose_preds[0][6][0]+pose_preds[0][5][0])/2),       # chest mid (artificial)
                pose_preds[0][0][0],     # nose
                0,                       # 
                pose_preds[0][5][0],     # left_shoulder
                pose_preds[0][7][0],     # left_elbow
                pose_preds[0][9][0],     # left_wrist
                pose_preds[0][11][0],    # left_hip
                pose_preds[0][13][0],    # left_knee
                pose_preds[0][15][0],    # left_ankle
                pose_preds[0][6][0],     # right_shoulder
                pose_preds[0][8][0],     # right_elbow
                pose_preds[0][10][0],    # right_wrist
                pose_preds[0][12][0],    # right_hip
                pose_preds[0][14][0],    # right_knee
                pose_preds[0][16][0],    # right_ankle
                pose_preds[0][2][0],     # right_eye
                pose_preds[0][1][0],     # left_eye
                pose_preds[0][4][0],     # right_ear
                pose_preds[0][3][0],     # left_ear
                0                       # 
                # pose_preds[0][][]     # right_heel        # only in mp
                # pose_preds[0][][]     # right_foot_index  # only in mp
                # pose_preds[0][][]     # left_heel         # only in mp
                # pose_preds[0][][]     # left_foot_index   # only in mp
            ]

            array_y = [
                abs((pose_preds[0][6][1]+pose_preds[0][5][1])/2),       # chest mid (artificial)
                pose_preds[0][0][1],     # nose
                0,                       # 
                pose_preds[0][5][1],     # left_shoulder
                pose_preds[0][7][1],     # left_elbow
                pose_preds[0][9][1],     # left_wrist
                pose_preds[0][11][1],    # left_hip
                pose_preds[0][13][1],    # left_knee
                pose_preds[0][15][1],    # left_ankle
                pose_preds[0][6][1],     # right_shoulder
                pose_preds[0][8][1],     # right_elbow
                pose_preds[0][10][1],    # right_wrist
                pose_preds[0][12][1],    # right_hip
                pose_preds[0][14][1],    # right_knee
                pose_preds[0][16][1],    # right_ankle
                pose_preds[0][2][1],     # right_eye
                pose_preds[0][1][1],     # left_eye
                pose_preds[0][4][1],     # right_ear
                pose_preds[0][3][1],     # left_ear
                0                       # 
                # pose_preds[0][][]     # right_heel        # only in mp
                # pose_preds[0][][]     # right_foot_index  # only in mp
                # pose_preds[0][][]     # left_heel         # only in mp
                # pose_preds[0][][]     # left_foot_index   # only in mp
            ]

            # visibility, NOT AVAILABLE BUT CAN BE INFERRED WITH NOSE AND EARS KPs
            array_v = [     
                0,    # chest mid (artificial)
                0,    # nose
                0,    # 
                0,    # left_shoulder
                0,    # left_elbow
                0,    # left_wrist
                0,    # left_hip
                0,    # left_knee
                0,    # left_ankle
                0,    # right_shoulder
                0,    # right_elbow
                0,    # right_wrist
                0,    # right_hip
                0,    # right_knee
                0,    # right_ankle
                0,    # right_eye
                0,    # left_eye
                0,    # right_ear
                0,    # left_ear
                0    # 
                # pose_preds[0][][]     # right_heel        # only in mp
                # pose_preds[0][][]     # right_foot_index  # only in mp
                # pose_preds[0][][]     # left_heel         # only in mp
                # pose_preds[0][][]     # left_foot_index   # only in mp
            ]

        # case no person detected in frame
        else:
            image_w_heatmap = image_gray_3chan
            cv2.imwrite(os.path.join(pose_dir, 'heatmap_{:08d}.jpg'.format(count)), image_w_heatmap)
            print("No person pose found at {:03.2f} fps".format(1/(then - now)))

            # append empty row on csv
            new_csv_row = []
            csv_output_rows.append(new_csv_row)

            # define detections as empty for ex eval.
            array_x=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            array_y=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            array_v=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        # write fps in image
        total_then = time.time()
        text = "{:03.2f} fps".format(1/(total_then - total_now))
        cv2.putText(image_debug, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        # write detections image
        img_file = os.path.join(pose_dir, 'pose_{:08d}.jpg'.format(count))
        cv2.imwrite(img_file, image_debug)

        # write detections and heatmap video
        outcap.write(np.uint8(image_debug))
        outcap_heatmap.write(np.uint8(image_w_heatmap))

        # after writing both dets and heatmaps videos, calculate angles
        poseEstimate = [array_x, array_y, array_v]
        poseEstimate = np.array(poseEstimate)
        # tag and side are examples by now
        # tag: 
        tag = 'A_T'
        # side: True: der, False: izq
        side = True
        exercise = dataScience.Exercises(tag, poseEstimate, side)
        angles = exercise.calculate()
        print(angles)

        # case angles are detected
        if angles != None:
            teta = []
            for i in range(0, len(angles)):
                teta.append(round(angles[i]['value'], 2))


            # time corresponding to the frame [in secs] (-1 to start at 0:00)
            frame_time = round((count-1)/fps,3)
            
            # frame data contains [time, angles, positions]
            frame_data = [
                str(frame_time), 
                teta, 
                tuple(poseEstimate[0:2, 0]), 
                tuple(poseEstimate[0:2, 1]),
                tuple(poseEstimate[0:2, 2]),
                tuple(poseEstimate[0:2, 3]), 
                tuple(poseEstimate[0:2, 4]), 
                tuple(poseEstimate[0:2, 5]),
                tuple(poseEstimate[0:2, 6]),
                tuple(poseEstimate[0:2, 7]), 
                tuple(poseEstimate[0:2, 8]), 
                tuple(poseEstimate[0:2, 9]),
                tuple(poseEstimate[0:2, 10]),
                tuple(poseEstimate[0:2, 11]), 
                tuple(poseEstimate[0:2, 12]), 
                tuple(poseEstimate[0:2, 13]),
                tuple(poseEstimate[0:2, 14]),
                tuple(poseEstimate[0:2, 15]), 
                tuple(poseEstimate[0:2, 16]), 
                tuple(poseEstimate[0:2, 17]),
                tuple(poseEstimate[0:2, 18])
            ]
            data.append(frame_data)

            if count == 90:
                break
        
    # create df with whole video info to pass to score.Exercises
    fieldnames = ['Second', 'Angle', 'kpt_0', 'kpt_1', 'kpt_2', 'kpt_3', 'kpt_4', 'kpt_5', 
                'kpt_6', 'kpt_7', 'kpt_8', 'kpt_9', 'kpt_10', 'kpt_11', 'kpt_12', 'kpt_13', 
                'kpt_14', 'kpt_15', 'kpt_16', 'kpt_17', 'kpt_18']
    df = pd.DataFrame(data, columns=fieldnames)
    df['Second'] = df['Second'].astype(float)
    # save df for further use
    df.to_csv('output/dataframe.csv')
    # evaluate exercise, get scores
    exercise_sc = score.Exercises(tag, df)
    print(df[['Second', 'Angle']].describe())
    score_result = exercise_sc.calculate()
    # convert to json and fill
    print(json.dumps(score_result, indent=4))
    json_path = 'output/json/'+tag+'.json'
    with open(json_path, 'w') as f:
        json.dump(score_result, f, indent=4)


    then_full= time.time()
    print("Processing complete at average {:03.2f} fps".format(count/(then_full - now_full)))
    # write csv
    csv_headers = ['frame']
    if cfg.DATASET.DATASET_TEST == 'coco':
        for keypoint in COCO_KEYPOINT_INDEXES.values():
            csv_headers.extend([keypoint+'_x', keypoint+'_y'])
    elif cfg.DATASET.DATASET_TEST == 'crowd_pose':
        for keypoint in COCO_KEYPOINT_INDEXES.values():
            csv_headers.extend([keypoint+'_x', keypoint+'_y'])
    else:
        raise ValueError('Please implement keypoint_index for new dataset: %s.' % cfg.DATASET.DATASET_TEST)

    csv_output_filename = os.path.join(args.outputDir, 'pose-data.csv')
    with open(csv_output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csv_headers)
        csvwriter.writerows(csv_output_rows)

    vidcap.release()
    outcap.release()
    outcap_heatmap.release()

    cv2.destroyAllWindows()

    ########## send output files to S3 bucket research-test-s3-bucket
    s3_client = boto3.client('s3')
    # upload dets video
    s3_client.upload_file(video_dets_name, 'research-test-s3-bucket', video_dets_name)
    # upload heatmaps video
    s3_client.upload_file(video_heatmaps_name, 'research-test-s3-bucket', video_heatmaps_name)
    # upload csv
    s3_client.upload_file(csv_output_filename, 'research-test-s3-bucket', csv_output_filename)
    # upload dataframe csv
    csv_dataframe_filename = os.path.join(args.outputDir, 'dataframe.csv')
    s3_client.upload_file(csv_dataframe_filename, 'research-test-s3-bucket', csv_dataframe_filename)
    # upload json
    json_filename = os.path.join(args.outputDir, 'json/'+tag+'.json')
    s3_client.upload_file(json_filename, 'research-test-s3-bucket', json_filename)

    # get download links
    download_link_dets = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': 'research-test-s3-bucket',
                                                            'Key': video_dets_name},
                                                    ExpiresIn=300)

    download_link_heatmaps = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': 'research-test-s3-bucket',
                                                            'Key': video_heatmaps_name},
                                                    ExpiresIn=300)
    download_link_dataframe = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': 'research-test-s3-bucket',
                                                            'Key': csv_dataframe_filename},
                                                    ExpiresIn=300)
    download_link_json = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': 'research-test-s3-bucket',
                                                            'Key': json_filename},
                                                    ExpiresIn=300)

    print('Files uploaded to S3 bucket.')
    print('Download DETECTIONS video:\n {}'.format(download_link_dets))
    print('Download HEATMAPS video:\n {}'.format(download_link_heatmaps))
    print('Download DATAFRAME CSV:\n {}'.format(download_link_dataframe))
    print('Download JSON:\n {}'.format(download_link_json))
    print('Download links expire in 5 min.')

if __name__ == '__main__':
    main()
