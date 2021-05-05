"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import math
import cv2
import numpy as np
import csv

from numpy.lib import stride_tricks
import modules.dataScienceMediapipe as dataScience

previous_position = []
theta, phi = math.pi / 4, -math.pi / 6
should_rotate = False
scale_dx = 800
scale_dy = 800


class Plotter3d:
    SKELETON_EDGES = np.array([[11, 10], [10, 9], [9, 0], [0, 3], [3, 4], [4, 5], [0, 6], [6, 7], [7, 8], [0, 12],
                               [12, 13], [13, 14], [0, 1], [1, 15], [15, 16], [1, 17], [17, 18]])

    def __init__(self, canvas_size, origin=(0.5, 0.5), scale=1):
        self.origin = np.array(
            [origin[1] * canvas_size[1], origin[0] * canvas_size[0]], dtype=np.float32)  # x, y
        self.scale = np.float32(scale)
        self.theta = 0
        self.phi = 0
        axis_length = 200
        axes = [
            np.array([[-axis_length/2, -axis_length/2, 0],
                      [axis_length/2, -axis_length/2, 0]], dtype=np.float32),
            np.array([[-axis_length/2, -axis_length/2, 0],
                      [-axis_length/2, axis_length/2, 0]], dtype=np.float32),
            np.array([[-axis_length/2, -axis_length/2, 0], [-axis_length/2, -axis_length/2, axis_length]], dtype=np.float32)]
        step = 20
        for step_id in range(axis_length // step + 1):  # add grid
            axes.append(np.array([[-axis_length / 2, -axis_length / 2 + step_id * step, 0],
                                  [axis_length / 2, -axis_length / 2 + step_id * step, 0]], dtype=np.float32))
            axes.append(np.array([[-axis_length / 2 + step_id * step, -axis_length / 2, 0],
                                  [-axis_length / 2 + step_id * step, axis_length / 2, 0]], dtype=np.float32))
        self.axes = np.array(axes)

    def plot(self, img, vertices, edges):
        global theta, phi
        img.fill(0)
        R = self._get_rotation(theta, phi)
        self._draw_axes(img, R)
        if len(edges) != 0:
            self._plot_edges(img, vertices, edges, R)

    def _draw_axes(self, img, R):
        axes_2d = np.dot(self.axes, R)
        axes_2d = axes_2d * self.scale + self.origin
        for axe in axes_2d:
            axe = axe.astype(int)
            cv2.line(img, tuple(axe[0]), tuple(axe[1]),
                     (128, 128, 128), 1, cv2.LINE_AA)

    def _plot_edges(self, img, vertices, edges, R):
        vertices_2d = np.dot(vertices, R)
        edges_vertices = vertices_2d.reshape(
            (-1, 2))[edges] * self.scale + self.origin
        for edge_vertices in edges_vertices:
            edge_vertices = edge_vertices.astype(int)
            cv2.line(img, tuple(edge_vertices[0]), tuple(
                edge_vertices[1]), (255, 255, 255), 1, cv2.LINE_AA)

    def _get_rotation(self, theta, phi):
        sin, cos = math.sin, math.cos
        return np.array([
            [cos(theta),  sin(theta) * sin(phi)],
            [-sin(theta),  cos(theta) * sin(phi)],
            [0,                       -cos(phi)]
        ], dtype=np.float32)  # transposed

    @staticmethod
    def mouse_callback(event, x, y, flags, params):
        global previous_position, theta, phi, should_rotate, scale_dx, scale_dy
        if event == cv2.EVENT_LBUTTONDOWN:
            previous_position = [x, y]
            should_rotate = True
        if event == cv2.EVENT_MOUSEMOVE and should_rotate:
            theta += (x - previous_position[0]) / scale_dx * 2 * math.pi
            phi -= (y - previous_position[1]) / scale_dy * 2 * math.pi * 2
            phi = max(min(math.pi / 2, phi), -math.pi / 2)
            previous_position = [x, y]
        if event == cv2.EVENT_LBUTTONUP:
            should_rotate = False


body_edges = np.array(
    [[0, 1],  # neck - nose
     [1, 16], [16, 18],  # nose - l_eye - l_ear
     [1, 15], [15, 17],  # nose - r_eye - r_ear
     [0, 3], [3, 4], [4, 5],     # neck - l_shoulder - l_elbow - l_wrist
     [0, 9], [9, 10], [10, 11],  # neck - r_shoulder - r_elbow - r_wrist
     [0, 6], [6, 7], [7, 8],        # neck - l_hip - l_knee - l_ankle
     [0, 12], [12, 13], [13, 14]])  # neck - r_hip - r_knee - r_ankle

# body_edges = np.array(
#     [[0,1]])  # neck - r_hip - r_knee - r_ankle
def detect_side(poses):
    right=0
    left=0
    for pose in poses:
        pose = np.array(pose[0:-1]).reshape((-1, 3)).transpose()
        right_points=(
            pose[2][17]+
            pose[2][9]+
            pose[2][10]+
            pose[2][11]+
            pose[2][12]+
            pose[2][13]+
            pose[2][14]
        )

        left_points=(
            pose[2][18]+
            pose[2][3]+
            pose[2][4]+
            pose[2][5]+
            pose[2][6]+
            pose[2][7]+
            pose[2][8]
        )
        right+=right_points
        left+=left_points
    if(left>right):
        side=False
    else:
        side=True

    return side

def draw_angle(img, teta, width, height, index, message):
    try:
        chart_position = [20+index*180, height-20]
        arrow = 80
        teta = teta * np.pi/180
        xf = int(np.cos(teta)*arrow)+chart_position[0]
        yf = chart_position[1] - int(np.sin(teta)*arrow)

        origin_rect = chart_position[0]-10
        if xf < chart_position[0]:
            chart_position[0] = chart_position[0] - xf + 20 + index*180
            xf = int(np.cos(teta)*arrow)+chart_position[0]
            origin_rect = xf - 10

        cv2.rectangle(img, (origin_rect, chart_position[1]+10),
                      (chart_position[0]+arrow+20, chart_position[1]-arrow-10), (255, 255, 255), -1)
        cv2.arrowedLine(img, (chart_position[0], chart_position[1]), (
            chart_position[0]+arrow, chart_position[1]), (255, 0, 0), 2)
        cv2.arrowedLine(
            img, (chart_position[0], chart_position[1]), (xf, yf), (255, 0, 0), 2)
        cv2.ellipse(img, (chart_position[0], chart_position[1]), (int(
            arrow/2), int(arrow/2)), 0, -teta*180/np.pi-10, 10, (0, 232, 255), 2)
        cv2.putText(img, str(round(teta*180/np.pi, 2)), (chart_position[0]+int(arrow/2), chart_position[1]-int(
            arrow/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, message, (chart_position[0]+5, chart_position[1]-70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    except Exception as e:
        print(e)


def draw_poses(img, poses_2d, width, height, n_frame, file_name, n_original_fps,tag):
    side = detect_side(poses_2d)
    for pose in poses_2d:
        pose = np.array(pose[0:-1]).reshape((-1, 3)).transpose()
        was_found = pose[2] > 0
        for edge in body_edges:
            if was_found[edge[0]] and was_found[edge[1]]:
                cv2.line(img, tuple(pose[0:2, edge[0]].astype(np.int32)), tuple(pose[0:2, edge[1]].astype(np.int32)),
                         (255, 255, 0), 4, cv2.LINE_AA)

        exercise = dataScience.Exercises(tag,pose,side)
        angles = exercise.calculate()
        teta = ''
        if angles != None:
            teta = []
            for i in range(0, len(angles)):
                teta.append(round(angles[i]['value'], 2))
            print(angles)
        with open('/home/kenny/media/data/'+file_name+'.csv', 'a') as csvfile:
            fieldnames = ['second', 'angle','kpt_0','kpt_1','kpt_2','kpt_3','kpt_4','kpt_5','kpt_6','kpt_7','kpt_8','kpt_9','kpt_10','kpt_11','kpt_12','kpt_13','kpt_14','kpt_15','kpt_16','kpt_17','kpt_18']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'second': str(n_frame/n_original_fps), 'angle': teta,'kpt_0':tuple(pose[0:2, 0].astype(np.int32)),'kpt_1':tuple(pose[0:2, 1].astype(np.int32)),'kpt_2':tuple(pose[0:2, 2].astype(np.int32)),'kpt_3':tuple(pose[0:2, 3].astype(np.int32)),'kpt_4':tuple(pose[0:2, 4].astype(np.int32)),'kpt_5':tuple(pose[0:2, 5].astype(np.int32)),'kpt_6':tuple(pose[0:2, 6].astype(np.int32)),'kpt_7':tuple(pose[0:2, 7].astype(np.int32)),'kpt_8':tuple(pose[0:2, 8].astype(np.int32)),'kpt_9':tuple(pose[0:2, 9].astype(np.int32)),'kpt_10':tuple(pose[0:2, 10].astype(np.int32)),'kpt_11':tuple(pose[0:2, 11].astype(np.int32)),'kpt_12':tuple(pose[0:2, 12].astype(np.int32)),'kpt_13':tuple(pose[0:2, 13].astype(np.int32)),'kpt_14':tuple(pose[0:2, 14].astype(np.int32)),'kpt_15':tuple(pose[0:2, 15].astype(np.int32)),'kpt_16':tuple(pose[0:2, 16].astype(np.int32)),'kpt_17':tuple(pose[0:2, 17].astype(np.int32)),'kpt_18':tuple(pose[0:2, 18].astype(np.int32))})

        for kpt_id in range(pose.shape[1]):
            if pose[2, kpt_id] != -1:
                if kpt_id == 100:
                    cv2.circle(img, tuple(pose[0:2, kpt_id].astype(
                        np.int32)), 7, (255, 0, 0), -1, cv2.LINE_AA)
                else:
                    cv2.circle(img, tuple(pose[0:2, kpt_id].astype(
                        np.int32)), 5, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(img, str(kpt_id), tuple(pose[0:2, kpt_id].astype(np.int32)), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 1, cv2.LINE_AA)
                # cv2.putText(img, 'teta-2D: '+str(teta), (50,20), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 1, cv2.LINE_AA)
                if angles != None:
                    for i in range(0,len(angles)):
                        draw_angle(img, round(angles[i]['value'],2), width, height, i, angles[i]['title'])
