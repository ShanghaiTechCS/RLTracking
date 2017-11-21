# -*- coding: utf-8 -*-
# @Time    : 2017/11/20 下午2:10
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

import numpy as np
import cv2
import sys
from time import time
import kcftracker


class App:
    def __init__(self):
        global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h

        selectingObject = False
        initTracking = False
        onTracking = False
        ix, iy, cx, cy = -1, -1, -1, -1
        w, h = 0, 0

        self.inteval = 1
        self.gt_rects = []

        # window
        self.win_name = 'tracking'

    # mouse callback function
    @staticmethod
    def draw_boundingbox(event, x, y, flags, param):
        global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h

        if event == cv2.EVENT_LBUTTONDOWN:
            selectingObject = True
            onTracking = False
            ix, iy = x, y
            cx, cy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            cx, cy = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            selectingObject = False
            if (abs(x - ix) > 10 and abs(y - iy) > 10):
                w, h = abs(x - ix), abs(y - iy)
                ix, iy = min(x, ix), min(y, iy)
                initTracking = True
            else:
                onTracking = False

        elif event == cv2.EVENT_RBUTTONDOWN:
            onTracking = False
            if (w > 0):
                ix, iy = x - w / 2, y - h / 2
                initTracking = True

    def read_gt(self, gt_path):
        for line in open(gt_path, 'r'):
            x, y, w, h = map(lambda x: float(x), line[:-1].split(','))
            self.gt_rects.append((x, y, w, h))

    def run(self, seq_name):
        global initTracking

        self.cap = cv2.VideoCapture('../../Sequences/%s/imgs/%%8d.jpg' % seq_name)
        self.read_gt('../../Sequences/%s/imgs/groundtruth.txt' % seq_name)
        self.tracker = kcftracker.KCFTracker(True, True, False)

        cv2.namedWindow(self.win_name)
        cv2.setMouseCallback(self.win_name, self.draw_boundingbox)
        self.start_window()

    def start_window(self):
        global initTracking, onTracking
        while (self.cap.isOpened()):
            ret, frame = self.cap.read()
            if not ret:
                break

            if selectingObject:
                cv2.rectangle(frame, (ix, iy), (cx, cy), (0, 255, 255), 1)
            elif initTracking:
                cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), (0, 255, 255), 2)

                self.tracker.init([ix, iy, w, h], frame)

                initTracking = False
                onTracking = True
            elif onTracking:
                t0 = time()
                bbox = self.tracker.update(frame)
                t1 = time()

                bbox = map(int, bbox)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 1)

                duration = t1 - t0
                cv2.putText(frame, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # a = np.concatenate([frame, (255 * np.random.rand(*frame.shape)).astype(np.uint8)], 1)
            cv2.imshow(self.win_name, frame)
            c = cv2.waitKey(self.inteval) & 0xFF
            if c == 27 or c == ord('q'):
                break


if __name__ == '__main__':
    app = App()
    app.run('basketball')
