from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import cv2
import time
import torch

import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker


class Argument():
    def __init__(self):
        self.video_name = ""
        self.config = ""
        self.snapshot = ""
        self.saveframe = False
        self.draw_moving_path = False
        self.framepath = ""

def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def draw_moving_path(polygon, points, frame, max_trackingpoint = 32):
                
    allpts = polygon.reshape((-1, 1, 2))
    allpts = (np.sum(allpts, axis = 0))/ len(allpts)
    _cx = allpts[0][0]
    _cy = allpts[0][1]
    center_point = (int(_cx), int(_cy))
   
    # First in first out
    if len(points) >= max_trackingpoint:
        points = points[1:]
    if len(points) != 0:
        dist = np.asarray(center_point) - np.asarray(points[-1])
        dist = np.sum(np.power(dist,2))
        # ignore the point which is too close 
        if dist > 32:
            points.append(center_point)
    else:
        points.append(center_point)

    # draw moving path according points 
    for i in range(1,len(points)):
        if points[i-1]is None or points[i]is None:
            continue
        thickness = int(np.sqrt(64 / float(i + 1)) * 1.5)
        cv2.line(frame, points[i - 1], points[i], (255, 255, 255), thickness)

    return points


def ObjectTracking():

    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    
    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)
    
    # parameters init
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

    first_frame = True
    cnt_ = 0
    pre_frame = 0
    pre_rect = 0
    points = []

    # main loop for getting frame and tracking object
    for frame in get_frames(args.video_name):
        if first_frame:
            try:
                # to select object
                init_rect = cv2.selectROI(video_name, frame, False, False)
                pre_rect = init_rect
            except:
                exit()

            # init model
            tracker.init(frame, init_rect)
            first_frame = False

        else:
            cnt_ +=1

            # make prediction
            outputs = tracker.track(frame)

            if outputs['best_score'] > 0.6:

                pre_frame = frame

                if 'polygon' in outputs:

                    polygon = np.array(outputs['polygon']).astype(np.int32)
                    cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                                  True, (0, 255, 0), 3)
                    #make mask
                    mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                    mask = mask.astype(np.uint8)
                    mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                    frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)

                else:
                    bbox = list(map(int, outputs['bbox']))
                    pre_rect = bbox
                    cv2.rectangle(frame, (bbox[0], bbox[1]),
                                  (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                                  (0, 255, 0), 3)
            else:
                # re-init model using previous location
                tracker.init(pre_frame, pre_rect)
 
            # draw moving path
            if args.draw_moving_path and len(polygon) != 0:
                points = draw_moving_path(polygon, points, frame)

            ## save frame as JPEG file
            if args.saveframe and os.path.isdir(args.framepath):
                fullpath = args.framepath + "frame{0:0>3}.jpg".format(cnt_)
                cv2.imwrite(fullpath, frame)     

            cv2.imshow(video_name, frame)
            
            # may need to adjust based on your hardware
            cv2.waitKey(20)

if __name__ == '__main__':

    args = Argument()
    #args.saveframe = True
    #args.draw_moving_path = True

    args.video_name = "demo/overcooked.mp4"
    args.config = "experiments/siammask_r50_l3/config.yaml"
    args.snapshot = "experiments/siammask_r50_l3/model.pth"
    args.framepath = "demo/output/"

    #check your path and file
    #print(os.path.isfile(args.video_name))
    #print(os.path.isfile(args.config))
    #print(os.path.isfile(args.snapshot))
    #print(os.path.isdir(args.framepath))

    torch.set_num_threads(1)
    if torch.cuda.device_count() > 0 :
        print(torch.cuda.get_device_name(0))


    ObjectTracking()

