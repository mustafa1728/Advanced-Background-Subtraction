""" This is the skeleton code for main.py
You need to complete the required functions. You may create addition source files and use them by importing here.
"""

import os
import cv2
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-i', '--inp_path', type=str, default='../COL780-A1-Data/baseline/input', required=False, \
                                                        help="Path for the input images folder")
    parser.add_argument('-o', '--out_path', type=str, default='../COL780-A1-Data/baseline/result', required=False, \
                                                        help="Path for the predicted masks folder")
    parser.add_argument('-c', '--category', type=str, default='b', required=False, \
                                                        help="Scene category. One of baseline, illumination, jitter, dynamic scenes, ptz (b/i/j/m/p)")
    parser.add_argument('-e', '--eval_frames', type=str, default='../COL780-A1-Data/baseline/eval_frames.txt', required=False, \
                                                        help="Path to the eval_frames.txt file")
    parser.add_argument('-g', '--gt_path', type=str, default='../COL780-A1-Data/baseline/groundtruth', required=False, \
                                                        help="Path for the ground truth masks folder")                                    
                                            
    args = parser.parse_args()
    return args


def get_background(prev_frames):
    back = np.median(np.asarray(prev_frames), axis = 0)
    return back

def update_prev_frames(prev_frames, cur_img, i):
    prev_frames[i%k] = cur_img
    return prev_frames

lower_threshold = (40,40,40)
higher_threshold = (60, 60, 60)

def threshold(diff):
    # diff = diff.astype('uint8')
    # mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    mask_confirm = cv2.inRange(diff, higher_threshold, (255,255,255))
    mask_unsure = cv2.inRange(diff, lower_threshold, higher_threshold)

    for iter in range(50):
        for i in range(1, diff.shape[0]-1):
            for j in range(1, diff.shape[1]-1):
                if mask_unsure[i, j]:
                    no_sure_neighbours = 0
                    if mask_confirm[i-1, j]:
                        no_sure_neighbours += 1
                    if mask_confirm[i+1, j]:
                        no_sure_neighbours += 1
                    if mask_confirm[i, j+1]:
                        no_sure_neighbours += 1
                    if mask_confirm[i, j-1]:
                        no_sure_neighbours += 1

                    if mask_confirm[i-1, j-1]:
                        no_sure_neighbours += 1
                    if mask_confirm[i+1, j-1]:
                        no_sure_neighbours += 1
                    if mask_confirm[i+1, j+1]:
                        no_sure_neighbours += 1
                    if mask_confirm[i-1, j+1]:
                        no_sure_neighbours += 1


                    if no_sure_neighbours>=1:
                        mask_confirm[i, j] = 1
                        mask_unsure[i, j] = 0
    return mask_confirm

k = 50

def baseline_bgs(args):

    os.makedirs(args.out_path, exist_ok=True) 
    with open(args.eval_frames) as f:
        eval_frames_lims = f.read().split(" ")
    eval_frames_lims = [int(x) for x in eval_frames_lims]

    #bg_img = cv2.GaussianBlur(bg_img,(5,5),cv2.BORDER_DEFAULT)

    prev_frames = [cv2.imread(os.path.join(args.inp_path,'in{:06d}.jpg'.format(i))) for i in range(eval_frames_lims[0] - k, eval_frames_lims[0])]
    for i in range(eval_frames_lims[0], eval_frames_lims[1] + 1):
        #print( os.path.join(args.inp_path,'in{:06d}.jpg'.format(i)) )
        #cur_img = cv2.GaussianBlur(cur_img,(5,5),cv2.BORDER_DEFAULT)

        cur_img = cv2.imread(os.path.join(args.inp_path,'in{:06d}.jpg'.format(i)))
        bg_img = get_background(prev_frames)

        cur_img = cur_img.astype(float)
        bg_img = bg_img.astype(float)
        diff = cv2.absdiff(bg_img, cur_img)
        mask = threshold(diff)

        Rcontours, hier_r = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        r_areas = [cv2.contourArea(c) for c in Rcontours]
        max_rarea = np.max(r_areas)
        CntExternalMask = np.zeros(mask.shape[:2], dtype="uint8")

        for c in Rcontours:
            if(( cv2.contourArea(c) > 20)):
                cv2.drawContours(CntExternalMask, [c], -1, 1, -1)

        mask = CntExternalMask
        kernel = np.ones((5,5), np.uint8)  
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)

        cv2.imwrite(args.out_path+'/gt{:06d}.png'.format(i), 255*mask)
        prev_frames = update_prev_frames(prev_frames, cur_img, i)
        print(i)


def illumination_bgs(args):
    #TODO complete this function
    pass


def jitter_bgs(args):
    #TODO complete this function
    pass


def dynamic_bgs(args):
    #TODO complete this function
    pass


def ptz_bgs(args):
    #TODO: (Optional) complete this function
    pass


def main(args):
    if args.category not in "bijdp":
        raise ValueError("category should be one of b/i/j/m/p - Found: %s"%args.category)
    FUNCTION_MAPPER = {
            "b": baseline_bgs,
            "i": illumination_bgs,
            "j": jitter_bgs,
            "m": dynamic_bgs,
            "p": ptz_bgs
        }

    FUNCTION_MAPPER[args.category](args)

if __name__ == "__main__":
    args = parse_args()
    main(args)