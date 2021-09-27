""" This is the skeleton code for main.py
You need to complete the required functions. You may create addition source files and use them by importing here.
"""

import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-i', '--inp_path', type=str, default='input', required=True, \
                                                        help="Path for the input images folder")
    parser.add_argument('-o', '--out_path', type=str, default='result', required=True, \
                                                        help="Path for the predicted masks folder")
    parser.add_argument('-c', '--category', type=str, default='b', required=True, \
                                                        help="Scene category. One of baseline, illumination, jitter, dynamic scenes, ptz (b/i/j/m/p)")
    parser.add_argument('-e', '--eval_frames', type=str, default='eval_frames.txt', required=True, \
                                                        help="Path to the eval_frames.txt file")
    args = parser.parse_args()
    return args


verbose = False
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  
normal_kernel = np.ones((5, 5), np.uint8)

def baseline_bgs(args):

    os.makedirs(args.out_path, exist_ok=True) 
    with open(args.eval_frames) as f:
        eval_frames_lims = f.read().split(" ")
    eval_frames_lims = [int(x) for x in eval_frames_lims]

    back_model = cv2.createBackgroundSubtractorKNN(	history = 850 , dist2Threshold = 450.0 , detectShadows = True)

    for i in range(1, eval_frames_lims[1] + 1):
        if verbose:
            print(i)

        cur_img = cv2.imread(os.path.join(args.inp_path,'in{:06d}.jpg'.format(i)))
        cur_img = cv2.medianBlur(cur_img, 5)
        mask = back_model.apply(cur_img)
        
        if i<eval_frames_lims[0]:
            continue

        mask = cv2.dilate(mask, dilate_kernel , iterations=1)
        mask = cv2.erode(mask, erode_kernel , iterations=2)

        Rcontours, _ = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        CntExternalMask = np.zeros(mask.shape[:2], dtype="uint8")
        for c in Rcontours:
            if(( cv2.contourArea(c) > 3 )):
                cv2.drawContours(CntExternalMask, [c], -1, 1, -1)
        mask = 255*CntExternalMask

        cv2.imwrite(args.out_path+'/gt{:06d}.png'.format(i), mask)


def illumination_bgs(args):

    os.makedirs(args.out_path, exist_ok=True) 
    with open(args.eval_frames) as f:
        eval_frames_lims = f.read().split(" ")
    eval_frames_lims = [int(x) for x in eval_frames_lims]

    back_model = cv2.createBackgroundSubtractorKNN(dist2Threshold = 2700, history=45)

    target_dims = (320, 240)
    original_means = []
    hist_means = []

    for i in range(1, eval_frames_lims[1] + 1):
        if verbose:
            print(i)

        cur_img = cv2.imread(os.path.join(args.inp_path,'in{:06d}.jpg'.format(i)))

        ### ============= Main Modification ============= ###
        cur_img_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
        cur_img = cv2.equalizeHist(cur_img_gray)
        ### ============================================= ###

        original_means.append(np.mean(cur_img_gray))
        hist_means.append(np.mean(cur_img))

        mask = back_model.apply(cur_img)

        if i<eval_frames_lims[0]:
            continue

        Rcontours, _ = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        CntExternalMask = np.zeros(mask.shape[:2], dtype="uint8")

        for c in Rcontours:
            if(( cv2.contourArea(c) > 3)):
                cv2.drawContours(CntExternalMask, [c], -1, 1, -1)
        mask = 255*CntExternalMask

        mask = cv2.dilate(mask, normal_kernel, iterations=2)
        mask = cv2.erode(mask, normal_kernel, iterations=1)

        mask = cv2.resize(mask, target_dims)

        cv2.imwrite(args.out_path+'/gt{:06d}.png'.format(i), mask)
        save_hist = False
        if save_hist:
            os.makedirs("../COL780-A1-Data/illumination/hist", exist_ok=True)
            cv2.imwrite("../COL780-A1-Data/illumination/hist"+'/hist{:06d}.png'.format(i), cur_img)

    plt.plot(original_means, label="Original Images")
    plt.plot(hist_means, label="Histogram Equalised")
    plt.xlabel("Image number")
    plt.ylabel("Average intensity of image")
    plt.legend()

def jitter_bgs(args):

    os.makedirs(args.out_path, exist_ok=True) 
    with open(args.eval_frames) as f:
        eval_frames_lims = f.read().split(" ")
    eval_frames_lims = [int(x) for x in eval_frames_lims]

    back_model = cv2.createBackgroundSubtractorKNN(dist2Threshold= 300.00 , history = 1000 , detectShadows = True)

    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 5000
    termination_eps = 1e-10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    count = 0

    for i in range(1, eval_frames_lims[1] + 1):
        if verbose:
            print(i)
        count+=1

        cur_img = cv2.imread(os.path.join(args.inp_path,'in{:06d}.jpg'.format(i)))

        ### ============= Main Modification ============= ###
        if count>10:
            back_img = back_model.getBackgroundImage()
            back_img_gray = cv2.cvtColor(back_img, cv2.COLOR_BGR2GRAY)
        else:
            back_img = cur_img
            back_img_gray = cv2.cvtColor(back_img, cv2.COLOR_BGR2GRAY)
        cur_img_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
        (_, warp_matrix) = cv2.findTransformECC (back_img_gray, cur_img_gray, warp_matrix, warp_mode, criteria, np.ones(back_img_gray.shape).astype("uint8"), gaussFiltSize=3)
        cur_img = cv2.warpAffine(cur_img, warp_matrix, (cur_img.shape[1], cur_img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        ### ============================================= ###

        mask = back_model.apply(cur_img)

        if i<eval_frames_lims[0]:
            continue

        Rcontours, _ = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        CntExternalMask = np.zeros(mask.shape[:2], dtype="uint8")

        for c in Rcontours:
            if(( cv2.contourArea(c) > 100)):
                cv2.drawContours(CntExternalMask, [c], -1, 1, -1)

        mask = 255*CntExternalMask

        mask = cv2.erode(mask, erode_kernel , iterations=1)

        # Applying inverse transformation using the same warp matrix estimated before
        mask = cv2.warpAffine(mask, warp_matrix, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR)

        cv2.imwrite(args.out_path+'/gt{:06d}.png'.format(i), mask)

def dynamic_bgs(args):
    os.makedirs(args.out_path, exist_ok=True) 
    with open(args.eval_frames) as f:
        eval_frames_lims = f.read().split(" ")
    eval_frames_lims = [int(x) for x in eval_frames_lims]


    back_model = cv2.createBackgroundSubtractorKNN(dist2Threshold = 1000,  history=1000)

    for i in range(1, eval_frames_lims[1] + 1):
        if verbose:
            print(i)
        cur_img = cv2.imread(os.path.join(args.inp_path,'in{:06d}.jpg'.format(i)))
        
        ### ============= Main Modification ============= ###
        cur_img = 1.5*cur_img
        cur_img[cur_img > 255] = 255
        cur_img = cur_img.astype("uint8")
        cur_img = cv2.pyrMeanShiftFiltering(cur_img, 15, 30)
        ### ============================================= ###

        mask = back_model.apply(cur_img)

        if i<eval_frames_lims[0]:
            continue

        Rcontours, _ = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        CntExternalMask = np.zeros(mask.shape[:2], dtype="uint8")

        for c in Rcontours:
            if(( cv2.contourArea(c) > 25)):
                cv2.drawContours(CntExternalMask, [c], -1, 1, -1)

        mask = 255*CntExternalMask

        kernel = np.ones((5,5), np.uint8)  
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=2)

        cv2.imwrite(args.out_path+'/gt{:06d}.png'.format(i), mask)

        


def ptz_bgs(args):

    os.makedirs(args.out_path, exist_ok=True) 
    with open(args.eval_frames) as f:
        eval_frames_lims = f.read().split(" ")
    eval_frames_lims = [int(x) for x in eval_frames_lims]

    back_model = cv2.createBackgroundSubtractorKNN(detectShadows = False, history = 100, dist2Threshold = 3100)

    warp_mode1 = cv2.MOTION_EUCLIDEAN
    warp_mode2 = cv2.MOTION_AFFINE
    warp_matrix1 = np.eye(2, 3, dtype=np.float32)
    warp_matrix2 = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 1000
    termination_eps = 1e-6
    first_img_gray = None
    count = 0
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    for i in range(eval_frames_lims[0] - 20, eval_frames_lims[1] + 1):
        if verbose:
            print(i)
        count+=1
        cur_img = cv2.imread(os.path.join(args.inp_path,'in{:06d}.jpg'.format(i)))
        cur_img = cv2.GaussianBlur(cur_img, (9, 9), cv2.BORDER_DEFAULT)
        
        
        
        ### ============= Main Modification ============= ###
        if count>10:
            back_img = back_model.getBackgroundImage()
            back_img_gray = cv2.cvtColor(back_img, cv2.COLOR_BGR2GRAY)
        else:
            back_img = cur_img
            back_img_gray = cv2.cvtColor(back_img, cv2.COLOR_BGR2GRAY)
        
        if first_img_gray is None:
            first_img = cur_img
            first_img_gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
            continue
        cur_img_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
        (_, warp_matrix1) = cv2.findTransformECC (back_img_gray, cur_img_gray, warp_matrix1, warp_mode1, criteria, np.ones(first_img_gray.shape).astype("uint8"), gaussFiltSize=3)
        cur_img = cv2.warpAffine(cur_img, warp_matrix1, (cur_img.shape[1], cur_img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        cur_img_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
        (_, warp_matrix2) = cv2.findTransformECC (back_img_gray, cur_img_gray, warp_matrix2, warp_mode2, criteria, np.ones(first_img_gray.shape).astype("uint8"), gaussFiltSize=3)
        cur_img = cv2.warpAffine(cur_img, warp_matrix2, (cur_img.shape[1], cur_img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        ### ============================================= ###

        mask = back_model.apply(cur_img)

        mask = cv2.dilate(mask, dilate_kernel , iterations=2)
        mask = cv2.erode(mask, erode_kernel , iterations=2)

        if i<eval_frames_lims[0]:
            continue

        Rcontours, _ = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        CntExternalMask = np.zeros(mask.shape[:2], dtype="uint8")

        for c in Rcontours:
            if(( cv2.contourArea(c) > 50) and ( cv2.contourArea(c) < 9000)):
                cv2.drawContours(CntExternalMask, [c], -1, 1, -1)

        mask = 255*CntExternalMask

        # Applying inverse transformation using the same warp matrices estimated before in reverse order
        mask = cv2.warpAffine(mask, warp_matrix2, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, warp_matrix1, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR)

        cv2.imwrite(args.out_path+'/gt{:06d}.png'.format(i), mask)



def main(args):
    if args.category not in "bijmp":
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