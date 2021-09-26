""" This is the skeleton code for main.py
You need to complete the required functions. You may create addition source files and use them by importing here.
"""

import os
import cv2
import argparse
import numpy as np
import timeit


def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-i', '--inp_path', type=str, default='../COL780-A1-Data/ptz/input', required=False, \
                                                        help="Path for the input images folder")
    parser.add_argument('-o', '--out_path', type=str, default='../COL780-A1-Data/ptz/result_fast', required=False, \
                                                        help="Path for the predicted masks folder")
    parser.add_argument('-c', '--category', type=str, default='b', required=False, \
                                                        help="Scene category. One of baseline, illumination, jitter, dynamic scenes, ptz (b/i/j/m/p)")
    parser.add_argument('-e', '--eval_frames', type=str, default='../COL780-A1-Data/ptz/eval_frames.txt', required=False, \
                                                        help="Path to the eval_frames.txt file")
    parser.add_argument('-g', '--gt_path', type=str, default='../COL780-A1-Data/ptz/groundtruth', required=False, \
                                                        help="Path for the ground truth masks folder")                                    
                                            
    args = parser.parse_args()
    return args


def baseline_bgs(args):

    os.makedirs(args.out_path, exist_ok=True) 
    with open(args.eval_frames) as f:
        eval_frames_lims = f.read().split(" ")
    eval_frames_lims = [int(x) for x in eval_frames_lims]

    start = timeit.timeit()

    back_model = cv2.createBackgroundSubtractorKNN(detectShadows = True)

    for i in range(1, eval_frames_lims[1] + 1):
        #print( os.path.join(args.inp_path,'in{:06d}.jpg'.format(i)) )
        #cur_img = cv2.GaussianBlur(cur_img,(5,5),cv2.BORDER_DEFAULT)

        cur_img = cv2.imread(os.path.join(args.inp_path,'in{:06d}.jpg'.format(i)))
        cur_img = cv2.medianBlur(cur_img, 5)
        mask = back_model.apply(cur_img)

        # _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)


        # print(np.max(mask), np.min(mask))
        # print(mask)
        # print(mask.shape)

        if i<eval_frames_lims[0]:
            continue

        Rcontours, hier_r = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        r_areas = [cv2.contourArea(c) for c in Rcontours]
        max_rarea = np.max(r_areas)
        CntExternalMask = np.zeros(mask.shape[:2], dtype="uint8")

        for c in Rcontours:
            if(( cv2.contourArea(c) > 50)):
                cv2.drawContours(CntExternalMask, [c], -1, 1, -1)

        mask = CntExternalMask

        kernel = np.ones((5,5), np.uint8)  
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)

        cv2.imwrite(args.out_path+'/gt{:06d}.png'.format(i), 255*mask)
        print(i)

    end = timeit.timeit()
    print(end - start)


def illumination_bgs(args):
    #TODO complete this function
    pass
    

def align(template, image = None):
    assert image is None or image.shape == template.shape
    pad = 6
    if image is None:
        black_img = np.zeros(template.shape)
        black_img[pad:-pad, pad:-pad, :] = template[pad:-pad, pad:-pad, :]
        return black_img
    min_distance = np.sum(image)**2
    least_diff_img_center = None
    for i in range(2*pad):
        for j in range(2*pad):
            img_center = image[i:-2*pad+i, j:-2*pad+j]
            dist = np.mean((img_center - template[pad:-pad, pad:-pad, :])**2)
            if dist<min_distance:
                least_diff_img_center = img_center
                min_distance = dist
    black_img = np.zeros(image.shape)
    black_img[pad:-pad, pad:-pad, :] = least_diff_img_center
    return black_img

def jitter_bgs(args):
    method = 1
    # method = 2

    os.makedirs(args.out_path, exist_ok=True) 
    with open(args.eval_frames) as f:
        eval_frames_lims = f.read().split(" ")
    eval_frames_lims = [int(x) for x in eval_frames_lims]

    back_model = cv2.createBackgroundSubtractorKNN(detectShadows = True)

    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 5000
    termination_eps = 1e-10
    first_img_gray = None
    first_img = None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    for i in range(1, eval_frames_lims[1] + 1):

        cur_img = cv2.imread(os.path.join(args.inp_path,'in{:06d}.jpg'.format(i)))
        # cur_img = cv2.medianBlur(cur_img, 5)

        
        if method == 1:
            if first_img_gray is None:
                first_img = cur_img
                first_img_gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
                continue
            cur_img_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
            (cc, warp_matrix) = cv2.findTransformECC (first_img_gray, cur_img_gray, warp_matrix, warp_mode, criteria, np.ones(first_img_gray.shape).astype("uint8"), gaussFiltSize=3)
            cur_img = cv2.warpAffine(cur_img, warp_matrix, (cur_img.shape[1], cur_img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        elif method == 2:
            if first_img is None:
                first_img = align(cur_img)
                mask = back_model.apply(cur_img)
                continue
            cur_img = align(first_img, cur_img)

        mask = back_model.apply(cur_img)
        # _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)

        if i<eval_frames_lims[0]:
            continue

        Rcontours, hier_r = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        r_areas = [cv2.contourArea(c) for c in Rcontours]
        max_rarea = np.max(r_areas)
        CntExternalMask = np.zeros(mask.shape[:2], dtype="uint8")

        for c in Rcontours:
            if(( cv2.contourArea(c) > 50)):
                cv2.drawContours(CntExternalMask, [c], -1, 1, -1)

        mask = CntExternalMask

        kernel = np.ones((5,5), np.uint8)  
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        print(i)
        cv2.imwrite(args.out_path+'/gt{:06d}.png'.format(i), 255*mask)
        # cv2.imwrite(args.out_path+'/gt{:06d}.png'.format(i), cur_img)


def dynamic_bgs(args):
    #TODO complete this function
    pass


def ptz_bgs(args):

    os.makedirs(args.out_path, exist_ok=True) 
    with open(args.eval_frames) as f:
        eval_frames_lims = f.read().split(" ")
    eval_frames_lims = [int(x) for x in eval_frames_lims]

    back_model = cv2.createBackgroundSubtractorKNN(detectShadows = True, history = 100)

    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 1000
    termination_eps = 1e-6
    first_img_gray = None
    count = 0
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    for i in range(eval_frames_lims[0] - 20, eval_frames_lims[1] + 1):
        count+=1
        cur_img = cv2.imread(os.path.join(args.inp_path,'in{:06d}.jpg'.format(i)))
        # cur_img = cv2.medianBlur(cur_img, 5)
        cur_img = cv2.GaussianBlur(cur_img, (9, 9), cv2.BORDER_DEFAULT)
        if count>10:
            back_img = back_model.getBackgroundImage()
            back_img_gray = cv2.cvtColor(back_img, cv2.COLOR_BGR2GRAY)
        else:
            back_img = cur_img
            back_img_gray = cv2.cvtColor(back_img, cv2.COLOR_BGR2GRAY)
        # back_img_gray = cv2.cvtColor(back_img, cv2.COLOR_BGR2GRAY)
        #cur_img = cv2.fastNlMeansDenoisingColored(cur_img, None, 10, 10, 7, 21)
        
        if first_img_gray is None:
            first_img = cur_img
            first_img_gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
            continue
        cur_img_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
        (cc, warp_matrix) = cv2.findTransformECC (back_img_gray, cur_img_gray, warp_matrix, warp_mode, criteria, np.ones(first_img_gray.shape).astype("uint8"), gaussFiltSize=3)
        cur_img = cv2.warpAffine(cur_img, warp_matrix, (cur_img.shape[1], cur_img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        # first_img_gray = cur_img_gray

        mask = back_model.apply(cur_img)
        # _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)

        if i<eval_frames_lims[0]:
            continue

        Rcontours, hier_r = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        CntExternalMask = np.zeros(mask.shape[:2], dtype="uint8")

        for c in Rcontours:
            if(( cv2.contourArea(c) > 50) and ( cv2.contourArea(c) < 9000)):
                cv2.drawContours(CntExternalMask, [c], -1, 1, -1)

        mask = 255*CntExternalMask

        kernel = np.ones((5,5), np.uint8)  
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        print(i)
        cv2.imwrite(args.out_path+'/gt{:06d}.png'.format(i), mask)
        # cv2.imwrite(args.out_path+'/gt{:06d}.png'.format(i), cur_img)



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