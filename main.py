""" This is the skeleton code for main.py
You need to complete the required functions. You may create addition source files and use them by importing here.
"""

import os
import cv2
import argparse
import numpy as np

#inp path --  /Users/tamajit/Desktop/IIT DELHI/Semester5/Computer Vision/Assignment1/COL780-A1-Data/baseline/input

#output path -- /Users/tamajit/Desktop/IIT DELHI/Semester5/Computer Vision/Assignment1/COL780-A1-Data/baseline/output

#category -- b

def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-i', '--inp_path', type=str, default='COL780-A1-Data/baseline/input', required=False, \
                                                        help="Path for the input images folder")
    parser.add_argument('-o', '--out_path', type=str, default='COL780-A1-Data/baseline/result', required=False, \
                                                        help="Path for the predicted masks folder")
    parser.add_argument('-c', '--category', type=str, default='b', required=False, \
                                                        help="Scene category. One of baseline, illumination, jitter, dynamic scenes, ptz (b/i/j/m/p)")
    parser.add_argument('-e', '--eval_frames', type=str, default='eval_frames.txt', required=False, \
                                                        help="Path to the eval_frames.txt file")
    parser.add_argument('-g', '--gt_path', type=str, default='COL780-A1-Data/baseline/groundtruth', required=False, \
                                                        help="Path for the ground truth masks folder")                                    
                                            
    args = parser.parse_args()
    return args


def baseline_bgs(args):

    image1 = cv2.imread(os.path.join(args.inp_path,'in000001.jpg'))

    image2 = cv2.imread(os.path.join(args.inp_path,'in000443.jpg'))

    print(image1.shape)
    print(image2.shape)



    gt = cv2.imread(os.path.join(args.gt_path,'gt000001.png'))
    print(gt.shape)

    height = image1.shape[0]

    bg_img = image1
    bg_img[:,160:,:] = image2[:,160:,:] 

    #back_ground = []

    #bg_img = cv2.GaussianBlur(bg_img,(5,5),cv2.BORDER_DEFAULT)

    cv2.imwrite("background_image.png" , bg_img)

    bg_mask = cv2.imread("background_mask.png",0)

    print(bg_mask)


    for i in range(470 , 1701):
        #print( os.path.join(args.inp_path,'in{:06d}.jpg'.format(i)) )
        cur_img = cv2.imread(os.path.join(args.inp_path,'in{:06d}.jpg'.format(i)))

        #cur_img = cv2.GaussianBlur(cur_img,(5,5),cv2.BORDER_DEFAULT)

        diff = cv2.absdiff(bg_img, cur_img )

        mask = cv2.inRange(diff,(35,35,35),(255,255,255))

        bg_mask = cv2.imread("background_mask.png",0)/255


        mask = mask * bg_mask

        cv2.imwrite(args.out_path+'/gt{:06d}.png'.format(i),mask)
    pass


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