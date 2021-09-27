# EBBE implementation  
import os
import cv2
import argparse
import numpy as np
import math




def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-i', '--inp_path', type=str, default='../COL780-A1-Data/baseline/input', required=False, \
                                                        help="Path for the input images folder")
    parser.add_argument('-o', '--out_path', type=str, default='../COL780-A1-Data/baseline/exp', required=False, \
                                                        help="Path for the predicted masks folder")
    parser.add_argument('-c', '--category', type=str, default='b', required=False, \
                                                        help="Scene category. One of baseline, illumination, jitter, dynamic scenes, ptz (b/i/j/m/p)")
    parser.add_argument('-e', '--eval_frames', type=str, default='../COL780-A1-Data/baseline/eval_frames.txt', required=False, \
                                                        help="Path to the eval_frames.txt file")                      
                                            
    args = parser.parse_args()
    return args


THD = 100
THmatch = 20 


def baseline_bgs(args):

    os.makedirs(args.out_path, exist_ok=True) 
    with open(args.eval_frames) as f:
        eval_frames_lims = f.read().split(" ")
    eval_frames_lims = [int(x) for x in eval_frames_lims]

    back_model = cv2.createBackgroundSubtractorKNN(detectShadows = True)


    #EBBE starting 

    cur_img = cv2.imread(os.path.join(args.inp_path,'in{:06d}.jpg'.format(1)))
    cur_img = cv2.medianBlur(cur_img, 5)
    # print(cur_img)
    # print(cur_img[1,0,:])
    # print(cur_img.shape)
    
    #Background Category color obtained array
    BCCO = np.zeros_like( cur_img , shape = ( cur_img.shape[0] , cur_img.shape[1] ) )

    #To store the set of categoies obtained in each pixel
    S_store = []
    
    #foreground and background pixel colour
    foreground = [255,255,255]
    background = [0,0,0]

    for i in range( 240 ):
        Row = []
        for j in range( 320 ):
            S = []
            S.append(cur_img[i,j,0])
            S.append(cur_img[i,j,1])
            S.append(cur_img[i,j,2])
            S.append(1)
            RC = [S]
            Row.append(RC)
        S_store.append(Row)
    
    #loop over all images
    for fn in range(1, eval_frames_lims[1] + 1):

        cur_img = cv2.imread(os.path.join(args.inp_path,'in{:06d}.jpg'.format(fn)))
        cur_img = cv2.medianBlur(cur_img, 5)

        mask = []
        cnt = 0
        cnt2 = 0
        #algortihm as stated in the paper
        for i in range( 240 ):
            cur_row = []
            for j in range( 320 ):
                R = cur_img[i,j,0]
                B = cur_img[i,j,1]
                G = cur_img[i,j,2]
                Dmin = 1e9
                ind = -1
                for k in range (len(S_store[i][j])) :
                    Dist = (R-S_store[i][j][k][0])**2 + (B-S_store[i][j][k][1])**2 + (G-S_store[i][j][k][2])**2
                    if Dist < Dmin:
                        Dmin = Dist
                        ind = k

                if Dist <= THD * THD:
                    cnt += 1

                    S_store[i][j][k][3] += 1

                    if( R > S_store[i][j][k][0] ):
                        S_store[i][j][k][0] += 1
                    elif ( B < S_store[i][j][k][1]  ):
                        S_store[i][j][k][1] -= 1

                    if( B > S_store[i][j][k][1] ):
                        S_store[i][j][k][1] += 1
                    elif ( B < S_store[i][j][k][1]  ):
                        S_store[i][j][k][1] -= 1
            
                    if( G > S_store[i][j][k][2] ):
                        S_store[i][j][k][2] += 1
                    elif ( G < S_store[i][j][k][2]  ):
                        S_store[i][j][k][2] -= 1

                    J = []
                    Take = [ False ]*len(S_store[i][j])
                    sum = 0
                    for k in range (len(S_store[i][j])) :
                        sum += S_store[i][j][k][3]
                        if( Take[k] ):
                            continue
                        Jd = [k]
                        Take[k] = True
                        for l in range ( len(S_store[i][j])):
                            if( Take[l] ):
                                continue
                            t = False
                            for m in range( len(Jd) ):
                                num = Jd[m]
                                if( (S_store[i][j][l][0]-S_store[i][j][num][0])**2 + (S_store[i][j][l][1]-S_store[i][j][num][1])**2 + (S_store[i][j][l][2]-S_store[i][j][num][2])**2 <= THD * THD ):
                                    t = True
                                    break
                            if t == True :
                                Take[l] = True
                                Jd.append(l)
                        J.append(Jd)
                    
                    P = []
                    Entropy = 0.0
                    for m in range (len(J) ) :
                        Prob = 0
                        for k in range (len(J[m]) ) :
                            Prob = Prob + S_store[i][j][J[m][k]][3]
                        Prob = (Prob*1.0)/sum
                        P.append(Prob)
                        Entropy -= Prob * math.log2(Prob)
                    
                    Relevant_samples = min((int)(2**Entropy),len(J))

                    A = [False] * (len(J))

                    temp = False
                    for m in range(Relevant_samples):
                        index = -1
                        maxi = 0
                        for k in range ( len(J) ):
                            if( A[k] ):
                                continue
                            if( P[k] > maxi ):
                                index = k
                                maxi = P[k]
                        A[index] = True
                        if ind in J[index] :
                            temp = True
                    if temp == True :
                        cur_row.append(background)
                        cnt2 += 1
                    else:
                        cur_row.append(foreground)
                else:
                    S = []
                    S.append(cur_img[i,j,0])
                    S.append(cur_img[i,j,1])
                    S.append(cur_img[i,j,2])
                    S.append(1)
                    S_store[i][j].append(S)
                    cur_row.append(foreground)
            mask.append(cur_row)
        fmask = cur_img
        for i in range (240) :
            for j in range (320) :
                fmask[i,j,0] = mask[i][j][0]
                fmask[i,j,1] = mask[i][j][1]
                fmask[i,j,2] = mask[i][j][2]
        
        imgray = cv2.cvtColor(fmask,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,254,255,cv2.THRESH_BINARY)
        Rcontours, hier_r = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        r_areas = [cv2.contourArea(c) for c in Rcontours]
        CntExternalMask = np.zeros(fmask.shape[:2], dtype="uint8")

        for c in Rcontours:
            if(( cv2.contourArea(c) > 0)):
                cv2.drawContours(CntExternalMask, [c], -1, 1, -1)

        fmask = CntExternalMask

        # kernel = np.ones((5,5), np.uint8)  
        # fmask = cv2.dilate(fmask, kernel, iterations=1)
        # fmask = cv2.erode(fmask, kernel, iterations=1)
        cv2.imwrite(args.out_path+'/gt{:06d}.png'.format(fn), fmask)



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