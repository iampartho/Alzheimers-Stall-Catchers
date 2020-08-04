import tqdm
import os
import cv2
import argparse
import numpy as np


def segfromframe(frame, img_dir):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 195, 205]) 
    upper_orange = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange) 


    im_floodfill = mask
    h, w = mask.shape[:2]
    masked = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, masked, (0,0), 255);
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    im_out = frame.copy()
    im_out[:,:,0] = im_floodfill_inv
    im_out[:,:,1] = im_floodfill_inv
    im_out[:,:,2] = im_floodfill_inv

    im_out = frame & im_out

    cnts = cv2.findContours(im_floodfill_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ROI = im_out[y:y+h, x:x+w]
        ROI = cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)
        ROI = cv2.medianBlur(ROI,5)
        break

    try:
    	return ROI
    except:
    	print("No contour found on the video in named " , img_dir)
    	return frame


def extract_frames(file_path, root_dir): # root_dir is something like "./your_folder_dir/dataset-name_frames"
    cap= cv2.VideoCapture(file_path)
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        folder_name = root_dir+'/' + file_path.split('/')[-1].split('.mp4')[0]
        os.makedirs(folder_name, exist_ok=True)
        img_dir = folder_name+'/'+str(i)+'.jpg'
        frame = segfromframe(frame, img_dir)
        cv2.imwrite(img_dir,frame)
        i+=1

    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", type=str, default="", help="Path to the data folder")
    parser.add_argument("--dataset_name", type=str, default="../../test", help="Name of the dataset")
    opt = parser.parse_args()
    print(opt)

    dataset_name = opt.dataset_name
    dataset_directory = opt.data_directory+dataset_name 
    frame_directory = dataset_directory+'_frames_gray'
    try:
    	all_videos = os.listdir(dataset_directory)
    except:
    	print("There is no directory named ", dataset_directory)
    	exit()
    os.makedirs(frame_directory, exist_ok=True)
    

    for each_video in tqdm.tqdm(all_videos, desc=f"processing total {len(all_videos)} videos"):
    	video_directory = dataset_directory+'/'+each_video
    	extract_frames(video_directory, frame_directory)


