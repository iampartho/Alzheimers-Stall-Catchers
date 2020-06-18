import tqdm
import os
import cv2
import argparse
import numpy as np
import pandas as pd

no_countour = {"Filename":[],"Frame_number":[]}

# def segfromframe(frame):
#     #ret, frame = cap.read()

#     ##getting area of ROI
#     #frame = cv2.imread(filename)
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     # Threshold of blue in HSV space 
#     lower_orange = np.array([9, 202, 210]) 
#     upper_orange = np.array([14, 255, 255])
#     # preparing the mask to overlay 
#     mask = cv2.inRange(hsv, lower_orange, upper_orange)
#     # The black region in the mask has the value of 0, 
#     # so when multiplied with original image removes all non-blue regions 
#     result = cv2.bitwise_and(frame, frame, mask = mask) 
#     ###floodfilling image
#     im_floodfill = mask
#     # Mask used to flood filling.
#     # Notice the size needs to be 2 pixels than the image.
#     h, w = mask.shape[:2]
#     masked = np.zeros((h+2, w+2), np.uint8)
#     # Floodfill from point (0, 0)
#     cv2.floodFill(im_floodfill, masked, (0,0), 255);
#     # Invert floodfilled image
#     im_floodfill_inv = cv2.bitwise_not(im_floodfill)

#     ###finding contour
#     cnts = cv2.findContours(im_floodfill_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#     cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

#     # Find bounding box and extract ROI
#     for c in cnts:
#         x,y,w,h = cv2.boundingRect(c)
#         ROI = frame[y-3:y+h+3, x-3:x+w+3]
#         ROI = cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)
#         ROI = cv2.medianBlur(ROI,5)
#         ret,ROI = cv2.threshold(ROI,127,255,cv2.THRESH_BINARY)
#         break

#     return ROI

def segfromframe(frame, img_dir):
    #ret, frame = cap.read()

    ##getting area of ROI
    #frame = cv2.imread(filename)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Threshold of blue in HSV space 
    lower_orange = np.array([5, 195, 205]) 
    upper_orange = np.array([20, 255, 255])
    # preparing the mask to overlay 
    mask = cv2.inRange(hsv, lower_orange, upper_orange) 
    # The black region in the mask has the value of 0, 
    # so when multiplied with original image removes all non-blue regions 
    result = cv2.bitwise_and(frame, frame, mask = mask) 

    ###floodfilling image
    im_floodfill = mask
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = mask.shape[:2]
    masked = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, masked, (0,0), 255);
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    im_out = frame.copy()
    im_out[:,:,0] = im_floodfill_inv
    im_out[:,:,1] = im_floodfill_inv
    im_out[:,:,2] = im_floodfill_inv
    # Combine the two images to get the foreground.
    #im_out = frame | im_floodfill_inv_img
    im_out = frame & im_out
    
    
    ###finding contour
    cnts = cv2.findContours(im_floodfill_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Find bounding box and extract ROI
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        #ROI = frame[y-3:y+h+3, x-3:x+w+3]
        ROI = im_out[y:y+h, x:x+w]
        ROI = cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)
        ROI = cv2.medianBlur(ROI,5)
        ret,ROI = cv2.threshold(ROI,127,255,cv2.THRESH_BINARY)
        break

    try:
    	return ROI
    except:
        print("No contour found on the video in named " , img_dir)
        video_name, frame_number = img_dir.split('/')[1:] #nano_frames/318455/50.jpg
        video_name = video_name+'.mp4'
        frame_number = frame_number.split('.jpg')[0]
        frame_number = int(frame_number)
        #global no_countour
        if not video_name in no_countour['Filename']:
            no_countour['Filename'].append(video_name)
            no_countour['Frame_number'].append([frame_number])
        else:
            idx = no_countour['Filename'].index(video_name)
            no_countour['Frame_number'][idx].append(frame_number)
        return frame


def extract_frames(file_path, root_dir): # root_dir is something like "./your_folder_dir/dataset-name_frames"
    # Opens the Video file
    #file_name = '105028.mp4'
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
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", type=str, default="", help="Path to the data folder")
    parser.add_argument("--dataset_name", type=str, default="micro", help="Name of the dataset")
    opt = parser.parse_args()
    print(opt)

    dataset_name = opt.dataset_name
    dataset_directory = opt.data_directory+dataset_name 
    frame_directory = dataset_directory+'_frames'
    try:
    	all_videos = os.listdir(dataset_directory)
    except:
    	print("There is no directory named ", dataset_directory)
    	exit()
    os.makedirs(frame_directory, exist_ok=True)
    

    for each_video in tqdm.tqdm(all_videos, desc=f"processing total {len(all_videos)} videos"):
    	video_directory = dataset_directory+'/'+each_video
    	extract_frames(video_directory, frame_directory)

    df = pd.DataFrame(no_countour)
    df.to_csv(f"{dataset_name}_corrupted_frames.csv", index=False)