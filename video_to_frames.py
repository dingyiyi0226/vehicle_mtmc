import os
import sys
import csv

import cv2 as cv

def video_to_frames(video, img_folder):
    cap = cv.VideoCapture(video)
    os.mkdir(img_folder)

    frame_num = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        path = os.path.join(img_folder, f'{frame_num:04d}.jpg')
        print(path)
        cv.imwrite(path, frame)

        # cv.imshow('Frame', frame)
        # if cv.waitKey(10) == ord('q'):
        #     break
        # time.sleep(1)

        frame_num += 1

def frame_to_vehicles(frame_folder, vehicle_folder, gt_path, anno_path):
    """ Frame number is file name """

    anno_filename = 'anno.csv'
    os.mkdir(vehicle_folder)

    anno_list = []

    gts = load_ground_truth(gt_path)
    for gt in gts:

        print(f'ID: {gt["id"]}, Frame: {gt["frame"]}')

        img_path = os.path.join(frame_folder, f'{gt["frame"]:04d}.jpg')
        img = cv.imread(img_path)

        cropped = img[gt["bbox"][1]:gt["bbox"][1]+gt["bbox"][3], gt["bbox"][0]:gt["bbox"][0]+gt["bbox"][2]]

        # cv.imshow('My Image', cropped)

        path = os.path.join(vehicle_folder, f'{gt["id"]:04d}-{gt["frame"]:04d}.jpg')
        cv.imwrite(path, cropped)

        anno_list.append([path, gt["id"]])

        with open(anno_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for anno in anno_list:
                writer.writerow(anno)


def load_ground_truth(file):

    """ [frame, ID, left, top, width, height, 1, -1, -1, -1] """

    gt = []

    with open(file) as f:
        for line in f.readlines():
            obj = line.split(',')

            gt.append({
                'frame': int(obj[0]),
                'id': int(obj[1]),
                'bbox': (int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5]))
            })

    return gt


if __name__ == '__main__':
    video_to_frames('train/S04/c016/vdo.avi', 'train/S04/c016/frames')
    frame_to_vehicles('train/S04/c016/frames', 'train/S04/c016/objs', 'train/S04/c016/gt/gt.txt', 'train/S04/c016/anno.txt')

    # for i in range(1, 6):
    #     video_to_frames(f'datasets/aic/train/S01/c0{i:02d}/vdo.avi', f'datasets/aic/train/S01/c0{i:02d}/frames')
    #     frame_to_vehicles(f'datasets/aic/train/S01/c0{i:02d}/frames', f'datasets/aic/train/S01/c0{i:02d}/objs', f'datasets/aic/train/S01/c0{i:02d}/gt/gt.txt', f'datasets/aic/train/S01/c0{i:02d}/anno.txt')

    # for i in range(6, 10):
    #     video_to_frames(f'datasets/aic/validation/S02/c0{i:02d}/vdo.avi', f'datasets/aic/validation/S02/c0{i:02d}/frames')
    #     frame_to_vehicles(f'datasets/aic/validation/S02/c0{i:02d}/frames', f'datasets/aic/validation/S02/c0{i:02d}/objs', f'datasets/aic/validation/S02/c0{i:02d}/gt/gt.txt', f'datasets/aic/validation/S02/c0{i:02d}/anno.txt')

    # for i in range(10, 16):
    #     video_to_frames(f'datasets/aic/train/S03/c0{i:02d}/vdo.avi', f'datasets/aic/train/S03/c0{i:02d}/frames')
    #     frame_to_vehicles(f'datasets/aic/train/S03/c0{i:02d}/frames', f'datasets/aic/train/S03/c0{i:02d}/objs', f'datasets/aic/train/S03/c0{i:02d}/gt/gt.txt', f'datasets/aic/train/S03/c0{i:02d}/anno.txt')

    # for i in range(16, 40):
    #     video_to_frames(f'datasets/aic/train/S04/c0{i:02d}/vdo.avi', f'datasets/aic/train/S04/c0{i:02d}/frames')
    #     frame_to_vehicles(f'datasets/aic/train/S04/c0{i:02d}/frames', f'datasets/aic/train/S04/c0{i:02d}/objs', f'datasets/aic/train/S04/c0{i:02d}/gt/gt.txt', f'datasets/aic/train/S04/c0{i:02d}/anno.txt')

    # for i in [10, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 33, 34, 35, 36]:
    #     video_to_frames(f'datasets/aic/validation/S05/c0{i:02d}/vdo.avi', f'datasets/aic/validation/S05/c0{i:02d}/frames')
    #     frame_to_vehicles(f'datasets/aic/validation/S05/c0{i:02d}/frames', f'datasets/aic/validation/S05/c0{i:02d}/objs', f'datasets/aic/validation/S05/c0{i:02d}/gt/gt.txt', f'datasets/aic/validation/S05/c0{i:02d}/anno.txt')


