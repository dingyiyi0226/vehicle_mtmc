import os
import csv

from collections import defaultdict


def split_anno(file, frame_window, anno_folder):
    annos = defaultdict(list)

    dataset = file.split('/')[-3]
    cam = file.split('/')[-2]
    print(f'Split anno on dataset {dataset} cam {cam}')

    with open(file, newline='') as csvfile:
        rows = csv.reader(csvfile)

        for row in rows:
            frame = int(row[0].split('/')[-1].split('-')[1].split('.')[0])
            annos[frame//frame_window].append(row)

    for k, v in annos.items():
        print(k, len(v))

        anno_path = os.path.join(anno_folder, f'{dataset}_{cam}_{k}.txt')
        with open(anno_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in v:
                writer.writerow(i)

def vric_anno(file, anno_folder):

    dataset = file.split('_')[-1].split('.')[0]
    image_folder = f'datasets/VRIC/{dataset}_images'
    anno_path = os.path.join(anno_folder, f'vric_{dataset}_anno.txt')

    data = []

    with open(file, newline='') as f:
        for row in f.readlines():
            image, id, cam = row.split()
            data.append([os.path.join(image_folder, image), id])

    with open(anno_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for d in data:
            writer.writerow(d)

def vric_split_anno(file, anno_folder, ratio):

    with open(file, newline='') as f:
        data = f.readlines()

    size = len(data)

    train = data[:int(size*ratio)]
    val = data[int(size*ratio):]

    train_anno = os.path.join(anno_folder, 'vric_train.txt')
    val_anno = os.path.join(anno_folder, 'vric_val.txt')

    with open(train_anno, 'w', newline='') as f:
        f.writelines(train)

    with open(val_anno, 'w', newline='') as f:
        f.writelines(val)


if __name__ == '__main__':
    # split_anno('datasets/aic/train/S01/c001/anno.txt', 1000, 'reid/vehicle_reid/datasets/annot/')
    # vric_anno('reid/vehicle_reid/datasets/datasets/VRIC/vric_probe.txt', 'reid/vehicle_reid/datasets/annot/')
    vric_split_anno('reid/vehicle_reid/datasets/annot/vric_train_anno.txt', 'reid/vehicle_reid/datasets/annot/', 0.8)
