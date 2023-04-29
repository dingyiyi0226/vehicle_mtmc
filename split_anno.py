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


if __name__ == '__main__':
    split_anno('datasets/aic/train/S01/c001/anno.txt', 1000, 'reid/vehicle_reid/datasets/annot/')
