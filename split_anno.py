import os
import csv
import random

from collections import defaultdict

import pandas as pd

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

    df = pd.read_csv(file, sep=",")
    df.columns = ["path", "id"]

    random.seed(42)

    train_size = int(0.75 * len(df))
    val_size = len(df) - train_size
    val_idxes = random.sample(range(len(df)), val_size)
    train_idxes = list(set(range(len(df))) - set(val_idxes))

    train_df, val_df = df.loc[train_idxes], df.loc[val_idxes]
    len(train_df), len(val_df)

    train_df.to_csv(os.path.join(anno_folder, 'vric_train.txt'), index=False)
    val_df.to_csv(os.path.join(anno_folder, 'vric_val.txt'), index=False)


def cityflow_split_anno(files, anno_folder, ratio=0.75, name='cityflow'):
    df = None
    for file in files:
        if df is None:
            df = pd.read_csv(file, sep=",")
            df.columns = ["path", "id"]
        else:
            dff = pd.read_csv(file, sep=",")
            dff.columns = ["path", "id"]
            df = pd.concat([df, dff], ignore_index=True)

    random.seed(42)

    train_size = int(ratio * len(df))
    val_size = len(df) - train_size
    val_idxes = random.sample(range(len(df)), val_size)
    train_idxes = list(set(range(len(df))) - set(val_idxes))

    train_df, val_df = df.loc[train_idxes], df.loc[val_idxes]
    print(len(train_df), len(val_df))

    train_df.to_csv(os.path.join(anno_folder, f'{name}_train.txt'), index=False)
    val_df.to_csv(os.path.join(anno_folder, f'{name}_val.txt'), index=False)

def cityflow_split_anno_frame(file, anno_folder, window=[0, 1000], ratio=0.75, name='cityflow'):
    df = pd.read_csv(file, sep=",")
    df.columns = ["path", "id"]
    print(df)

    df = df.loc[window[0]:window[1]]

    window_size = window[1]-window[0]
    train_size = int(ratio*window_size)
    val_size = len(df) - train_size

    random.seed(42)
    val_idxes = random.sample(range(len(df)), val_size)
    train_idxes = list(set(range(len(df))) - set(val_idxes))
    train_df, val_df = df.loc[train_idxes], df.loc[val_idxes]
    print(len(train_df), len(val_df))

    # train_df.to_csv(os.path.join(anno_folder, f'{name}_w_train.txt'), index=False)
    # val_df.to_csv(os.path.join(anno_folder, f'{name}_w_val.txt'), index=False)

def cityflow_gt_anno(file, anno_folder, name='cityflow'):
    df = pd.read_csv(file, sep=",")
    df.columns = ["frame", "id", "left", "top", "width", "height", "1", "-1", "-2", "-3"]
    print(df)

    return

def filter_data(file, frame=None, output_file=None, ratio=0.75):
    """ filter data by frame window
    Args:
    - Frame: [start_frame, end_frame]

    File Format: /path/[id]-[frame].jpg, [id]

    """

    df = pd.read_csv(file, sep=",")
    df.columns = ["path", "id"]
    df["frame"] = df["path"].str.extract(r'-(\d+)\.jpg').astype(int)
    # print(df)

    if frame is None:
        filtered_df = df
    else:
        filtered_df = df[(df["frame"] >= frame[0]) & (df["frame"] <= frame[1])]

    return filtered_df[["path", "id"]]

def split_data(df, ratio=0.75):
    """ split train/val set by ratio """

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate the split index
    split_index = int(len(df) * ratio)

    # Split the DataFrame into training and validation sets
    train_df = df[:split_index]
    val_df = df[split_index:]

    return train_df, val_df

def retrain_data(file, frame=None, ratio=0.75, output_prefix=None):
    """ generate retraining data """

    filtered_df = filter_data(file, frame)
    print('Retrained size:', len(filtered_df))

    train_df, val_df = split_data(filtered_df, ratio)

    if output_prefix is not None:
        train_file = output_prefix + '_train.txt'
        val_file = output_prefix + '_val.txt'
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)

    return train_df, val_df


if __name__ == '__main__':
    # split_anno('datasets/aic/train/S01/c001/anno.txt', 1000, 'reid/vehicle_reid/datasets/annot/')
    # vric_anno('reid/vehicle_reid/datasets/datasets/VRIC/vric_probe.txt', 'reid/vehicle_reid/datasets/annot/')
    # vric_split_anno('reid/vehicle_reid/datasets/annot/vric_train_anno.txt', 'reid/vehicle_reid/datasets/annot/', 0.8)
    # cityflow_split_anno([
    #     'reid/vehicle_reid/datasets/annot/S01_c001.txt',
    #     'reid/vehicle_reid/datasets/annot/S01_c002.txt',
    #     'reid/vehicle_reid/datasets/annot/S01_c003.txt',
    #     'reid/vehicle_reid/datasets/annot/S01_c004.txt',
    #     'reid/vehicle_reid/datasets/annot/S01_c005.txt',
    # ], 'reid/vehicle_reid/datasets/annot/', 0.75, 'cityflow_S01')

    # cityflow_gt_anno('./datasets/aic/train/S01/c001/gt/gt.txt','reid/vehicle_reid/datasets/annot/')
    # cityflow_split_anno_frame('datasets/aic/train/S01/c001/anno.txt', 'reid/vehicle_reid/datasets/annot/', [0, 1000], name='c001_0-1000')
    retrain_data('datasets/aic/train/S01/c001/anno.txt', frame=[60, 100], output_prefix='reid/vehicle_reid/datasets/annot/test')



