import pandas as pd

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
    retrain_data('datasets/aic/train/S01/c001/anno.txt', frame=[60, 100], output_prefix='reid/vehicle_reid/datasets/annot/test')
