import concurrent.futures
import multiprocessing


from reid.vehicle_reid.train_func import train
from tools import log

def retrain_job(cfg, train_data, val_data, epoch, result_queue):

    model_name = cfg.MOT.REID_MODEL_OPTS.split('/')[-2]
    # train_data = "reid/vehicle_reid/datasets/annot/c001_0-1000_w_train.txt"
    # val_data = "reid/vehicle_reid/datasets/annot/c001_0-1000_w_val.txt"

    # REID_MODEL_CKPT: "models/model/model_name/net_{int}.pth"
    last_epoch = int(cfg.MOT.REID_MODEL_CKPT.rsplit('_', 1)[1].split('.')[0])

    config = {
        "name": model_name,
        "data_dir": "reid/vehicle_reid/datasets",
        "train_csv_path": train_data,
        "val_csv_path": val_data,
        "checkpoint": cfg.MOT.REID_MODEL_CKPT,
        "gpu_ids": str(cfg.SYSTEM.GPU_IDS[0]),
        "save_freq": epoch/2,
        "total_epoch": last_epoch + 1 + epoch,
        "start_epoch": last_epoch + 1,

        "batchsize": 16,
        "mixstyle": "True",
        "contrast": "True",
    }

    log.info("Retraining start")
    train(config)
    log.info("Retraining Complete")

    new_cfg = cfg.clone()
    new_cfg.defrost()

    new_cfg.MOT.REID_MODEL_CKPT = cfg.MOT.REID_MODEL_CKPT.rsplit('_', 1)[0] + '_' + str(last_epoch+epoch) + '.pth'
    new_cfg.freeze()

    result_queue.put(new_cfg)

    return True

def retrain(cfg, train_data, val_data, epoch, result_queue):

    process = multiprocessing.Process(target=retrain_job, args=(cfg, train_data, val_data, epoch, result_queue))
    process.start()

    return process

# def retrain(cfg, train_data, val_data, epoch):
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         future = executor.submit(retrain_job, cfg, train_data, val_data, epoch)
#         return future
