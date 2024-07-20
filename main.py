import numpy
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list_prevDbs,
                        Dataset_train_prevDbs, Dataset_PrevDbs)
from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)


config = {
    "database_path": "/Users/ashispradhan/Downloads/RnD_infrasonic/FoR_Dataset/for_norm_restructured",
#     "asv_score_path": "ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt",
    "model_path": "./models/weights/AASIST-L.pth",
    "batch_size": 24,
    "num_epochs": 100,
    "loss": "CCE",
    "track": "LA",
    "eval_all_best": "True",
    "eval_output": "eval_scores_cmu_cmu.txt",
    "cudnn_deterministic_toggle": "True",
    "cudnn_benchmark_toggle": "False",
    "model_config": {
        "architecture": "AASIST",
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
        "gat_dims": [24, 32],
        "pool_ratios": [0.4, 0.5, 0.7, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0]
    },
    "optim_config": {
        "optimizer": "adam", 
        "amsgrad": "False",
        "base_lr": 0.0001,
        "lr_min": 0.000005,
        "betas": [0.9, 0.999],
        "weight_decay": 0.0001,
        "scheduler": "cosine"
    }
}





model_config = config["model_config"]
optim_config = config["optim_config"]
optim_config["epochs"] = config["num_epochs"]
# track = config["track"]






if "eval_all_best" not in config:
    config["eval_all_best"] = "True"
if "freq_aug" not in config:
    config["freq_aug"] = "False"






set_seed(1234, config)





output_dir = Path('./exp_result')
# prefix_2019 = "cmu.{}".format(track)
database_path = Path(config["database_path"])
dev_trial_path = Path('./protocols/for_split_dev.txt')
# (database_path /
#                     "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
#                         track, prefix_2019))
# eval_trial_path = Path('/nas/rishith/datasets/smaller_dbs/restructured/protocols/for_test_protocol.txt')
eval_trial_path = Path('./protocols/for_test_protocol.txt')
# (
#     database_path /
#     "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
#         track, prefix_2019))







# define model related paths
model_tag = "{}_{}_ep{}_bs{}".format(
    'CMU',
    'DNN-classifier',
    config["num_epochs"], config["batch_size"])
# if args.comment:
#     model_tag = model_tag + "_{}".format(args.comment)
model_tag = output_dir / model_tag
model_save_path = model_tag / "weights"
eval_score_path = model_tag / config["eval_output"]
writer = SummaryWriter(model_tag)
os.makedirs(model_save_path, exist_ok=True)
# copy(args.config, model_tag / "config.conf")







def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model






# set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: {}".format(device))
# if device == "cpu":
#     raise ValueError("GPU not detected!")

# define model architecture
model = get_model(model_config, device)
model.load_state_dict(
torch.load(config['model_path'], map_location = device))
print("Model loaded : {}".format(config['model_path']))

# # define dataloaders
# trn_loader, dev_loader, eval_loader = get_loader(
#     database_path, 1234, config)






"""Make PyTorch DataLoaders for train / developement / evaluation"""
# track = config["track"]
# prefix_2019 = "ASVspoof2019.{}".format(track)

trn_database_path = database_path / "train"#"ASVspoof2019_{}_train/".format(track)
dev_database_path = database_path / "validation"#"ASVspoof2019_{}_dev/".format(track)
eval_database_path = database_path / "test"#"ASVspoof2019_{}_eval/".format(track)

trn_list_path = Path('./protocols/for_split_train.txt')
# (database_path /
#                  "ASVspoof2019_{}_cm_protocols/{}.cm.train.trn.txt".format(
#                      track, prefix_2019))
dev_trial_path = './protocols/for_split_dev.txt'
# (database_path /
#                   "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
#                       track, prefix_2019))
eval_trial_path = './protocols/for_test_protocol.txt'
# (
#     database_path /
#     "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
#         track, prefix_2019))

d_label_trn, file_train = genSpoof_list_prevDbs(dir_meta=trn_list_path,
                                        is_train=True,
                                        is_eval=False)
print("no. training files:", len(file_train))

train_set = Dataset_train_prevDbs(list_IDs=file_train,
                                       labels=d_label_trn,
                                       base_dir=trn_database_path)
gen = torch.Generator()
gen.manual_seed(1234)
trn_loader = DataLoader(train_set,
                        batch_size=config["batch_size"],
                        shuffle=True,
                        drop_last=True,
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                        generator=gen)

_, file_dev = genSpoof_list_prevDbs(dir_meta=dev_trial_path,
                            is_train=False,
                            is_eval=False)
print("no. validation files:", len(file_dev))

dev_set = Dataset_PrevDbs(list_IDs=file_dev,
                                        base_dir=dev_database_path)
dev_loader = DataLoader(dev_set,
                        batch_size=config["batch_size"],
                        shuffle=False,
                        drop_last=False,
                        pin_memory=True)

file_eval = genSpoof_list_prevDbs(dir_meta=eval_trial_path,
                          is_train=False,
                          is_eval=True)
eval_set = Dataset_PrevDbs(list_IDs=file_eval,
                                         base_dir=eval_database_path)
eval_loader = DataLoader(eval_set,
                         batch_size=config["batch_size"],
                         shuffle=False,
                         drop_last=False,
                         pin_memory=True)

# return trn_loader, dev_loader, eval_loader






# get optimizer and scheduler
optim_config["steps_per_epoch"] = len(trn_loader)
optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
optimizer_swa = SWA(optimizer)

best_dev_eer = 1.
best_eval_eer = 100.
best_dev_tdcf = 0.05
best_eval_tdcf = 1.
n_swa_update = 0  # number of snapshots of model to use in SWA
f_log = open(model_tag / "metric_log.txt", "a")
f_log.write("=" * 5 + "\n")






# make directory for metric logging
metric_path = model_tag / "metrics"
os.makedirs(metric_path, exist_ok=True)






def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    for batch_x, utt_id in data_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _,batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            utt_id, key = trl.strip().split(' ')
            assert fn == utt_id
            fh.write("{} {} {} \n".format(utt_id, key, sco))
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in trn_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _,batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss






from evaluation import compute_eer
def getEER(score_path):
    cm_data = np.genfromtxt(score_path, dtype=str)
    cm_keys = cm_data[:, 1]
    cm_scores = cm_data[:, 2].astype(float)
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']
    eer_cm = compute_eer(bona_cm, spoof_cm)[0]
    return eer_cm * 100






feature, label = next(iter(trn_loader))





# Training
for epoch in range(config["num_epochs"]):
    print("Start training epoch{:03d}".format(epoch))
    running_loss = train_epoch(trn_loader, model, optimizer, device,
                               scheduler, config)
    produce_evaluation_file(dev_loader, model, device,
                            metric_path/"dev_score.txt", dev_trial_path)
    dev_eer = getEER(metric_path/"dev_score.txt")
#     , dev_tdcf = calculate_tDCF_EER(
#         cm_scores_file=metric_path/"dev_score.txt",
#         asv_score_file=database_path/config["asv_score_path"],
#         output_file=metric_path/"dev_t-DCF_EER_{}epo.txt".format(epoch),
#         printout=False)
    print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}".format(
        running_loss, dev_eer))
    writer.add_scalar("loss", running_loss, epoch)
    writer.add_scalar("dev_eer", dev_eer, epoch)
#     writer.add_scalar("dev_tdcf", dev_tdcf, epoch)

#     best_dev_tdcf = min(dev_tdcf, best_dev_tdcf)
    if best_dev_eer >= dev_eer:
        print("best model find at epoch", epoch)
        best_dev_eer = dev_eer
        torch.save(model.state_dict(),
                   model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))

        # do evaluation whenever best model is renewed
        if str_to_bool(config["eval_all_best"]):
            produce_evaluation_file(eval_loader, model, device,
                                    eval_score_path, eval_trial_path)
            dev_eer = getEER(metric_path/"dev_score.txt")
            eval_eer = getEER(eval_score_path)
#             , eval_tdcf = calculate_tDCF_EER(
#                 cm_scores_file=eval_score_path,
#                 asv_score_file=database_path / config["asv_score_path"],
#                 output_file=metric_path /
#                 "t-DCF_EER_{:03d}epo.txt".format(epoch))

            log_text = "epoch{:03d}, ".format(epoch)
            if eval_eer < best_eval_eer:
                log_text += "best eer, {:.4f}%".format(eval_eer)
                best_eval_eer = eval_eer
                torch.save(model.state_dict(),
                           model_save_path / "best.pth")
#             if eval_tdcf < best_eval_tdcf:
#                 log_text += "best tdcf, {:.4f}".format(eval_tdcf)
#                 best_eval_tdcf = eval_tdcf
#                 torch.save(model.state_dict(),
#                            model_save_path / "best.pth")
            if len(log_text) > 0:
                print(log_text)
                f_log.write(log_text + "\n")

        print("Saving epoch {} for swa".format(epoch))
        optimizer_swa.update_swa()
        n_swa_update += 1
    writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
    writer.add_scalar("best_dev_tdcf", best_dev_tdcf, epoch)






print("Start final evaluation")
epoch += 1
if n_swa_update > 0:
    optimizer_swa.swap_swa_sgd()
    optimizer_swa.bn_update(trn_loader, model, device=device)
produce_evaluation_file(eval_loader, model, device, eval_score_path,
                        eval_trial_path)
eval_eer = getEER(eval_score_path)
# , eval_tdcf = calculate_tDCF_EER(cm_scores_file=eval_score_path,
#                                          asv_score_file=database_path /
#                                          config["asv_score_path"],
#                                          output_file=model_tag / "t-DCF_EER.txt")
f_log = open(model_tag / "metric_log.txt", "a")
f_log.write("=" * 5 + "\n")
f_log.write("EER: {:.3f}".format(eval_eer))
f_log.close()

torch.save(model.state_dict(),
           model_save_path / "swa.pth")

if eval_eer <= best_eval_eer:
    best_eval_eer = eval_eer
# if eval_tdcf <= best_eval_tdcf:
#     best_eval_tdcf = eval_tdcf
    torch.save(model.state_dict(),
               model_save_path / "best.pth")
print("Exp FIN. EER: {:.3f}".format(
    best_eval_eer))
