import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np
import time

def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)

def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = f"logs/{args['model_name']}/{args['dataset']}/{init_cls}/{args['increment']}"

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = f"logs/{args['model_name']}/{args['dataset']}/{init_cls}/{args['increment']}/{args['prefix']}_{args['seed']}_{args['backbone_type']}"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )

    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks

    # Choose the model based on whether CLS token is enabled
    from backbone.moment_cllora import moment_1_small_cllora as model_loader
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []
    start_time = time.time()

    for task in range(data_manager.nb_tasks):
        logging.info(f"All params: {count_parameters(model._network)}")
        logging.info(f"Trainable params: {count_parameters(model._network, True)}")

        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None:
            logging.info(f"CNN: {cnn_accy['grouped']}")
            logging.info(f"NME: {nme_accy['grouped']}")

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_values = [nme_accy["grouped"][key] for key in nme_keys]
            nme_matrix.append(nme_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])
            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info(f"CNN top1 curve: {cnn_curve['top1']}")
            logging.info(f"CNN top5 curve: {cnn_curve['top5']}")
            logging.info(f"NME top1 curve: {nme_curve['top1']}")
            logging.info(f"NME top5 curve: {nme_curve['top5']}\n")

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))

        else:
            logging.info("No NME accuracy.")
            logging.info(f"CNN: {cnn_accy['grouped']}")

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            cnn_curve["top1"].append(cnn_accy["top1"])

            logging.info(f"CNN top1 curve: {cnn_curve['top1']}")
            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time/60:.2f} min")

def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{device}")
        gpus.append(device)

    args["device"] = gpus

def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_args(args):
    for key, value in args.items():
        logging.info(f"{key}: {value}")
