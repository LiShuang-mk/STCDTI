from models import STCDTI
from protst.model import HierarchicalRNN
from protst.util import load_config
import protst.data as protst_data
import time
from utils import set_seed, graph_collate_func, mkdir
from dataloader import DTIDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import argparse
import warnings, os
import pandas as pd
from datetime import datetime
from easydict import EasyDict
import yaml


parser = argparse.ArgumentParser(description="MGNDTI for DTI prediction")
parser.add_argument("-c", "--config", type=str, help="the path to configure file")
parser.add_argument(
    "--data",
    type=str,
    metavar="TASK",
    help="dataset",
    default="sample",
    choices=["Bindingdb", "BioSNAP", "KIBA", "Davis"],
)
parser.add_argument(
    "--model_epoch_name",
    type=str,
    help="pretrained modelEpoch name, "
    + "this will use the value as the name of a subdirectory in the output-dir to distinguish different protein-encoder",
    default=100,
)
parser.add_argument(
    "-p",
    "--protein_encoder_checkpoint",
    type=str,
    help="protein encoder model state-dict file path",
)
parser.add_argument("--seed", type=int, help="seed", default=42)
parser.add_argument("--cuda_id", type=int, help="cuda_id", default=3)
parser.add_argument(
    "--data_path", type=str, help="the path to datasets directory", default="./data"
)

args = parser.parse_args()
device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def main():
    torch.cuda.empty_cache()
    cfg = load_config(args.config)
    warnings.filterwarnings("ignore", category=Warning)
    output_path = os.path.join(
        cfg.RESULT.OUTPUT_DIR, args.data, str(args.model_epoch_name), str(args.seed)
    )
    cfg.RESULT.OUTPUT_DIR = output_path
    mkdir(output_path)
    cfg.SOLVER.SEED = args.seed

    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = os.path.join(args.data_path, args.data) + "/"

    """load data"""
    train_set = pd.read_csv(dataFolder + "train.csv")
    # seenBoth测试与验证
    seenBothTest_set = pd.read_csv(dataFolder + "seenBothTest.csv")
    seenBothVal_set = pd.read_csv(dataFolder + "seenBothVal.csv")
    # unseenDrug测试与验证
    unseenDrugTest_set = pd.read_csv(dataFolder + "unseenDrugTest.csv")
    unseenDrugVal_set = pd.read_csv(dataFolder + "unseenDrugVal.csv")
    # unseenProtein测试与验证
    unseenProteinTest_set = pd.read_csv(dataFolder + "unseenProteinTest.csv")
    unseenProteinVal_set = pd.read_csv(dataFolder + "unseenProteinVal.csv")
    # unseenBoth测试与验证
    unseenBothTest_set = pd.read_csv(dataFolder + "unseenBothTest.csv")
    unseenBothVal_set = pd.read_csv(dataFolder + "unseenBothVal.csv")

    print(f"train_set: {len(train_set)}")

    set_seed(cfg.SOLVER.SEED)

    def encode_protein(seq):
        graph = protst_data.Protein._residue_from_sequence(seq)
        return graph

    train_dataset = DTIDataset(
        train_set.index.values,
        train_set,
        protein_encoder=encode_protein,
    )

    seenBothVal_dataset = DTIDataset(
        seenBothVal_set.index.values,
        seenBothVal_set,
        protein_encoder=encode_protein,
    )
    seenBothTest_dataset = DTIDataset(
        seenBothTest_set.index.values,
        seenBothTest_set,
        protein_encoder=encode_protein,
    )

    unseenDrugVal_dataset = DTIDataset(
        unseenDrugVal_set.index.values,
        unseenDrugVal_set,
        protein_encoder=encode_protein,
    )
    unseenDrugTest_dataset = DTIDataset(
        unseenDrugTest_set.index.values,
        unseenDrugTest_set,
        protein_encoder=encode_protein,
    )

    unseenProteinVal_dataset = DTIDataset(
        unseenProteinVal_set.index.values,
        unseenProteinVal_set,
        protein_encoder=encode_protein,
    )
    unseenProteinTest_dataset = DTIDataset(
        unseenProteinTest_set.index.values,
        unseenProteinTest_set,
        protein_encoder=encode_protein,
    )

    unseenBothVal_dataset = DTIDataset(
        unseenBothVal_set.index.values,
        unseenBothVal_set,
        protein_encoder=encode_protein,
    )
    unseenBothTest_dataset = DTIDataset(
        unseenBothTest_set.index.values,
        unseenBothTest_set,
        protein_encoder=encode_protein,
    )

    params = {
        "batch_size": cfg.SOLVER.BATCH_SIZE,
        "shuffle": True,
        "num_workers": cfg.SOLVER.NUM_WORKERS,
        "drop_last": True,
        "collate_fn": graph_collate_func,
        "pin_memory": True,
    }
    training_generator = DataLoader(train_dataset, **params)
    params["shuffle"] = False
    params["drop_last"] = False

    seenBothVal_generator = DataLoader(seenBothVal_dataset, **params)
    seenBothTest_generator = DataLoader(seenBothTest_dataset, **params)

    unseenDrugVal_generator = DataLoader(unseenDrugVal_dataset, **params)
    unseenDrugTest_generator = DataLoader(unseenDrugTest_dataset, **params)

    unseenProteinVal_generator = DataLoader(unseenProteinVal_dataset, **params)
    unseenProteinTest_generator = DataLoader(unseenProteinTest_dataset, **params)

    unseenBothVal_generator = DataLoader(unseenBothVal_dataset, **params)
    unseenBothTest_generator = DataLoader(unseenBothTest_dataset, **params)

    proteinModel = HierarchicalRNN(cfg.HRNN.PN, cfg.HRNN.QN, cfg.HRNN.OUT_DIM)
    proteinModel.load_state_dict(
        torch.load(
            args.protein_encoder_checkpoint,
            map_location=device,
            weights_only=True,
        )
    )

    # 冻结模型参数
    for param in proteinModel.parameters():
        param.requires_grad = False

    model = STCDTI(proteinModel, **cfg).to(device=device)
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY
    )
    torch.backends.cudnn.benchmark = True

    # trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, **cfg)
    trainer = Trainer(
        model,
        opt,
        device,
        training_generator,
        seenBothVal_generator,
        seenBothTest_generator,
        unseenDrugVal_generator,
        unseenDrugTest_generator,
        unseenProteinVal_generator,
        unseenProteinTest_generator,
        unseenBothVal_generator,
        unseenBothTest_generator,
        **cfg,
    )

    result = trainer.train()
    
    # trainer.best_unseenBothModel = model
    # result = trainer.preTest("unseenBoth")

    with open(os.path.join(output_path, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))

    return result


if __name__ == "__main__":
    print(f"start: {datetime.now()}")
    start_time = time.time()
    """ train """
    result = main()
    """"""
    end_time = time.time()
    total_time_seconds = end_time - start_time
    hours = total_time_seconds // 3600
    minutes = (total_time_seconds % 3600) // 60
    seconds = total_time_seconds % 60
    print(
        "Total running time of the model: {} hours {} minutes {} seconds".format(
            int(hours), int(minutes), int(seconds)
        )
    )
    print(f"end: {datetime.now()}")
