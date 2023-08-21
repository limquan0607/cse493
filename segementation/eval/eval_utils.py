import numpy as np
import torch
from tqdm.notebook import tqdm
from config import DEVICE
from eval.loss import DiceScore
import lightning
import time
import os


def build_eval_array(val_dataset, model: torch.nn.Module):
    preds = []
    labels = []
    dice = 0
    total = 0

    for slice, label in tqdm(val_dataset):
        # print(np.expand_dims(slice, axis=0).shape)
        slice = np.expand_dims(slice, axis=0)
        label = np.expand_dims(label, axis=0)
        slice = torch.tensor(slice).float().to(DEVICE)
        label = torch.tensor(label).float().to(DEVICE)
        # for i in range(label.shape[0]):
        if label.sum() != 0:
            with torch.no_grad():
                logit = model(slice)
                pred = logit
                mask = label
                # print(mask.shape, mask.dtype)
                # print(pred.shape, pred.dtype)
                dice += DiceScore()(pred, mask)
                total += 1
    #     preds.append(pred.cpu().numpy())
    #     labels.append(label)
    return dice / total
    return 1
    # preds = np.array(preds)
    # labels = np.array(labels)
    # return preds, labels


def get_dice_score(val_dataset, model):
    print(
        "------------------------------------------------------------------------------------------------------"
    )
    # preds, labels = build_eval_array(val_dataset, model)
    s = build_eval_array(val_dataset, model)
    # score = DiceScore()(torch.from_numpy(preds), torch.from_numpy(labels))
    # print(f"The Val uDice Score is: {score}")
    print(float(s))
    print(
        "------------------------------------------------------------------------------------------------------"
    )
    return float(s)


def write_experiment_to_file(
    save_dir, data_dir, loss_fn, logger, optimizer, model, dice_score
):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_name = "model_" + model.get_name() + str(timestamp) + ".txt"
    file_path = os.path.join(save_dir, file_name)
    with open(file_path, "w") as file:
        file.write(f"Time: {timestamp}   Model: {model.get_name()}\n")
        file.write(
            "------------------------------------------------------------------\n"
        )
        file.write(f"Using dataset from {data_dir}:\n")
        file.write(
            "------------------------------------------------------------------\n"
        )
        file.write(f"Dice score on validation: {dice_score}\n")
        file.write(
            "------------------------------------------------------------------\n"
        )
        file.write(f"Save at: {logger.log_dir}\n")
        file.write(
            "------------------------------------------------------------------\n"
        )
        file.write(f"Loss function: {str(loss_fn)}\n")
        file.write(
            "------------------------------------------------------------------\n"
        )
        file.write("Optimizer detail:\n")
        file.write(str(optimizer) + "\n")
        file.write(
            "------------------------------------------------------------------\n"
        )
        file.write("Model detail:\n")
        file.write(str(model) + "\n")
        file.write(
            "------------------------------------------------------------------\n"
        )
    print("Write successful at " + str(file_path))
    file.close()


def load_model_from_checkpoint(model: lightning.LightningModule, dir):
    print(
        "------------------------------------------------------------------------------------------------------"
    )
    model.load_from_checkpoint(dir)
    model.eval()
    model.to(DEVICE)
    print("Loading model completed")
    print(
        "------------------------------------------------------------------------------------------------------"
    )
