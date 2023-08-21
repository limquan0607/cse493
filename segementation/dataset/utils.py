import torch
import os
from tqdm.notebook import tqdm
from collections import Counter


def get_class_balancing_weights(training_data):
    img_class_count = Counter(
        [1 if mask.sum() > 0 else 0 for data, mask in tqdm(training_data)]
    )
    pos_weight = img_class_count[0] / sum(img_class_count.values())
    neg_weight = img_class_count[1] / sum(img_class_count.values())
    return neg_weight, pos_weight


def get_weighted_random_sampler(training_data, neg_weight, pos_weight):
    weighted_list = [
        pos_weight if mask.sum() > 0 else neg_weight for (_, mask) in training_data
    ]
    return torch.utils.data.sampler.WeightedRandomSampler(
        weighted_list, len(weighted_list)
    )


def get_data_loader(
    train_dataset, val_dataset, batch_size=16, num_workers=os.cpu_count()
):
    neq, pos = get_class_balancing_weights(train_dataset)
    sampler = get_weighted_random_sampler(train_dataset, neq, pos)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return train_loader, val_loader
