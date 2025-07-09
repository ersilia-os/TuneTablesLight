import os
import math
import argparse
import random
import datetime
import itertools
import json
import subprocess
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from sklearn.utils import check_X_y
from sklearn.base import BaseEstimator

class RandomUnderSampler(BaseEstimator):
    def __init__(self, positive_fraction=0.5, random_state=None):
        self.positive_fraction = positive_fraction
        self.random_state = random_state

    def fit_resample(self, X, y):
        X, y = check_X_y(X, y)
        rng = np.random.RandomState(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise ValueError("Only binary classification supported")
        pos_label = classes[np.argmax(counts)] if counts[np.argmax(counts)] < counts[np.argmin(counts)] else classes[np.argmin(counts)]
        neg_label = classes[1] if pos_label == classes[0] else classes[0]
        idx_pos = np.where(y == pos_label)[0]
        idx_neg = np.where(y == neg_label)[0]
        n_pos = len(idx_pos)
        desired_neg = int(n_pos * (1 - self.positive_fraction) / self.positive_fraction)
        desired_neg = min(len(idx_neg), desired_neg)
        idx_neg_down = rng.choice(idx_neg, size=desired_neg, replace=False)
        idx_new = np.concatenate([idx_pos, idx_neg_down])
        rng.shuffle(idx_new)
        return X[idx_new], y[idx_new]
    
def install_psutil():
    subprocess.run(["pip", "install", "psutil"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

def seed_all(seed=0):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# copied from huggingface
def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1
):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_restarting_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    steps_per_restart,
    num_cycles=0.5,
    last_epoch=-1,
):
    assert num_training_steps % steps_per_restart == 0

    def inner_lr_lambda(current_step, num_warmup_steps, num_training_steps):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    def lr_lambda(current_step):
        inner_step = current_step % steps_per_restart
        return inner_lr_lambda(
            inner_step,
            num_warmup_steps if current_step < steps_per_restart else 0,
            steps_per_restart,
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_openai_lr(transformer_model):
    num_params = sum(p.numel() for p in transformer_model.parameters())
    return 0.003239 - 0.0001395 * math.log(num_params)


def get_weighted_single_eval_pos_sampler(max_len):
    return lambda: random.choices(
        range(max_len), [1 / (max_len - i) for i in range(max_len)]
    )[0]


def get_uniform_single_eval_pos_sampler(max_len, min_len=0):
    return lambda: random.choices(range(min_len, max_len))[0]


def get_fixed_batch_sampler(max_len):
    return lambda: random.choices([max_len])[0]


class SeqBN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)
        self.d_model = d_model

    def forward(self, x):
        assert self.d_model == x.shape[-1]
        flat_x = x.view(-1, self.d_model)
        flat_x = self.bn(flat_x)
        return flat_x.view(*x.shape)


def set_locals_in_self(locals):
    self = locals["self"]
    for var_name, val in locals.items():
        if var_name != "self":
            setattr(self, var_name, val)


default_device = "cuda:0" if torch.cuda.is_available() else "cpu:0"


class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split("=")
            try:
                my_dict[k] = eval(v)
            except NameError:
                my_dict[k] = v
        setattr(namespace, self.dest, my_dict)
        print("dict values: {}".format(my_dict))


def get_nan_value(v, set_value_to_nan=0.0):
    if random.random() < set_value_to_nan:
        return v
    else:
        return random.choice([-999, 0, 1, 999])


def to_ranking(data):
    x = data >= data.unsqueeze(-3)
    x = x.sum(0)
    return x


# TODO: Is there a better way to do this?
#   1. Cmparing to unique elements: When all values are different we still get quadratic blowup
#   2. Argsort(Argsort()) returns ranking, but with duplicate values there is an ordering which is problematic
#   3. Argsort(Argsort(Unique))->Scatter seems a bit complicated, doesn't have quadratic blowup, but how fast?
def to_ranking_low_mem(data):
    x = torch.zeros_like(data)
    for col in range(data.shape[-1]):
        x_ = data[:, :, col] >= data[:, :, col].unsqueeze(-2)
        x_ = x_.sum(0)
        x[:, :, col] = x_
    return x


def nan_handling_missing_for_unknown_reason_value(set_value_to_nan=0.0):
    return get_nan_value(float("nan"), set_value_to_nan)


def nan_handling_missing_for_no_reason_value(set_value_to_nan=0.0):
    return get_nan_value(float("-inf"), set_value_to_nan)


def nan_handling_missing_for_a_reason_value(set_value_to_nan=0.0):
    return get_nan_value(float("inf"), set_value_to_nan)


def torch_masked_mean(x, mask, dim=0, return_share_of_ignored_values=False):
    num = torch.where(mask, torch.full_like(x, 1), torch.full_like(x, 0)).sum(dim=dim)
    value = torch.where(mask, x, torch.full_like(x, 0)).sum(dim=dim)
    if return_share_of_ignored_values:
        return value / num, 1.0 - num / x.shape[dim]
    return value / num


def torch_masked_std(x, mask, dim=0):
    num = torch.where(mask, torch.full_like(x, 1), torch.full_like(x, 0)).sum(dim=dim)
    value = torch.where(mask, x, torch.full_like(x, 0)).sum(dim=dim)
    mean = value / num
    mean_broadcast = torch.repeat_interleave(mean.unsqueeze(dim), x.shape[dim], dim=dim)
    quadratic_difference_from_mean = torch.square(
        torch.where(mask, mean_broadcast - x, torch.full_like(x, 0))
    )
    return torch.sqrt(torch.sum(quadratic_difference_from_mean, dim=dim) / (num - 1))


def torch_nanmean(x, dim=0, return_nanshare=False):
    return torch_masked_mean(
        x, ~torch.isnan(x), dim=dim, return_share_of_ignored_values=return_nanshare
    )


def torch_nanstd(x, dim=0):
    return torch_masked_std(x, ~torch.isnan(x), dim=dim)


def normalize_data(data, normalize_positions=-1):
    if normalize_positions > 0:
        mean = torch_nanmean(data[:normalize_positions], dim=0)
        std = torch_nanstd(data[:normalize_positions], dim=0) + 0.000001
    else:
        mean = torch_nanmean(data, dim=0)
        std = torch_nanstd(data, dim=0) + 0.000001
    data = (data - mean) / std
    data = torch.clip(data, min=-100, max=100)

    return data


def remove_outliers(X, n_sigma=4, normalize_positions=-1):
    # Expects T, B, H
    assert len(X.shape) == 3, "X must be T,B,H"

    data = X if normalize_positions == -1 else X[:normalize_positions]

    data_mean, data_std = torch_nanmean(data, dim=0), torch_nanstd(data, dim=0)
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    mask = (data <= upper) & (data >= lower) & ~torch.isnan(data)
    data_mean, data_std = torch_masked_mean(data, mask), torch_masked_std(data, mask)

    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    X = torch.maximum(-torch.log(1 + torch.abs(X)) + lower, X)
    X = torch.minimum(torch.log(1 + torch.abs(X)) + upper, X)
    return X


def bool_mask_to_att_mask(mask):
    return (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )


def print_on_master_only(is_master):
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_dist(device):
    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
        print("torch.distributed.launch and my rank is", rank)
        torch.cuda.set_device(rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=20),
            world_size=torch.cuda.device_count(),
            rank=rank,
        )
        torch.distributed.barrier()
        print_on_master_only(rank == 0)
        print(
            f"Distributed training on {torch.cuda.device_count()} GPUs, this is rank {rank}, "
            "only I can print, but when using print(..., force=True) it will print on all ranks."
        )
        return True, rank, f"cuda:{rank}"
    elif "SLURM_PROCID" in os.environ and torch.cuda.device_count() > 1:
        assert device != "cpu:0"
        rank = int(os.environ["SLURM_PROCID"])
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        torch.cuda.set_device(rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        print("distributed submitit launch and my rank is", rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=20),
            world_size=torch.cuda.device_count(),
            rank=rank,
        )
        torch.distributed.barrier()
        print_on_master_only(rank == 0)
        print(
            f"Distributed training on {torch.cuda.device_count()} GPUs, this is rank {rank}, "
            "only I can print, but when using print(..., force=True) it will print on all ranks."
        )

        return True, rank, f"cuda:{rank}"
    else:
        print_on_master_only(True)
        return False, 0, device


class NOP:
    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


def check_compatibility(dl):
    if hasattr(dl, "num_outputs"):
        print(
            "`num_outputs` for the DataLoader is deprecated. It is assumed to be 1 from now on."
        )
        assert dl.num_outputs != 1, (
            "We assume num_outputs to be 1. Instead of the num_ouputs change your loss."
            "We specify the number of classes in the CE loss."
        )


def product_dict(dic):
    keys = dic.keys()
    vals = dic.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def normalize_by_used_features_f(
    x, num_features_used, num_features, normalize_with_sqrt=False
):
    if normalize_with_sqrt:
        return x / (num_features_used / num_features) ** (1 / 2)
    return x / (num_features_used / num_features)


class EmbeddingConcatenator:
    def __init__(self, model, method, prefix_weights) -> None:
        self.model = model
        self.original_prefix_size = model.prefix_size
        self.original_embedding = self.model.prefix_embedding.weight.data
        self.original_y_embedding = self.model.prefix_y_embedding
        self.prefix_weights = prefix_weights
        self.prefix_size = None
        self.concatenated_embedding = None
        self.concatenated_y_embedding = None
        self.method = method

    def concat_embedding(self):
        if self.concatenated_embedding is not None:
            return
        # extract embedding parameters
        if self.method == "duplicate":
            self.concatenated_embedding = torch.cat(
                [self.original_embedding, self.original_embedding], dim=0
            ).to(self.model.prefix_embedding.weight.device)
            self.concatenated_y_embedding = torch.cat(
                [self.original_y_embedding, self.original_y_embedding], dim=0
            ).to(self.model.prefix_embedding.weight.device)
            self.prefix_size = self.original_prefix_size * 2
        else:
            raise NotImplementedError("Method {} not implemented!".format(self.method))

    def get_model(self):
        return self.model

    def replace_embedding(self):
        if self.concatenated_embedding is None:
            raise ValueError("Please concat embedding first!")
        self.model.prefix_embedding.weight = nn.Parameter(self.concatenated_embedding)
        self.model.prefix_y_embedding = self.concatenated_y_embedding
        self.model.prefix_size = self.prefix_size

    def restore_embedding(self):
        self.model.prefix_embedding.weight = nn.Parameter(self.original_embedding)
        self.model.prefix_y_embedding = self.original_y_embedding
        self.model.prefix_size = self.original_prefix_size
        self.model.freeze_parameters_except_prefix()


def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False


def make_serializable(config_sample):
    if isinstance(config_sample, torch.Tensor):
        config_sample = "tensor"
    if isinstance(config_sample, dict):
        config_sample = {k: make_serializable(config_sample[k]) for k in config_sample}
    if isinstance(config_sample, list):
        config_sample = [make_serializable(v) for v in config_sample]
    if callable(config_sample):
        config_sample = str(config_sample)
    if not is_json_serializable(config_sample):
        config_sample = str(config_sample)
    return config_sample
