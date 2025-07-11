import argparse
import copy
import itertools
import json
import os
import numpy as np
import time
import yaml
import warnings

import torch
import torch.nn.functional as F
import tunetables_light.priors as priors
import tunetables_light.utils as utils
from torch import nn
from torch import autograd
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)

from tunetables_light.utils import (
    get_cosine_schedule_with_warmup,
    get_openai_lr,
    StoreDictKeyPair,
    get_weighted_single_eval_pos_sampler,
    get_uniform_single_eval_pos_sampler,
)
from tunetables_light.priors.real import (
    SummarizeAfter,
    process_data,
    loop_translate,
    TabDS,
    preprocess_input,
    get_train_dataloader,
    get_shuffle_index,
    get_subset_dl,
)
from tunetables_light.losses import kl_divergence
from tunetables_light.transformer import TransformerModel
import tunetables_light.encoders as encoders
import tunetables_light.positional_encodings as positional_encodings
from tunetables_light.utils import init_dist, seed_all, EmbeddingConcatenator
from contextlib import nullcontext
from tqdm.auto import tqdm

torch.backends.cudnn.benchmark = True


class GPULossTracker:
    def __init__(self, device=None):
        self.running_loss = torch.tensor(0.0, device=device)
        self.count = 0

    def update(self, loss: torch.Tensor) -> None:
        self.running_loss += loss.detach()
        self.count += 1

    def average(self) -> float:
        avg = self.running_loss / max(self.count, 1)
        return avg.cpu().item()

    def reset(self) -> None:
        self.running_loss.zero_()
        self.count = 0


def real_data_eval_out(
    r_model,
    cl: int = 1152,
    train_data=None,
    val_dl=None,
    softmax_temperature: torch.Tensor = torch.log(torch.tensor([0.8])),
    return_probs: bool = False,
):
    device = next(r_model.parameters()).device
    r_model.eval()

    td0 = train_data[0][:cl].to(device, non_blocking=True).to(torch.float32)
    td1 = train_data[1][:cl].to(device, non_blocking=True).to(torch.float32)
    single_eval_pos = td0.size(0)
    num_classes_local = len(torch.unique(td1))

    softmax_temperature = softmax_temperature.to(device)

    start_time = time.time()
    prediction_list = []
    target_list = []
    output_list = []

    with torch.inference_mode():
        for data, targets, _ in tqdm(val_dl,                     
                                     total=(len(val_dl)),
                    desc="Inference Batches",
                    ncols=100,
                    colour="magenta",
                    dynamic_ncols=True,
                    bar_format=(
                        "{desc} |" 
                        " {bar:30} |" 
                        " {percentage:3.0f}% " 
                        "[{n_fmt}/{total_fmt} batches] " 
                        "⏱️ {elapsed}<{remaining}"
                    )
        ):
            batch_x = data[0].to(device, non_blocking=True).to(torch.float32)
            batch_y = data[1].to(device, non_blocking=True).to(torch.float32)

            perm = torch.randperm(batch_y.nelement(), device=device)
            batch_y = batch_y.view(-1)[perm].view_as(batch_y)

            inp0 = torch.cat((td0, batch_x), dim=0)
            inp1 = torch.cat((td1, batch_y), dim=0)
            model_input = (inp0, inp1)

            with torch.amp.autocast(device_type="cuda"):
                out = r_model(model_input, single_eval_pos=cl)

            out = out[:, :num_classes_local] / torch.exp(softmax_temperature)
            out = F.softmax(out, dim=-1)

            output_list.append(out)
            _, preds = torch.max(out, 1)
            prediction_list.append(preds.cpu())
            target_list.append(targets)

    outputs = torch.cat(output_list, dim=0).cpu().numpy()
    predictions = torch.cat(prediction_list, dim=0).numpy()
    targets_np = torch.cat(target_list, dim=0).numpy()

    results = {}
    results["Eval_Time"] = float(np.round(time.time() - start_time, 3))
    results["Accuracy"] = float(np.round(accuracy_score(targets_np, predictions), 3))
    try:
        results["Log_Loss"] = float(
            np.round(
                log_loss(targets_np, outputs, labels=np.arange(num_classes_local)), 3
            )
        )
    except Exception:
        results["Log_Loss"] = 0.0
    results["F1_Weighted"] = float(
        np.round(f1_score(targets_np, predictions, average="weighted"), 3)
    )
    results["F1_Macro"] = float(
        np.round(f1_score(targets_np, predictions, average="macro"), 3)
    )
    try:
        if num_classes_local == 2:
            results["ROC_AUC"] = float(
                np.round(roc_auc_score(targets_np, outputs[:, 1]), 3)
            )
        else:
            results["ROC_AUC"] = float(
                np.round(
                    roc_auc_score(
                        targets_np,
                        outputs,
                        multi_class="ovr",
                        labels=np.arange(num_classes_local),
                    ),
                    3,
                )
            )
    except Exception:
        results["ROC_AUC"] = 0.0

    if return_probs:
        return results, outputs, targets_np
    else:
        return results, predictions, targets_np


def train(
    args,
    dataset,
    criterion,
    encoder_generator,
    emsize=200,
    nhid=200,
    nlayers=6,
    nhead=2,
    dropout=0.0,
    epochs=10,
    steps_per_epoch=100,
    batch_size=200,
    bptt=10,
    lr=None,
    weight_decay=0.0,
    warmup_epochs=10,
    input_normalization=False,
    y_encoder_generator=None,
    pos_encoder_generator=None,
    decoder=None,
    extra_prior_kwargs_dict={},
    scheduler=get_cosine_schedule_with_warmup,
    load_weights_from_this_state_dict=None,
    validation_period=10,
    single_eval_pos_gen=None,
    bptt_extra_samples=None,
    gpu_device="cuda:0",
    aggregate_k_gradients=1,
    verbose=False,
    style_encoder_generator=None,
    epoch_callback=None,
    initializer=None,
    initialize_with_model=None,
    train_mixed_precision=False,
    efficient_eval_masking=True,
    boosting=False,
    boosting_lr=1e-3,
    boosting_n_iters=10,
    rand_init_ensemble=False,
    do_concat="",
    is_wrapper=False,
    x_wrapper=None,
    y_wrapper=None,
    **model_extra_args,
):
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.multiprocessing.set_start_method("spawn", force=True)
    device = gpu_device if torch.cuda.is_available() else "cpu:0"
    print(f"Training will be on {device} device")
    using_dist, rank, device = init_dist(device)
    start_time = time.time()

    if not verbose:
        verbose = True
        print(
            "Currently, verbose must be set to True (pass --verbose); this will change in a future release"
        )

    if not os.path.exists(extra_prior_kwargs_dict.get("save_path")):
        try:
            os.makedirs(extra_prior_kwargs_dict.get("save_path"))
        except Exception as e:
            print("Error creating save path: ", e)
            print("Using current directory instead")
            extra_prior_kwargs_dict["save_path"] = os.getcwd()

    max_time = extra_prior_kwargs_dict.get("max_time", 0)
    do_kl_loss = extra_prior_kwargs_dict.get("kl_loss", False)
    n_workers = extra_prior_kwargs_dict.get("num_workers", 1)
    extra_prior_kwargs_dict["do_impute"] = True
    extra_prior_kwargs_dict["ohe"] = False
    linear = extra_prior_kwargs_dict.get("linear", False)

    if extra_prior_kwargs_dict.get("pad_features", None):
        num_features = 100
    else:
        num_features = extra_prior_kwargs_dict.get("num_features", 100)

    if extra_prior_kwargs_dict.get("prior_type") == "real":
        real_prior = True
    else:
        real_prior = False

    if extra_prior_kwargs_dict.get("prompt_tuning"):
        do_prompt_tuning = True
        prefix_size = extra_prior_kwargs_dict.get("tuned_prompt_size", 100)
    else:
        do_prompt_tuning = False
        prefix_size = 0

    single_eval_pos_gen = (
        single_eval_pos_gen
        if callable(single_eval_pos_gen)
        else lambda: single_eval_pos_gen
    )
    real_data_qty = extra_prior_kwargs_dict.get("real_data_qty", 0)
    if real_data_qty <= 0:
        real_data_qty = bptt

    def make_datasets(
        extra_prior_kwargs_dict,
        do_permute=True,
        bptt=0,
        steps_per_epoch=None,
        is_wrapper=False,
    ):
        args.summerize_after_prep = extra_prior_kwargs_dict.get(
            "summerize_after_prep", "False"
        )
        args.preprocess_type = extra_prior_kwargs_dict.get("preprocess_type", "none")
        args.rand_seed = extra_prior_kwargs_dict.get("rand_seed", 0)

        if is_wrapper:
            train_index = dataset.split_indeces[0]
            val_index = dataset.split_indeces[1]
            test_index = dataset.split_indeces[1]
        else:
            for i, split_dictionary in enumerate(dataset.split_indeces):
                if i != extra_prior_kwargs_dict.get("split"):
                    continue
                train_index = split_dictionary["train"]
                val_index = split_dictionary["val"]
                test_index = split_dictionary["test"]

        if True:
            processed_data = process_data(
                dataset,
                train_index,
                val_index,
                test_index,
                verbose=extra_prior_kwargs_dict.get("verbose"),
                scaler="None",
                one_hot_encode=extra_prior_kwargs_dict.get("ohe", True),
                impute=extra_prior_kwargs_dict.get("do_impute", True),
                args=args,
            )
            X_train, y_train = processed_data["data_train"]
            X_val, y_val = processed_data["data_val"]
            X_test, y_test = processed_data["data_test"]

            if is_wrapper:
                extra_prior_kwargs_dict["shuffle_index"] = {
                    "train": np.arange(0, len(X_train)),
                    "val": np.arange(0, len(X_val)),
                    "test": np.arange(0, len(X_test)),
                }

            if extra_prior_kwargs_dict.get("shuffle_index", None) == None:
                extra_prior_kwargs_dict["shuffle_index"] = {
                    "train": get_shuffle_index(X_train),
                    "val": get_shuffle_index(X_val),
                    "test": get_shuffle_index(X_test),
                }

            X_train = X_train[extra_prior_kwargs_dict["shuffle_index"]["train"]]
            y_train = y_train[extra_prior_kwargs_dict["shuffle_index"]["train"]]
            X_val = X_val[extra_prior_kwargs_dict["shuffle_index"]["val"]]
            y_val = y_val[extra_prior_kwargs_dict["shuffle_index"]["val"]]
            X_test = X_test[extra_prior_kwargs_dict["shuffle_index"]["test"]]
            y_test = y_test[extra_prior_kwargs_dict["shuffle_index"]["test"]]

            n_features = X_train.shape[1]
            n_samples = X_train.shape[0]
            num_classes = len(set(y_train))
            steps_per_epoch = len(X_train) // bptt

            if bptt > n_samples:
                if verbose:
                    print(
                        f"WARNING: bptt {bptt} is larger than the number of samples in the training set, {n_samples}. Setting bptt=128."
                    )
                bptt = 128

        seed_all(extra_prior_kwargs_dict.get("rand_seed", 0))

        X, y = X_train, y_train

        if do_permute and (not is_wrapper):
            label_perm = np.random.permutation(num_classes)
        else:
            label_perm = np.arange(num_classes)

        invert_perm_map = {label_perm[i]: i for i in range(num_classes)}
        rev_invert_perm_map = {i: label_perm[i] for i in range(num_classes)}

        if do_permute and (not is_wrapper):
            feat_idx = np.random.permutation(X.shape[1])
        else:
            feat_idx = np.arange(X.shape[1])

        idx = np.random.permutation(X.shape[0])
        X = X[idx, ...]
        y = y[idx, ...]

        y = loop_translate(y, rev_invert_perm_map)

        X = X[:, feat_idx, ...]
        X_val = X_val[:, feat_idx, ...]
        X_test = X_test[:, feat_idx, ...]

        num_classes = len(np.unique(np.unique(y)))
        if (
            do_prompt_tuning
            and extra_prior_kwargs_dict.get("tuned_prompt_label_balance", "equal")
            == "proportional"
        ):
            int_y = y.astype(int)
            label_weights = np.bincount(int_y) / len(int_y)
            label_weights = torch.from_numpy(label_weights).float().to(device)
        else:
            label_weights = None

        if extra_prior_kwargs_dict.get("do_preprocess", False):
            preprocess_type = extra_prior_kwargs_dict.get("preprocess_type", "none")
            summerize_after_prep = extra_prior_kwargs_dict.get(
                "summerize_after_prep", "False"
            )

            X = preprocess_input(
                torch.from_numpy(X.copy().astype(np.float32)),
                preprocess_type,
                summerize_after_prep,
                args,
                is_train=True,
            )
            X_val = preprocess_input(
                torch.from_numpy(X_val.copy().astype(np.float32)),
                preprocess_type,
                summerize_after_prep,
                args,
                is_train=False,
            )
            X_test = preprocess_input(
                torch.from_numpy(X_test.copy().astype(np.float32)),
                preprocess_type,
                summerize_after_prep,
                args,
                is_train=False,
            )
            if args.summerize_after_prep:
                X, X_val, X_test = SummarizeAfter(
                    X, X_val, X_test, y, y_val, y_test, num_features, args
                )
        else:
            X = torch.from_numpy(X.copy().astype(np.float32))
            X_val = torch.from_numpy(X_val.copy().astype(np.float32))
            X_test = torch.from_numpy(X_test.copy().astype(np.float32))

        do_pf = extra_prior_kwargs_dict.get("pad_features", True)
        if do_pf:

            def pad_data(data):
                return torch.cat(
                    [data, torch.zeros(data.shape[0], num_features - data.shape[1])],
                    dim=1,
                )

            if X.shape[1] < num_features:
                X = pad_data(X)
            if X_val.shape[1] < num_features:
                X_val = pad_data(X_val)
            if X_test.shape[1] < num_features:
                X_test = pad_data(X_test)
        if epochs == 0:
            train_ds = TabDS(X, y)
            val_ds = TabDS(X_val, y_val, show_shape=False)
            test_ds = TabDS(X_test, y_test, show_shape=False)
        else:
            train_ds = TabDS(X, y)
            val_ds = TabDS(X_val, y_val)
            test_ds = TabDS(X_test, y_test)

        return (
            X,
            y,
            X_val,
            y_val,
            X_test,
            y_test,
            invert_perm_map,
            steps_per_epoch,
            num_classes,
            label_weights,
            train_ds,
            val_ds,
            test_ds,
        )

    def make_dataloaders(bptt=bptt, not_zs=True):
        dl, bptt = get_train_dataloader(
            train_ds,
            bptt=bptt,
            shuffle=False,
            num_workers=n_workers,
            drop_last=True,
            agg_k_grads=aggregate_k_gradients,
            not_zs=not_zs,
        )

        val_dl = DataLoader(
            val_ds,
            batch_size=min(bptt, y_val.shape[0] // 2),
            shuffle=False,
            num_workers=n_workers,
        )

        test_dl = DataLoader(
            test_ds,
            batch_size=min(bptt, y_val.shape[0] // 2),
            shuffle=False,
            num_workers=n_workers,
        )
        X_data_for_fitting = []
        y_data_for_fitting = []
        for idx, (td, _, _) in enumerate(dl):
            X_data_for_fitting.append(td[0])
            y_data_for_fitting.append(td[1])
            X_data_concat = torch.cat(X_data_for_fitting, dim=0)
            y_data_concat = torch.cat(y_data_for_fitting, dim=0)
            if X_data_concat.shape[0] >= real_data_qty:
                break
        data_for_fitting = [X_data_concat, y_data_concat]
        return dl, val_dl, test_dl, bptt, data_for_fitting

    if real_prior:
        not_zs = extra_prior_kwargs_dict.get("zs_eval_ensemble", 0) == 0
        do_zs = (not not_zs) and (not do_kl_loss)
        seed_all(extra_prior_kwargs_dict.get("rand_seed"))

        if do_kl_loss:
            if extra_prior_kwargs_dict["uniform_bptt"] == False:
                print("KL loss with TabPFN-zs only supports uniform bptt")
                extra_prior_kwargs_dict["uniform_bptt"] = True

        data_for_fitting = None
        (
            X,
            y,
            X_val,
            y_val,
            X_test,
            y_test,
            invert_perm_map,
            steps_per_epoch,
            num_classes,
            label_weights,
            train_ds,
            val_ds,
            test_ds,
        ) = make_datasets(
            extra_prior_kwargs_dict,
            do_permute=not_zs,
            bptt=bptt,
            steps_per_epoch=steps_per_epoch,
            is_wrapper=is_wrapper,
        )
        old_bptt = bptt
        dl, val_dl, test_dl, bptt, data_for_fitting = make_dataloaders(
            bptt=bptt, not_zs=not_zs
        )
        val_dl = get_subset_dl(extra_prior_kwargs_dict, val_dl)
        if epochs == 0:
            return None, None, None, test_dl

        if verbose:
            if data_for_fitting:
                print("Size of data for fitting: ", len(data_for_fitting[0]))

        if do_zs or do_kl_loss:
            from tunetables_light.scripts.transformer_prediction_interface import (
                TuneTablesZeroShotClassifier,
            )

            if extra_prior_kwargs_dict.get("zs_eval_ensemble", 0) > 0:
                ens_size = extra_prior_kwargs_dict.get("zs_eval_ensemble", 0)
            else:
                ens_size = 32
            eval_model = TuneTablesZeroShotClassifier(
                device=device,
                N_ensemble_configurations=ens_size,
                base_path=".",
                seed=extra_prior_kwargs_dict.get("rand_seed", 0),
                batch_size_inference=1,
            )
            if do_kl_loss:
                eval_model.fit(
                    data_for_fitting[0], data_for_fitting[1], overwrite_warning=True
                )
        else:
            eval_model = None

        if old_bptt != bptt:
            max_pos = int((len(data_for_fitting[0]) // 10) * (0.8))
            if verbose:
                print("bptt changed from {} to {}".format(old_bptt, bptt))
            if extra_prior_kwargs_dict.get("uniform_bptt", False):
                single_eval_pos_gen = lambda: np.random.randint(0, max_pos)
            else:
                single_eval_pos_gen = max_pos
        if do_zs:

            def tpc_data_eval(
                cl=1000, X=None, y=None, X_val=None, y_val=None, ens_size=1
            ):
                num_classes_local = len(np.unique(y))
                start_time = time.time()
                results = dict()
                if cl > len(X):
                    cl = len(X) - 1
                eval_model.fit(X[:cl, ...], y[:cl, ...], overwrite_warning=True)
                predictions = eval_model.predict(X_val).astype(np.int64)
                outputs = np.zeros((len(X_val), num_classes_local))
                output_eval = eval_model.predict_proba(X_val)
                for j in range(output_eval.shape[1]):
                    outputs[:, invert_perm_map[j]] = output_eval[:, j]
                for i in range(num_classes_local):
                    # try:
                    outputs[:, i] = outputs[:, invert_perm_map[i]]
                targets = y_val
                warnings.filterwarnings("ignore")
                end_time = time.time()
                results["Eval_Time"] = np.round(end_time - start_time, 3).item()
                results["Accuracy"] = np.round(
                    accuracy_score(targets, predictions), 3
                ).item()
                try:
                    results["Log_Loss"] = np.round(
                        log_loss(targets, outputs, labels=np.arange(num_classes_local)),
                        3,
                    ).item()
                except Exception as e:
                    if verbose:
                        print("Error calculating log loss: ", e)
                    results["Log_Loss"] = 0.0
                results["F1_Weighted"] = np.round(
                    f1_score(targets, predictions, average="weighted"), 3
                ).item()
                results["F1_Macro"] = np.round(
                    f1_score(targets, predictions, average="macro"), 3
                ).item()
                try:
                    if num_classes == 2:
                        results["ROC_AUC"] = np.round(
                            roc_auc_score(
                                targets,
                                outputs[:, 1],
                                labels=np.arange(num_classes_local),
                            ),
                            3,
                        ).item()
                    else:
                        results["ROC_AUC"] = np.round(
                            roc_auc_score(
                                targets,
                                outputs,
                                labels=np.arange(num_classes_local),
                                multi_class="ovr",
                            ),
                            3,
                        ).item()
                except Exception as e:
                    if verbose:
                        print("Error calculating ROC AUC: ", e)
                    results["ROC_AUC"] = 0.0
                warnings.filterwarnings("default")
                return results

            res_dict = dict()
            val_results = tpc_data_eval(
                cl=real_data_qty,
                X=data_for_fitting[0],
                y=data_for_fitting[1],
                X_val=X_val,
                y_val=y_val,
                ens_size=extra_prior_kwargs_dict.get("zs_eval_ensemble", 0),
            )
            res_dict = dict(res_dict, **{"Val_" + k: v for k, v in val_results.items()})
            test_results = tpc_data_eval(
                cl=real_data_qty,
                X=data_for_fitting[0],
                y=data_for_fitting[1],
                X_val=X_test,
                y_val=y_test,
                ens_size=extra_prior_kwargs_dict.get("zs_eval_ensemble", 0),
            )
            res_dict = dict(
                res_dict, **{"Test_" + k: v for k, v in test_results.items()}
            )
            with open(
                os.path.join(
                    extra_prior_kwargs_dict.get("save_path"), "zs_eval_ensemble.json"
                ),
                "w",
            ) as f:
                json.dump(res_dict, f)
    else:
        raise Exception("Excepted a real dataset")

    if do_zs:
        return "", res_dict, None, None

    encoder = encoder_generator(num_features, emsize)
    style_def = None
    style_encoder = (
        style_encoder_generator(style_def.shape[1], emsize)
        if (style_def is not None)
        else None
    )
    if do_kl_loss:
        assert num_classes < 11, (
            "KL loss with TabPFN-zs only supports 10 classes or fewer"
        )
        n_out = 10
        criterion = kl_divergence
    elif isinstance(criterion, nn.GaussianNLLLoss):
        n_out = 2
    elif isinstance(criterion, nn.CrossEntropyLoss):
        n_out = criterion.weight.shape[0]
    else:
        n_out = 1
    model = TransformerModel(
        encoder,
        n_out,
        emsize,
        nhead,
        nhid,
        nlayers,
        dropout,
        style_encoder=style_encoder,
        y_encoder=y_encoder_generator(1, emsize),
        input_normalization=input_normalization,
        pos_encoder=(
            pos_encoder_generator or positional_encodings.NoPositionalEncoding
        )(emsize, bptt * 2),
        decoder=decoder,
        init_method=initializer,
        efficient_eval_masking=efficient_eval_masking,
        prefix_size=prefix_size,
        n_classes=num_classes,
        prefix_label_probs=label_weights,
        num_features=extra_prior_kwargs_dict.get("num_features", 100),
        linear=extra_prior_kwargs_dict.get("linear", False),
        **model_extra_args,
    )
    model.criterion = criterion
    if load_weights_from_this_state_dict is not None:
        encoder_mismatch = False
        decoder_mismatch = False

        if do_kl_loss:
            load_weights_from_this_state_dict.pop("criterion.weight")
        if num_classes > 10:
            decoder_mismatch = True
            load_weights_from_this_state_dict["decoder.2.weight"] = model.state_dict()[
                "decoder.2.weight"
            ]
            load_weights_from_this_state_dict["decoder.2.bias"] = model.state_dict()[
                "decoder.2.bias"
            ]
            load_weights_from_this_state_dict["criterion.weight"] = model.state_dict()[
                "criterion.weight"
            ]
        if (
            load_weights_from_this_state_dict.get("prefix_embedding.weight", None)
            is None
            and model.state_dict().get("prefix_embedding.weight", None) is not None
        ):
            load_weights_from_this_state_dict["prefix_embedding.weight"] = (
                model.state_dict()["prefix_embedding.weight"]
            )
        if load_weights_from_this_state_dict.get("encoder.weight", None) is not None:
            load_shape = load_weights_from_this_state_dict.get(
                "encoder.weight", None
            ).shape
            model_shape = model.state_dict().get("encoder.weight", None).shape
            if load_shape != model_shape:
                encoder_mismatch = True
                if verbose:
                    print(
                        "Encoder weight shape mismatch: ",
                        load_shape,
                        model_shape,
                        "Using randomly initialized encoder weights from model instead",
                    )
                load_weights_from_this_state_dict["encoder.weight"] = (
                    model.state_dict()["encoder.weight"]
                )
        model.load_state_dict(load_weights_from_this_state_dict)
    if initialize_with_model is not None:
        model.init_from_small_model(initialize_with_model)

    params_to_optimize = []
    if do_prompt_tuning:
        params_to_optimize.append("prefix_embedding")
    if encoder_mismatch:
        params_to_optimize.append("encoder")
    if decoder_mismatch:
        params_to_optimize.append("decoder.2")
        params_to_optimize.append("criterion")
    if verbose:
        print("Params to optimize: ", params_to_optimize)

        print(
            f"Using a Transformer with {sum(p.numel() for p in model.parameters()) / 1000 / 1000:.{2}f} M parameters"
        )

    try:
        for (k, v), (k2, v2) in zip(
            model.state_dict().items(), initialize_with_model.state_dict().items()
        ):
            print(k, ((v - v2) / v).abs().mean(), v.shape)
    except Exception:
        pass

    model.to(device)
    if using_dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], output_device=rank, broadcast_buffers=False
        )

    if not real_prior:
        dl.model = model

    if lr is None:
        lr = get_openai_lr(model)
        if verbose:
            print(f"Using OpenAI max lr of {lr}.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched_obj = scheduler(
        optimizer, warmup_epochs, epochs if epochs is not None else 100
    )

    scaler = GradScaler() if train_mixed_precision else None

    utils.check_compatibility(dl)

    master_epoch_count = []

    def real_data_eval(
        r_model,
        cl=1000,
        train_data=None,
        val_dl=None,
        softmax_temperature=torch.log(torch.tensor([0.8])),
    ):
        start_time = time.time()
        td = copy.deepcopy(train_data)
        num_classes_local = len(torch.unique(td[1]))
        td[0] = td[0][:cl, ...]
        td[1] = td[1][:cl, ...]
        single_eval_pos = len(td[0])
        softmax_temperature = softmax_temperature.to(device)
        with torch.inference_mode():
            prediction_list = []
            target_list = []
            output_list = []
            for batch, (data, targets, _) in enumerate(
                tqdm(
                    val_dl,
                    total=(len(val_dl)),
                    desc="Training Batches",
                    ncols=100,
                    colour="magenta",
                    dynamic_ncols=True,
                    bar_format=(
                        "{desc} |" 
                        " {bar:30} |" 
                        " {percentage:3.0f}% " 
                        "[{n_fmt}/{total_fmt} batches] " 
                        "⏱️ {elapsed}<{remaining}"
                    ),
                )
            ):
                if extra_prior_kwargs_dict.get("debug", False):
                    data_temp_idx = torch.randperm(data[1].nelement())
                    data[1] = data[1].view(-1)[data_temp_idx].view(data[1].size())

                batch_data = tuple([
                    torch.cat((td[0], data[0]), dim=0).to(torch.float32),
                    torch.cat((td[1], data[1]), dim=0).to(torch.float32),
                ])
                output = r_model(
                    tuple(e.to(device) if torch.is_tensor(e) else e for e in batch_data)
                    if isinstance(batch_data, tuple)
                    else batch_data.to(device),
                    single_eval_pos=single_eval_pos,
                )
                new_output = loop_translate(output, invert_perm_map)
                output = new_output
                output = output[:, 0:num_classes_local] / torch.exp(softmax_temperature)
                output = torch.nn.functional.softmax(output, dim=-1)
                output_list.append(output)
                _, predicted = torch.max(output.cpu().data, 1)
                prediction_list.append(predicted)
                target_list.append(targets)
            outputs = torch.cat(output_list, dim=0).cpu().numpy()
            predictions = torch.cat(prediction_list, dim=0).cpu().numpy()
            targets = torch.cat(target_list, dim=0).cpu().numpy()

        results = dict()
        warnings.filterwarnings("ignore")
        results["Eval_Time"] = np.round(time.time() - start_time, 3).item()
        accuracy = np.round(accuracy_score(targets, predictions), 3).item()
        results["Accuracy"] = accuracy
        try:
            results["Log_Loss"] = np.round(
                log_loss(targets, outputs, labels=np.arange(num_classes_local)), 3
            ).item()
        except Exception as e:
            if verbose:
                print("Error calculating log loss: ", e)
            results["Log_Loss"] = 0.0
        results["F1_Weighted"] = np.round(
            f1_score(targets, predictions, average="weighted"), 3
        ).item()
        results["F1_Macro"] = np.round(
            f1_score(targets, predictions, average="macro"), 3
        ).item()
        try:
            if num_classes_local == 2:
                results["ROC_AUC"] = np.round(
                    roc_auc_score(
                        targets, outputs[:, 1], labels=np.arange(num_classes_local)
                    ),
                    3,
                ).item()
            else:
                results["ROC_AUC"] = np.round(
                    roc_auc_score(
                        targets,
                        outputs,
                        labels=np.arange(num_classes_local),
                        multi_class="ovr",
                    ),
                    3,
                ).item()
        except Exception as e:
            if verbose:
                print("Error calculating ROC AUC: ", e)
            results["ROC_AUC"] = 0.0

        warnings.filterwarnings("default")

        return results, outputs, targets

    def train_epoch(
        e_model, e_optimizer, boost_this_epoch=False, eval_model=None, bptt_search=False
    ):
        tracker = GPULossTracker(device=device)
        if max_time > 0 and time.time() - start_time > max_time:
            print("Max time reached. Exiting")
            exit(0)
        epoch_start_time = time.time()
        time_to_get_batch = 0
        time_to_get_batches = 0
        forward_time = 0
        forward_times = 0
        backward_times = 0
        loss_times = 0
        grad_times = 0
        step_time = 0
        before_get_batch = time.time()
        batches_seen = 0
        shuffle_every_epoch = extra_prior_kwargs_dict.get("shuffle_every_epoch", False)
        permute_feature_pos = extra_prior_kwargs_dict.get(
            "permute_feature_position_in_ensemble", False
        )
        for batch, (data, targets, single_eval_pos) in enumerate(dl):
            if isinstance(data, list):
                data = tuple(data)
            if (
                isinstance(single_eval_pos, torch.Tensor)
                and single_eval_pos.numel() == 0
            ):
                single_eval_pos = None
            if using_dist and not (
                batch % aggregate_k_gradients == aggregate_k_gradients - 1
            ):
                cm = e_model.no_sync()
            else:
                cm = nullcontext()

            if permute_feature_pos:
                data = tuple([data[0][:, torch.randperm(data[0].shape[1])], data[1]])
            elif shuffle_every_epoch:
                seed_all(
                    extra_prior_kwargs_dict.get("rand_seed", 0)
                    + len(master_epoch_count)
                )
                perm_idx = torch.randperm(data[0].shape[0])
                data = tuple([data[0][perm_idx, ...], data[1][perm_idx, ...]])
            with cm:
                time_to_get_batch = time.time() - before_get_batch
                time_to_get_batches += time_to_get_batch
                before_forward = time.time()
                if boosting:
                    single_eval_pos = len(targets) // 2
                elif bptt_extra_samples is None:
                    single_eval_pos = (
                        single_eval_pos_gen()
                        if callable(single_eval_pos_gen)
                        else single_eval_pos_gen
                    )
                else:
                    single_eval_pos = max(targets.shape[0] - bptt_extra_samples, 0)
                with autocast("cuda", enabled=scaler is not None):
                    # If style is set to None, it should not be transferred to device
                    output = e_model(
                        tuple(
                            e.to(torch.float32).to(device) if torch.is_tensor(e) else e
                            for e in data
                        )
                        if isinstance(data, tuple)
                        else data.to(device),
                        single_eval_pos=single_eval_pos,
                    )
                    if not bptt_search:
                        assert output.requires_grad, "Output does not require gradients"
                    forward_time = time.time() - before_forward
                    forward_times += forward_time
                    before_backward = time.time()
                    if single_eval_pos is not None:
                        targets = targets[single_eval_pos:]
                    if isinstance(criterion, nn.GaussianNLLLoss):
                        assert output.shape[-1] == 2, (
                            "need to write a little bit of code to handle multiple regression targets at once"
                        )
                        mean_pred = output[..., 0]
                        var_pred = output[..., 1].abs()
                        losses = criterion(
                            mean_pred.flatten(),
                            targets.to(device).flatten(),
                            var=var_pred.flatten(),
                        )
                    elif isinstance(criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
                        losses = criterion(
                            output.flatten(), targets.to(device).flatten()
                        )
                    elif isinstance(criterion, nn.CrossEntropyLoss):
                        losses = criterion(
                            output.reshape(-1, n_out),
                            targets.to(device).long().flatten(),
                        )
                    elif do_kl_loss:
                        # TODO: investigate shape mismatches
                        real_data_preds = eval_model.predict_proba(data[0])
                        if real_data_preds.shape[1] < output.shape[1]:
                            real_data_preds = np.concatenate(
                                [
                                    real_data_preds,
                                    np.zeros((
                                        real_data_preds.shape[0],
                                        output.shape[1] - real_data_preds.shape[1],
                                    )),
                                ],
                                axis=1,
                            )
                        if real_data_preds.shape[0] != output.shape[0]:
                            if verbose:
                                print(
                                    f"Real data preds and tuned prompt output have different shapes: ",
                                    real_data_preds.shape,
                                    output.shape,
                                )
                            smaller_shape = min(
                                real_data_preds.shape[0], output.shape[0]
                            )
                            real_data_preds = real_data_preds[:smaller_shape, :]
                            output = output[:smaller_shape, :]
                        real_data_preds = torch.tensor(real_data_preds).to(device)
                        assert real_data_preds.shape == output.shape, (
                            f"Real data preds and tuned prompt output have different shapes: {real_data_preds.shape} and {output.shape}"
                        )
                        losses = criterion(real_data_preds, output)
                    else:
                        losses = criterion(output, targets)
                    if boosting or do_kl_loss:
                        loss = losses.mean()
                        nan_share = torch.tensor([0])
                    else:
                        if len(output.shape) == 2:
                            output = output.unsqueeze(1)
                        losses = losses.view(*output.shape[0:2])

                        loss, nan_share = utils.torch_nanmean(
                            losses.mean(0), return_nanshare=True
                        )
                        loss = loss / aggregate_k_gradients

                if scaler:
                    loss = scaler.scale(loss)
                if boosting and boost_this_epoch:
                    cur_grads = []
                    if prior_grad_dict is None:
                        prior_grad_iter = None
                    else:
                        prior_grad_iter = prior_grad_dict[batch].to(output.device)
                    output_grad = autograd.grad(loss, output)[0]
                    gradient_dict[batch] = output_grad.detach().cpu().clone()

                    if prior_grad_iter is not None:
                        grad_shape = output_grad.shape
                        flat_grad = output_grad.flatten()
                        grad_signs = torch.sign(flat_grad)
                        flat_prior_grad = prior_grad_iter.flatten()
                        cur_weight = 0.65
                        flat_grad_new = torch.sqrt(
                            cur_weight * torch.pow(flat_grad, 2)
                            + (1 - cur_weight) * torch.pow(flat_prior_grad, 2)
                        )
                        flat_grad_new_signs = torch.sign(flat_grad_new)
                        flat_grad_new[flat_grad_new_signs != grad_signs] *= -1
                        output_grad = flat_grad_new.reshape(grad_shape)

                    output.backward(output_grad)
                elif bptt_search:
                    pass
                else:
                    loss.backward()
                backward_times += time.time() - before_backward
                tracker.update(loss)
                if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                    if scaler:
                        scaler.unscale_(e_optimizer)
                    torch.nn.utils.clip_grad_norm_(e_model.parameters(), 1.0)
                    try:
                        if scaler:
                            scaler.step(e_optimizer)
                            scaler.update()
                        else:
                            e_optimizer.step()
                    except:
                        print("Invalid optimization step encountered")
                    e_optimizer.zero_grad()

                step_time = time.time() - before_forward
            before_get_batch = time.time()
            batches_seen += 1
        if batches_seen < extra_prior_kwargs_dict.get("min_batches_per_epoch", 1):
            raise ValueError(
                "Not enough batches seen in epoch: saw {} batches, expected at least {}".format(
                    batches_seen,
                    extra_prior_kwargs_dict.get("min_batches_per_epoch", 1),
                )
            )

        total_loss = tracker.average()

        return total_loss, None, time_to_get_batch, forward_time, step_time, None, None

    def concat_embedding(ec, model, method):
        device = ec.model.prefix_embedding.weight.device
        if method == "duplicate":
            ec.concatenated_embedding = torch.cat(
                [ec.original_embedding, ec.original_embedding], dim=0
            ).to(device)
            ec.concatenated_y_embedding = torch.cat(
                [ec.original_y_embedding, ec.original_y_embedding], dim=0
            ).to(device)
            ec.prefix_size = ec.original_prefix_size * 2
        elif method.startswith("rand-init"):
            num_to_concat = min(int(method.split("-")[-1]), len(ec.prefix_weights) + 1)
            if verbose:
                print("Concatenating {} embeddings".format(num_to_concat))
            if num_to_concat == 1:
                ec.concatenated_embedding = ec.original_embedding
                ec.concatenated_y_embedding = ec.original_y_embedding
                ec.prefix_size = ec.original_prefix_size
            else:
                ec.concatenated_embedding = torch.cat(
                    [ec.original_embedding.to(device)]
                    + [
                        ec.prefix_weights[i]["prefix_weights"].to(device)
                        for i in range(num_to_concat - 1)
                    ],
                    dim=0,
                ).to(device)
                ec.concatenated_y_embedding = torch.cat(
                    [ec.original_y_embedding.to(device)]
                    + [
                        ec.prefix_weights[i]["prefix_y_labels"].to(device)
                        for i in range(num_to_concat - 1)
                    ],
                    dim=0,
                ).to(device)
                if "size-ctl" in method:
                    if "perm" in method:
                        sel = torch.randperm(ec.concatenated_embedding.shape[0])[
                            : ec.original_prefix_size
                        ].to(device)
                    else:
                        total_emb_size = ec.original_prefix_size
                        emb_size = total_emb_size // num_to_concat
                        orig_emb_size = ec.original_embedding.shape[0]
                        start_pos = [j * orig_emb_size for j in range(num_to_concat)]
                        sel = torch.cat(
                            [torch.arange(i, i + emb_size) for i in start_pos], dim=0
                        ).to(device)

                    ec.concatenated_embedding = ec.concatenated_embedding[sel]
                    ec.concatenated_y_embedding = ec.concatenated_y_embedding[sel]
                    ec.prefix_size = sel.shape[0]
                else:
                    ec.prefix_size = ec.original_prefix_size * num_to_concat
        else:
            raise NotImplementedError("Method {} not implemented!".format(method))
        model.prefix_embedding.weight = nn.Parameter(ec.concatenated_embedding)
        model.prefix_y_embedding = ec.concatenated_y_embedding
        model.prefix_size = ec.prefix_size
        return model

    def restore_embedding(ec, model):
        model.prefix_embedding.weight = nn.Parameter(ec.original_embedding)
        model.prefix_y_embedding = ec.original_y_embedding
        model.prefix_size = ec.original_prefix_size
        model.freeze_parameters_except_named(params_to_optimize)
        return model

    def save_prefix_weights(model, path, i, do_concat, prefix_weights_l):
        prefix_weights = model.state_dict()["prefix_embedding.weight"].cpu().numpy()
        prefix_fn = f"prefix_weights_{i}.npy"
        prefix_save_path = os.path.join(path, prefix_fn)
        np.save(prefix_save_path, prefix_weights)
        prefix_y_labels = model.prefix_y_embedding.cpu().numpy()
        prefix_y_fn = f"prefix_y_labels_{i}.npy"
        prefix_y_save_path = os.path.join(path, prefix_y_fn)

        np.save(prefix_y_save_path, prefix_y_labels)
        if do_concat:
            prefix_weights_l.append({
                "prefix_weights": torch.from_numpy(prefix_weights).float(),
                "prefix_y_labels": torch.from_numpy(prefix_y_labels),
            })
        return prefix_weights_l

    def update_ensemble_acc(
        ens_acc, ens_acc_nc, ens_acc_test, ens_acc_test_nc, num_classes
    ):
        num_classes_local_val = len(np.unique(labels_np))
        num_classes_local_test = len(np.unique(labels_np_test))
        predictions_np = np.argmax(probs_np, axis=1)
        predictions_np_test = np.argmax(probs_np_test, axis=1)
        try:
            if num_classes == 2:
                roc_auc = np.round(
                    roc_auc_score(
                        labels_np,
                        probs_np[:, 1],
                        labels=np.arange(num_classes_local_val),
                    ),
                    3,
                ).item()
                test_roc_auc = np.round(
                    roc_auc_score(
                        labels_np_test,
                        probs_np_test[:, 1],
                        labels=np.arange(num_classes_local_test),
                    ),
                    3,
                ).item()
            else:
                roc_auc = np.round(
                    roc_auc_score(
                        labels_np,
                        probs_np,
                        labels=np.arange(num_classes_local_val),
                        multi_class="ovr",
                    ),
                    3,
                ).item()
                test_roc_auc = np.round(
                    roc_auc_score(
                        labels_np_test,
                        probs_np_test,
                        labels=np.arange(num_classes_local_test),
                        multi_class="ovr",
                    ),
                    3,
                ).item()
        except Exception as e:
            if verbose:
                print("Error calculating ROC AUC: ", e)
            roc_auc = 0.0
            test_roc_auc = 0.0
        f1_weighted = np.round(
            f1_score(labels_np, predictions_np, average="weighted"), 3
        ).item()
        f1_macro = np.round(
            f1_score(labels_np, predictions_np, average="macro"), 3
        ).item()
        try:
            ll = np.round(
                log_loss(labels_np, probs_np, labels=np.arange(num_classes_local_val)),
                3,
            )
        except Exception as e:
            if verbose:
                print("Error calculating ll/ECE/TACE: ", e)
            ll = 0.0
        test_f1_weighted = np.round(
            f1_score(labels_np_test, predictions_np_test, average="weighted"), 3
        ).item()
        test_f1_macro = np.round(
            f1_score(labels_np_test, predictions_np_test, average="macro"), 3
        ).item()
        try:
            test_ll = np.round(
                log_loss(
                    labels_np_test,
                    probs_np_test,
                    labels=np.arange(num_classes_local_test),
                ),
                3,
            )
        except Exception as e:
            if verbose:
                print("Error calculating ll/ECE/TACE: ", e)
            test_ll = 0.0
        if do_prompt_tuning:
            predictions_np_nc = np.argmax(probs_np_nc, axis=1)
            predictions_np_nc_test = np.argmax(probs_np_nc_test, axis=1)
            nc_f1_weighted = np.round(
                f1_score(labels_np_nc, predictions_np_nc, average="weighted"), 3
            ).item()
            nc_f1_macro = np.round(
                f1_score(labels_np_nc, predictions_np_nc, average="macro"), 3
            ).item()
            try:
                if num_classes == 2:
                    roc_auc_nc = np.round(
                        roc_auc_score(
                            labels_np_nc,
                            probs_np_nc[:, 1],
                            labels=np.arange(num_classes_local_val),
                        ),
                        3,
                    ).item()
                    test_roc_auc_nc = np.round(
                        roc_auc_score(
                            labels_np_nc_test,
                            probs_np_nc_test[:, 1],
                            labels=np.arange(num_classes_local_test),
                        ),
                        3,
                    ).item()
                else:
                    roc_auc_nc = np.round(
                        roc_auc_score(
                            labels_np_nc,
                            probs_np_nc,
                            labels=np.arange(num_classes_local_val),
                            multi_class="ovr",
                        ),
                        3,
                    ).item()
                    test_roc_auc_nc = np.round(
                        roc_auc_score(
                            labels_np_nc_test,
                            probs_np_nc_test,
                            labels=np.arange(num_classes_local_test),
                            multi_class="ovr",
                        ),
                        3,
                    ).item()
            except Exception as e:
                if verbose:
                    print("Error calculating ROC AUC: ", e)
                roc_auc_nc = 0.0
                test_roc_auc_nc = 0.0
            try:
                nc_ll = np.round(
                    log_loss(
                        labels_np_nc,
                        probs_np_nc,
                        labels=np.arange(num_classes_local_val),
                    ),
                    3,
                )
            except Exception as e:
                if verbose:
                    print("Error calculating ll/ECE/TACE: ", e)
                nc_ll = 0.0
            nc_test_f1_weighted = np.round(
                f1_score(labels_np_nc_test, predictions_np_nc_test, average="weighted"),
                3,
            ).item()
            nc_test_f1_macro = np.round(
                f1_score(labels_np_nc_test, predictions_np_nc_test, average="macro"), 3
            ).item()
            try:
                nc_test_ll = np.round(
                    log_loss(
                        labels_np_nc_test,
                        probs_np_nc_test,
                        labels=np.arange(num_classes_local_test),
                    ),
                    3,
                )
            except Exception as e:
                if verbose:
                    print("Error calculating ll/ECE/TACE: ", e)
                nc_test_ll = 0.0
        else:
            nc_f1_weighted = 0
            nc_f1_macro = 0
            roc_auc_nc = 0
            test_roc_auc_nc = 0
            nc_test_f1_weighted = 0
            nc_test_f1_macro = 0
            nc_ll = 0
            nc_test_ll = 0
        if verbose:
            print(
                "Ensemble accuracy: ", ens_acc, "Ensemble accuracy (NC): ", ens_acc_nc
            )
        new_res = {
            "Ens_Val_Accuracy": ens_acc,
            "Ens_Val_Accuracy_NC": ens_acc_nc,
            "Ens_Val_F1_Weighted": f1_weighted,
            "Ens_Val_F1_Macro": f1_macro,
            "Ens_Val_F1_Weighted_NC": nc_f1_weighted,
            "Ens_Val_F1_Macro_NC": nc_f1_macro,
            "Ens_Val_Log_Loss": ll,
            "Ens_Val_Log_Loss_NC": nc_ll,
            "Ens_Val_ROC_AUC": roc_auc,
            "Ens_Val_ROC_AUC_NC": roc_auc_nc,
            "Ens_Test_Accuracy": ens_acc_test,
            "Ens_Test_Accuracy_NC": ens_acc_test_nc,
            "Ens_Test_F1_Weighted": test_f1_weighted,
            "Ens_Test_F1_Macro": test_f1_macro,
            "Ens_Test_F1_Weighted_NC": nc_test_f1_weighted,
            "Ens_Test_F1_Macro_NC": nc_test_f1_macro,
            "Ens_Test_Log_Loss": test_ll,
            "Ens_Test_Log_Loss_NC": nc_test_ll,
            "Ens_Test_ROC_AUC": test_roc_auc,
            "Ens_Test_ROC_AUC_NC": test_roc_auc_nc,
        }
        return new_res

    def train_test_loop(t_model, t_optim, t_sched, eval_model, dl, val_dl, test_dl):
        return_outputs = None
        return_targets = None
        res_dict = None
        best_val_score = best_val_score_nc = 0
        best_total_loss = 1e9
        if do_prompt_tuning:
            best_val_embed = t_model.prefix_embedding.weight.detach().cpu()
        best_res_dict = None
        best_outputs = None
        best_targets = None
        is_best = False
        patience = 0

        for epoch in range(1, epochs + 1) if epochs is not None else itertools.count(1):
            is_best = False
            if verbose:
                print("epoch", epoch, "of", epochs)
            boost_this_epoch = True if epoch == 1 else False
            epoch_start_time = time.time()
            master_epoch_count.append(1)
            if do_prompt_tuning:
                t_model.freeze_parameters_except_named(params_to_optimize)
                for n, p in t_model.named_parameters():
                    grad_reqd = False
                    for s in params_to_optimize:
                        if s in n:
                            grad_reqd = True
                    assert p.requires_grad == grad_reqd, (
                        "Parameter {} does not have the correct grad requirement!".format(
                            n
                        )
                    )
            t_model.train()
            total_loss, _, time_to_get_batch, forward_time, step_time, _, _ = (
                train_epoch(
                    t_model,
                    t_optim,
                    boost_this_epoch,
                    eval_model=eval_model,
                    bptt_search=False,
                )
            )
            val_score = val_score_nc = val_score_concat = val_score_nc_concat = (
                test_score
            ) = test_score_nc = None
            res_dict = dict()
            res_dict["epoch_train_time"] = np.round(
                time.time() - epoch_start_time, 3
            ).item()
            res_dict["master_epoch_count"] = len(master_epoch_count)
            if real_prior:
                val_start_time = time.time()
                val_results, val_outputs, val_targets = real_data_eval(
                    r_model=t_model,
                    cl=real_data_qty,
                    train_data=data_for_fitting,
                    val_dl=val_dl,
                )
                res_dict = dict(
                    res_dict, **{"Val_" + k: v for k, v in val_results.items()}
                )
                val_score = res_dict["Val_Accuracy"]

                return_outputs = [val_outputs]
                return_targets = [val_targets]
                if do_prompt_tuning:
                    if do_concat != "":
                        print(
                            f"We are doing the prompting and concatenation: {do_concat}"
                        )
                        ec = EmbeddingConcatenator(t_model, do_concat, prefix_weights_l)
                        t_model = concat_embedding(ec, t_model, do_concat)
                        val_score_concat, _, _ = real_data_eval(
                            r_model=ec.get_model(),
                            cl=real_data_qty,
                            train_data=data_for_fitting,
                            val_dl=val_dl,
                        )
                        res_dict = dict(
                            res_dict,
                            **{
                                "Val_concat_" + k: v
                                for k, v in val_score_concat.items()
                            },
                        )
                        val_score_nc_concat, _, _ = real_data_eval(
                            r_model=ec.get_model(),
                            cl=0,
                            train_data=data_for_fitting,
                            val_dl=val_dl,
                        )
                        res_dict = dict(
                            res_dict,
                            **{
                                "Val_concat_nc_" + k: v
                                for k, v in val_score_nc_concat.items()
                            },
                        )
                        t_model = restore_embedding(ec, t_model)
                        t_optim = torch.optim.AdamW(
                            t_model.parameters(), lr=lr, weight_decay=weight_decay
                        )
                        t_sched = scheduler(
                            t_optim,
                            warmup_epochs,
                            epochs if epochs is not None else 100,
                        )
                    else:
                        val_score_nc_concat = ""
                        val_score_concat = ""
                    val_score_nc, val_outputs, val_targets = real_data_eval(
                        r_model=t_model,
                        cl=0,
                        train_data=data_for_fitting,
                        val_dl=val_dl,
                    )
                    return_outputs.append(val_outputs)
                    return_targets.append(val_targets)
                    res_dict = dict(
                        res_dict, **{"Val_nc_" + k: v for k, v in val_score_nc.items()}
                    )

                # Early stopping logic
                score_condition = round(total_loss, 2) < round(best_total_loss, 2)

                if score_condition:
                    patience = 0
                    best_total_loss = total_loss
                    is_best = True
                    if do_prompt_tuning:
                        best_val_embed = t_model.prefix_embedding.weight.detach().cpu()
                else:
                    patience += 1
                if verbose:
                    print("val_epoch time: ", round(time.time() - val_start_time, 2))

            elif hasattr(dl, "validate") and epoch % validation_period == 0:
                with torch.no_grad():
                    print(f"VALIDATION WITH VALIDATE ATTRIBUTE: {val_score}")
                    val_score = dl.validate(model)

            NO_PATIENCE = patience > extra_prior_kwargs_dict.get(
                "early_stopping_patience", 2
            )
            if is_best or (NO_PATIENCE and "Test_Accuracy" not in res_dict):
                test_results, test_outputs, test_targets = real_data_eval(
                    r_model=t_model,
                    cl=real_data_qty,
                    train_data=data_for_fitting,
                    val_dl=test_dl,
                )
                res_dict = dict(
                    res_dict, **{"Test_" + k: v for k, v in test_results.items()}
                )
                return_outputs = (
                    return_outputs[:1] + [test_outputs] + return_outputs[1:]
                )
                return_targets = (
                    return_targets[:1] + [test_targets] + return_targets[1:]
                )
                if do_prompt_tuning:
                    test_score_nc, test_outputs, test_targets = real_data_eval(
                        r_model=t_model,
                        cl=0,
                        train_data=data_for_fitting,
                        val_dl=test_dl,
                    )
                    res_dict = dict(
                        res_dict,
                        **{"Test_nc_" + k: v for k, v in test_score_nc.items()},
                    )
                    return_outputs.append(test_outputs)
                    return_targets.append(test_targets)
                if is_best:
                    best_outputs = return_outputs
                    best_targets = return_targets
                    best_res_dict = res_dict
            if verbose:
                get_time = time.time() - epoch_start_time
                print("-" * 78)
                print(
                    f"| end of epoch {epoch:3d} | time: {get_time:5.2f}s | mean loss {total_loss:5.2f} | "
                    f" | val score {val_score}"
                    if val_score is not None
                    else f" | val score nc {res_dict.get('Val_nc_Accuracy', 0)}"
                    if val_score_nc is not None
                    else f" | test score {res_dict.get('Test_Accuracy', 0)}"
                    if res_dict.get("Test_Accuracy", 0) is not None
                    else f" | test score nc {res_dict.get('Test_nc_Accuracy', 0)}"
                    if res_dict.get("Test_nc_Accuracy", 0) is not None
                    else ""
                )
                print("-" * 78)
                if epoch_callback is not None and rank == 0:
                    epoch_callback(model, epoch / epochs, res_dict)
                if val_score is not None:
                    res_dict = dict(
                        res_dict,
                        **{
                            "epoch": epoch,
                        },
                    )
                    if is_best:
                        best_res_dict = res_dict
                        best_outputs = return_outputs
                        best_targets = return_targets
                    mstr = extra_prior_kwargs_dict.get("model_string")
                    boost_iter = (
                        f"ensemble_iter_{cur_boost_iter}" if is_ensemble else ""
                    )
                    log_path = os.path.join(
                        extra_prior_kwargs_dict.get("save_path"),
                        f"{mstr}_{boost_iter}_log_{epoch}.json",
                    )
                    with open(log_path, "w") as f:
                        json.dump(res_dict, f, indent=4)

                if NO_PATIENCE:
                    break

            t_sched.step()

        if (
            do_prompt_tuning
            and not do_kl_loss
            and isinstance(best_val_embed, torch.Tensor)
        ):
            t_model.prefix_embedding.weight = nn.Parameter(best_val_embed.to(device))
            t_model.prefix_embedding.weight.requires_grad = True
            t_optim = torch.optim.AdamW(
                t_model.parameters(), lr=lr, weight_decay=weight_decay
            )
            t_sched = scheduler(
                t_optim, warmup_epochs, epochs if epochs is not None else 100
            )
            v_scr, val_outputs, val_targets = real_data_eval(
                r_model=t_model,
                cl=real_data_qty,
                train_data=data_for_fitting,
                val_dl=val_dl,
            )
            if (v_scr["Accuracy"] != best_res_dict["Val_Accuracy"]) and verbose:
                print(
                    "WARNING: Best embedding score {} does not match best score {}!".format(
                        v_scr, best_res_dict["Val_Accuracy"]
                    )
                )

        return best_outputs, best_targets, best_res_dict

    if extra_prior_kwargs_dict.get("bptt_search", False):
        backup_epochs = epochs
        epochs = 1
        backup_unif_bptt = extra_prior_kwargs_dict.get("uniform_bptt", False)
        extra_prior_kwargs_dict["uniform_bptt"] = True
        bptt_intervals = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
        STOP = False
        for bptt_idx, bptt in enumerate(bptt_intervals):
            if verbose:
                print("Trying bptt: ", bptt)
            try:
                dl, bptt = get_train_dataloader(
                    dl.dataset,
                    bptt=bptt,
                    shuffle=True,
                    num_workers=n_workers,
                    drop_last=True,
                    agg_k_grads=aggregate_k_gradients,
                )
                with torch.no_grad():
                    (
                        total_loss,
                        _,
                        time_to_get_batch,
                        forward_time,
                        step_time,
                        nan_share,
                        _,
                    ) = train_epoch(
                        model, optimizer, False, eval_model=eval_model, bptt_search=True
                    )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if verbose:
                        print(f"OOM with batch size {bptt}")
                    STOP = True
                    search_idx = max(bptt_idx, 2)
                    bptt = bptt_intervals[search_idx - 2]
                    print("Setting bptt to ", bptt)
                    dl, bptt = get_train_dataloader(
                        dl.dataset,
                        bptt=bptt,
                        shuffle=True,
                        num_workers=n_workers,
                        drop_last=True,
                        agg_k_grads=aggregate_k_gradients,
                    )
                else:
                    raise e
            if STOP:
                break
        epochs = backup_epochs
        extra_prior_kwargs_dict["uniform_bptt"] = backup_unif_bptt

    bagging = extra_prior_kwargs_dict.get("bagging", False)
    if bagging:
        split_size = extra_prior_kwargs_dict.get("subset_rows_bagging", 10000)
        if split_size == 0:
            if verbose:
                print("WARNING: subsampling was 0, using full dataset for bagging")
            split_size = len(dl.dataset)
        dl_backup = dl
        split_indices = []
        for i in range(boosting_n_iters):
            np.random.seed(extra_prior_kwargs_dict.get("rand_seed") + i)
            split_indices.append(
                np.random.choice(
                    np.arange(len(dl_backup.dataset)), size=split_size, replace=True
                )
            )
    is_ensemble = boosting or bagging or rand_init_ensemble
    prefix_weights_l = []
    cur_boost_iter = 0
    total_loss = float("inf")
    total_positional_losses = float("inf")
    output_dict = {}
    i = 0
    ensembling_acc = dict()
    res_dict_ensemble = dict()
    best_results = dict()
    try:
        ens_patience = 0
        topk_key = extra_prior_kwargs_dict.get("topk_key", "Val_Accuracy")
        if "nc_" in topk_key:
            topk_ens_key = "Ens_" + topk_key.replace("nc_", "") + "_NC"
        else:
            topk_ens_key = "Ens_" + topk_key
        if bagging:
            subset_dataset = Subset(dl_backup.dataset, split_indices[i])
            dl, bptt = get_train_dataloader(
                subset_dataset,
                bptt=bptt,
                shuffle=True,
                num_workers=n_workers,
                drop_last=True,
                agg_k_grads=aggregate_k_gradients,
            )
        prior_grad_dict = None
        gradient_dict = {}
        output_dict[i], test_targets, results_dict = train_test_loop(
            model, optimizer, sched_obj, eval_model, dl, val_dl, test_dl
        )
        res_dict_ensemble[i] = best_results = results_dict
        prior_grad_dict = gradient_dict

        probs_np = output_dict[0][0]
        labels_np = test_targets[0]
        probs_np_test = output_dict[0][1]
        labels_np_test = test_targets[1]
        if do_prompt_tuning:
            probs_np_nc = output_dict[0][2]
            labels_np_nc = test_targets[2]
            probs_np_nc_test = output_dict[0][3]
            labels_np_nc_test = test_targets[3]
        if is_ensemble:
            master_epoch_count.append(1)
            best_ens_acc = res_dict_ensemble[i][topk_key]
            ensembling_acc[i] = update_ensemble_acc(
                res_dict_ensemble[i]["Val_Accuracy"],
                res_dict_ensemble[i]["Val_nc_Accuracy"],
                res_dict_ensemble[i]["Test_Accuracy"],
                res_dict_ensemble[i]["Test_nc_Accuracy"],
                len(np.unique(labels_np)),
            )
            if not do_concat:
                with open(
                    os.path.join(
                        extra_prior_kwargs_dict.get("save_path"), "ensembling_acc.json"
                    ),
                    "w",
                ) as f:
                    json.dump(ensembling_acc, f, indent=4)
        if do_prompt_tuning:
            prefix_weights_l = save_prefix_weights(
                model,
                extra_prior_kwargs_dict.get("save_path"),
                i,
                do_concat,
                prefix_weights_l,
            )
    except KeyboardInterrupt:
        pass

    if is_ensemble:
        for i in range(1, boosting_n_iters):
            next_seed = extra_prior_kwargs_dict.get("rand_seed") + i
            seed_all(next_seed)

            if extra_prior_kwargs_dict.get("reseed_data", True):
                extra_prior_kwargs_dict["do_impute"] = np.random.choice([True, False])
                extra_prior_kwargs_dict["ohe"] = np.random.choice([True, False])
                extra_prior_kwargs_dict["preprocess_type"] = np.random.choice([
                    "none",
                    "power_all",
                    "robust_all",
                    "quantile_all",
                ])
                (
                    X,
                    y,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    invert_perm_map,
                    steps_per_epoch,
                    num_classes,
                    label_weights,
                    train_ds,
                    val_ds,
                    test_ds,
                ) = make_datasets(
                    extra_prior_kwargs_dict,
                    do_permute=not_zs,
                    bptt=bptt,
                    steps_per_epoch=steps_per_epoch,
                    is_wrapper=is_wrapper,
                )
                old_bptt = bptt
                dl, val_dl, test_dl, bptt, data_for_fitting = make_dataloaders(
                    bptt=bptt
                )
                if old_bptt != bptt:
                    if verbose:
                        print("bptt changed from {} to {}".format(old_bptt, bptt))
                    if extra_prior_kwargs_dict.get("uniform_bptt", False):
                        single_eval_pos_gen = lambda: np.random.randint(0, bptt)
                    else:
                        single_eval_pos_gen = bptt
                if bagging:
                    dl_backup = dl
            if bagging:
                subset_dataset = Subset(dl_backup.dataset, split_indices[i])
                dl = DataLoader(
                    subset_dataset,
                    batch_size=bptt,
                    shuffle=False,
                    num_workers=n_workers,
                    drop_last=True,
                )
            cur_boost_iter = i
            print("Ensembling iteration: ", i + 1, " of ", boosting_n_iters, "\n \n")
            model.init_prefix_weights()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            sched_obj = scheduler(
                optimizer, warmup_epochs, epochs if epochs is not None else 100
            )
            output_dict[i], test_targets, results_dict = train_test_loop(
                model, optimizer, sched_obj, eval_model, dl, val_dl, test_dl
            )
            res_dict_ensemble[i] = results_dict
            if do_prompt_tuning:
                prefix_weights_l = save_prefix_weights(
                    model,
                    extra_prior_kwargs_dict.get("save_path"),
                    i,
                    do_concat,
                    prefix_weights_l,
                )
            prior_grad_dict = gradient_dict

            if do_concat != "":
                continue

            current_outs = dict()
            current_preds = dict()
            boosting_accs = dict()
            topk_ens_val = extra_prior_kwargs_dict.get("keep_topk_ensemble", 0)
            if topk_ens_val > 0:
                if verbose:
                    print(
                        "keeping top {} of {} models, per provided key {}".format(
                            topk_ens_val, i + 1, topk_key
                        )
                    )
                sorted_res = sorted(
                    res_dict_ensemble.items(),
                    key=lambda x: x[1][topk_key],
                    reverse=True,
                )
                models_to_include = [x[0] for x in sorted_res][:topk_ens_val]
            else:
                models_to_include = list(range(i + 1))
            for m in range(len(output_dict[0])):
                total = len(test_targets[m])
                if extra_prior_kwargs_dict.get("average_ensemble"):
                    current_outs[m] = torch.zeros_like(
                        torch.from_numpy(output_dict[0][m])
                    )
                    for j in range(i + 1):
                        if j not in models_to_include:
                            continue
                        current_outs[m] += output_dict[j][m]
                    current_outs[m] /= i + 1
                else:
                    current_outs[m] = torch.from_numpy(output_dict[0][m]).to(
                        device=device, dtype=torch.float32
                    )
                    for j in range(1, i + 1):
                        if j not in models_to_include:
                            continue

                        boost_res = boosting_lr * torch.from_numpy(
                            output_dict[j][m]
                        ).to(device=device, dtype=torch.float32)
                        current_outs[m] = current_outs[m] + boost_res
                _, current_preds[m] = torch.max(current_outs[m].cpu().data, 1)
                correct = (
                    (current_preds[m] == torch.from_numpy(test_targets[m])).sum().item()
                )
                boosting_accs[m] = np.round(correct / total, 3)
            probs_np = output_dict[0][0]
            labels_np = test_targets[0]
            probs_np_test = output_dict[0][1]
            labels_np_test = test_targets[1]
            if do_prompt_tuning:
                probs_np_nc = output_dict[0][2]
                labels_np_nc = test_targets[2]
                probs_np_nc_test = output_dict[0][3]
                labels_np_nc_test = test_targets[3]
            best_results = ensembling_acc[i] = update_ensemble_acc(
                boosting_accs[0],
                boosting_accs[2],
                boosting_accs[1],
                boosting_accs[3],
                len(np.unique(labels_np)),
            )
            cur_ens_acc = ensembling_acc[i][topk_ens_key]
            if cur_ens_acc > best_ens_acc:
                ens_patience = 0
                best_ens_acc = cur_ens_acc
            else:
                ens_patience += 1
            if do_prompt_tuning:
                prefix_weights_l = save_prefix_weights(
                    model,
                    extra_prior_kwargs_dict.get("save_path"),
                    i,
                    do_concat,
                    prefix_weights_l,
                )
            with open(
                os.path.join(
                    extra_prior_kwargs_dict.get("save_path"), "ensembling_acc.json"
                ),
                "w",
            ) as f:
                json.dump(ensembling_acc, f, indent=4)

                master_epoch_count.append(1)

            if ens_patience > extra_prior_kwargs_dict.get("early_stopping_patience", 2):
                print("Early stopping after {} ensembles".format(i))
                break

    if rank == 0:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
            dl = None
        return model, best_results, data_for_fitting, None

    return model, best_results, data_for_fitting, None


def _parse_args(config_parser, parser):
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)

    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


if __name__ == "__main__":
    config_parser = argparse.ArgumentParser(
        description="Only used as a first parser for the config file path."
    )
    config_parser.add_argument("--config")
    parser = argparse.ArgumentParser()
    parser.add_argument("prior")
    parser.add_argument("--loss_function", default="gaussnll")
    parser.add_argument(
        "--min_y",
        type=float,
        help="barnll can only model y in strict ranges, this is the minimum y can take.",
    )
    parser.add_argument(
        "--max_y",
        type=float,
        help="barnll can only model y in strict ranges, this is the maximum y can take.",
    )
    parser.add_argument(
        "--num_features",
        default=None,
        type=int,
        help="Specify depending on the prior (can be None).",
    )
    parser.add_argument(
        "--extra_prior_kwargs_dict",
        default={},
        dest="extra_prior_kwargs_dict",
        action=StoreDictKeyPair,
        nargs="+",
        metavar="KEY=VAL",
        help="Specify depending on the prior.",
    )
    parser.add_argument(
        "--encoder", default="linear", type=str, help="Specify depending on the prior."
    )
    parser.add_argument(
        "--y_encoder",
        default="linear",
        type=str,
        help="Specify depending on the prior. You should specify this if you do not fuse x and y.",
    )
    parser.add_argument(
        "--pos_encoder",
        default="none",
        type=str,
        help="Specify depending on the prior.",
    )
    parser.add_argument("--bptt", default=10, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--warmup_epochs", default=3, type=int)
    parser.add_argument("--validation_period", default=10, type=int)
    parser.add_argument(
        "--permutation_invariant_max_eval_pos",
        default=None,
        type=int,
        help="Set this to an int to ",
    )
    parser.add_argument(
        "--permutation_invariant_sampling",
        default="weighted",
        help="Only relevant if --permutation_invariant_max_eval_pos is set.",
    )
    parser.add_argument("--train_mixed_precision", action="store_true")

    # these can likely be mostly left at defaults
    parser.add_argument(
        "--emsize", default=512, type=int
    )  # sometimes even larger is better e.g. 1024
    parser.add_argument("--nlayers", default=6, type=int)
    parser.add_argument("--nhid", default=None, type=int)  # 2*emsize is the default
    parser.add_argument(
        "--nhead", default=4, type=int
    )  # nhead = emsize / 64 in the original paper
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--steps_per_epoch", default=10, type=int)
    parser.add_argument("--batch_size", default=1000, type=int)
    parser.add_argument(
        "--lr", "--learning_rate", default=0.001, type=float
    )  # try also .0003, .0001, go lower with lower batch size

    args, _ = _parse_args(config_parser, parser)

    if args.nhid is None:
        args.nhid = 2 * args.emsize

    prior = args.__dict__.pop("prior")

    if prior == "gp":
        prior = priors.fast_gp.DataLoader
    elif prior == "ridge":
        prior = priors.ridge.DataLoader
    elif prior == "stroke":
        prior = priors.stroke.DataLoader
    elif prior == "mix_gp":
        prior = priors.fast_gp_mix.DataLoader
    else:
        raise NotImplementedError(f"Prior == {prior}.")

    loss_function = args.__dict__.pop("loss_function")

    criterion = nn.GaussianNLLLoss(reduction="none", full=True)
    classificiation_criterion = nn.CrossEntropyLoss(reduction="none")
    max_y = args.__dict__.pop("max_y")
    min_y = args.__dict__.pop("min_y")
    # criterion = nn.MSELoss(reduction='none')

    if loss_function == "ce":
        criterion = nn.CrossEntropyLoss(reduction="none")
    elif loss_function == "gaussnll":
        criterion = nn.GaussianNLLLoss(reduction="none", full=True)
    elif loss_function == "mse":
        criterion = nn.MSELoss(reduction="none")
    else:
        raise NotImplementedError(f"loss_function == {loss_function}.")

    encoder = args.__dict__.pop("encoder")
    y_encoder = args.__dict__.pop("y_encoder")

    def get_encoder_generator(encoder):
        if encoder == "linear":
            encoder_generator = encoders.Linear
        elif encoder == "mlp":
            encoder_generator = encoders.MLP
        elif encoder == "positional":
            encoder_generator = encoders.Positional
        else:
            raise NotImplementedError(f"A {encoder} encoder is not valid.")
        return encoder_generator

    encoder_generator = get_encoder_generator(encoder)
    y_encoder_generator = get_encoder_generator(y_encoder)

    pos_encoder = args.__dict__.pop("pos_encoder")

    if pos_encoder == "none":
        pos_encoder_generator = None
    elif pos_encoder == "sinus":
        pos_encoder_generator = positional_encodings.PositionalEncoding
    elif pos_encoder == "learned":
        pos_encoder_generator = positional_encodings.LearnedPositionalEncoding
    elif pos_encoder == "paired_scrambled_learned":
        pos_encoder_generator = positional_encodings.PairedScrambledPositionalEncodings
    else:
        raise NotImplementedError(f"pos_encoer == {pos_encoder} is not valid.")

    permutation_invariant_max_eval_pos = args.__dict__.pop(
        "permutation_invariant_max_eval_pos"
    )
    permutation_invariant_sampling = args.__dict__.pop("permutation_invariant_sampling")
    if permutation_invariant_max_eval_pos is not None:
        if permutation_invariant_sampling == "weighted":
            get_sampler = get_weighted_single_eval_pos_sampler
        elif permutation_invariant_sampling == "uniform":
            get_sampler = get_uniform_single_eval_pos_sampler
        else:
            raise ValueError()
        args.__dict__["single_eval_pos_gen"] = get_sampler(
            permutation_invariant_max_eval_pos
        )

    train(
        prior,
        criterion,
        encoder_generator,
        y_encoder_generator=y_encoder_generator,
        pos_encoder_generator=pos_encoder_generator,
        **args.__dict__,
    )
