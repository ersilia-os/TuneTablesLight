import torch
import random
import pathlib
import itertools
import sys
import os
import pickle
import io
import numpy as np
import warnings
import joblib
import json
import glob
import shutil

sys.path.append("../")

from torch.utils.checkpoint import checkpoint
from tunetables_light.train_loop import reload_config, train_function
from tunetables_light.utils import (
    normalize_data,
    to_ranking_low_mem,
    remove_outliers,
    normalize_by_used_features_f,
)
from tunetables_light.scripts.model_builder import (
    load_model,
    load_model_only_inference,
    get_model,
)
from tunetables_light.train import real_data_eval_out
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import column_or_1d
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from pathlib import Path
from tqdm.auto import tqdm


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Manager":
            from settings import Manager

            return Manager
        try:
            return self.find_class_cpu(module, name)
        except:
            return None

    def find_class_cpu(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def check_file(model_base_path, model_dir, file_name):
    model_path = os.path.join(model_base_path, model_dir)
    file_path = os.path.join(model_path, file_name)
    if not Path(file_path).is_file():
        import requests

        print("No checkpoint found at", file_path)
        print("Downloading Base TabPFN checkpoint (~100 MB)…")

        os.makedirs(model_path, exist_ok=True)
        url = (
            "https://raw.githubusercontent.com/"
            "Abellegese/TuneTables/main/"
            "tunetables_light/models/"
            "prior_diff_real_checkpoint_n_0_epoch_42.cpkt"
        )
        r = requests.get(url, allow_redirects=True)
        r.raise_for_status()

        with open(file_path, "wb") as f:
            f.write(r.content)

    return file_path


def load_model_workflow(
    i,
    e,
    add_name,
    base_path,
    device="cpu",
    eval_addition="",
    only_inference=True,
    prefix_size=0,
    n_classes=2,
):
    def get_file(e):
        model_file = (
            f"models_diff/prior_diff_real_checkpoint{add_name}_n_{i}_epoch_{e}.cpkt"
        )
        model_path = os.path.join(base_path, model_file)
        results_file = os.path.join(
            base_path,
            f"models_diff/prior_diff_real_results{add_name}_n_{i}_epoch_{e}_{eval_addition}.pkl",
        )
        return model_file, model_path, results_file

    def check_file(e):
        model_file, model_path, results_file = get_file(e)
        if not Path(model_path).is_file() and e == -1:
            print(
                "We have to download the TabPFN, as there is no checkpoint at ",
                model_path,
            )
            print("It has about 100MB, so this might take a moment.")
            import requests

            url = "https://github.com/automl/TabPFN/raw/main/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt"
            r = requests.get(url, allow_redirects=True)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            open(model_path, "wb").write(r.content)
        elif not Path(model_path).is_file():
            return None, None, None
        return model_file, model_path, results_file

    model_file = None
    if e == -1:
        for e_ in range(100, -2, -1):
            model_file_, model_path_, results_file_ = check_file(e_)
            if model_file_ is not None:
                e = e_
                model_file, model_path, results_file = (
                    model_file_,
                    model_path_,
                    results_file_,
                )
                print(f"Loading model {model_path}")
                break
    else:
        raise Exception("Not implemented")

    if model_file is None:
        model_file, model_path, results_file = get_file(e)
        raise Exception("No checkpoint found at " + str(model_path))

    if only_inference:
        model, c = load_model_only_inference(
            base_path, model_file, device, prefix_size=prefix_size, n_classes=n_classes
        )
    else:
        model, c = load_model(
            base_path, model_file, device, eval_positions=[], verbose=False
        )

    return model, c, results_file


class TabPFNClassifier(BaseEstimator, ClassifierMixin):
    models_in_memory = {}

    def __init__(
        self,
        device="cpu",
        base_path=pathlib.Path(__file__).parent.parent.resolve(),
        model_string="",
        N_ensemble_configurations=3,
        no_preprocess_mode=False,
        multiclass_decoder="permutation",
        feature_shift_decoder=True,
        only_inference=True,
        seed=0,
        no_grad=True,
        batch_size_inference=4,
        subsample_features=False,
        prefix_size=0,
        use_memory=False,
        n_classes=2,
    ):
        i = 0
        model_key = model_string + "|" + str(device)
        if model_key in self.models_in_memory and use_memory:
            print("Loading model from memory")
            model, c, results_file = self.models_in_memory[model_key]
        else:
            model, c, results_file = load_model_workflow(
                i,
                -1,
                add_name=model_string,
                base_path=base_path,
                device=device,
                eval_addition="",
                only_inference=only_inference,
                prefix_size=prefix_size,
                n_classes=n_classes,
            )
            if use_memory:
                self.models_in_memory[model_key] = (model, c, results_file)
                if len(self.models_in_memory) == 2:
                    print(
                        "Multiple models in memory. This might lead to memory issues. Consider calling remove_models_from_memory()"
                    )
        self.device = device
        self.model = model
        self.c = c
        self.style = None
        self.temperature = None
        self.N_ensemble_configurations = N_ensemble_configurations
        self.base__path = base_path
        self.base_path = base_path
        self.i = i
        self.model_string = model_string

        self.max_num_features = self.c["num_features"]
        self.max_num_classes = self.c["max_num_classes"]
        self.differentiable_hps_as_style = self.c["differentiable_hps_as_style"]

        self.no_preprocess_mode = no_preprocess_mode
        self.feature_shift_decoder = feature_shift_decoder
        self.multiclass_decoder = multiclass_decoder
        self.only_inference = only_inference
        self.seed = seed
        self.no_grad = no_grad
        self.subsample_features = subsample_features
        self.prefix_size = prefix_size
        self.num_classes = n_classes

        assert self.no_preprocess_mode if not self.no_grad else True, (
            "If no_grad is false, no_preprocess_mode must be true, because otherwise no gradient can be computed."
        )

        self.batch_size_inference = batch_size_inference

    def remove_models_from_memory(self):
        self.models_in_memory = {}

    def load_result_minimal(self, path, i, e):
        with open(path, "rb") as output:
            _, _, _, style, temperature, optimization_route = CustomUnpickler(
                output
            ).load()

            return style, temperature

    def _validate_targets(self, y):
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        cls, y = np.unique(y_, return_inverse=True)
        if len(cls) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d class"
                % len(cls)
            )

        self.classes_ = cls

        return np.asarray(y, dtype=np.float64, order="C")

    def fit(self, X, y, overwrite_warning=False):
        if self.no_grad:
            X, y = check_X_y(X, y, force_all_finite=False)
        y = self._validate_targets(y)
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        self.X_ = X
        self.y_ = y

        if X.shape[1] > self.max_num_features:
            if self.subsample_features:
                print(
                    "WARNING: The number of features for this classifier is restricted to ",
                    self.max_num_features,
                    " and will be subsampled.",
                )
            else:
                raise ValueError(
                    "The number of features for this classifier is restricted to ",
                    self.max_num_features,
                )
        if len(np.unique(y)) > self.max_num_classes:
            raise ValueError(
                "The number of classes for this classifier is restricted to ",
                self.max_num_classes,
            )
        if X.shape[0] > 1024 and not overwrite_warning:
            raise ValueError(
                "WARNING: TabPFN is not made for datasets with a trainingsize > 1024. \Prediction might take a while, be less reliable. We advise not to run datasets > 10k samples, which might lead to your machine crashing (due to quadratic memory scaling of TabPFN). Please confirm you want to run by passing overwrite_warning=True to the fit function."
            )

        return self

    def predict_proba(
        self, X, normalize_with_test=False, return_logits=False, return_early=False
    ):
        check_is_fitted(self)

        if self.no_grad:
            X = check_array(X, force_all_finite=False)
            X_full = np.concatenate([self.X_, X], axis=0)
            X_full = torch.tensor(X_full, device=self.device).float().unsqueeze(1)
        else:
            assert torch.is_tensor(self.X_) & torch.is_tensor(X), (
                "If no_grad is false, this function expects X as "
                "a tensor to calculate a gradient"
            )
            X_full = torch.cat((self.X_, X), dim=0).float().unsqueeze(1).to(self.device)

            if int(torch.isnan(X_full).sum()):
                print(
                    "X contains nans and the gradient implementation is not designed to handel nans."
                )

        y_full = np.concatenate([self.y_, np.zeros(shape=X.shape[0])], axis=0)
        y_full = torch.tensor(y_full, device=self.device).float().unsqueeze(1)

        eval_pos = self.X_.shape[0]
        prediction = transformer_predict(
            self.model[2],
            X_full,
            y_full,
            eval_pos,
            device=self.device,
            style=self.style,
            inference_mode=True,
            preprocess_transform="none" if self.no_preprocess_mode else "mix",
            normalize_with_test=normalize_with_test,
            N_ensemble_configurations=self.N_ensemble_configurations,
            softmax_temperature=self.temperature,
            multiclass_decoder=self.multiclass_decoder,
            feature_shift_decoder=self.feature_shift_decoder,
            differentiable_hps_as_style=self.differentiable_hps_as_style,
            seed=self.seed,
            return_logits=return_logits,
            no_grad=self.no_grad,
            batch_size_inference=self.batch_size_inference,
            return_early=return_early,
            **get_params_from_config(self.c),
        )
        prediction_, y_ = prediction.squeeze(0), y_full.squeeze(1).long()[eval_pos:]

        return prediction_.detach().cpu().numpy() if self.no_grad else prediction_

    def predict(
        self,
        X,
        return_winning_probability=False,
        normalize_with_test=False,
        return_early=False,
    ):
        p = self.predict_proba(
            X, normalize_with_test=normalize_with_test, return_early=return_early
        )
        y = np.argmax(p, axis=-1)
        y = self.classes_.take(np.asarray(y, dtype=np.intp))
        if return_winning_probability:
            return y, p.max(axis=-1)
        return y


import time


def transformer_predict(
    model,
    eval_xs,
    eval_ys,
    eval_position,
    device="cpu",
    max_features=100,
    style=None,
    inference_mode=False,
    num_classes=2,
    extend_features=True,
    normalize_with_test=False,
    normalize_to_ranking=False,
    softmax_temperature=0.0,
    multiclass_decoder="permutation",
    preprocess_transform="mix",
    categorical_feats=[],
    feature_shift_decoder=False,
    N_ensemble_configurations=10,
    batch_size_inference=16,
    differentiable_hps_as_style=False,
    average_logits=True,
    fp16_inference=False,
    normalize_with_sqrt=False,
    seed=0,
    no_grad=True,
    return_logits=False,
    return_early=False,
    **kwargs,
):
    num_classes = len(torch.unique(eval_ys))

    def predict(eval_xs, eval_ys, used_style, softmax_temperature, return_logits):
        inference_mode_call = (
            torch.inference_mode() if inference_mode and no_grad else NOP()
        )
        with inference_mode_call:
            start = time.time()
            output = model(
                (
                    used_style.repeat(eval_xs.shape[1], 1)
                    if used_style is not None
                    else None,
                    eval_xs,
                    eval_ys.float(),
                ),
                single_eval_pos=eval_position,
            )[:, :, 0:num_classes]

            output = output[:, :, 0:num_classes] / torch.exp(softmax_temperature)
            if not return_logits:
                output = torch.nn.functional.softmax(output, dim=-1)
        return output

    def preprocess_input(eval_xs, preprocess_transform):
        if eval_xs.shape[1] > 1:
            raise Exception("Transforms only allow one batch dim - TODO")

        if eval_xs.shape[2] > max_features:
            eval_xs = eval_xs[
                :,
                :,
                sorted(np.random.choice(eval_xs.shape[2], max_features, replace=False)),
            ]

        if preprocess_transform != "none":
            if preprocess_transform == "power" or preprocess_transform == "power_all":
                pt = PowerTransformer(standardize=True)
            elif (
                preprocess_transform == "quantile"
                or preprocess_transform == "quantile_all"
            ):
                pt = QuantileTransformer(output_distribution="normal")
            elif (
                preprocess_transform == "robust" or preprocess_transform == "robust_all"
            ):
                pt = RobustScaler(unit_variance=True)

        eval_xs = normalize_data(
            eval_xs, normalize_positions=-1 if normalize_with_test else eval_position
        )

        eval_xs = eval_xs[:, 0, :]
        sel = [
            len(torch.unique(eval_xs[0 : eval_ys.shape[0], col])) > 1
            for col in range(eval_xs.shape[1])
        ]
        eval_xs = eval_xs[:, sel]

        warnings.simplefilter("error")
        if preprocess_transform != "none":
            eval_xs = eval_xs.cpu().numpy()
            feats = (
                set(range(eval_xs.shape[1]))
                if "all" in preprocess_transform
                else set(range(eval_xs.shape[1])) - set(categorical_feats)
            )
            for col in feats:
                try:
                    pt.fit(eval_xs[0:eval_position, col : col + 1])
                    trans = pt.transform(eval_xs[:, col : col + 1])
                    eval_xs[:, col : col + 1] = trans
                except:
                    pass
            eval_xs = torch.tensor(eval_xs).float()
        warnings.simplefilter("default")

        eval_xs = eval_xs.unsqueeze(1)

        eval_xs = (
            remove_outliers(
                eval_xs,
                normalize_positions=-1 if normalize_with_test else eval_position,
            )
            if not normalize_to_ranking
            else normalize_data(to_ranking_low_mem(eval_xs))
        )
        eval_xs = normalize_by_used_features_f(
            eval_xs,
            eval_xs.shape[-1],
            max_features,
            normalize_with_sqrt=normalize_with_sqrt,
        )

        return eval_xs.to(device)

    eval_xs, eval_ys = eval_xs.to(device), eval_ys.to(device)
    eval_ys = eval_ys[:eval_position]

    model.to(device)

    model.eval()

    if not differentiable_hps_as_style:
        style = None

    if style is not None:
        style = style.to(device)
        style = style.unsqueeze(0) if len(style.shape) == 1 else style
        num_styles = style.shape[0]
        softmax_temperature = (
            softmax_temperature
            if softmax_temperature.shape
            else softmax_temperature.unsqueeze(0).repeat(num_styles)
        )
    else:
        num_styles = 1
        style = None
        softmax_temperature = torch.log(torch.tensor([0.8]))

    styles_configurations = range(0, num_styles)

    def get_preprocess(i):
        if i == 0:
            return "power_all"
        if i == 1:
            return "none"

    preprocess_transform_configurations = (
        ["none", "power_all"]
        if preprocess_transform == "mix"
        else [preprocess_transform]
    )

    if seed is not None:
        torch.manual_seed(seed)

    feature_shift_configurations = (
        torch.randperm(eval_xs.shape[2]) if feature_shift_decoder else [0]
    )
    class_shift_configurations = (
        torch.randperm(len(torch.unique(eval_ys)))
        if multiclass_decoder == "permutation"
        else [0]
    )

    ensemble_configurations = list(
        itertools.product(class_shift_configurations, feature_shift_configurations)
    )
    rng = random.Random(seed)
    rng.shuffle(ensemble_configurations)
    ensemble_configurations = list(
        itertools.product(
            ensemble_configurations,
            preprocess_transform_configurations,
            styles_configurations,
        )
    )
    ensemble_configurations = ensemble_configurations[0:N_ensemble_configurations]

    output = None
    eval_xs_transformed = {}
    inputs, labels = [], []
    start = time.time()
    for ensemble_configuration in ensemble_configurations:
        (
            (class_shift_configuration, feature_shift_configuration),
            preprocess_transform_configuration,
            styles_configuration,
        ) = ensemble_configuration

        style_ = (
            style[styles_configuration : styles_configuration + 1, :]
            if style is not None
            else style
        )
        softmax_temperature_ = softmax_temperature[styles_configuration]

        eval_xs_, eval_ys_ = eval_xs.clone(), eval_ys.clone()

        if preprocess_transform_configuration in eval_xs_transformed:
            eval_xs_ = eval_xs_transformed[preprocess_transform_configuration].clone()
        else:
            eval_xs_ = preprocess_input(
                eval_xs_, preprocess_transform=preprocess_transform_configuration
            )
            if no_grad:
                eval_xs_ = eval_xs_.detach()
            eval_xs_transformed[preprocess_transform_configuration] = eval_xs_

        eval_ys_ = ((eval_ys_ + class_shift_configuration) % num_classes).float()

        eval_xs_ = torch.cat(
            [
                eval_xs_[..., feature_shift_configuration:],
                eval_xs_[..., :feature_shift_configuration],
            ],
            dim=-1,
        )

        if extend_features:
            eval_xs_ = torch.cat(
                [
                    eval_xs_,
                    torch.zeros((
                        eval_xs_.shape[0],
                        eval_xs_.shape[1],
                        max_features - eval_xs_.shape[2],
                    )).to(device),
                ],
                -1,
            )
        inputs += [eval_xs_]
        labels += [eval_ys_]

    inputs = torch.cat(inputs, 1)
    inputs = torch.split(inputs, batch_size_inference, dim=1)
    labels = torch.cat(labels, 1)
    labels = torch.split(labels, batch_size_inference, dim=1)
    outputs = []

    for batch_input, batch_label in zip(inputs, labels):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="None of the inputs have requires_grad=True. Gradients will be None",
            )
            warnings.filterwarnings(
                "ignore",
                message="torch.cuda.amp.autocast only affects CUDA ops, but CUDA is not available.  Disabling.",
            )
            if device == "cpu":
                output_batch = checkpoint(
                    predict,
                    batch_input,
                    batch_label,
                    style_,
                    softmax_temperature_,
                    True,
                    use_reentrant=True,
                )
            else:
                with torch.amp.autocast(device_type=device, enabled=fp16_inference):
                    output_batch = checkpoint(
                        predict,
                        batch_input,
                        batch_label,
                        style_,
                        softmax_temperature_,
                        True,
                        use_reentrant=True,
                    )
        outputs += [output_batch]

    outputs = torch.cat(outputs, 1)
    if return_early:
        return outputs

    for i, ensemble_configuration in enumerate(
        tqdm(
            ensemble_configurations,
            desc="Running Inference",
            unit="batch",
            colour="blue",
            ncols=80,
        )
    ):
        (
            (class_shift_configuration, feature_shift_configuration),
            preprocess_transform_configuration,
            styles_configuration,
        ) = ensemble_configuration
        output_ = outputs[:, i : i + 1, :]
        output_ = torch.cat(
            [
                output_[..., class_shift_configuration:],
                output_[..., :class_shift_configuration],
            ],
            dim=-1,
        )

        if not average_logits and not return_logits:
            output_ = torch.nn.functional.softmax(output_, dim=-1)
        output = output_ if output is None else output + output_

    output = output / len(ensemble_configurations)
    if average_logits and not return_logits:
        output = torch.nn.functional.softmax(output, dim=-1)

    output = torch.transpose(output, 0, 1)

    return output


def get_params_from_config(c):
    return {
        "max_features": c["num_features"],
        "rescale_features": c.get("normalize_by_used_features", True),
        "normalize_to_ranking": c["normalize_to_ranking"],
        "normalize_with_sqrt": c.get("normalize_with_sqrt", False),
    }


class TuneTablesClassifierLight(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        device="cpu",
        epoch=10,
        batch_size=16,
        lr=0.1,
        dropout=0.2,
        tuned_prompt_size = 10,
        early_stopping=2,
        prompt_tuning=True,
        no_preprocess_mode=False,
        no_grad=True,
        boosting=False,
        bagging=False, 
        ensemble_size=5,
        average_ensemble=False

    ):
        self.boosting = boosting
        self.dropout = dropout
        self.average_ensemble = average_ensemble
        self.ensemble_size = ensemble_size
        self.bagging = bagging
        self.early_stopping = early_stopping
        self.prompt_tuning = prompt_tuning
        self.tuned_prompt_size = tuned_prompt_size
        self.batch_size_inference = 1
        self.no_preprocess_mode = no_preprocess_mode
        self.no_grad = no_grad
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.base_path = pathlib.Path(__file__).parent.parent.resolve()
        self.model_base_path = pathlib.Path(__file__).parent.parent.resolve()
        self.log_path = os.path.join(self.base_path, "logs")
        self.pretrained_model_file = os.path.join(
            self.model_base_path,
            "models",
            "prior_diff_real_checkpoint_n_0_epoch_42.cpkt",
        )
        check_file(
            self.model_base_path,
            "models",
            "prior_diff_real_checkpoint_n_0_epoch_42.cpkt",
        )
        self.device = device
        self.classes_ = 2
        self.model_file = self.pretrained_model_file

        class Args:
            pass

        self.args = Args()

        self.args = self.get_default_config(self.args)

        self.config, self.model_string = reload_config(config_type="real", longer=1, args=self.args)
        self.config["wandb_log"] = False

        import ConfigSpace

        for k, v in self.config.items():
            if isinstance(v, ConfigSpace.hyperparameters.CategoricalHyperparameter):
                self.config[k] = v.default_value

    def get_default_config(self, args):
        args.resume = self.pretrained_model_file
        args.save_path = self.log_path
        args.prior_type = "real"
        args.data_path = ""
        args.dropout = self.dropout
        args.prompt_tuning = True
        args.tuned_prompt_size = self.tuned_prompt_size
        args.tuned_prompt_label_balance = "equal"
        args.lr = self.lr
        args.batch_size = self.batch_size
        args.bptt = 1152
        args.uniform_bptt = False
        args.seed = 42
        args.early_stopping = self.early_stopping
        args.epochs = self.epoch
        args.num_eval_fitting_samples = 1000
        args.split = 0
        args.boosting = self.boosting
        args.bagging = self.bagging
        args.bptt_search = False
        args.workers = 4
        args.val_subset_size = 1000000
        args.subset_features = 100
        args.subsampling = 0
        args.rand_init_ensemble = False
        args.ensemble_lr = 0.5
        args.ensemble_size = self.ensemble_size
        args.reseed_data = False
        args.aggregate_k_gradients = 1
        args.average_ensemble = self.average_ensemble
        args.permute_feature_position_in_ensemble = False
        args.concat_method = ""
        args.save_every_k_epochs = 2
        args.validation_period = 3
        args.wandb_name = "tabpfn_pt_airlines"
        args.wandb_log = False
        args.wandb_group = "openml__colic__27_pt10_rdq_0_split_0"
        args.wandb_project = "tabpfn-pt"
        args.wandb_entity = "nyu-dice-lab"
        args.subset_features_method = "pca"
        args.pad_features = True
        args.do_preprocess = True
        args.zs_eval_ensemble = 0
        args.min_batches_per_epoch = 1
        args.keep_topk_ensemble = 0
        args.topk_key = "Val_Accuracy"
        args.max_time = 36000
        args.preprocess_type = "none"
        args.optuna_objective = "Val_Accuracy"
        args.verbose = True
        args.shuffle_every_epoch = False
        args.max_num_classes = 10
        args.real_data_qty = 0
        args.summerize_after_prep = False
        args.kl_loss = False
        args.private_model = False
        args.private_data = False
        args.edg = ["50", "1e-4", "1.2"]

        return args

    def _fit_only_prefitted(self, X, y):
        model, c = load_model_only_inference(
            self.base_path, self.model_file, self.device, prefix_size=10, n_classes=2
        )
        return model[2], c

    def _fit(self, X, y):
        model, data_for_fitting, _ = train_function(
            self.config.copy(),
            0,
            self.model_string,
            is_wrapper=True,
            x_wrapper=X,
            y_wrapper=y,
            cat_idx=[],
        )

        return model, data_for_fitting

    def fit(self, X, y):
        x = X
        assert isinstance(x, np.ndarray), "x must be a numpy array"
        assert isinstance(y, np.ndarray), "x must be a numpy array"
        assert len(x.shape) == 2, "x must be a 2D array (samples, features)"
        assert len(y.shape) == 1, "y must be a 1D array"

        self.model, self.data_for_fitting = self._fit(X, y)
        self.eval_pos = self.data_for_fitting[0].shape[0]
        self.num_classes = len(np.unique(y))

        self._x_train = x
        self._y_train = y


    def predict_proba(
        self, X, normalize_with_test=False, return_logits=False, return_early=False
    ):
        self.no_grad = True
        self.model, self.c = self._fit_only_prefitted(self._x_train, self._y_train)

        if self.no_grad:
            X = check_array(X, force_all_finite=False)
            X_full = np.concatenate([self._x_train, X], axis=0)
            X_full = torch.tensor(X_full, device=self.device).float().unsqueeze(1)
        else:
            assert torch.is_tensor(self._x_train) & torch.is_tensor(X), (
                "If no_grad is false, this function expects X as "
                "a tensor to calculate a gradient"
            )
            X_full = (
                torch.cat((self._x_train, X), dim=0)
                .float()
                .unsqueeze(1)
                .to(self.device)
            )

            if int(torch.isnan(X_full).sum()):
                print(
                    "X contains nans and the gradient implementation is not designed to handel nans."
                )

        y_full = np.concatenate([self._y_train, np.zeros(shape=X.shape[0])], axis=0)
        y_full = torch.tensor(y_full, device=self.device).float().unsqueeze(1)

        eval_pos = self._x_train.shape[0]
        prediction = transformer_predict(
            self.model,
            X_full,
            y_full,
            eval_pos,
            device=self.device,
            inference_mode=True,
            preprocess_transform="none" if self.no_preprocess_mode else "mix",
            normalize_with_test=normalize_with_test,
            softmax_temperature=0,
            return_logits=return_logits,
            no_grad=self.no_grad,
            batch_size_inference=self.batch_size_inference,
            return_early=return_early,
            **get_params_from_config(self.c),
        )
        prediction_, y_ = prediction.squeeze(0), y_full.squeeze(1).long()[eval_pos:]

        return prediction_.detach().cpu().numpy() if self.no_grad else prediction_

    def predict_(
        self,
        X,
        return_winning_probability=False,
        normalize_with_test=False,
        return_early=False,
    ):
        p = self.predict_proba(
            X, normalize_with_test=normalize_with_test, return_early=return_early
        )
        y = np.argmax(p, axis=-1)
        y = self.classes_.take(np.asarray(y, dtype=np.intp))
        if return_winning_probability:
            return y, p.max(axis=-1)
        return y

    def save_model(self, model_dir: str):
        def copy_ckpts(src_dir, dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
            files = glob.glob(os.path.join(src_dir, "*.cpkt"))
            if files:
                latest_file = max(files, key=os.path.getmtime)
                dst_file = os.path.join(dst_dir, "model.cpkt")
                shutil.copy(latest_file, dst_file)
            else:
                print("No .ckpt files found.")

            npy_files = glob.glob(os.path.join(src_dir, "*.npy"))
            for npy_file in npy_files:
                shutil.copy(npy_file, dst_dir)

        args = self.args
        config = self.config

        other_metadata = {
            "model_string": self.model_string,
            "eval_pos": self.eval_pos,
            "num_classes": self.num_classes,
            "device": self.device,
            "pretrained_model_file": self.pretrained_model_file,
        }

        data = {
            "x_train": self._x_train,
            "y_train": self._y_train,
            "data_for_fitting": self.data_for_fitting,
        }
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(
                {"args": args, "config": config, "other": other_metadata},
                f,
                default=str,
            )
        joblib.dump(data, os.path.join(model_dir, "training_data.joblib"))

        copy_ckpts(self.log_path, model_dir)

    @classmethod
    def load_model(cls, model_dir: str):
        obj = cls()
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        obj.args = metadata["args"]
        obj.config = metadata["config"]
        obj.model_string = metadata["other"]["model_string"]
        obj.num_classes = metadata["other"]["num_classes"]
        obj.eval_pos = metadata["other"]["eval_pos"]
        obj.device = metadata["other"]["device"]
        data = joblib.load(os.path.join(model_dir, "training_data.joblib"))
        obj._x_train = data["x_train"]
        obj._y_train = data["y_train"]
        obj.data_for_fitting = data["data_for_fitting"]

        obj.model_file = os.path.join(model_dir, "model.cpkt")
        return obj
