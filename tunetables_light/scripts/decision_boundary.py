import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from functools import reduce

import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_matplotlib_support
from sklearn.utils import _safe_indexing
from sklearn.base import is_regressor
from sklearn.utils.validation import check_is_fitted


def _check_boundary_response_method(estimator, response_method):
    has_classes = hasattr(estimator, "classes_")
    if has_classes and len(estimator.classes_) > 2:
        if response_method not in {"auto", "predict"}:
            msg = (
                "Multiclass classifiers are only supported when response_method is"
                " 'predict' or 'auto'"
            )
            raise ValueError(msg)
        methods_list = ["predict"]
    elif response_method == "auto":
        methods_list = ["decision_function", "predict_proba", "predict"]
    else:
        methods_list = [response_method]

    prediction_method = [getattr(estimator, method, None) for method in methods_list]
    prediction_method = reduce(lambda x, y: x or y, prediction_method)
    if prediction_method is None:
        raise ValueError(
            f"{estimator.__class__.__name__} has none of the following attributes: "
            f"{', '.join(methods_list)}."
        )

    return prediction_method


class DecisionBoundaryDisplay:
    def __init__(self, *, xx0, xx1, response, xlabel=None, ylabel=None):
        self.xx0 = xx0
        self.xx1 = xx1
        self.response = response
        self.xlabel = xlabel
        self.ylabel = ylabel

    def plot(self, plot_method="contourf", ax=None, xlabel=None, ylabel=None, **kwargs):
        check_matplotlib_support("DecisionBoundaryDisplay.plot")
        import matplotlib.pyplot as plt  # noqa

        if plot_method not in ("contourf", "contour", "pcolormesh"):
            raise ValueError(
                "plot_method must be 'contourf', 'contour', or 'pcolormesh'"
            )

        if ax is None:
            _, ax = plt.subplots()

        plot_func = getattr(ax, plot_method)
        self.surface_ = plot_func(self.xx0, self.xx1, self.response, **kwargs)

        if xlabel is not None or not ax.get_xlabel():
            xlabel = self.xlabel if xlabel is None else xlabel
            ax.set_xlabel(xlabel)
        if ylabel is not None or not ax.get_ylabel():
            ylabel = self.ylabel if ylabel is None else ylabel
            ax.set_ylabel(ylabel)

        self.ax_ = ax
        self.figure_ = ax.figure
        return self

    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        *,
        grid_resolution=100,
        eps=1.0,
        plot_method="contourf",
        response_method="auto",
        xlabel=None,
        ylabel=None,
        ax=None,
        **kwargs,
    ):
        check_matplotlib_support(f"{cls.__name__}.from_estimator")
        check_is_fitted(estimator)

        if not grid_resolution > 1:
            raise ValueError(
                f"grid_resolution must be greater than 1. Got {grid_resolution} instead."
            )

        if not eps >= 0:
            raise ValueError(
                f"eps must be greater than or equal to 0. Got {eps} instead."
            )

        possible_plot_methods = ("contourf", "contour", "pcolormesh")
        if plot_method not in possible_plot_methods:
            available_methods = ", ".join(possible_plot_methods)
            raise ValueError(
                f"plot_method must be one of {available_methods}. Got {plot_method} instead."
            )

        x0, x1 = _safe_indexing(X, 0, axis=1), _safe_indexing(X, 1, axis=1)

        x0_min, x0_max = x0.min() - eps, x0.max() + eps
        x1_min, x1_max = x1.min() - eps, x1.max() + eps

        xx0, xx1 = np.meshgrid(
            np.linspace(x0_min, x0_max, grid_resolution),
            np.linspace(x1_min, x1_max, grid_resolution),
        )
        if hasattr(X, "iloc"):
            X_grid = X.iloc[[], :].copy()
            X_grid.iloc[:, 0] = xx0.ravel()
            X_grid.iloc[:, 1] = xx1.ravel()
        else:
            X_grid = np.c_[xx0.ravel(), xx1.ravel()]

        pred_func = _check_boundary_response_method(estimator, response_method)
        response = pred_func(X_grid)

        if pred_func.__name__ == "predict" and hasattr(estimator, "classes_"):
            encoder = LabelEncoder()
            encoder.classes_ = estimator.classes_
            response = encoder.transform(response)

        if response.ndim != 1:
            if is_regressor(estimator):
                raise ValueError("Multi-output regressors are not supported")

            # TODO: Support pos_label
            response = response[:, 1]

        if xlabel is None:
            xlabel = X.columns[0] if hasattr(X, "columns") else ""

        if ylabel is None:
            ylabel = X.columns[1] if hasattr(X, "columns") else ""

        display = DecisionBoundaryDisplay(
            xx0=xx0,
            xx1=xx1,
            response=response.reshape(xx0.shape),
            xlabel=xlabel,
            ylabel=ylabel,
        )
        return display.plot(ax=ax, plot_method=plot_method, **kwargs)
