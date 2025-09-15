"""
WandB or Neptune logging helpers
"""
import os
from argparse import Namespace
from copy import deepcopy
from typing import Any, Callable

import numpy as np

from omegaconf import OmegaConf, DictConfig


class OurLogger():
    """
    Thin wrapper around neptune.Run or wandb.Run
    - Mainly used for clarity around neptune.Run tracking
    """
    def __init__(self, logger) -> None:
        self.logger = logger
        self.step_count = 0

    def get_reduce_fn(self, reduction: str) -> Callable:
        """
        Get the reduction function
        """
        match reduction:
            case "mean":
                return np.mean
            case "sum":
                return np.sum
            case "max":
                return np.max
            case "min":
                return np.min
            case _:
                raise ValueError(f"Invalid reduction: {reduction}")

    def log(
        self,
        metrics: dict[str, list[int | float]],
        step: int | None = None,
        reduction: str = "mean",
        bin_metric: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Log iterable metrics
        """
        if step is None:
            step = self.step_count
        logging_metrics = {}

        reduce_fn = self.get_reduce_fn(reduction)

        # if bin_metric is not None:
        #     bin_name = bin_metric.split("/")[-1]
        #     bin_ids = np.unique(metrics[bin_metric])
        #     bin_values = np.array(metrics[bin_metric])
        #     for key, value in metrics.items():
        #         if key == bin_metric:
        #             continue
        #         for bin_id in bin_ids:
        #             val_by_bin = reduce_fn(np.array(value)[bin_values == bin_id])
        #             metric_name = f"{key}/{bin_name}_{bin_id}"
        #             logging_metrics[metric_name] = val_by_bin
        # else:
        #     logging_metrics = {key: reduce_fn(val) for key, val in metrics.items()}
        if bin_metric is not None:
            bin_name = bin_metric.split("/")[-1]
            bin_ids = np.unique(metrics[bin_metric])
            bin_values = np.array(metrics[bin_metric])
            for key, value in metrics.items():
                if isinstance(value, list):
                    if key == bin_metric:
                        continue
                    for bin_id in bin_ids:
                        val_by_bin = reduce_fn(np.array(value)[bin_values == bin_id])
                        metric_name = f"{key}/{bin_name}_{bin_id}"
                        logging_metrics[metric_name] = val_by_bin
                else:
                    logging_metrics[key] = value
        else:
            logging_metrics = {
                key: reduce_fn(val) if isinstance(val, list) else val
                for key, val in metrics.items()
            }
        self.logger.log(logging_metrics, step=step, **kwargs)
        self.step_count += 1

    def finish(self) -> None:
        """
        Finish logging
        """
        self.logger.finish()

    def get_url(self) -> str:
        """
        Get the URL of the logger
        """
        try:
            return self.logger.get_url()
        except AttributeError:
            return self.logger.url


def flatten_dict(
    d: dict[str, Any] | DictConfig,
    parent_key: str = "",
    sep: str = "_",
) -> dict[str, Any]:
    """
    Recursively flattens a nested dict.
    """
    items = {}
    for key, val in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(val, dict):
            # recurse into sub-dict
            items.update(flatten_dict(val, new_key, sep=sep))
        else:
            items[new_key] = val
    return items


def init_logger(args: Namespace, **kwargs: Any) -> OurLogger | None:
    """
    Initialize logger (either WandB or Neptune)
    """

    logging_kwargs = deepcopy(kwargs)
    for k, v in logging_kwargs.items():
        if isinstance(v, DictConfig):
            logging_kwargs[k] = flatten_dict(OmegaConf.to_container(v))

    if "WANDB_API_KEY" in os.environ:
        args.logger = 'wandb'
        print(f"-> Defaulting to WandB logging at {args.run_name}")
        return OurLogger(init_wandb(args, **logging_kwargs))

    if "NEPTUNE_API_TOKEN" in os.environ:
        args.logger = 'neptune'
        logger = init_neptune(args, **logging_kwargs)
        logger.log = logger.log_metrics  # Set same api as WandB
        logger.finish = logger.close
        logger.get_url = logger.get_run_url
        print(f"-> Using Neptune logging at {args.run_name}")
        return OurLogger(logger)

    # No logger initialized
    print(
        "-> No logger initialized. If you want to log, please set "
        "NEPTUNE_API_TOKEN or WANDB_API_KEY in .env file"
    )
    return None


def init_wandb(
    args: Namespace,
    model_config: dict | None = None,
    trainer_config: dict | None = None,
    **other_config: Any,
) -> Any:
    """Initialize WandB"""
    if args.no_wandb:
        return None

    import wandb
    _attrs = [a for a in dir(args) if a[0] != "_"]
    config = {a: getattr(args, a) for a in _attrs}

    # Hacky logging
    if model_config is not None:
        for k, v in model_config["model_config"].items():
            config["model_" + k] = v

    if trainer_config is not None:
        for k, v in trainer_config.items():
            if isinstance(v, dict):
                for _k, _v in v.items():
                    config[f"{k}_{_k}"] = _v
            else:
                config[k] = v

    # Add any other attributes
    for k, v in other_config.items():
        config[k] = v

    wandb.init(
        config=config,
        entity=args.wandb_entity,
        name=args.run_name,
        project=args.project_name,
    )
    return wandb


def init_neptune(
    args: Namespace,
    model_config: dict | None = None,
    trainer_config: dict | None = None,
    **other_config: Any,
) -> Any:
    """Initialize Neptune"""
    if args.no_neptune:
        return None

    # Start the run
    from neptune_scale import Run as NeptuneRun
    run = NeptuneRun(
        experiment_name=args.run_name,
        project=f"{args.logger_entity}/{args.project_name}",  # logger_entity should be "workspace-name"
        api_token=os.environ["NEPTUNE_API_TOKEN"],
        tags=getattr(args, "tags", None),  # if we track tags
    )
    _attrs = [a for a in dir(args) if not a.startswith("_")]
    config = {a: getattr(args, a) for a in _attrs}

    if model_config:
        for k, v in model_config.get("model_config", {}).items():
            config[f"model_{k}"] = v

    if trainer_config:
        for k, v in trainer_config.items():
            if isinstance(v, dict):
                for subk, subv in v.items():
                    config[f"{k}_{subk}"] = subv
            else:
                config[k] = v

    # Add any other attributes
    for k, v in other_config.items():
        config[k] = v
    # log all hyperparameters under “parameters”
    run.log_configs(config)
    return run
