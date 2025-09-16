"""
Base trainer class. Provides some common functionality for all trainers.
Inherit and override to implement:
- `generate_and_score`: generate rollouts, score them, save to
                        replay buffer as training data
- `update_model`: train model(s) over collected rollouts
- `evaluate`: similar to `generate_and_score`, but cases where it's
              different (e.g., we don't do multiple rollouts per sample)
"""
from copy import deepcopy
from os.path import join
from typing import Any

from omegaconf import DictConfig
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from ..environments.base import Environment
from ..model.checkpoints import save_trainable_weights, load_model_checkpoint
from ..optimizers import get_optimizer, get_scheduler
from ..replay_buffer import get_replay_buffer, ReplayBuffer
from ..utils.logging import print_header, OurLogger

from .dataloaders import get_update_dataloader


class BaseTrainer:
    """
    Parent trainer class. Inherit and override to implement:
    - `generate_and_score`: generate rollouts, score them, save to
                            replay buffer as training data
    - `update_model`: train model(s) over collected rollouts
    - `evaluate`: similar to `generate_and_score`, but cases where it's
                  different (e.g., we don't do multiple rollouts per sample)

    Notable arguments:
    - num_steps: total number of environment samples to train over (potentially repeating)
    - samples_per_prompt: number of rollouts to generate per step
    - prompts_per_update: number of prompts to train over each update
    - update_num_steps: number of steps to train model(s) for each policy update
    - update_num_epochs: number of epochs to train model(s) for each policy update
    - update_batch_size: effective batch size for each policy update
    - gradient_accumulation_steps: number of optimization steps to accumulate before updating the model
    - eval_step: number of steps between evaluation
    """

    def __init__(
        self,
        name: str,
        num_steps: int,
        samples_per_prompt: int,
        prompts_per_update: int,
        max_tries: int,
        generation_config: dict[str, Any],
        replay_buffer_config: dict[str, Any],
        update_num_steps: int | None,
        update_num_epochs: int | None,
        update_batch_size: int,
        gradient_accumulation_steps: int,
        update_eval_step: int,
        update_dataset_config: dict[str, Any] | DictConfig,
        dataloader_config: dict[str, Any] | DictConfig,
        optimizer_config: dict[str, Any] | DictConfig,
        scheduler_config: dict[str, Any] | DictConfig,
        eval_step: int | None,
        eval_batch_step: int | None,
        eval_generation_config: dict[str, Any] | DictConfig | None,
        eval_samples_per_prompt: int,
        eval_max_tries: int,
        max_eval_samples: int,
        return_best_model: bool,
        early_stopping_limit: int,
        keep_best_eval_model: bool,
        checkpoint_dir: str,
        generation_dir: str,
        run_name: str,
        logger: OurLogger | None,
        verbose: bool,
        save_iter: int = 1,
        save_rollout_data_step: int = 50,
        debug: bool = False,
        run_url: str = "",
        save_try: int | str = "max",
        rollout_metric: str = "return",  # "return" or "correct" or "accuracy"
        **kwargs: Any,
    ):
        self.name = name
        # Training parameters
        self.num_steps = num_steps

        # 1. Rollouts
        self.samples_per_prompt = samples_per_prompt
        self.prompts_per_update = prompts_per_update
        self.max_tries = max_tries
        self.generation_config = generation_config
        # 1.1. Replay buffer
        self.replay_buffer_config = replay_buffer_config

        # 2. Updates
        self.update_num_steps = update_num_steps
        self.update_num_epochs = update_num_epochs  # redundant with `update_num_steps`
        self.update_batch_size = update_batch_size  # effective batch size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.update_minibatch_size = self.update_batch_size // self.gradient_accumulation_steps
        self.update_eval_step = update_eval_step

        # 3. More configs
        self.update_dataset_config = update_dataset_config
        self.dataloader_config = dataloader_config
        self.dataloader_config["batch_size"] = self.update_minibatch_size
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        # Evaluation parameters
        self.eval_step = eval_step or (eval_batch_step * self.prompts_per_update)
        self.eval_generation_config = eval_generation_config
        if self.eval_generation_config is None:
            self.eval_generation_config = deepcopy(self.generation_config)
        self.eval_samples_per_prompt = eval_samples_per_prompt

        self.eval_max_tries = eval_max_tries
        self.max_eval_samples = max_eval_samples

        self.save_try = save_try
        self.saved_checkpoint = False

        self.return_best_model = return_best_model
        self.early_stopping_limit = early_stopping_limit
        self.keep_best_eval_model = keep_best_eval_model

        self.checkpoint_dir = checkpoint_dir
        self.generation_dir = generation_dir

        print(f"-> Checkpoint dir: {self.checkpoint_dir}")
        print(f"-> Generation dir: {self.generation_dir}")

        # Logging and miscellaneous
        self.run_name = run_name
        self.run_url = run_url
        self.logger = logger
        self.verbose = verbose

        self.save_iter = save_iter
        self.save_rollout_data_step = save_rollout_data_step
        self.debug = debug

        self.temperature = self.generation_config.get("temperature", 1.0)
        print(f"-> Generation Temperature: {self.temperature}")
        self.checkpoint_path = f"{self.checkpoint_dir}/{self.run_name}.pt"
        self.generation_path = f"{self.generation_dir}/{self.run_name}"  # can add suffixes to this

        # Initialize step counters
        self.update_train_step_counter = 0
        self.update_eval_step_counter = 0
        self.logger_update_metrics = {"train": {}, "eval": {}}

        # Model and data arguments; override in child classes if needed
        self.model_kwargs = {
            "use_cache": False,
            "output_hidden_states": False,
            "output_attentions": False,
        }
        self.data_args = ["input_ids", "attention_mask", "labels"]

        self.rollout_metric = rollout_metric
        self.init_metrics()

    def generate_and_score(
        self,
        model: AutoModelForCausalLM | PreTrainedModel,
        env: Environment,
        replay_buffer: ReplayBuffer,
        tokenizer: AutoTokenizer | PreTrainedTokenizer,
        samples_per_prompt: int,
        max_tries: int,
        data_sample_id: int,
        batch_id: int,
        unique_sample_id: int,
        split: str,
        pbar_position: int = 1,
        generation_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], int]:
        """
        Generate rollouts and score them
        """
        raise NotImplementedError("Generate and score not implemented")

    def update_model(
        self,
        model: AutoModelForCausalLM | PreTrainedModel | nn.Module,
        tokenizer: AutoTokenizer | PreTrainedTokenizer,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        num_epochs: int,
        **kwargs: Any,
    ) -> tuple[
        AutoModelForCausalLM | PreTrainedModel | nn.Module,
        torch.optim.Optimizer,
        torch.optim.lr_scheduler.LRScheduler,
    ]:
        """
        Update model with replay buffer data
        -> returns updated model, optimizer, and scheduler
        """
        raise NotImplementedError("Update model not implemented")

    def get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer | Any:
        """
        Get optimizer
        """
        return get_optimizer(model=model, **self.optimizer_config)

    def get_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler | Any:
        """
        Get LR scheduler
        """
        return get_scheduler(optimizer=optimizer, **self.scheduler_config)

    def get_replay_buffer(
        self,
        model: AutoModelForCausalLM | PreTrainedModel,
    ) -> ReplayBuffer:
        """
        Get replay buffer
        """
        if getattr(model, "lm", None) is not None:
            embed_dim = model.lm.config.hidden_size
            embed_dtype = model.lm.dtype
            if embed_dtype is None:
                embed_dtype = torch.float32
        else:
            embed_dim = model.config.hidden_size
            embed_dtype = model.dtype
        replay_buffer = get_replay_buffer(
            embed_dim=embed_dim,
            embed_dtype=embed_dtype,
            **self.replay_buffer_config
        )
        return replay_buffer

    def get_update_data(
        self,
        replay_buffer: ReplayBuffer,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Get training data from replay buffer
        """
        return replay_buffer.get_data(**kwargs)[0]

    def train(
        self,
        model: AutoModelForCausalLM | PreTrainedModel,
        env: Environment,
        replay_buffer: ReplayBuffer,
        tokenizer: PreTrainedTokenizer,
        optimizer: torch.optim.Optimizer | Any,
        scheduler: torch.optim.lr_scheduler.LRScheduler | Any,
        eval_env: Environment,
        generate_and_score_kwargs: dict[str, Any] | None = None,
        update_kwargs: dict[str, Any] | None = None,
        debug: bool = False,
        **kwargs: Any,
    ) -> tuple[AutoModelForCausalLM | PreTrainedModel, ReplayBuffer, int]:
        """
        Main training loop
        """
        generate_and_score_kwargs = generate_and_score_kwargs or {}
        update_kwargs = update_kwargs or {}

        dataset_epochs = 0
        data_sample_ix = 0    # sample index in the environment dataset
        update_batch_ix = 0   # batch index for each model update (policy, value, etc.)
        unique_sample_id = 0  # unique sample id for every rollout (max = num_steps * num_generations - 1)
        env.shuffle(seed=dataset_epochs)
        pbar = tqdm(
            range(self.num_steps),
            desc="Training",
            colour="blue",
            leave=True,
            position=0,
        )
        train_rollout_metrics = {"step": []}
        eval_rollout_metrics  = {"step": []}
        rollout_ckpt_path = self.checkpoint_path.replace(".pt", "_rollout.pt")

        if not debug:
            # Initial evaluation (TODO: refactor)
            step_ix = -1
            eval_step_rollout_metrics = self.evaluate(
                model,
                eval_env,
                replay_buffer,
                tokenizer=tokenizer,
                split="eval",
                data_sample_offset=100000,  # hack to distinguish from train
            )
            _num_entries = len(eval_step_rollout_metrics["return"])
            eval_step_rollout_metrics["step"] = [step_ix] * _num_entries
            for k, v in eval_step_rollout_metrics.items():
                try:
                    if k.startswith("er_"):
                        v = [np.mean(v)] * _num_entries  # hacks
                    if k not in eval_rollout_metrics:
                        eval_rollout_metrics[k] = []
                    eval_rollout_metrics[k].extend(v)
                except Exception as e:
                    print(e)
            # Save rollout metrics
            try:
                pd.DataFrame(eval_rollout_metrics).to_csv(
                    join(self.generation_dir, f"{self.run_name}-eval_metrics.csv"),
                    index=False,
                )
            except ValueError as e:
                print(e)
                for k, v in eval_rollout_metrics.items():
                    print(k, len(v))
                breakpoint()
            if self.logger is not None:
                self.logger.log(
                    {
                        f"rollout_eval/{k}": v
                        for k, v in eval_rollout_metrics.items()
                        if k not in ["img_save_path"]
                    },
                    step=None,
                    reduction="mean",  # average the metrics by `bin_metric`
                    bin_metric="rollout_eval/trials",
                )
            # Get metric for evaluation (last try return)
            save_try = (
                np.unique(eval_step_rollout_metrics["trials"]).max()
                if self.save_try == "max" else self.save_try
            )
            all_trials = np.array(eval_step_rollout_metrics["trials"])
            val_metric = np.array(eval_step_rollout_metrics[self.rollout_metric])
            val_metric = val_metric[all_trials == save_try].mean()
            if self.save_try == "max":
                try:
                    assert save_try == self.eval_max_tries - 1, (  # 0-indexing
                        f"save_try: {save_try} !=" + 
                        f"eval_max_tries - 1: {self.eval_max_tries-1}"
                    )
                except Exception as e:
                    print(e)
                    breakpoint()

            self.update_best_metric(
                metric_name="rollout",
                metric_val=val_metric,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                total_grad_steps=step_ix,  # hack, but not grad steps
                checkpoint_path=rollout_ckpt_path,
            )
        
        # Main training loop
        for step_ix in pbar:
            # Collect `num_generations` rollouts for each sample
            # -> Saves these to replay buffer
            _gen_metrics, unique_sample_id = self.generate_and_score(
                model,
                env,
                replay_buffer,
                tokenizer,
                samples_per_prompt=self.samples_per_prompt,
                max_tries=self.max_tries,
                data_sample_id=data_sample_ix,
                batch_id=update_batch_ix,
                unique_sample_id=unique_sample_id,
                split="train",
                pbar_position=1,
                generation_config=self.generation_config,
                **generate_and_score_kwargs,
            )
            _gen_metrics["step"] = [step_ix] * len(_gen_metrics["return"])
            if self.logger is not None:
                self.logger.log(
                    {
                        f"rollout_train/{k}": v
                        for k, v in _gen_metrics.items()
                        if k not in ["img_save_path"]
                    },
                    step=None,
                    reduction="mean",  # average the metrics by `bin_metric`
                    bin_metric="rollout_train/trials",
                )
            data_sample_ix += 1
            for k, v in _gen_metrics.items():
                if k not in train_rollout_metrics:
                    train_rollout_metrics[k] = []
                train_rollout_metrics[k].extend(v)
            if (step_ix + 1) % self.save_iter == 0:  # Save rollout metrics
                pd.DataFrame(train_rollout_metrics).to_csv(
                    join(self.generation_dir, f"{self.run_name}-train_metrics.csv"),
                    index=False,
                )
            # Update model with (update_batch_size x num_generations) rollouts
            # if (step_ix + 1) % self.update_batch_size == 0:
            if (step_ix + 1) % self.prompts_per_update == 0:
                # Create dataloaders
                # update_data = replay_buffer.get_data(batch_id=update_batch_ix)[0]
                # update_data = self.get_update_data(
                #     replay_buffer, batch_id=update_batch_ix,
                # )
                update_data_dict = replay_buffer.get_data_dict(
                    batch_id=update_batch_ix
                )[0]
                try:
                    update_dataloader = get_update_dataloader(
                        dataset_dict=update_data_dict,
                        dataloader_config=self.dataloader_config,
                        tokenizer=tokenizer,
                        **self.update_dataset_config,
                    )
                except Exception as e:
                    print(e)
                    try:  # hacks
                        # update_data_dict = replay_buffer.get_data_dict(batch_id=update_batch_ix)[0]
                        update_data = replay_buffer.get_data(batch_id=update_batch_ix)[0]
                        update_dataloader = get_update_dataloader(
                            data=update_data,
                            dataloader_config=self.dataloader_config,
                            tokenizer=tokenizer,
                            **self.update_dataset_config,
                        )
                    except Exception as e2:
                        print(e2)
                        breakpoint()

                # Load rollout data
                train_loader = update_dataloader.train_loader
                eval_loader = update_dataloader.eval_loader
                # Train model
                model, optimizer, scheduler = self.update_model(
                    model=model,
                    tokenizer=tokenizer,
                    train_loader=train_loader,
                    eval_loader=eval_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    num_epochs=self.update_num_epochs,
                    pbar_position=1,
                    **update_kwargs,
                )
                # replay_buffer.clean_up()
                update_batch_ix += 1

            # "Epoch" finished; reshuffle environment data
            if (step_ix + 1) % len(env) == 0:
                env.shuffle(seed=dataset_epochs)
                data_sample_ix = 0
                dataset_epochs += 1

            # Evaluate model on held-out environment
            if (step_ix + 1) % self.eval_step == 0:
                # In case env and eval_env are the same object
                eval_env.split = "eval"
                eval_env.max_tries = self.eval_max_tries
                eval_step_rollout_metrics = self.evaluate(
                    model,
                    eval_env,
                    replay_buffer,
                    tokenizer=tokenizer,
                    # unique_sample_id=unique_sample_id,
                    split="eval",
                    data_sample_offset=100000 * (step_ix + 1),  # hack to distinguish from train
                )
                _num_entries = len(eval_step_rollout_metrics["return"])
                eval_step_rollout_metrics["step"] = [step_ix] * _num_entries
                for k, v in eval_step_rollout_metrics.items():
                    try:
                        if k.startswith("er_"):
                            v = [np.mean(v)] * _num_entries  # hacks
                        if k not in eval_rollout_metrics:
                            eval_rollout_metrics[k] = []
                        eval_rollout_metrics[k].extend(v)
                    except Exception as e:
                        print(e)

                if (step_ix + 1) % self.save_iter == 0:  # Save rollout metrics
                    try:
                        pd.DataFrame(eval_rollout_metrics).to_csv(
                            f"{self.generation_path}-eval_metrics.csv", index=False,
                        )
                    except Exception as e:
                        print(e)
                        for k, v in eval_rollout_metrics.items():
                            print(k, len(v))
                        breakpoint()
                if self.logger is not None:
                    self.logger.log(
                        {
                            f"rollout_eval/{k}": v
                            for k, v in eval_rollout_metrics.items()
                            if k not in ["img_save_path"]  # or if not isinstance(v[0], str)
                        },
                        step=None,
                        reduction="mean",  # average the metrics by `bin_metric`
                        bin_metric="rollout_eval/trials",
                    )
                # Get metric for evaluation
                save_try = (
                    np.unique(eval_step_rollout_metrics["trials"]).max()
                    if self.save_try == "max" else self.save_try
                )
                all_trials = np.array(eval_step_rollout_metrics["trials"])
                val_metric = np.array(eval_step_rollout_metrics[self.rollout_metric])
                val_metric = val_metric[all_trials == save_try].mean()
                if self.save_try == "max":
                    try:
                        assert save_try == self.eval_max_tries - 1, (  # 0-indexing
                            f"save_try: {save_try} != eval_max_tries - 1: {self.eval_max_tries-1}"
                        )
                    except Exception as e:
                        print(e)
                        breakpoint()
                self.update_best_metric(
                    metric_name="rollout",
                    metric_val=val_metric,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    total_grad_steps=step_ix,  # hack, but not grad steps
                    checkpoint_path=rollout_ckpt_path,
                )
                if self.keep_best_eval_model:
                    try:
                        print_header(f"Loading best model from {rollout_ckpt_path}")
                        model = load_model_checkpoint(model, rollout_ckpt_path)[0]
                    except Exception as e:
                        print(e)
                        breakpoint()
                # In case env and eval_env are the same object
                env.split = "train"
                env.max_tries = self.max_tries
        if self.return_best_model:
            try:
                print_header(f"Loading best model from {rollout_ckpt_path}")
                model = load_model_checkpoint(
                    model,  # type: ignore
                    rollout_ckpt_path,
                )[0]
            except Exception as e:
                print(e)
                breakpoint()
        return model, replay_buffer, unique_sample_id

    def evaluate(
        self,
        model: AutoModelForCausalLM | PreTrainedModel,
        env: Environment,
        replay_buffer: ReplayBuffer,
        tokenizer: AutoTokenizer | PreTrainedTokenizer,
        split: str,
        data_sample_offset: int,
        max_eval_samples: int | None = None,
        **generate_and_score_kwargs: Any,
    ) -> dict[str, Any]:
        """
        Evaluate the model
        """
        eval_step_rollout_metrics = {}
        model_was_training = getattr(model, "training", False)
        model.eval()
        eval_samples = (
            max_eval_samples if max_eval_samples is not None else self.max_eval_samples
        )
        num_episodes = min(len(env), eval_samples)
        pbar = tqdm(range(num_episodes), desc="Evaluating", colour="green", position=1)

        # update_batch_ix = 0   # batch index for each model update (policy, value, etc.)
        update_batch_ix = 100
        with torch.no_grad():
            for data_sample_ix in pbar:  # sample index in the environment dataset
                _sample_id = data_sample_ix + data_sample_offset
                _gen_metrics, unique_sample_id = self.generate_and_score(
                    model,
                    env,
                    replay_buffer,
                    tokenizer=tokenizer,
                    samples_per_prompt=self.eval_samples_per_prompt,
                    max_tries=self.eval_max_tries,
                    data_sample_id=data_sample_ix,
                    batch_id=update_batch_ix,
                    unique_sample_id=_sample_id,
                    split=split,  # is_train=False, update_replay_buffer=False
                    pbar_position=2,
                    generation_config=self.eval_generation_config,
                    **generate_and_score_kwargs,
                )
                for k, v in _gen_metrics.items():
                    if k not in eval_step_rollout_metrics:
                        eval_step_rollout_metrics[k] = []
                    eval_step_rollout_metrics[k].extend(v)
                _pbar_postfix = {
                    self.rollout_metric: eval_step_rollout_metrics[self.rollout_metric][-1]
                }
                pbar.set_postfix(**_pbar_postfix)

        # replay_buffer.clean_up()  # clear buffer or at least remove eval & test samples
        model.train(mode=model_was_training)
        return eval_step_rollout_metrics

    def summarize_metrics(self, rollout_metrics: dict[str, Any]) -> dict[str, Any]:
        """
        Save rollout metrics to dict (e.g., for logger logging)
        """
        metrics: dict[str, Any] = {
            "total": len(rollout_metrics[self.rollout_metric]),
            "success": sum(rollout_metrics[self.rollout_metric]),
        }
        for k, v in rollout_metrics.items():
            metrics[f"{k}_mean"] = np.mean(v)
            metrics[f"{k}_std"] = np.std(v)
        return metrics

    def init_metrics(self) -> None:
        """
        Reset metrics for training
        - 'rollout': success metrics with rollouts / interacting with environment
        - 'model_update': loss metrics for updating policy
        - 'value_update': loss metrics for updating critic (optional)
        """
        # replicate 200 for >= or <=
        self.best_metrics = {
            "rollout": {
                "reward": -1e12,
                "return": -1e12,
                "correct": -1,
                "accuracy": -1,
                "early_stopping_count": 0,
                "is_better": lambda x, y: x >= y,
                "total_grad_steps": -1,
            },
            "model_update": {
                "lm_loss": 1e12,
                "early_stopping_count": 0,
                "is_better": lambda x, y: x <= y,
                "total_grad_steps": -1,
            },
            "critic_update": {
                "loss": 1e12,
                "early_stopping_count": 0,
                "is_better": lambda x, y: x <= y,
                "total_grad_steps": -1,
            },
        }

    def reset_metrics(self, metric_name: str) -> None:
        """
        Reset metrics for training
        """
        if metric_name == "rollout":
            self.best_metrics["rollout"] = {
                "reward": -1e12,
                "return": -1e12,
                "correct": -1,
                "accuracy": -1,
                "early_stopping_count": 0,
                "is_better": lambda x, y: x >= y,
                "total_grad_steps": -1,
            }
        elif metric_name == "model_update":
            self.best_metrics[metric_name] = {
                "lm_loss": 1e12,
                "early_stopping_count": 0,
                "is_better": lambda x, y: x <= y,
                "total_grad_steps": -1,
            }
        elif metric_name == "critic_update":
            self.best_metrics[metric_name] = {
                "loss": 1e12,
                "early_stopping_count": 0,
                "is_better": lambda x, y: x <= y,
                "total_grad_steps": -1,
            }

    def update_best_metric(
        self,
        metric_name: str,
        metric_val: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        total_grad_steps: int,
        checkpoint_path: str,
    ) -> bool:
        """
        Update best metric, save checkpoint if it's the best so far,
        and return whether we should early stop
        """
        early_stopping = False
        metric_name_to_metric_key = {
            "rollout": self.rollout_metric,
            "model_update": "lm_loss",
            "critic_update": "loss",
        }
        metric_key = metric_name_to_metric_key[metric_name]
        is_better = self.best_metrics[metric_name]["is_better"]

        # Update best metric
        prior_best_metric = self.best_metrics[metric_name][metric_key]
        prior_best_step = self.best_metrics[metric_name]["total_grad_steps"]
        step_text = f"(step {total_grad_steps})"
        if is_better(metric_val, self.best_metrics[metric_name][metric_key]):
            self.best_metrics[metric_name][metric_key] = metric_val
            self.best_metrics[metric_name]["total_grad_steps"] = total_grad_steps
            torch.save(
                {
                    "model_state_dict": save_trainable_weights(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "grad_step": total_grad_steps,
                },
                checkpoint_path,
            )
            self.best_metrics[metric_name]["early_stopping_count"] = 0
            print(
                f"-> {metric_name} {metric_key} improved to {metric_val}! {step_text}"
            )
            print(
                f"-> Prior best {metric_key}: {prior_best_metric}. "
                f"(step {prior_best_step})"
            )
            print(f"-> Saving checkpoint to {checkpoint_path}")
            self.saved_checkpoint = True
        else:
            print(f"-> {metric_name} {metric_key}: {metric_val} {step_text}")
            print(
                f"-> Best so far: {prior_best_metric} (step {prior_best_step})"
            )
            self.best_metrics[metric_name]["early_stopping_count"] += 1
            if (
                self.best_metrics[metric_name]["early_stopping_count"]
                >= self.early_stopping_limit
            ):
                early_stopping = True

        return early_stopping
