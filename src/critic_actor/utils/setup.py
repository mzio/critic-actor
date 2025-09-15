"""
Setup utilities
"""
import os
from os.path import join
import random
from typing import Any, cast
from argparse import Namespace
from omegaconf import OmegaConf

import numpy as np
import torch

from .logging import print_config


def seed_everything(seed: int) -> None:
    """Seed everything"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_run_name(
    args: Namespace,
    prefix: str = "",
    ignore_args: list[str] | None = None,
) -> str:
    """Return run name"""
    run_name = prefix
    if ignore_args is None:
        ignore_args = []

    for argname, argval in vars(args).items():
        if argval is None or argname in ignore_args:
            continue
        argn = "".join([c[0] for c in argname.split("_")])
        # remove hyphens and dots, e.g., --model_name gpt-4.1-nano-2025-04-14
        argval = str(argval).replace("-", "_").replace(".", "_")
        run_name += f"-{argn}={argval}"

    if getattr(args, "lora_checkpoint_path", None) is not None:
        ckpt_id = "_".join(
            [
                "=".join(["".join([x[0] for x in c.split("_")]) for c in s.split("=")])
                for s in args.lora_checkpoint_path.split("/")[-1].split("-")
            ]
        )
        run_name += f"-ckpt={ckpt_id}"

    run_name = run_name.replace("False", "0").replace("True", "1")

    return run_name


def get_configs(
    args: Namespace,
    verbose: bool = False,
) -> dict[str, dict[str, Any] | None]:
    """
    Load and process experiment configs
    """
    configs = {
        "llm_config": None,
        "model_config": None,
        "critic_actor_config": None,
        "lora_config": None,
        "replay_buffer_config": None,
        "env_config": None,
        "trainer_config": None,
    }

    # LLM config
    if getattr(args, "llm_config", None) is not None:
        llm_config_path = join(args.config_dir, "model", f"{args.llm_config}.yaml")
        llm_config = OmegaConf.load(llm_config_path)
        configs["llm_config"] = llm_config

    # Model config
    if getattr(args, "model_config", None) is not None:
        model_config_path = join(args.config_dir, "model", f"{args.model_config}.yaml")
        model_config = OmegaConf.load(model_config_path)
        configs["model_config"] = model_config

    # Critic-Actor config
    if getattr(args, "critic_actor_config", None) is not None:
        critic_actor_config_path = join(
            args.config_dir, "model", f"{args.critic_actor_config}.yaml"
        )
        critic_actor_config = OmegaConf.load(critic_actor_config_path)
        configs["critic_actor_config"] = critic_actor_config

    # LoRA config
    if getattr(args, "lora_config", None) is not None:
        lora_config_path = join(args.config_dir, "lora", f"{args.lora_config}.yaml")
        lora_config = OmegaConf.load(lora_config_path)
        configs["lora_config"] = lora_config

    # Replay buffer config
    if getattr(args, "replay_buffer_config", None) is not None:
        replay_buffer_config_path = join(
            args.config_dir, "replay_buffer", f"{args.replay_buffer_config}.yaml",
        )
        replay_buffer_config = OmegaConf.load(replay_buffer_config_path)
        for arg in ["discount_factor"]:
            if getattr(args, arg, None) is not None:
                setattr(replay_buffer_config, arg, getattr(args, arg))
        configs["replay_buffer_config"] = replay_buffer_config

    # Environment config
    env_config_path = join(args.config_dir, "environments", f"{args.env_config}.yaml")
    env_config = OmegaConf.load(env_config_path)
    for arg in ["max_turns", "num_samples", "verbose"]:
        if getattr(args, arg, None) is not None:
            setattr(env_config, arg, getattr(args, arg))
    # Adjust environment tokenizer to model
    if configs["model_config"] is not None:
        if getattr(env_config, "pretrained_model_config", None) is None:
            setattr(env_config, "pretrained_model_config", model_config.model_config)
        else:
            for attr in model_config.model_config:
            # for attr in env_config.pretrained_model_config:
                v = getattr(model_config.model_config, attr)
                setattr(env_config.pretrained_model_config, attr, v)
    configs["env_config"] = env_config

    # Trainer config
    if getattr(args, "trainer_config", None) is not None:
        trainer_config_path = join(
            args.config_dir, "trainers", f"{args.trainer_config}.yaml"
        )
        trainer_config = OmegaConf.load(trainer_config_path)
        # Override defaults
        for arg in ["optim", "lr", "weight_decay"]:
            if getattr(args, arg, None) is not None:
                setattr(trainer_config.optimizer_config, arg, getattr(args, arg))

        for arg in [
            "num_steps",
            "samples_per_prompt",
            "prompts_per_update",
            "update_batch_size",
            "gradient_accumulation_steps",
            "update_num_steps",
            "update_num_epochs",
            "update_eval_step",
            "no_importance_weight",
            "max_tries",
            "max_steps",
            # LLM / Critic-Actor
            "num_return_sequences",
            "reasoning_effort",
            "single_response_only",
            "include_reasoning_for_critic",
            "truncation",
            # Evaluation
            "eval_samples_per_prompt",
            "eval_generation_batch_size",
            "eval_max_tries",
            "max_eval_samples",
            # Miscellaneous
            "verbose",
            "checkpoint_dir",
            "generation_dir",
            "run_name",
            "debug",
            # Experimental / Trying stuff
            "keep_best_eval_model",
        ]:
            if getattr(args, arg, None) is not None:
                if arg[:3] == "no_":
                    setattr(trainer_config, arg[3:], not getattr(args, arg))
                else:
                    setattr(trainer_config, arg, getattr(args, arg))

        # Set replay buffer config
        trainer_config.replay_buffer_config = replay_buffer_config
        configs["trainer_config"] = trainer_config

        # Update shared environment configs to match trainer configs
        env_config.max_tries = trainer_config.max_tries

        # Update generation config used in trainer
        if getattr(llm_config, "generation_config", None) is not None:
            trainer_config["generation_config"] = llm_config["generation_config"]
            trainer_config["eval_generation_config"] = llm_config["generation_config"]

    # Cast all configs consistently to dictionaries
    for k, v in configs.items():
        if v is not None:
            if verbose:
                print_config(v, name=k.upper())
            configs[k] = cast(dict[str, Any], OmegaConf.to_container(v))

    return configs
