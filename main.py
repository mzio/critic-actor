"""
Main training script
"""
import argparse
import os
import sys
from os.path import join
from copy import deepcopy

from dotenv import load_dotenv

from critic_actor.environments import load_env
from critic_actor.model import load_model
from critic_actor.model.critic_actor import load_critic_actor
from critic_actor.model.peft import create_peft_config, count_parameters
from critic_actor.llm import load_llm
from critic_actor.replay_buffer import get_replay_buffer
from critic_actor.trainers import load_trainer
from critic_actor.utils import (
    get_configs,
    get_run_name,
    init_logger,
    print_config,
    print_header,
    seed_everything,
)


def get_args() -> argparse.Namespace:
    """
    Load and process experiment arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default="critic-actor")

    # Main configs (see ./configs)
    parser.add_argument("--llm_config", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--critic_actor_config", type=str)
    parser.add_argument("--replay_buffer_config", type=str)
    parser.add_argument("--env_config", type=str)
    parser.add_argument("--trainer_config", type=str)
    parser.add_argument("--lora_config", type=str, default=None)

    # Override defaults
    ## Environment
    parser.add_argument("--max_turns", type=int, default=100,
                        help="Maximum number of steps per episode")
    parser.add_argument("--max_tries", type=int, default=1,
                        help="Maximum number of tries per episode")

    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--samples_per_prompt", type=int, default=None)
    parser.add_argument("--prompts_per_update", type=int, default=None)
    parser.add_argument("--update_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--update_num_steps", type=int, default=None)
    parser.add_argument("--update_num_epochs", type=int, default=None)
    parser.add_argument("--update_eval_step", type=int, default=None)

    ## LLM / critic-actor
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--reasoning_effort", type=str, default=None,
                        choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--num_return_sequences", type=int, default=None,
                        help="Number of actions to pick from")
    parser.add_argument("--single_response_only", action="store_true", default=None)
    parser.add_argument("--include_reasoning_for_critic", action="store_true", default=None)
    parser.add_argument("--truncation", type=str, default=None, choices=["auto", "disabled"])

    ## Returns and advantage
    parser.add_argument("--discount_factor", type=float, default=None,
                        help="Discount factor for returns")
    ## Policy updates
    parser.add_argument("--no_importance_weight", action="store_true", default=None,
                        help="Disable importance weighting during policy updates")
    parser.add_argument("--optim", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)

    ## Evaluation
    parser.add_argument("--eval_step", type=int, default=None,
                        help="Number of steps between evaluations")
    parser.add_argument("--eval_samples_per_prompt", type=int, default=None,
                        help="Number of rollouts to generate per data sample")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Max samples to evaluate on, e.g., for debugging")
    parser.add_argument("--eval_max_tries", type=int, default=None,
                        help="Max sequential tries allowed for evaluation")
    # Logging
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--generation_dir", type=str, default="./generations")
    parser.add_argument("--no_wandb", action="store_true", default=False)
    parser.add_argument("--no_neptune", action="store_true", default=False)
    parser.add_argument("--logger_entity", type=str, default='periodiclabs')
    parser.add_argument("--wandb_entity", type=str, default=None)

    # Miscellaneous
    parser.add_argument("--config_dir", type=str, default="./configs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replicate", type=str, default="0")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)

    # Experimental / Trying stuff
    parser.add_argument("--keep_best_eval_model", action="store_true", default=None)

    args = parser.parse_args()

    # Get experiment name
    _ignore_args = [
        "project_name", "checkpoint_dir", "generation_dir",
        "logger_entity", "no_wandb", "no_neptune",
        "config_dir", "verbose", "debug",
    ]
    args.run_name = get_run_name(args, prefix="cria", ignore_args=_ignore_args)
    args.run_name = args.run_name.replace("/", "_")  # some configs are nested
    args.checkpoint_name = f"{args.run_name}.pt"

    # Setup checkpoint directories
    args.checkpoint_dir = join(args.checkpoint_dir, args.model_config)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    print("-> Saving checkpoints to", args.checkpoint_dir)

    # Setup generation directories
    for _dir in [args.model_config, args.env_config, args.trainer_config]:
        _dir = _dir.replace("/", "_")
        args.generation_dir = join(args.generation_dir, _dir)
        if not os.path.isdir(args.generation_dir):
            os.makedirs(args.generation_dir)
    args.generation_path = join(args.generation_dir, f"{args.run_name}.csv")
    print("-> Saving generations to", args.generation_path)
    return args


def main() -> None:
    """
    Main training loop
    """
    load_dotenv()  # Load global environment variables
    args = get_args()
    seed_everything(args.seed)
    args.run_cmd = ' '.join([sys.executable] + sys.argv)

    # Load configs
    configs = get_configs(args, verbose=args.verbose)
    llm_config = configs["llm_config"]
    model_config = configs["model_config"]
    critic_actor_config = configs["critic_actor_config"]
    lora_config = configs["lora_config"]
    replay_buffer_config = configs["replay_buffer_config"]
    env_config = configs["env_config"]
    trainer_config = configs["trainer_config"]

    if args.verbose:
        for k, v in configs.items():
            if v is not None:
                print_config(v, name=f"{k.upper()} Config")

    # Get LLM and Critic-Actor encoder model
    if "model_type" in llm_config:
        llm_config.pop("model_type")
    llm = load_llm(**llm_config)
    critic_model, critic_tokenizer = load_model(**model_config)
    critic_model.eval()  # Freeze the encoder model
    for p in critic_model.parameters():
        p.requires_grad = False

    # Get Critic-Actor head
    critic_actor = load_critic_actor(llm_encoder=critic_model, **critic_actor_config)
    if lora_config is not None:
        critic_actor = create_peft_config(critic_actor, lora_config)[0]
    critic_actor.to(critic_model.device)
    critic_actor.train()

    # Log some info
    args.model_train_params = count_parameters(critic_actor, requires_grad=True)
    args.model_total_params = count_parameters(critic_actor, requires_grad=False)
    args.pct_trainable = args.model_train_params / args.model_total_params * 100
    if args.verbose:
        print_header("Critic-Actor Encoder Model")
        print(critic_model)
        print_header("Critic-Actor Head")
        print(critic_actor)
        print_header("Trainable Parameters")
        for n, p in critic_actor.named_parameters():
            if p.requires_grad:
                print(f"├── {n}")
        print_header("Trainable Parameter Counts")
        print(f"├── Train parameters: {args.model_train_params}")
        print(f"├── Total parameters: {args.model_total_params}")
        print(f"├── Percent training: {args.pct_trainable:.3f}%")

    logger = init_logger(
        args,
        model_config=model_config,
        trainer_config=trainer_config,
        **critic_actor_config,
    )
    args.run_url = logger.get_url() if logger is not None else ''
    print_header("Run attributes")
    print(f"├── Run command: \033[36m{args.run_cmd}\033[0m")
    print(f"├── Run URL: \033[36m{args.run_url}\033[0m")

    # Load environment
    train_env = load_env(**env_config)
    eval_config = deepcopy(env_config)
    eval_config["split"] = "eval"
    eval_config["max_tries"] = trainer_config["eval_max_tries"]
    eval_env = load_env(**eval_config)
    eval_env.max_eval_samples = args.max_eval_samples

    # Setup environment for Critic-Actors
    train_env.llm = llm
    train_env.critic_model = critic_model
    eval_env.llm = llm
    eval_env.critic_model = critic_model

    generation_config = {
        "max_new_tokens": args.max_new_tokens,
    }

    # Load trainer
    trainer = load_trainer(
        **trainer_config,
        model_config=model_config["model_config"],
        generation_config=generation_config,
        logger=logger,
        run_url=args.run_url,
    )
    # Get replay buffer and optimizers
    replay_buffer = trainer.get_replay_buffer(**replay_buffer_config)
    optimizer = trainer.get_optimizer(critic_actor)
    scheduler = trainer.get_scheduler(optimizer)

    print("replay_buffer.discount_factor:", replay_buffer.discount_factor)
    print("replay_buffer.normalize_returns:", replay_buffer.normalize_returns)
    print("replay_buffer.normalize_advantages:", replay_buffer.normalize_advantages)
    print("replay_buffer.negative_returns:", replay_buffer.negative_returns)
    print("replay_buffer.advantage_fn:", replay_buffer.advantage_fn)

    critic_actor, replay_buffer, unique_sample_id = trainer.train(
        model=critic_actor,
        env=train_env,
        replay_buffer=replay_buffer,
        tokenizer=critic_tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        eval_env=eval_env,
    )
    # Final evaluation
    eval_env.split = "test"
    eval_env.max_tries = trainer_config["eval_max_tries"]
    # eval_env.max_eval_samples = None
    test_rollout_metrics = trainer.evaluate(
        model=critic_actor,
        env=eval_env,
        replay_buffer=replay_buffer,
        tokenizer=critic_tokenizer,
        split="test",
        data_sample_offset=-100000,  # hack to distinguish from train and eval
    )
    if logger is not None:
        logger.log(
            {f"rollout_test/{k}": v for k, v in test_rollout_metrics.items()},
            bin_metric="rollout_test/trials",
        )
        logger.finish()

    print("-> Done training!")
    print(f"-> See run at: \033[36m{args.run_url}\033[0m")
    print(f"-> Checkpoint: \033[36m{join(args.checkpoint_dir, args.checkpoint_name)}\033[0m")


if __name__ == "__main__":
    main()
