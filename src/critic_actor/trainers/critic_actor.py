"""
Critic-Actor Trainer
"""
from copy import copy, deepcopy
from os.path import join
from typing import Any

import json
import numpy as np

from rich import print as rich_print
from rich.console import Console
from rich.panel import Panel
# from rich.progress import Progress
from rich.table import Table
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer

from ..environments import Environment
from ..llm import LLM
from ..model.critic_actor import CriticActor
from ..model.checkpoints import load_model_checkpoint
from ..replay_buffer.critic_actor import CriticActorReplayBuffer
from ..utils.logging import print_header

from .base import BaseTrainer


class CriticActorTrainer(BaseTrainer):
    """
    Critic-Actor Trainer
    """
    def __init__(
        self,
        num_return_sequences: int,
        generation_config: dict[str, Any],
        reasoning_effort: str,
        include_reasoning_for_critic: bool = False,
        single_response_only: bool = False,
        truncation: str = "auto",  # or "disabled"
        use_early_stopping: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(generation_config=generation_config, **kwargs)
        self.num_return_sequences = num_return_sequences
        self.generation_config = generation_config

        self.reasoning_effort = reasoning_effort
        self.include_reasoning_for_critic = include_reasoning_for_critic
        self.single_response_only = single_response_only
        self.truncation = truncation

        self.replay_buffer_config = kwargs.get("replay_buffer_config", {})
        self.dataloader_config = kwargs.get("dataloader_config", {})

        self.update_train_step_counter = 0
        self.update_eval_step_counter = 0

        self.console = Console()
        self.saved_checkpoint = False
        # Early stopping during policy updates
        # -> (default off bc early stopping on loss may not make sense here)
        self.use_early_stopping = use_early_stopping

    def get_replay_buffer(self, **kwargs: Any) -> CriticActorReplayBuffer:
        """
        Return a Critic-Actor replay buffer object
        """
        return CriticActorReplayBuffer(**self.replay_buffer_config)

    def get_update_data(
        self,
        replay_buffer: CriticActorReplayBuffer,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Get training data from replay buffer
        """
        return replay_buffer.get_data(**kwargs)[0]

    def generate_and_score(
        self,
        model: CriticActor,
        env: Environment,
        replay_buffer: CriticActorReplayBuffer,
        tokenizer: AutoTokenizer,
        data_sample_id: int,
        batch_id: int,
        unique_sample_id: int,
        split: str,
        samples_per_prompt: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate and score a batch of data
        """
        # See earlier initialization code:
        # -> The LLM and (frozen) critic model are part of the environment
        llm = env.llm
        critic_model = env.critic_model
        critic_head = model
        critic_tokenizer = tokenizer

        # GRPO-like defaults
        replay_buffer.discount_factor = 1.0
        replay_buffer.normalize_returns = False
        replay_buffer.normalize_advantages = False
        replay_buffer.negative_returns = False

        all_batch_gen_metrics = {}
        reasoning = (
            {"effort": self.reasoning_effort, "summary": "auto"}
        ) if llm.model_name.startswith("o") or "gpt-5" in llm.model_name else None

        print(f"Reasoning: {reasoning}")

        critic_model_kwargs = {
            "output_hidden_states": True,
            "use_cache": False,
            "return_dict": True,
        }
        critic_model.eval()

        model_was_training = critic_head.training
        critic_head.eval()
        critic_head.to(critic_model.device, dtype=critic_model.dtype)

        # Specify this in self.evaluate() as samples_per_prompt = 1
        samples_per_prompt = samples_per_prompt or self.samples_per_prompt
        with torch.no_grad():
            # Generate rollouts sequentially
            for generation_idx in range(samples_per_prompt):
                all_gen_metrics = {
                    "reward": [],
                    "return": [],
                    "length": [],
                    "trials": [],
                    "truncated": [],
                }
                sample_id = unique_sample_id
                obs, info = env.reset(
                    sample_idx=data_sample_id,
                    generation_idx=generation_idx,
                )
                done = False
                timestep = 0
                try_step = 0
                rollout_len = 0

                # Initial critic messages
                # -> May also include system prompt
                prior_critic_messages = deepcopy(obs.prior_messages + obs.messages)

                while not done:
                    # Messages for critic (critic_model, critic_head)
                    query_state_message = prior_critic_messages
                    try:
                        _user_content = obs.messages[-1]["content"]
                    except Exception as e:
                        print(e)
                        _user_content = ""
                    key_action_messages = [
                        [{"role": "user", "content": _user_content}]
                        for _ in range(self.num_return_sequences)
                    ]
                    # Messages for actor LLM
                    messages = llm.update_messages(
                        messages=obs.messages,
                        model_response=obs.model_response,
                        prior_messages=obs.prior_messages,
                        interleave=False,
                    )

                    if self.verbose:
                        print_header(f"Message inputs (Data Sample {data_sample_id} | Gen {generation_idx} | Step {timestep})")
                        for _idx, _msg in enumerate(messages):
                            print(f"{_idx}. {str(_msg)[:1024]}")

                    # Generate actions
                    print_header("* Generating actions... *")
                    actions_generated = False
                    while not actions_generated:
                        try:
                            llm_responses = llm.sample(
                                system_prompt=obs.system_prompt,
                                messages=messages,
                                tools=obs.tools,
                                num_return_sequences=self.num_return_sequences, 
                                reasoning=reasoning,
                                **self.generation_config,
                                truncation=self.truncation,
                            )
                            llm_action_lists = [
                                llm.get_actions(response) for response in llm_responses
                            ]
                            # Score actions
                            # 1. Build up action for critic
                            for act_idx, action_list in enumerate(llm_action_lists):
                                print_header(f"Raw actions for action {act_idx}")
                                for subact_idx, action in enumerate(action_list):
                                    print(f"Action {subact_idx}: {action.text}")
                                    print("-" * 100)
                                reasoning_content = ""
                                for subact_idx, action in enumerate(action_list):
                                    if action.type == "reasoning" and self.include_reasoning_for_critic:
                                        reasoning_content += f"{action.text}\n\n"

                                    elif action.type != "reasoning" and action.type != "function_call":  #  and action.text.startswith("<think>"):
                                        _text = action.text.replace("<think>", "").replace("</think>", "").strip()
                                        if subact_idx + 1 == len(action_list) or self.single_response_only:
                                            key_action_messages[act_idx].append(
                                                {"role": "assistant", "content": _text}
                                            )

                                    elif action.type == "function_call":
                                        # if reasoning_content != "":
                                        #     reasoning_content = f"<think>\n{reasoning_content}\n</think>"
                                        #     reasoning_content = reasoning_content.replace("\n\n\n", "\n\n")
                                        content = f"{reasoning_content}\n{action.text}".replace("\n\n\n", "\n\n").strip()
                                        # Final content
                                        key_action_messages[act_idx].append(
                                            {"role": "assistant", "content": content}
                                        )

                                    if self.verbose:
                                        if subact_idx == 0:
                                            _title = (
                                                f"Data Sample {data_sample_id} | "
                                                f"Gen {generation_idx} | "
                                                f"Step {timestep} | "
                                                f"Action {act_idx}"
                                            )
                                            table = Table(title=_title)
                                            table.add_column("role")
                                            table.add_column("content")
                                        style = (
                                            f"italic color({act_idx + 8})"
                                            if action.type == "reasoning"
                                            else f"color({act_idx + 8})"  # +8 for bright color
                                        )
                                        table.add_row(
                                            f"{subact_idx:>2}. {action.type}",
                                            action.text,
                                            style=style,
                                            end_section=True,
                                        )
                                        if subact_idx + 1 == len(action_list) or self.single_response_only:
                                            self.console.print(table)
                                    if self.single_response_only:
                                        break
                                # if len(key_action_messages[act_idx]) == 1:
                                #     key_action_messages[act_idx] = [
                                #         {"role": "assistant", "content": "error"}  # dummy message, hopefully not picked
                                #     ]
                            for _act_idx in range(len(key_action_messages)):
                                if len(key_action_messages[_act_idx]) == 1:
                                    raise ValueError(f"No actions generated for act_idx {_act_idx}")
                            actions_generated = True
                        except Exception as e:
                            print(f"Error generating actions: {e}")

                    critic_tokenizer.pad_token = critic_tokenizer.eos_token
                    critic_tokenizer.padding_side = "right"

                    # 2. Tokenize actions for critic
                    critic_input_query = critic_tokenizer.apply_chat_template(
                        query_state_message,  # critic_messages[0][:state_len],
                        add_generation_prompt=True,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                        enable_thinking=True,
                    )
                    critic_input_keys = critic_tokenizer.apply_chat_template(
                        key_action_messages,
                        add_generation_prompt=False,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                        padding=True,
                        enable_thinking=True,
                    )
                    # _critic_input_keys = critic_tokenizer.apply_chat_template(key_action_messages, add_generation_prompt=False, tokenize=False, padding=True, enable_thinking=True)
                    # breakpoint()

                    if self.verbose:
                        _title = (
                            f"DECODED TOKENS ("
                            f"Data Sample {data_sample_id} | "
                            f"Gen {generation_idx} | "
                            f"Step {timestep})"
                        )
                        table = Table(title=_title)
                        table.add_column("role")
                        table.add_column("content")
                        _query_text = critic_tokenizer.decode(critic_input_query["input_ids"][0])
                        table.add_row("query", _query_text, style="magenta", end_section=True)
                        for _idx, _input_ids in enumerate(critic_input_keys["input_ids"]):
                            _text = critic_tokenizer.decode(_input_ids)
                            style = f"color({_idx + 8})"  # bright colors
                            table.add_row(f"key {_idx}", _text, style=style, end_section=True)
                        self.console.print(table)

                    # 3. Compute critic embeddings (for now no overlap)
                    embed_q = critic_model.model(  # backbone output has last_hidden_state
                        **critic_input_query.to(critic_model.device),
                        **critic_model_kwargs,
                    ).last_hidden_state
                    embed_k = critic_model.model(
                        **critic_input_keys.to(critic_model.device),
                        **critic_model_kwargs,
                    ).last_hidden_state

                    # 4. Compute critic projections
                    mask_q = critic_input_query["attention_mask"].bool()
                    mask_k = critic_input_keys["attention_mask"].bool()

                    critic_logits = critic_head(
                        embed_q=embed_q,
                        embed_k=embed_k.unsqueeze(0),  # batch_size, num_actions, seq_len, embed_dim
                        mask_q=mask_q,
                        mask_k=mask_k.unsqueeze(0),    # batch_size, num_actions, seq_len
                    )
                    critic_log_probs = F.log_softmax(critic_logits, dim=-1)[0]  # (num_actions)
                    critic_pick = critic_logits.argmax(dim=-1)
                    key_action_message = key_action_messages[critic_pick]
                    # prior_critic_messages.extend(key_action_message)  # [user_prompt, assistant_action
                    prior_critic_messages.append(key_action_message[-1])  # no extra user message from above

                    llm_response = llm_responses[critic_pick]
                    llm_action_list = llm_action_lists[critic_pick]
                    rollout_len += critic_input_keys["input_ids"][critic_pick].shape[-1]

                    if self.verbose:
                        _title = (
                            f"PICKED ACTION ({critic_pick.item()}) (Sample {data_sample_id} | "
                            f"Gen {generation_idx} | Step {timestep})"
                        )
                        table = Table(title=_title)
                        table.add_column("role")
                        table.add_column("content")
                        for _act_idx, _action in enumerate(llm_action_list):
                            style = "italic bright_magenta" if _action.type == "reasoning" else "bright_magenta"
                            table.add_row(f"{_act_idx:>2}. {_action.type}", _action.text, style=style, end_section=True)
                        self.console.print(table)

                    obs.prior_messages = messages  # messages is extended in llm.update_messages
                    next_obs, reward, done, trunc, info = env.step(
                        parsed_actions=llm_action_list,
                        model_response=llm_response,
                        current_state=obs,
                        timestep=timestep,
                        try_step=try_step,
                        verbose=self.verbose,
                        sample_idx=sample_id,
                        generation_idx=generation_idx,
                    )
                    obs = next_obs

                    # Save to replay buffer
                    embeds_q = [_embed[_mask] for _embed, _mask in zip(embed_q.cpu(), mask_q.cpu())]
                    embeds_k = [_embed[_mask] for _embed, _mask in zip(embed_k.cpu(), mask_k.cpu())]

                    _tokens_q = [_token[_mask] for _token, _mask in zip(critic_input_query["input_ids"], mask_q)]
                    _tokens_k = [_token[_mask] for _token, _mask in zip(critic_input_keys["input_ids"], mask_k)]

                    replay_buffer.add(
                        state=embeds_q[0],
                        action=embeds_k[critic_pick],
                        all_actions=embeds_k,
                        action_label=critic_pick.item(),
                        state_input_ids=_tokens_q[0],
                        action_input_ids=_tokens_k,
                        old_logprobs=critic_log_probs,
                        reward=reward,
                        done=done,
                        sample_id=sample_id,
                        timestep=timestep,
                        batch_id=batch_id,
                        is_train=split == "train",
                        data_sample_id=data_sample_id,
                        try_step=try_step,
                        generation_id=generation_idx,
                        temperature=critic_head.temperature,
                        next_obs=None,
                    )

                    # Describe and handle outputs
                    if len(obs.messages) > 0:
                        if (
                            obs.messages[-1].get("type") == "function_call_output"
                        ):
                            fc_output = json.loads(obs.messages[-1]["output"])["stdout"]
                            critic_messages = [
                                {"role": "user", "content": fc_output}
                            ]
                        else:
                            critic_messages = [
                                {"role": "user", "content": obs.messages[-1]["content"]}
                            ]
                    else:
                        _content = (
                            "Error with last response. Please reflect and try again."
                        )
                        obs.messages = [
                            {"role": "user", "content": _content}
                        ]
                        critic_messages = copy(obs.messages)

                    # [user_prompt, assistant_action, response]
                    prior_critic_messages.extend(critic_messages)

                    if self.verbose:
                        rich_print(Panel(
                            f"Run name: [green]{self.run_name}[/green]\n"
                            f"Run URL: [cyan]{self.run_url}[/cyan]",
                            title="Logging Info",
                            border_style="cyan",
                        ))

                    if done:
                        returns = replay_buffer.compute_returns(
                            last_value=0.0,
                            sample_id=sample_id,
                            is_train=split == "train",
                            try_step=try_step,
                            print_returns=self.verbose,
                        )
                        # Can use first-step return to benchmark perf
                        # -> Higher if complete task in fewer steps
                        first_return = returns[0].item()

                        all_gen_metrics["reward"].append(reward)
                        all_gen_metrics["return"].append(first_return)
                        all_gen_metrics["length"].append(rollout_len)
                        all_gen_metrics["truncated"].append(trunc)
                        all_gen_metrics["trials"].append(try_step)

                        for k, v in all_gen_metrics.items():
                            if k not in all_batch_gen_metrics:
                                all_batch_gen_metrics[k] = []
                            all_batch_gen_metrics[k].extend(v)

                    timestep = info["timestep"]
                    try_step = info["try_step"]

                unique_sample_id += 1

            if done:  # Now compute advantages
                _advantages = replay_buffer.compute_advantages(
                    data_sample_id=data_sample_id,
                    is_train=split == "train",
                    try_step=try_step - 1,  # adjust bc we updated already
                    print_advantages=self.verbose,
                )
        critic_head.train(model_was_training)
        return all_batch_gen_metrics, unique_sample_id


    def update_model(
        self,
        model: AutoModelForCausalLM,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        num_epochs: int,
        **kwargs: Any,
    ) -> tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """
        Update model over (multiple) epoch(s)
        """
        total_steps = 0
        total_grad_steps = 0
        eval_in_grad_step = False
        early_stopping = False
        self.saved_checkpoint = False

        # Reset metrics fo each round of policy updates
        self.logger_update_metrics = {"train": {}, "eval": {}}
        self.reset_metrics("model_update")

        for epoch in range(num_epochs):
            _outputs = self._update_model(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                checkpoint_path=self.checkpoint_path,
                is_train=True,
                total_steps=total_steps,
                total_grad_steps=total_grad_steps,
                eval_in_grad_step=eval_in_grad_step,
                epoch=epoch,
                eval_loader=eval_loader,
                **kwargs
            )
            model             = _outputs["model"]
            total_steps       = _outputs["total_steps"]
            total_grad_steps  = _outputs["total_grad_steps"]
            eval_in_grad_step = _outputs["eval_in_grad_step"]
            early_stopping    = _outputs["early_stopping"]

            # Could return these or save them locally
            # train_loss_metrics = _outputs["loss_metrics"]
            # eval_loss_metrics  = _outputs["eval_loss_metrics"]

            if early_stopping and self.use_early_stopping:
                print(f"Early stopping policy updates at {total_grad_steps} grad steps")
                break

        if (
            self.return_best_model and self.saved_checkpoint and self.use_early_stopping
            # and len(eval_loader) > 32  # arbitrary but large enough to not be high variance
        ):
            try:
                print_header(f"Loading best model from {self.checkpoint_path}")
                model, optimizer, scheduler = load_model_checkpoint(
                    model,
                    self.checkpoint_path,
                    optimizer=optimizer,  # type: ignore
                    scheduler=scheduler,
                )
            except Exception as e:
                print(e)
                breakpoint()
        return model, optimizer, scheduler

    def _update_model(
        self,
        model: CriticActor,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer | Any,
        scheduler: torch.optim.lr_scheduler.LRScheduler | Any,
        gradient_accumulation_steps: int,
        checkpoint_path: str,
        is_train: bool,
        total_steps: int,
        total_grad_steps: int,
        eval_in_grad_step: bool,
        epoch: int,
        eval_loader: DataLoader | None = None,
        **kwargs: Any,
    ) -> tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """
        Update model with replay buffer data
        """
        if is_train:
            model.train()
            split = "train"
        else:
            model.eval()
            split = "eval"

        if is_train:
            desc = f"\033[36m{split.capitalize()} Epoch {epoch}...\033[0m"
        else:
            desc = f"\033[92m{split.capitalize()} Epoch {epoch}...\033[0m"
        pbar = tqdm(
            dataloader, desc=desc, leave=True,
            colour="blue" if is_train else "green",
        )

        mean_loss_metrics = {}
        early_stopping = False
        grad_steps_per_update = 0
        with torch.set_grad_enabled(is_train):
            for _, data in enumerate(pbar):
                model_inputs = {
                    k: v.to(model.device)
                    for k, v in data.items()
                    if k in ["embed_q", "embed_k", "mask_q", "mask_k"]
                }
                label = data["label"].to(model.device)
                old_logprobs = data["old_logprobs"].to(model.device)
                return_ = data["return_"]
                advantage = data["advantage"].to(model.device)
                temperature = data["temperature"]
                try:
                    assert 1. / temperature[0] == model.inv_temp, (
                        f"Temperature {temperature} does not match inverse temperature {model.inv_temp} ({1 / model.inv_temp})"
                    )
                except Exception as e:
                    print(e)
                    breakpoint()

                # For transparency, we'll do the computation explicitly
                embed_q = model_inputs["embed_q"]  # (b, l, d)
                embed_k = model_inputs["embed_k"]  # (b, a, l, d)
                mask_q  = model_inputs["mask_q"]   # (b, l)
                mask_k  = model_inputs["mask_k"]   # (b, a, l)
                b, a, l, d = embed_k.shape

                embed_q = model.q_head(embed_q, mask_q).unsqueeze(1)  # so cosine works
                embed_k = model.k_head(embed_k.view(-1, l, d), mask_k.view(-1, l))
                embed_k = embed_k.view(b, a, -1)
                if model.cos_sim:
                    logits = torch.cosine_similarity(embed_q, embed_k, dim=-1) * model.inv_temp
                else:
                    logits = torch.einsum("b1e,bne->bn", embed_q, embed_k) * model.inv_temp
                try:
                    # log_probs = F.log_softmax(logits, dim=-1)[label]
                    log_probs = F.log_softmax(logits, dim=-1)
                    log_probs = torch.gather(log_probs, 1, label.long().unsqueeze(1)).squeeze(1)  # (batch_size,)
                except Exception as e:
                    print(e)
                    breakpoint()
                adv_abs = torch.abs(advantage)

                # Compute loss
                # -> If adv > 0, loss = -1 * ratio * log(p) * adv (normal)
                # -> If adv < 0, loss = -1 * ratio * log(1 - p) * |adv|
                # ratio = torch.exp(log_probs.detach() - old_logprobs[label])
                with torch.no_grad():
                    old_logprobs = torch.gather(
                        old_logprobs, 1, label.long().unsqueeze(1)
                    ).squeeze(1)  # (batch_size,)
                ratio = torch.exp(log_probs.detach() - old_logprobs)
                loss = -1 * ratio * (
                    log_probs * adv_abs * (advantage >= 0)
                    + torch.log(1 - log_probs.exp() + 1e-8) * adv_abs * (advantage < 0)
                )
                loss = loss.mean()

                if torch.isnan(loss).any():
                    breakpoint()

                if is_train:  # Update model
                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                    if (total_steps + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        # scheduler.step()
                        optimizer.zero_grad()
                        total_grad_steps += 1
                        grad_steps_per_update += 1
                        eval_in_grad_step = False
                    logger_step = copy(self.update_train_step_counter)
                    self.update_train_step_counter += 1
                else:
                    logger_step = copy(self.update_eval_step_counter)
                    self.update_eval_step_counter += 1

                loss_metrics = {
                    "loss": loss.item(),
                    "returns": return_.mean().item(),
                    "advantages": advantage.mean().item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "update_grad_steps": total_grad_steps,
                    "total_steps": logger_step,
                }
                pbar.set_postfix(**loss_metrics)

                for k, v in loss_metrics.items():
                    if k not in self.logger_update_metrics[split]:
                        self.logger_update_metrics[split][k] = []
                    self.logger_update_metrics[split][k].append(v)
                    # Log for mean loss metrics
                    if k not in mean_loss_metrics:
                        mean_loss_metrics[k] = []
                    mean_loss_metrics[k].append(v)

                # Log to logger
                if self.logger is not None:
                    _logger_metrics = {}
                    for k, v in self.logger_update_metrics[split].items():
                        # Log moving average of last 100 steps
                        _logger_metrics[f"update_{split}/{k}"] = np.mean(v[-100:])
                        _logger_metrics[f"update_{split}/{k}_std"] = np.std(v[-100:])
                    self.logger.log(_logger_metrics, step=None)
                
                # Evaluate model
                if (
                    is_train
                    and total_grad_steps > 0
                    and total_grad_steps % self.update_eval_step == 0
                    and not eval_in_grad_step  # if accumulating gradients, only eval once
                ):
                    _eval_loader = eval_loader if len(eval_loader) > 32 else dataloader
                    eval_outputs = self._update_model(
                        model, _eval_loader, optimizer, scheduler,
                        # model, dataloader, optimizer, scheduler,  # hack, just do on train again
                        gradient_accumulation_steps,
                        checkpoint_path=checkpoint_path,
                        is_train=False,
                        total_steps=total_steps,
                        total_grad_steps=total_grad_steps,
                        eval_in_grad_step=True,
                        epoch=epoch,
                        eval_loader=eval_loader,
                    )
                    eval_metrics = eval_outputs["loss_metrics"]
                    if scheduler is not None:  # assume plateau scheduler
                        scheduler.step(eval_metrics["loss"])
                    eval_in_grad_step = True

                    early_stopping = self.update_best_metric(
                        metric_name="model_update",
                        metric_val=eval_metrics["loss"],
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        total_grad_steps=total_grad_steps,
                        checkpoint_path=checkpoint_path,
                    )

                    if early_stopping and self.use_early_stopping:
                        print(
                            f"Early stopping policy updates at {total_grad_steps} grad steps"
                        )
                        break

                del model_inputs, label, return_, advantage, log_probs, adv_abs, ratio, loss
                torch.cuda.empty_cache()

                if is_train:
                    total_steps += 1

            for k, v in mean_loss_metrics.items():
                mean_loss_metrics[k] = np.mean(v)

        return {
            "model": model,
            "total_steps": total_steps,
            "total_grad_steps": total_grad_steps,
            "eval_in_grad_step": eval_in_grad_step,
            "early_stopping": early_stopping and self.use_early_stopping,  # redundant
            "loss_metrics": mean_loss_metrics,
        }
