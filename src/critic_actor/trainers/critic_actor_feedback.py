"""
Critic-Actor Trainer For a Feedback Model
"""
from copy import copy, deepcopy
from os.path import join
from typing import Any

import json
import numpy as np

from rich import print as rich_print
from rich.panel import Panel
from rich.table import Table

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer

from ..environments import Environment
from ..model.critic_actor import CriticActor
from ..replay_buffer.critic_actor import CriticActorReplayBuffer
from ..utils.logging import print_header

from .critic_actor import CriticActorTrainer



FEEDBACK_DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant guiding a user to solve a challenging question.

You first gave the user the question to solve, and the may observe the user's step-by-step thoughts or progress so far.

Now based on the user's progress so far, provide concise and constructive feedback to guide the user on what to do next.

Do NOT offer to do anything else for the user.

## Requirements
- If the user seems stuck, offer feedback on what the user is doing wrong.
- Pay attention to all parts of the user's current approach.
- *IMPORTANT*: If the user seems to have solved the question, you *SHOULD ENCOURAGE* the user to submit their solution.

You should remind the user that the final answer should be a concise sentence, in the following format: 'Final Answer: <put your answer here>'.  
"""

FEEDBACK_SOCRATIC_SYSTEM_PROMPT = """You are a helpful assistant guiding a user to solve a challenging question.

You first gave the user the question to solve, and the may observe the user's step-by-step thoughts or progress so far.

Now based on the user's progress so far, provide concise and constructive feedback to guide the user on what to do next.

Do NOT offer to do anything else for the user. Instead, use the *SOCRATIC METHOD* and *guided questioning* to help the user solve the question through their own understanding.

## Requirements
- If the user seems stuck, offer feedback on what the user is doing wrong.
- Do NOT tell them what to do. Never mention explicit next steps they should try.
- Instead, use the Socratic method and guided questioning to help the user solve the question through their own understanding.
- Do be targeted in your feedback. Considering focusing on a single hypothesis to not confuse the user.
- Pay attention to all parts of the user's current approach, including their thoughts.
- *IMPORTANT*: If the user seems to have solved the question, you *MUST ENCOURAGE* the user to submit their solution ASAP and NOW.

You should remind the user that the final answer should be a concise sentence, in the following format: 'Final Answer: <put your answer here>'.
"""

FEEDBACK_SYSTEM_PROMPT_CHOICES = {
    "default": FEEDBACK_DEFAULT_SYSTEM_PROMPT,
    "socratic": FEEDBACK_SOCRATIC_SYSTEM_PROMPT,
}

SUMMARY_SYSTEM_PROMPT = """Please *accurately yet concisely* summarize and describe the current state of the user's progress. 

# Requirements

- Pretend you are the user, so your response should be a reflection of your own progress. 
- Carefully describe the current and past approaches taken, and their outcomes so far. 
- Do *NOT* add extra details. Only describe what is striclty observed. Do *NOT* speculate what to do next.
"""


class CriticActorFeedbackTrainer(CriticActorTrainer):
    """
    Feedback Critic-Actor Trainer
    """
    def __init__(
        self,
        truncation: str = "auto",  # or "disabled"
        feedback_system_prompt_choice: str = "default",
        feedback_reasoning_effort: str = "high",
        feedback_on_reasoning: bool = True,
        feedback_truncation: str = "auto",  # or "disabled"
        discount_factor: float = 0.9,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs, truncation=truncation)
        self.truncation = truncation
        self.feedback_system_prompt_choice = feedback_system_prompt_choice
        self.feedback_reasoning_effort = feedback_reasoning_effort
        self.feedback_system_prompt = FEEDBACK_SYSTEM_PROMPT_CHOICES[
            feedback_system_prompt_choice
        ]
        self.feedback_on_reasoning = feedback_on_reasoning
        self.feedback_truncation = feedback_truncation
        self.discount_factor = discount_factor

        print(self.generation_config)

    def get_replay_buffer(self, **replay_buffer_config: Any) -> CriticActorReplayBuffer:
        """
        Return a Critic-Actor replay buffer object
        -> By default, we use discount_factor = 1 and GRPO-like "advantage"
        """
        def advantage_fn(
            returns: torch.Tensor,
            **kwargs: Any,
        ) -> torch.Tensor:
            """Dummy function which just returns the returns"""
            return returns

        replay_buffer = CriticActorReplayBuffer(**replay_buffer_config)
        replay_buffer.discount_factor = self.discount_factor
        replay_buffer.normalize_returns = False
        replay_buffer.normalize_advantages = False
        replay_buffer.negative_returns = True
        replay_buffer.advantage_fn = advantage_fn
        return replay_buffer

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
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate and score a batch of data
        """
        llm = env.llm
        critic_model = env.critic_model
        critic_actor = model
        critic_tokenizer = tokenizer

        all_gen_metrics = {
            "reward": [],
            "return": [],
            "length": [],
            "trials": [],
            "advantage": [],
            "truncated": [],
            "correct": [],
            "total": [],
            "accuracy": [],
        }
        reasoning = (
            {"effort": self.reasoning_effort, "summary": "auto"}
        ) if llm.model_name.startswith("o") or "gpt-5" in llm.model_name else None

        feedback_reasoning = (
            {"effort": self.feedback_reasoning_effort, "summary": "auto"}
        ) if llm.model_name.startswith("o") or "gpt-5" in llm.model_name else None

        print(f"Solution Reasoning: {reasoning}")
        print(f"Feedback Reasoning: {feedback_reasoning}")

        critic_model_kwargs = {
            "output_hidden_states": True,
            "use_cache": False,
            "return_dict": True,
        }
        critic_model.eval()

        model_was_training = critic_actor.training
        critic_actor.eval()
        critic_actor.to(critic_model.device, dtype=critic_model.dtype)

        with torch.no_grad():
            # Generate rollouts sequentially
            for generation_idx in range(self.samples_per_prompt):
                # sample_id = data_sample_id * self.samples_per_prompt + generation_idx
                sample_id = unique_sample_id
                obs, info = env.reset(
                    sample_idx=data_sample_id,
                    generation_idx=generation_idx,
                )
                done = False
                timestep = 0
                try_step = 0
                rollout_len = 0

                local_action_list = []

                # Chat history for feedback LLM
                prior_feedback_messages = [
                    {"role": "user", "content": "Please give me an challenging question to solve."},
                    {"role": "assistant", "content": deepcopy(obs.messages[-1]["content"])},
                ]

                # # Initial critic messages
                # # -> May also include system prompt
                # prior_critic_messages = deepcopy(obs.prior_messages + obs.messages)

                # To fill in
                embed_q = None
                mask_q = None
                embed_k = None
                mask_k = None
                critic_input_query = None
                critic_input_keys = None
                critic_log_probs = None
                critic_pick = None

                while not done:
                    # Generate solution with policy LLM
                    print_header("* Generating actions... *")
                    messages = llm.update_messages(
                        messages=obs.messages,
                        model_response=obs.model_response,
                        prior_messages=obs.prior_messages,
                        interleave=False,
                    )
                    model_response_generated = False
                    while not model_response_generated:
                        model_responses = llm.sample(
                            system_prompt=obs.system_prompt,
                            messages=messages,
                            tools=obs.tools,
                            num_return_sequences=1,
                            reasoning=reasoning,
                            **self.generation_config,
                        )
                        action_lists = [
                            llm.get_actions(_response) for _response in model_responses
                        ]
                        action_list = action_lists[0]
                        model_response = model_responses[0]
                        if len(action_list) > 0:
                            model_response_generated = True

                    if not action_list[-1].text.startswith("<summary>"):
                        for _action in action_list:
                            local_action_list.append(_action)

                    # Build up actions for feedback model
                    policy_content = ""
                    for _idx, action in enumerate(action_list):
                        if action.type == "reasoning" and self.feedback_on_reasoning:
                            # prior_critic_messages.append(
                            #     {"role": "user", "content": action.text}
                            # )
                            if policy_content == "":
                                policy_content += "<think>\n"
                            policy_content += f"{action.text}\n\n"

                        elif action.type != "reasoning" and action.type != "function_call":
                            _text = copy(action.text)
                            if _text.startswith("<think>"):
                                _text = _text[len("<think>"):]
                            if _text.endswith("</think>"):
                                _text = _text[:-len("</think>")]
                            # prior_critic_messages.append(
                            #     {"role": "user", "content": _text}
                            # )
                            if policy_content == "":
                                policy_content += "<think>\n"
                            policy_content += f"{_text}\n\n"

                        elif action.type == "function_call":
                            if policy_content != "":
                                policy_content += "\n</think>\n\n"
                            _text = f"<tool_call>\n{action.text}\n</tool_call>"
                            policy_content += f"{_text}\n\n"
                            
                            prior_feedback_messages.append(
                                {"role": "user", "content": policy_content}
                            )

                        elif _idx + 1 == len(action_list):
                            if policy_content != "":
                                policy_content += "\n</think>\n\n"
                            prior_feedback_messages.append(
                                {"role": "user", "content": policy_content}
                            )

                    obs.prior_messages = messages
                    next_obs, reward, done, trunc, info = env.step(
                        parsed_actions=action_list,
                        model_response=model_response,
                        current_state=obs,
                        timestep=timestep,
                        try_step=try_step,
                        verbose=True,
                        sample_idx=sample_id,
                        generation_idx=generation_idx,
                    )

                    obs = next_obs

                    # Save to replay buffer (from last round)
                    if embed_q is not None:
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
                            temperature=critic_actor.temperature,
                            next_obs=None,
                        )

                        if done:
                            returns = replay_buffer.compute_returns(
                                last_value=0.0,
                                sample_id=sample_id,
                                is_train=split == "train",
                                try_step=try_step,
                                print_returns=self.verbose,
                            )
                            advantages = replay_buffer.compute_advantages(
                                data_sample_id=data_sample_id,
                                is_train=split == "train",
                                try_step=try_step,
                                print_advantages=self.verbose,
                            )

                            if len(returns) == 1:
                                breakpoint()

                            # breakpoint()
                            # Can use first-step return to benchmark perf
                            # -> Higher if complete task in fewer steps
                            first_return = returns[0].item()
                            first_advantage = advantages[0].item()

                            all_gen_metrics["reward"].append(reward)
                            all_gen_metrics["return"].append(first_return)
                            all_gen_metrics["length"].append(rollout_len)
                            all_gen_metrics["advantage"].append(first_advantage)
                            all_gen_metrics["truncated"].append(trunc)
                            all_gen_metrics["trials"].append(try_step)

                            if self.verbose:
                                print_header(f"({self.run_name})")

                            unique_sample_id += 1

                        timestep = info["timestep"]
                        try_step = info["try_step"]

                        if done and try_step < self.max_tries:  # more attempts
                            done = False
                            timestep = 0

                    # Describe / handle outputs
                    if not done and len(obs.messages) > 0:
                        if obs.messages[-1].get("type") == "function_call_output":
                            fc_output = json.loads(obs.messages[-1]["output"])["stdout"]
                            content = f"The code returned the following output(s):\n\n{fc_output}"
                            feedback_messages = [
                                {"role": "user", "content": content}
                            ]
                        else:
                            feedback_messages = [
                                {"role": "user", "content": obs.messages[-1]["content"]}
                            ]
                        
                        prior_feedback_messages.extend(feedback_messages)
                    
                    # Get feedback
                    if not done:
                        print_header("* Thinking about feedback... *")
                        print("Prior critic messages:")
                        for _idx, msg in enumerate(prior_feedback_messages):
                            if isinstance(msg["content"], list):
                                _msg = deepcopy(msg)
                                _msg["content"] = [_msg["content"][0]]
                            else:
                                _msg = deepcopy(msg)
                            print(f"{_idx:2d}. {str(_msg)[:1024]}")

                        # May want to summarize the current state
                        summary_response = llm.sample(
                            system_prompt=SUMMARY_SYSTEM_PROMPT,
                            messages=prior_feedback_messages,
                            num_return_sequences=1,
                            **self.generation_config,
                            # temperature not always supported for OAI Responses API
                        )
                        summary_list = [
                            llm.get_actions(_response) for _response in summary_response
                        ][0]
                        summary_content = summary_list[-1].text.strip()

                        print_header("SUMMARY:")
                        print(summary_content)

                        # Messages for critic (critic_model, critic_actor)
                        # query_state_message = prior_critic_messages
                        # key_action_messages = [
                        #     [{"role": "user", "content": ""}]
                        #     for _ in range(self.num_return_sequences)
                        # ]
                        query_state_message = [
                            {"role": "user", "content": summary_content}
                        ]
                        key_action_messages = [
                            [{"role": "user", "content": summary_content}]
                            if self.include_reasoning_for_critic
                            else []
                            for _ in range(self.num_return_sequences)
                        ]
                        # Generate potential feedback
                        if self.feedback_truncation is not None:
                            feedback_gen_config = deepcopy(self.generation_config)
                            feedback_gen_config["truncation"] = self.feedback_truncation

                        print_header("** Generating potential feedback... **")
                        print(f"Include reasoning for critic: {self.include_reasoning_for_critic}")

                        sampled_feedback = False
                        while not sampled_feedback:
                            try:
                                feedback_model_responses = llm.sample(
                                    system_prompt=self.feedback_system_prompt,
                                    messages=prior_feedback_messages,
                                    num_return_sequences=self.num_return_sequences,
                                    reasoning=feedback_reasoning,
                                    **feedback_gen_config,
                                )
                                # Process and parse the feedback
                                feedback_action_lists = [
                                    llm.get_actions(_response) for _response in feedback_model_responses
                                ]
                                feedback_final_list = ["" for _ in range(self.num_return_sequences)]
                                # Score actions
                                # 1. Build up action for critic
                                num_good = 0
                                for act_idx, action_list in enumerate(feedback_action_lists):
                                    reasoning_content = ""  # for Qwen models
                                    for subact_idx, action in enumerate(action_list):
                                        if action.type == "reasoning" and self.include_reasoning_for_critic:
                                            reasoning_content += f"{action.text}\n\n"
                                        elif action.text.startswith("<think>"):
                                            _text = action.text[len("<think>"):-len("</think>")].strip()
                                            reasoning_content += f"{_text}\n\n"
                                        elif action.type != "reasoning":
                                            # if reasoning_content != "":
                                            #     reasoning_content = f"<think>\n{reasoning_content}\n</think>"
                                            #     reasoning_content = reasoning_content.replace("\n\n\n", "\n\n")
                                            content = f"{reasoning_content}\n{action.text}".replace("\n\n\n", "\n\n").strip()
                                            # Final content
                                            key_action_messages[act_idx].append(
                                                {"role": "assistant", "content": content}
                                            )
                                            feedback_final_list[act_idx] = action.text
                                        # reasoning response (total generation was truncated)
                                        elif subact_idx + 1 == len(action_list) or self.single_response_only:
                                            key_action_messages[act_idx].append(
                                                {"role": "assistant", "content": action.text}
                                            )
                                            feedback_final_list[act_idx] = action.text
                                        
                                        if self.verbose:
                                            if subact_idx == 0:
                                                _title = (
                                                    f"Sample {data_sample_id} | "
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
                                    
                                    if key_action_messages[act_idx] == []:
                                        key_action_messages[act_idx] = [
                                            {"role": "assistant", "content": "error"}  # dummy message, hopefully not picked
                                        ]
                                    else:
                                        num_good += 1

                                assert num_good == self.num_return_sequences
                            except Exception as e:
                                print(f"Error generating feedback: {e}")
                                sampled_feedback = False

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

                        _critic_decoded_tokens = critic_tokenizer.batch_decode(
                            critic_input_keys["input_ids"], skip_special_tokens=True,
                        )
                        for _key_idx, _key_tokens in enumerate(_critic_decoded_tokens):
                            print_header(f"key {_key_idx} tokens")
                            print(_key_tokens)
                            print("-" * 100)

                        if self.verbose:
                            _title = (
                                f"DECODED TOKENS ("
                                f"Sample {data_sample_id} | "
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

                        critic_logits = critic_actor(
                            embed_q=embed_q,
                            embed_k=embed_k.unsqueeze(0),  # batch_size, num_actions, seq_len, embed_dim
                            mask_q=mask_q,
                            mask_k=mask_k.unsqueeze(0),    # batch_size, num_actions, seq_len
                        )
                        critic_log_probs = F.log_softmax(critic_logits, dim=-1)[0]  # (num_actions)
                        critic_pick = critic_logits.argmax(dim=-1)
                        # key_action_message = key_action_messages[critic_pick]
                        
                        # prior_critic_messages.extend(key_action_message)  # [user_prompt, assistant_action

                        feedback_response    = feedback_model_responses[critic_pick]
                        feedback_action_list = feedback_action_lists[critic_pick]
                        feedback_final       = feedback_final_list[critic_pick]
                        rollout_len += critic_input_keys["input_ids"][critic_pick].shape[-1]

                        # prior_feedback_messages.extend(feedback_response)  # [user_prompt, assistant_action, response]
                        prior_feedback_messages.append(
                            {"role": "assistant", "content": feedback_final}
                        )

                        if self.verbose:
                            _title = (
                                f"PICKED ACTION (Sample {data_sample_id} | "
                                f"Gen {generation_idx} | Step {timestep})"
                            )
                            table = Table(title=_title)
                            table.add_column("role")
                            table.add_column("content")
                            for _act_idx, _action in enumerate(feedback_action_list):
                                style = "italic bright_magenta" if _action.type == "reasoning" else "bright_magenta"
                                table.add_row(f"{_act_idx:>2}. {_action.type}", _action.text, style=style, end_section=True)
                            self.console.print(table)

                        # Pass the feedback back to the policy model
                        obs.messages.append({
                            "role": "user",
                            "content": feedback_final,
                        })

                        # Reset actions leading up to feedback
                        local_action_list = []

                    else:  # done
                        if self.verbose:
                            print_header(f"({self.run_name})")

                        # # Save to replay buffer (from last round)
                        # embeds_q = [_embed[_mask] for _embed, _mask in zip(embed_q.cpu(), mask_q.cpu())]
                        # embeds_k = [_embed[_mask] for _embed, _mask in zip(embed_k.cpu(), mask_k.cpu())]

                        # _tokens_q = [_token[_mask] for _token, _mask in zip(critic_input_query["input_ids"], mask_q)]
                        # _tokens_k = [_token[_mask] for _token, _mask in zip(critic_input_keys["input_ids"], mask_k)]

                        # replay_buffer.add(
                        #     state=embeds_q[0],
                        #     action=embeds_k[critic_pick],
                        #     all_actions=embeds_k,
                        #     action_label=critic_pick.item(),
                        #     state_input_ids=_tokens_q[0],
                        #     action_input_ids=_tokens_k,
                        #     old_logprobs=critic_log_probs,
                        #     reward=reward,
                        #     done=done,
                        #     sample_id=sample_id,
                        #     timestep=timestep,
                        #     batch_id=batch_id,
                        #     is_train=split == "train",
                        #     data_sample_id=data_sample_id,
                        #     try_step=try_step,
                        #     generation_id=generation_idx,
                        #     temperature=critic_actor.temperature,
                        #     next_obs=None,
                        # )
                        # returns = replay_buffer.compute_returns(
                        #     last_value=0.0,
                        #     sample_id=sample_id,
                        #     is_train=split == "train",
                        #     try_step=try_step,
                        #     print_returns=self.verbose,
                        # )
                        # advantages = replay_buffer.compute_advantages(
                        #     data_sample_id=data_sample_id,
                        #     is_train=split == "train",
                        #     try_step=try_step,
                        #     print_advantages=self.verbose,
                        # )

                        # breakpoint()
                        # # Can use first-step return to benchmark perf
                        # # -> Higher if complete task in fewer steps
                        # first_return = returns[0].item()
                        # first_advantage = advantages[0].item()

                        # all_gen_metrics["reward"].append(reward)
                        # all_gen_metrics["return"].append(first_return)
                        # all_gen_metrics["length"].append(rollout_len)
                        # all_gen_metrics["advantage"].append(first_advantage)
                        # all_gen_metrics["truncated"].append(trunc)
                        # all_gen_metrics["trials"].append(try_step)
                        # # all_gen_metrics["correct"].append(correct)
                        # # all_gen_metrics["total"].append(total)
                        # # all_gen_metrics["accuracy"].append(accuracy)
                        # # all_gen_metrics["img_save_path"].append(img_save_path)

                        # if not trunc:  # we submitted something
                        #     eval_result = info["eval_result"]
                        #     all_gen_metrics["img_save_path"].append(img_save_path)
                        # else:
                        #     # Hack for ARC
                        #     _metadata = obs.metadata
                        #     # score_input_grids = _metadata["score_input_grids"]
                        #     eval_result = {
                        #         "correct": [0],
                        #         "accuracy": 0,
                        #     }
                        #     all_gen_metrics["img_save_path"].append("")

                        # all_gen_metrics["correct"].append(sum(eval_result["correct"]))
                        # all_gen_metrics["total"].append(len(eval_result["correct"]))
                        # all_gen_metrics["accuracy"].append(eval_result["accuracy"])

                        # # timestep = info["timestep"]
                        # # try_step = info["try_step"]

                        # if try_step < self.max_tries:  # more attempts
                        #     done = False
                        #     timestep = 0

                        # if self.verbose:
                        #     print_header(f"({self.run_name})")

                    # timestep = info["timestep"]
                    # try_step = info["try_step"]
                    # obs = next_obs

                # unique_sample_id += 1

        critic_actor.train(model_was_training)

        if self.verbose:
            rich_print(Panel(
                f"Run name: [green]{self.run_name}[/green]\n"
                f"Run URL: [cyan]{self.run_url}[/cyan]",
                title="Logging Info",
                border_style="cyan",
            ))

        return all_gen_metrics, unique_sample_id
