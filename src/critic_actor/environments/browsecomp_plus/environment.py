"""
BrowseComp Plus Environment
"""
from copy import copy, deepcopy
from functools import partial
from os.path import join
from typing import Any
import json
import pprint

import numpy as np

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

from ...llm import ActionFromLLM, Response
from ...utils.logging import print_header
from ..base import Environment, EnvironmentStateWithAnswer
from .data import process_sample, render_prompt
from .grader import BrowseCompEval
from .fewshot_prompts import get_react_messages


class BrowseCompPlusEnv(Environment):
    """
    BrowseComp Plus Environment
    """
    def __init__(
        self,
        dataset_config: dict[str, Any],
        grader_model_config: dict[str, Any],
        pretrained_model_config: dict[str, Any] | None = None,
        num_distractors: int | None = None,  # number of non-gold titles to use
        ambiguous_titles: bool = False,
        include_titles_in_prompt: bool = True,
        max_doc_tokens: int | None = None,
        num_train_samples: int = 750,
        num_val_samples: int = 30,
        num_test_samples: int = 50,
        max_steps: int = 20,
        max_tries: int = 1,
        seed: int = 0,
        split: str = "train",
        system_prompt_choice: str = "default",
        truncation_message: str = "Sorry, you have reached the maximum number of steps. Please try again.",
        prior_obs_not_in_state: bool = False,
        use_fewshot_samples: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_steps=max_steps, max_tries=max_tries, seed=seed, **kwargs)
        self.dataset_config = dataset_config
        self.pretrained_model_config = pretrained_model_config
        self.grader_model_config = grader_model_config
        self.grader_model = BrowseCompEval(grader_model_config=grader_model_config)

        # Tokenization and few-shot examples
        self.use_fewshot_samples = use_fewshot_samples
        func_call_argname = "arguments"
        if pretrained_model_config is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(**pretrained_model_config)
            if "llama" in pretrained_model_config["pretrained_model_name_or_path"].lower():
                func_call_argname = "parameters"
        else:
            self.tokenizer = None
        # For fewshot examples
        self.get_react_messages = partial(get_react_messages, func_call_argname=func_call_argname)
        print(f"Using {func_call_argname} for function calls in few-shot examples")

        self.num_distractors = num_distractors
        self.ambiguous_titles = ambiguous_titles
        self.include_titles_in_prompt = include_titles_in_prompt
        self.max_doc_tokens = max_doc_tokens
    
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.split = split

        self.system_prompt_choices = {
            "default": "You are a helpful assistant.",
            "default_hf": "You are a helpful assistant.",
        }
        self.system_prompt = self.system_prompt_choices[system_prompt_choice]
        self.truncation_message = truncation_message

        # Load data
        self.dataset = self.init_data()
        self.data_by_split = self.init_splits()

        self.console = Console()  # logging
        self.rollouts = {}  # store rollouts for each sample and generation

        # If True, we remove the prior observations from the next_state
        # -> Model should write down thoughts / only sees its prior responses
        self.prior_obs_not_in_state = prior_obs_not_in_state

    def init_data(self) -> list[dict[str, Any]]:
        """
        Initialize dataset (from pre-downloaded file)
        """
        cache_dir = self.dataset_config["cache_dir"]
        path = self.dataset_config.get("path", "decrypted_browsecomp_plus.jsonl")
        dataset_path = join(cache_dir, path.replace(".jsonl", ""))
        # try:
        #     dataset = load_from_disk(dataset_path)
        
        # except Exception as e:
        if True:
            # print(f"Error loading dataset from {dataset_path}: {e}")
            pbar = tqdm(
                total=self.num_train_samples + self.num_val_samples + self.num_test_samples,
                desc=f"-> Loading data from {join(cache_dir, path)}",
            )
            dataset = []
            with open(join(cache_dir, path), "r") as f:
                for _, line in enumerate(f):
                    sample = json.loads(line)
                    # Process individual samples
                    query_id = sample["query_id"]
                    query    = sample["query"]
                    answer   = sample["answer"]

                    all_docs_dict, all_titles, all_docs = process_sample(
                        sample=sample,
                        max_distractors=self.num_distractors,
                        ambiguous_titles=self.ambiguous_titles,
                        max_doc_tokens=self.max_doc_tokens,
                        tokenizer=self.tokenizer,
                    )
                    prompt, search_tool_desc = render_prompt(
                        query=query,
                        all_titles=all_titles,
                        include_titles_in_prompt=self.include_titles_in_prompt,
                    )
                    dataset.append({
                        "prompt": prompt,
                        "search_tool_desc": search_tool_desc,
                        "answer": answer,
                        "all_docs_dict": all_docs_dict,
                        # Extra
                        "query_id": query_id,
                        "query": query,
                        "all_docs": all_docs,
                        "all_titles": all_titles,
                    })
                    pbar.update(1)
            # dataset = Dataset.from_list(dataset)
            # dataset.save_to_disk(dataset_path)
            # print(f"Saved dataset to {dataset_path}!")
        return dataset

    def init_splits(self) -> dict[str, np.ndarray]:
        """
        Initialize splits
        """
        np.random.seed(self.seed)
        shuffle_indices = np.arange(len(self.dataset))
        np.random.shuffle(shuffle_indices)

        # Get splits
        last_val_idx  = self.num_train_samples + self.num_val_samples
        last_test_idx = last_val_idx + self.num_test_samples
        train_indices = np.arange(self.num_train_samples)
        val_indices   = np.arange(self.num_train_samples, last_val_idx)
        test_indices  = np.arange(last_val_idx, last_test_idx)

        dataset = np.array(self.dataset)[shuffle_indices]

        return {
            "train": dataset[train_indices],
            "eval":  dataset[val_indices],
            "test":  dataset[test_indices],
        }

    def __len__(self) -> int:
        """
        Number of samples
        """
        return len(self.data_by_split[self.split])

    def shuffle(self, seed: int | None = None) -> None:
        """
        Shuffle dataset
        """
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
        indices = np.arange(len(self.data_by_split[self.split]))
        np.random.shuffle(indices)
        self.data_by_split[self.split] = self.data_by_split[self.split][indices]
        
    def reset(
        self,
        sample_idx: int = 0,
        generation_idx: int = 0,
    ) -> tuple[EnvironmentStateWithAnswer, dict[str, Any]]:
        """
        Reset the environment
        """
        info = {
            "sample_idx": sample_idx,
            "generation_idx": generation_idx,
            "timestep": 0,
            "try_step": 0,
        }
        sample = self.data_by_split[self.split][sample_idx]
        messages = [
            {"role": "user", "content": sample["prompt"]}
        ]
        metadata = sample
        # Track for accuracy eval
        metadata["correct"] = 0
        metadata["total"] = 1

        if self.use_fewshot_samples:
            fewshot_messages, _ = self.get_react_messages(
                system_prompt=None,
                num_samples=1,
                # prompt_type="webthink_mc_simple6",
                include_titles_in_prompt=self.include_titles_in_prompt,
                prior_obs_not_in_state=self.prior_obs_not_in_state,
            )
            messages = fewshot_messages[0] + messages
        
        return EnvironmentStateWithAnswer(
            system_prompt=self.system_prompt,
            messages=messages,
            answer=sample["answer"],
            model_response=None,
            prior_messages=[],
            tools=[sample["search_tool_desc"]],
            metadata=metadata,
        ), info

    def step(
        self,
        **kwargs: Any,
    ) -> tuple[EnvironmentStateWithAnswer, float, bool, bool, dict[str, Any]]:
        """
        Step through the environment; see _step_impl for details
        - Returns (next_state, reward, done, truncated, info)
        """
        return self._step_impl(**kwargs)

    def _step_impl(
        self,
        parsed_actions: list[ActionFromLLM],
        model_response: list[Response | str],
        current_state: EnvironmentStateWithAnswer,
        timestep: int,
        try_step: int = 0,
        verbose: bool = False,
        sample_idx: int = 0,
        generation_idx: int = 0,
        **kwargs: Any,
    ) -> tuple[EnvironmentStateWithAnswer, float, bool, bool, dict[str, Any]]:
        """
        Subclass implementation of step
        """
        metadata = deepcopy(current_state.metadata)
        metadata["sample_idx"] = sample_idx
        metadata["generation_idx"] = generation_idx

        answer = current_state.answer
        prior_messages = current_state.prior_messages
        question = metadata["query"]

        # Use to retrieve documents
        all_docs_dict = metadata["all_docs_dict"]

        done = False
        truncated = False
        reward = 0
        updated_try_step = False

        info = {
            "sample_idx": sample_idx,
            "generation_idx": generation_idx,
            "timestep": copy(timestep),
            "try_step": copy(try_step),
            "num_messages": 0,
            "num_function_calls": 0,
        }
        # Store rollouts (and view them if verbose is True)
        messages = []
        _rollout_key = f"{sample_idx}-{generation_idx}"
        if _rollout_key not in self.rollouts:
            _title = f"(Data) Sample {sample_idx} | Generation {generation_idx}"
            self.rollouts[_rollout_key] = {
                "messages": [],
                "table": Table(title=_title),
            }
            self.rollouts[_rollout_key]["table"].add_column("role", style="cyan")
            self.rollouts[_rollout_key]["table"].add_column("content")
            self.rollouts[_rollout_key]["table"].add_column("info", style="cyan")

        if (
            len(prior_messages) > 0
            and isinstance(prior_messages[-1], dict)
            and prior_messages[-1].get("role", None) == "user"
        ):
            last_message = prior_messages[-1]
            self.rollouts[_rollout_key]["messages"].append(last_message)
            self.rollouts[_rollout_key]["table"].add_row(
                last_message['role'],
                str(last_message["content"]),
                '',
                end_section=True,
                style="magenta",
            )

        print_header("Parsed Actions")
        print(parsed_actions)
        
        # Parse actions (messages and tool calls)
        for action_idx, action in enumerate(parsed_actions):
            info_keys = ["role", "type", "name", "call_id"]
            info_dict = {"try ": try_step, "step": timestep,}
            info_dict.update({
                _k: _v
                for _k, _v in vars(action).items()
                if _v is not None and _k in info_keys
            })
            info_dict["answer"] = answer
            info_text = "\n".join([
                f"{_k}: {_v}" for _k, _v in info_dict.items()
            ])

            if action.type == "function_call":
                fc_name = action.name
                fc_args = action.arguments
                try:
                    # Execute the tool call (search in this case)
                    title = fc_args["title"]
                    assert fc_name == "search", "Only search tool is supported."
                    assert title in all_docs_dict, f"Title '{title}' not found in all_docs_dict."
                    result = all_docs_dict[title]["text"]

                    if self.prior_obs_not_in_state:
                        result = (
                            f"## Result:\n{result}\n\n"
                            f"## Next:\nWrite down your thoughts on the results and "
                            "what to do next (you won't be able to see these results "
                            "later unless you search for the title again). "
                            "Then proceed with another tool call or your final answer."
                        )
                
                except Exception as e:
                    if title not in all_docs_dict:
                        result = f"## Result:\nCould not find [{title}]"
                    else:
                        result = f"## Result:\nError during execution:\n{str(e)}"

                env_response = {
                    "role": "tool",
                    "type": "function_call_output",
                    "call_id": action.call_id,
                    "output": result,
                }
                messages.append(env_response)
                info["num_function_calls"] += 1
                if verbose:
                    style = f"color({generation_idx + 8})"  # +8 for bright colors
                    _stdout = {"name": fc_name, "args": fc_args,}
                    self.rollouts[_rollout_key]["table"].add_row(
                        f"tool_call\n(action {action_idx+1}/{len(parsed_actions)})",
                        pprint.pformat(_stdout),
                        info_text,
                        style=style,
                        end_section=True,
                    )
                    self.rollouts[_rollout_key]["table"].add_row(
                        f"tool_call_result\n(action {action_idx+1}/{len(parsed_actions)})",
                        str(result),
                        info_text,
                        style=style + " bold",
                        end_section=True,
                    )
            
            elif action.type in ["message", "reasoning"]:
                info["num_messages"] += 1
                if verbose:
                    style = f"color({generation_idx + 8})"
                    style += " italic" if action.type == "reasoning" else ''
                    self.rollouts[_rollout_key]["table"].add_row(
                        f"{action.type}\n(action {action_idx+1}/{len(parsed_actions)})",
                        action.text,
                        info_text,
                        style=style,
                        end_section=True,
                    )

                if (
                    action.type == "message"
                    and action_idx + 1 == len(parsed_actions)
                    and "Final Answer: " in action.text
                ):
                    # Last action was an answer submission
                    reward, grader_text = self.grader_model(
                        question=question,
                        correct_answer=answer,
                        response=action.text,
                        track_metrics=True,  # keep track for now
                        verbose=verbose,
                        sample_id=sample_idx,
                        generation_id=generation_idx,
                        split=self.split,
                    )

                    reward = float(reward)  # convert bool to float for reward
                    done = True
                    info["try_step"] += 1
                    updated_try_step = True

                    if reward == 1:
                        user_content = "Correct!"
                    else:  # wrong; provide feedback
                        user_content = grader_text
                    messages.append({
                        "role": "user",
                        "content": user_content,
                    })
                    metadata["correct"]  = reward
                    metadata["total"]    = 1  # explicit here
                    metadata["accuracy"] = metadata["correct"] / metadata["total"]

                elif action_idx + 1 == len(parsed_actions) and "```json" in action.text:
                    # Last action was not a proper tool call. Ask assistant to call tools
                    messages.append({
                        "role": "user",
                        "content": "Remember to invoke tools correctly!",
                    })

                elif action_idx + 1 == len(parsed_actions):
                    messages.append({
                        "role": "user",
                        "content": "Error with last response. No tool calls or Final Answer found.",
                    })

                    # self.rollouts[_rollout_key]["messages"].append(
                    #     messages[-1]
                    # )
                if len(messages) > 0:
                    try:
                        self.rollouts[_rollout_key]["table"].add_row(
                            messages[-1]["role"],
                            messages[-1]["content"],
                            info_text,
                            style="cyan",
                            end_section=True,
                        )
                    except Exception as e:
                        print(e)
                        print(messages[-1])
                        breakpoint()

            if verbose and action_idx + 1 == len(parsed_actions):
                self.console.print(self.rollouts[_rollout_key]["table"])

        metadata["timestep"] = timestep + 1

        if self.prior_obs_not_in_state:
            # may be hardcoded for LLMs with list[dict[str, Any]] chat format
            prior_messages = [
                msg for msg in prior_messages if msg.get("role", None) != "tool"
            ]

        print_header("Prior Messages")
        for _idx, _msg in enumerate(prior_messages):
            print(f"{_idx}. {str(_msg)[:1024]}")

        if len(messages) == 0:
            messages.append({
                "role": "user",
                "content": "Error with last response. No tool calls or Final Answer found.",
            })

        # Return new state
        new_state = EnvironmentStateWithAnswer(
            system_prompt=current_state.system_prompt,
            messages=messages,
            answer=answer,
            model_response=model_response,
            prior_messages=prior_messages,
            tools=current_state.tools,
            metadata=metadata,
        )
        info["timestep"] += 1  # could be metadata["timestep"]
        if info["timestep"] == self.max_steps:
            truncated = True
            done = True
            messages.append({"role": "user", "content": self.truncation_message})
            if not updated_try_step:
                info["try_step"] += 1
                updated_try_step = True
        return new_state, reward, done, truncated, info
            