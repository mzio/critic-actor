"""
GPQA Environment
"""
from copy import copy, deepcopy
from typing import Any

import numpy as np

from datasets import Dataset as HFDataset, load_dataset

from ...llm import ActionFromLLM, Response
from ..base import Environment, EnvironmentStateWithAnswer
from .grader import score_response


SYSTEM_PROMPT_CHOICES = {
    "default": (
        "You are a step-by-step reasoning assistant."
        "\n\nFor a given question, respond with **ONLY** your *next-step* thought. "
        # " thought. the next result of your prior reasoning steps." 
        " This can be a plan, a reflection, or the result of your prior reasoning steps."
        " You *must* pay attention to your prior reasoning: if you previously said you will do"
        "do something, do it.\n\n"
        "Only when the user *allows*, provide the single most likely answer choice based on your reasoning." 
        " Answer in the format \"The correct answer is (insert answer here).\""
    )
}

USER_PROMPT = (
    "Ok, please proceed. Use your past reasoning to respond with a plan, "
    "a reflection or summary of your progress, your next reasoning step, "
    "or your carrying out prior plans. Remember if you previously said you would do "
    "something, you must do it."
)
# USER_PROMPT = "Ok, please proceed. Carry out your prior plans."


class GPQAEnv(Environment):
    """
    GPQA Environment
    """
    def __init__(
        self,
        dataset_config: dict[str, Any],
        prompt_column: str = "Question",
        answer_column: str = "Correct Answer",
        remove_columns: list[str] = None,
        keep_columns: list[str] = None,
        system_prompt: str = "default",
        prompt_prefix: str = "What is the correct answer to this question:\n\n",
        prompt_suffix: str = "",
        val_split_size: float = 0.1,
        split: str = "train",
        max_turns: int = 10,
        max_tries: int = 1,
        seed: int = 0,
        data_process_config: dict[str, Any] = None,
        truncation_message: str = "Sorry, too many turns.",
        reasoning_steps: int = 5,
        **kwargs: Any,
    ):
        super().__init__(
            max_turns=max_turns,
            max_tries=max_tries,
            seed=seed,
            truncation_message=truncation_message,
            **kwargs,
        )
        # Setup dataset
        self.dataset_config = dataset_config
        self.prompt_column = prompt_column
        self.answer_column = answer_column
        self.remove_columns = remove_columns
        self.keep_columns = keep_columns
        
        # Setup prompts
        self.system_prompt = SYSTEM_PROMPT_CHOICES[system_prompt]
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix

        self.val_split_size = val_split_size
        self.split = split

        # Setup environment
        self.max_turns = max_turns
        self.max_tries = max_tries
        self.seed = seed

        self.reasoning_steps = reasoning_steps

        self.data_process_config = data_process_config or {}
        self.data_process_config["with_indices"] = True

        self.data_by_split = self._init_data()
        if self.split == "train":
            self.shuffle()

    def __len__(self) -> int:
        return len(self.data_by_split[self.split])

    def shuffle(self, seed: int = None) -> None:
        """Shuffle dataset"""
        seed = self.seed if seed is None else seed
        self.data_by_split[self.split] = (
            self.data_by_split[self.split].shuffle(seed=seed)
        )

    def reset(
        self,
        sample_idx: int,
        generation_idx: int,
    ) -> tuple[EnvironmentStateWithAnswer, dict[str, Any]]:
        """
        Load a new sample and reset the environment
        """
        sample = self.data_by_split[self.split][sample_idx]
        info = {
            "sample_idx": sample_idx,
            "generation_idx": generation_idx,
            "timestep": 0,
            "try_step": 0,
            "question": sample["prompt"],
            "answer": sample["answer"],
        }
        messages = [
            {"role": "user", "content": sample["prompt"]},
        ]
        return EnvironmentStateWithAnswer(
            system_prompt=self.system_prompt,
            messages=messages,
            answer=sample["answer"],
            model_response=None,
            prior_messages=[],
            tools=None,
            metadata=info,
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
        question = metadata["question"]

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
        
        # Parse actions (messages and tool calls)
        for action_idx, action in enumerate(parsed_actions):
            print(f"(DEBUGGING action_idx {action_idx + 1} / {len(parsed_actions)})\n", action)
            if action.type in ["message", "reasoning"]:
                info["num_messages"] += 1
                if (
                    action.type == "message"
                    and action_idx + 1 == len(parsed_actions)
                    and "The correct answer is " in action.text
                ):
                    # Last action was an answer submission
                    reward = score_response(
                        question=question,
                        correct_answer=answer,
                        response=action.text,
                    )
                    reward = float(reward)  # convert bool to float for reward
                    done = True

                    user_content = "Correct!" if reward == 1 else "Incorrect!"
                    messages.append({
                        "role": "user",
                        "content": user_content,
                    })
                    metadata["correct"]  = reward
                    metadata["total"]    = 1  # explicit here
                    metadata["accuracy"] = metadata["correct"] / metadata["total"]

                elif action_idx + 1 == len(parsed_actions):  # 
                    user_prompt = copy(USER_PROMPT)
                    if timestep + 1 >= self.reasoning_steps:  # +1 bc timesteps start at 0
                        user_prompt += (
                            " You may also provide your final answer." 
                            " Answer in the format \"The correct answer is (insert answer here).\""
                        )
                    messages.append({
                        "role": "user",
                        "content": user_prompt,
                    })

        metadata["timestep"] = timestep + 1
        if len(messages) == 0:
            messages.append({
                "role": "user",
                "content": "Error with last response. No messages or thoughts found.",
            })

        # Return new state
        new_state = EnvironmentStateWithAnswer(
            system_prompt=current_state.system_prompt,
            messages=messages,
            answer=answer,
            model_response=model_response,
            prior_messages=prior_messages,
            tools=[],
            metadata=metadata,
        )
        info["timestep"] += 1  # could be metadata["timestep"]
        if info["timestep"] == self.max_turns:
            truncated = True
            done = True
            messages.append({"role": "user", "content": self.truncation_message})
            if not updated_try_step:
                info["try_step"] += 1
                updated_try_step = True
        return new_state, reward, done, truncated, info

    def _init_data(self) -> dict[str, HFDataset]:
        """
        Initialize and preprocess samples
        """
        # Get training set of samples not in the diamond set
        # 1. First get diamond prompts
        dataset_config = deepcopy(self.dataset_config)
        dataset_config["name"] = "gpqa_diamond"
        diamond_dataset = load_dataset(**dataset_config)['train']
        diamond_prompts = set(diamond_dataset[self.prompt_column])

        # 2. Then get extended prompts and filter
        dataset_config["name"] = "gpqa_extended"
        extended_dataset = load_dataset(**dataset_config)["train"]
        extended_dataset = extended_dataset.filter(
            lambda x: x[self.prompt_column] not in diamond_prompts
        )
        datasets_extended = extended_dataset.train_test_split(
            test_size=self.val_split_size, 
            seed=self.seed
        )
        datasets = {
            "train": datasets_extended['train'],
            "eval": datasets_extended['test'],
            "test": diamond_dataset
        }
        # Apply additional data processing
        if self.remove_columns is not None:
            remove_columns = self.remove_columns
        else:
            remove_columns = datasets["train"].column_names
        if self.keep_columns is not None:
            remove_columns = [
                col for col in remove_columns if col not in self.keep_columns
            ]
        self.data_process_config["remove_columns"] = remove_columns

        for split in datasets:
            datasets[split] = datasets[split].map(self._preprocess_data_fn,
                                                  **self.data_process_config)
            # print(f'{split} samples:')
            # for i in range(min(5, len(datasets[split]))):
            #     print(f'Question {i}: {datasets[split][i]["prompt"]}')
            #     print(f'Answer {i}: {datasets[split][i]["answer"]}')
        return datasets

    def _preprocess_data_fn(self, sample: dict[str,any], idx: int,) -> dict[str,any]:
        """
        Preprocess individual data samples into multiple-choice questions.
        """
        # Format MC question
        choices = [
            sample["Incorrect Answer 1"],
            sample["Incorrect Answer 2"],
            sample["Incorrect Answer 3"],
            sample["Correct Answer"],
        ]
        np.random.shuffle(choices)  # Shuffle the choices
        correct_idx = [
            idx for idx, choice in enumerate(choices)
            if choice == sample["Correct Answer"]
        ][0]
        choices = [  # Get alphabet bullet (A. B. C. D.)
            f"{chr(97 + i).capitalize()}. {choice}".strip()
            for i, choice in enumerate(choices)
        ]
        answer = choices[correct_idx][:1]

        choices_str = "\n".join(choices)
        question = (
            f"{self.prompt_prefix}{sample[self.prompt_column]}\n\n"
            f"Choices:\n{choices_str}{self.prompt_suffix}"
        )
        return {
            "system_prompt": self.system_prompt,
            "prompt": question,
            "answer": answer,
            "idx": idx,
        }
