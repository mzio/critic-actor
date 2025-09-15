"""
GPQA Environment
"""
from typing import Any

import numpy as np

from copy import deepcopy

from datasets import Dataset as HFDataset, load_dataset


class GPQAEnv():
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
        system_prompt: str = "",
        prompt_prefix: str = "What is the correct answer to this question:\n\n",
        prompt_suffix: str = "",
        val_split_size: float = 0.1,
        split: str = "train",
        max_steps: int = 20,
        max_tries: int = 1,
        seed: int = 0,
        data_process_config: dict[str, Any] = None,
    ):
        # Setup dataset
        self.dataset_config = dataset_config
        self.prompt_column = prompt_column
        self.answer_column = answer_column
        self.remove_columns = remove_columns
        self.keep_columns = keep_columns
        
        # Setup prompts
        self.system_prompt = system_prompt
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix

        self.val_split_size = val_split_size
        self.split = split

        # Setup environment
        self.max_steps = max_steps
        self.max_tries = max_tries
        self.seed = seed

        self.data_process_config = data_process_config or {}
        self.data_process_config["with_indices"] = True

        self.datasets = self._init_data()


    def __len__(self) -> int:
        return len(self.datasets[self.split])

    def shuffle(self, seed: int = None) -> None:
        """Shuffle dataset"""
        seed = self.seed if seed is None else seed
        self.datasets[self.split] = self.datasets[self.split].shuffle(seed=seed)

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
        question = f"{self.prompt_prefix}{sample[self.prompt_column]}\n\nChoices:\n{choices_str}{self.prompt_suffix}"
        messages = [
            {"role": "user", "content": question},
        ]
        return {
            "system_prompt": self.system_prompt,
            "prompt": messages,
            "answer": answer,
            "idx": idx,
        }

