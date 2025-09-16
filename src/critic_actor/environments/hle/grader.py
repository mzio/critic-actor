"""
LLM-based grader for BrowseComp Plus Environment

Copied and modified from https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py
"""
import re
from typing import Any

import numpy as np

from ...llm import LLM, load_llm

# from: https://github.com/centerforaisafety/hle/blob/7b6be5aad6f9b43af3857de7867f3b52f6e4acb3/hle_eval/run_judge_results.py#L16-L33
GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.
```

If the extracted_final_answer is incorrect, provide a detailed explanation of why it is incorrect."""  # No confidence score for now
# confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
# """.strip()

CHOICE_STRINGS = ["yes", "no"]


class HumanitiesLastExamEval():
    """
    LLM Grader for Humanities Last Exam
    """
    def __init__(
        self,
        grader_model: LLM | None = None,
        grader_model_config: dict[str, Any] | None = None,
        max_new_tokens: int = 4096,
    ):
        self.grader_model = grader_model or load_llm(**grader_model_config)
        self.max_new_tokens = max_new_tokens
        # Optional for tracking metrics
        self.metrics = {
            "correct": [],
            "sample_id": [],
            "generation_id": [],
            "split": [],
        }
        self.running_metrics = {}

    def grade_sample(
        self,
        question: str,
        correct_answer: str,
        response: str,
    ) -> tuple[str, str]:
        """
        Grade a sample response to a question
        """
        grader_prompt = GRADER_TEMPLATE.format(
            question=question,
            correct_answer=correct_answer,
            response=response,
        ).strip()
        prompt_messages = [
            {"role": "user", "content": grader_prompt}
        ]
        sampler_response = self.grader_model.sample(
            system_prompt=None,
            messages=prompt_messages,
            tools=None,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=1,
            temperature=0.0,
        )[0]
        grading_response = self.grader_model.get_actions(sampler_response)[-1].text

        # match = re.search(r"correct: (yes|no)", grading_response)
        # match = match.group(0) if match else "no"  # Default to "no" if no match
        # MZ 09/07/25: above match grouping seems wrong, below is corrected
        match = re.search(r"correct:\s*(yes|no)\b", grading_response, flags=re.IGNORECASE)
        match = match.group(1).lower() if match else "no"   # -> 'yes' or 'no'
        return match, grading_response.strip()

    def __call__(self,
        question: str,
        correct_answer: str,
        response: str,
        track_metrics: bool = False,
        verbose: bool = False,
        sample_id: int = 0,
        generation_id: int = 0,
        split: str = "train",
    ) -> tuple[bool, str]:
        """
        Call grader on a single sample
        """
        grade_result, grade_msg = self.grade_sample(question, correct_answer, response)
        # Metrics based on grading response
        is_correct = grade_result == "yes"
        # is_incorrect = grade_result == "no"
        
        if track_metrics:
            self.metrics["correct"].append(is_correct)
            self.metrics["sample_id"].append(sample_id)
            self.metrics["generation_id"].append(generation_id)
            self.metrics["split"].append(split)

            for split in np.unique(self.metrics["split"]):
                _mask = np.array(self.metrics["split"]) == split
                _correct = np.array(self.metrics["correct"])[_mask].sum()
                _total = np.sum(_mask)
                _acc = _correct / _total * 100
                self.running_metrics[f"{split}/acc"] = _acc
                self.running_metrics[f"{split}/correct"] = _correct
                self.running_metrics[f"{split}/total"] = _total
            
            if verbose:
                for name, value in self.running_metrics.items():
                    _prefix = '%' if name.endswith('/acc') else ''
                    print(f"running {name}: {value:.2f}{_prefix}")
        
        return is_correct, grade_msg
