"""
OpenAI Responses API LLMs
"""

import asyncio
import json
from copy import deepcopy
from typing import Any, Sequence

from openai import AsyncOpenAI, BadRequestError, OpenAI
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ResponseOutputMessage,
)

from .base import ActionFromLLM, LLM


class OpenAIResponsesLLM(LLM):
    """
    OpenAI LLM
    """

    def __init__(
        self,
        model: str,
        use_vision: bool = False,
        generation_config: Any | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, generation_config=generation_config, **kwargs)
        self.use_vision = use_vision
        self.client = OpenAI(base_url=base_url)
        self.last_usage = None

    def _sample_single(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_new_tokens: int | None = None,
        **generation_kwargs: Any,
    ) -> Response | None:
        """
        Generate a single response from the model
        """
        generation_kwargs = dict(self.generation_config, **generation_kwargs)
        if max_new_tokens is not None:
            generation_kwargs["max_output_tokens"] = max_new_tokens
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=messages,
                tools=tools,
                **generation_kwargs,
            )
            # Track token usage *before* returning
            self.last_usage = self._track_tokens(getattr(response, "usage", None))
            if self.last_usage is not None:
                print("OpenAI token usage:")
                for k, v in self.last_usage.items():
                    print(f"-> {k}: {v}")
            return response

        except BadRequestError as e:
            if e.code == "invalid_prompt":
                # log.exception("Openai invalid prompt error: %s", e)
                print("Openai invalid prompt error: %s", e)
                return None
            
            if e.type == "invalid_request_error":
                print("OpenAI invalid request error: %s", e)
                print(f"self.last_usage: {self.last_usage}")
                return None

            print(f"OpenAI error: {e}")
            print(f"type(e): {type(e)}")
            print(vars(e))
            print(f"self.last_usage: {self.last_usage}")
            raise e

    def _sample_batch(
        self,
        num_return_sequences: int = 1,
        **kwargs: Any,
    ) -> list[Response | None]:
        """
        Generate a batch of responses from the model
        """
        return [self._sample_single(**kwargs) for _ in range(num_return_sequences)]

    def sample(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_new_tokens: int = 1024,
        num_return_sequences: int = 1,
        **generation_kwargs: Any,
    ) -> list[Response | None]:
        """
        Generate response(s) from the model
        """
        return self._sample_batch(
            num_return_sequences=num_return_sequences,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            max_new_tokens=max_new_tokens,
            **generation_kwargs,
        )

    def update_messages(
        self,
        messages: list[dict[str, Any]],
        model_response: Response | None,
        prior_messages: list[
            dict[str, Any] | ResponseOutputMessage | ResponseFunctionToolCall
        ],
        interleave: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Return updated messages for the model
        - If interleave, the model_response will be interleaved with the messages:
        [model_response[0], messages[0], model_response[1], messages[1], ...]
        - Otherwise, the messages will be appended to the end of the model_response:
        [model_response[0], ..., model_response[-1], messages[0], ..., messages[-1]]
        """
        if interleave or model_response is None:
            new_messages: Sequence[ResponseOutputItem | dict[str, Any]] = []
        else:
            new_messages = deepcopy(model_response.output)
            print(f"OpenAIResponsesLLM.update_messages: {len(new_messages)}")
            for _idx, msg in enumerate(new_messages):
                print(f"new_messages {_idx}:\n\n{type(msg)} {msg}")

        for idx, message in enumerate(messages):
            if interleave and model_response is not None:
                new_messages.append(model_response.output[idx])
            if message.get("type", None) == "function_call_output":
                message.pop("role")
                try:
                    image_message = message.pop("image_output")
                    new_messages.extend([message, image_message])
                except KeyError:
                    new_messages.append(message)
            else:
                new_messages.append(message)
        return prior_messages + new_messages  # new copy

    def get_actions(self, response: Response | None) -> list[ActionFromLLM]:
        """
        Process response from OpenAI Responses API
        """
        action_list = []
        if response is None:
            return action_list
        for output in response.output:
            if output.type == "function_call":
                arguments = json.loads(output.arguments)
                text_args = ", ".join([f'{k}="{v}"' for k, v in arguments.items()])
                text_repr = f"{output.name}({text_args})"
                action_list.append(
                    ActionFromLLM(
                        role="assistant",
                        type=output.type,
                        text=text_repr,
                        call_id=output.call_id,
                        name=output.name,
                        arguments=arguments,
                    )
                )
            elif output.type == "message":  # Regular message
                action_list.append(
                    ActionFromLLM(
                        role=output.role,
                        type=output.type,
                        text=output.content[0].text,
                        call_id=None,
                        name=None,
                        arguments=None,
                    )
                )
            elif output.type == "reasoning":
                try:
                    for summary in output.summary:
                        reasoning_text = summary.text
                        action_list.append(
                            ActionFromLLM(
                                role="assistant",
                                type=output.type,
                                text=reasoning_text,
                                call_id=None,
                                name=None,
                                arguments=None,
                            )
                        )
                except Exception as e:
                    print("-> Error with OpenAIResponsesLLM.get_actions:")
                    print(f"  -> output.type: {output.type}\n  -> error: {e}")
            else:
                raise ValueError(f"Unknown output type: {output.type}")
        return action_list


class AsyncOpenAIResponsesLLM(OpenAIResponsesLLM):
    """
    Async version of OpenAI Responses API
    """

    async def _sample_single_async(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_new_tokens: int | None = None,
        **generation_kwargs: Any,
    ) -> Response | None:
        """
        Generate a single response from the model
        """
        generation_kwargs = dict(self.generation_config, **generation_kwargs)
        if max_new_tokens is not None:
            generation_kwargs["max_output_tokens"] = max_new_tokens
        async with AsyncOpenAI(base_url=None) as _ac:
            try:
                response = await _ac.responses.create(
                    model=self.model,
                    instructions=system_prompt,
                    input=messages,
                    tools=tools,
                    **generation_kwargs,
                )
                # Track token usage *before* returning
                self.last_usage = self._track_tokens(getattr(response, "usage", None))
                if self.last_usage is not None:
                    print("OpenAI token usage:")
                    for k, v in self.last_usage.items():
                        print(f"-> {k}: {v}")
                return response
            except BadRequestError as e:
                if e.code == "invalid_prompt":
                    # log.exception("Openai invalid prompt error: %s", e)
                    print("Openai invalid prompt error: %s", e)
                    return None
                raise e

    async def _sample_batch_async(
        self,
        num_return_sequences: int = 1,
        **kwargs: Any,
    ) -> list[Response | None]:
        """
        Generate a batch of responses from the model
        """
        return await asyncio.gather(
            *[self._sample_single_async(**kwargs) for _ in range(num_return_sequences)],
            return_exceptions=False,
        )

    def sample(
        self,
        system_prompt: str,
        *,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_new_tokens: int = 1024,
        num_return_sequences: int = 1,
        **generation_kwargs: Any,
    ) -> list[Response | None]:
        """
        Generate a single response from the model
        """
        try:
            return asyncio.run(
                self._sample_batch_async(
                    system_prompt=system_prompt,
                    messages=messages,
                    tools=tools,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=num_return_sequences,
                    **generation_kwargs,
                )
            )
        except Exception as e:
            print(f"asyncio.run error: {e}")
            raise e
