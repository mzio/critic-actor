"""
Dataloader classes for handling model training over rollouts
(policy, value updates)
"""
from typing import Any


def get_update_dataloader(name: str, **kwargs: Any) -> Any | None:
    """
    Get a dataloader by name
    """
    if name == "critic_actor":
        from .critic_actor import CriticActorLoader
        if "dataset_dict" in kwargs:
            kwargs["data"] = kwargs["dataset_dict"]
            del kwargs["dataset_dict"]
        return CriticActorLoader(**kwargs)

    raise ValueError(f"Unknown dataloader: {name}")
