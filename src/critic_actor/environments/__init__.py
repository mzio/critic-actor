"""
RL Environments
"""

from .base import Environment


def load_env(name: str, **kwargs: any) -> Environment:
    """
    Load an environment by name
    """
    if name == "gpqa":
        from .gpqa.environment import GPQAEnv

        return GPQAEnv(**kwargs)

    elif name == "browsecomp_plus":
        from .browsecomp_plus.environment import BrowseCompPlusEnv

        return BrowseCompPlusEnv(**kwargs)

    else:
        raise ValueError(f"Sorry environment '{name}' not implemented.")


def get_env(name: str, **kwargs: any) -> Environment:
    """
    Alias for `load_env`
    """
    return load_env(name, **kwargs)
