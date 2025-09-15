"""
Basic logging helpers
"""
from omegaconf import OmegaConf, DictConfig, ListConfig
import rich.syntax
import rich.tree


def print_header(x, border="both") -> None:
    """Print with borders"""
    print("-" * len(x))
    print(x)
    print("-" * len(x))


def print_config(config: DictConfig, name: str = "CONFIG", style="bright") -> None:
    """
    Prints content of DictConfig using Rich library and its tree structure.
    """
    tree = rich.tree.Tree(name, style=style, guide_style=style)
    fields = config.keys()
    for field in fields:
        branch = tree.add(str(field), style=style, guide_style=style)
        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=True)
        elif isinstance(config_section, ListConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=True)
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)
