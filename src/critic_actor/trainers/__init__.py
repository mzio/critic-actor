"""
Model trainers
"""

def load_trainer(name: str, **kwargs: any):
    """
    Load a trainer by name
    """
    if name == "critic_actor":
        from .critic_actor import CriticActorTrainer
        return CriticActorTrainer(name=name, **kwargs)

    elif name == "critic_actor_reinforce":
        from .critic_actor_reinforce import CriticActorTrainer
        return CriticActorTrainer(name=name, **kwargs)

    elif name == "critic_actor_feedback":
        from .critic_actor_feedback import CriticActorTrainer
        return CriticActorTrainer(name=name, **kwargs)

    raise NotImplementedError(f"Sorry trainer '{name}' not implemented yet.")
