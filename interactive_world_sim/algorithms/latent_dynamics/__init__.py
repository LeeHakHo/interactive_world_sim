def __getattr__(name):
    if name == "LatentWorldModel":
        from .latent_world_model import LatentWorldModel  # noqa
        return LatentWorldModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["LatentWorldModel"]
