"""Allow ``python -m diffusionlab`` as an alternative to the console script."""

from diffusionlab.cli import entrypoint

if __name__ == "__main__":
    entrypoint()
