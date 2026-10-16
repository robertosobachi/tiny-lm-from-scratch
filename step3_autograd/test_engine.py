"""Step 3: numerical gradient check for the autograd engine.

Goal: for a few small expressions built from Value, compare the gradients from
backward() with central finite differences. If they disagree, the engine is wrong.

Run: uv run pytest step3_autograd
"""
