# tiny-lm-from-scratch

A character-level language model built in five steps, from counting bigrams to a
tiny transformer. Each step is self-contained: no code is shared between steps.
See [PLAN.md](PLAN.md) for what each step teaches and what "done" looks like.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv python install 3.12
uv add numpy
uv add --dev ruff pytest
```

Then download the datasets listed in [data/README.md](data/README.md).

## Running

Run everything from the repo root.

```sh
uv run python step1_bigram_counts/bigram_counts.py
uv run python step2_bigram_nn/bigram_nn.py
uv run python step3_autograd/demo.py
uv run python step4_mlp/mlp.py
uv run python step5_transformer/transformer.py   # run `uv add torch` first
```

Tests (steps 3 and 4 only):

```sh
uv run pytest
```

Lint and format:

```sh
uv run ruff check .
uv run ruff format .
```

## Steps

1. Bigram model by counting
2. Bigram model as a single-layer neural net
3. Scalar autograd engine
4. MLP with character embeddings
5. Tiny transformer (PyTorch)
