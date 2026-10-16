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

Each step builds on the ideas of the one before it, but not on its code.

1. **Bigram model by counting** (`step1_bigram_counts/`). Counts how often each
   character follows another in a list of names, turns the counts into
   probabilities and samples new names. This is the baseline every later step
   should beat.
2. **Bigram model as a neural net** (`step2_bigram_nn/`). The same model, but
   learned by gradient descent instead of counted. It introduces logits, softmax
   and cross-entropy loss, with the gradient derived by hand, and should reach
   about the same loss as step 1.
3. **Scalar autograd engine** (`step3_autograd/`). A small engine that records
   arithmetic on single numbers and computes gradients automatically with the
   chain rule. `engine.py` is the engine, `test_engine.py` checks its gradients
   against finite differences and `demo.py` trains a tiny network with it.
4. **MLP with character embeddings** (`step4_mlp/`). Predicts the next character
   from the previous three, using learned vectors for each character and a hidden
   layer. Backprop is written by hand in numpy, `test_mlp_grads.py` checks it,
   and a train/val split shows whether the model generalises.
5. **Tiny transformer** (`step5_transformer/`). A small GPT-style model trained
   on Shakespeare. Self-attention lets each position look at all earlier ones
   rather than a fixed window. PyTorch handles the gradients from here.

## Layout

- `PLAN.md`: concepts, "done" criteria and review questions for each step.
- `data/`: datasets (gitignored). `data/README.md` says where to get them.
- `pyproject.toml`: dependencies and ruff config. `uv.lock` pins exact versions.
