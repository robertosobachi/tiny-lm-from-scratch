# Plan

Five steps, each self-contained. No code is shared between steps.

Reference numbers come from Karpathy's makemore and nanoGPT. Treat them as
ballpark figures, not targets.

Why step 4 uses numpy and not the step 3 engine: a scalar autograd engine is far
too slow for roughly 200k training examples, and PyTorch only comes in at step 5.
So step 4 does backprop by hand, with a gradient-check test to catch mistakes.

## Step 1: bigram by counting

- **Concepts:** a language model is a probability distribution over the next
  character. Count every (char, next char) pair, normalise each row into
  probabilities and sample from them. Average negative log likelihood (NLL)
  measures how good the model is.
- **Files:** `step1_bigram_counts/bigram_counts.py`
- **Done:** prints the shape of the 27x27 count matrix, the most common bigrams,
  10 sampled names (name-shaped, mostly bad) and the average NLL over the dataset,
  around 2.45. The same seed gives the same names.
- **Questions:**
  - Why add 1 to every count? What happens to the loss without it if a name
    contains a bigram the model never saw?
  - Why are the samples bad when the probabilities are exactly right for this
    model?
- **New syntax:** `with open(...)`, `str.splitlines()`, list and dict
  comprehensions, `zip(w, w[1:])`, `np.zeros((27, 27), dtype=np.int32)`,
  `N[i, j] += 1`, `N.sum(axis=1, keepdims=True)` and broadcasting,
  `rng = np.random.default_rng(42)`, `rng.choice(27, p=row)`, `np.log`, f-strings.

## Step 2: the same bigram as a neural net

- **Concepts:** a one-hot input times a weight matrix gives logits, softmax turns
  the logits into probabilities and cross-entropy is the same NLL as in step 1.
  There is no autograd yet, so you derive the gradient of softmax plus
  cross-entropy by hand and run gradient descent. Weight regularisation does the
  job that count smoothing did in step 1.
- **Files:** `step2_bigram_nn/bigram_nn.py`
- **Done:** the loss starts near ln(27) = 3.30 and falls to roughly 2.46 to 2.5,
  close to step 1. The samples look like step 1's.
- **Questions:**
  - Why does a starting loss of about 3.30 mean the initialisation is sane?
  - Why does `W` end up close to the log of step 1's normalised counts?
- **New syntax:** `@` for matrix multiplication, `np.eye(27)[xs]` for one-hot
  vectors, fancy indexing with `probs[np.arange(n), ys]`, subtracting
  `logits.max(axis=1, keepdims=True)` for a numerically stable softmax, `.shape`,
  in-place `W -= lr * dW`.

## Step 3: scalar autograd engine

- **Concepts:** every scalar operation builds a graph node that knows its local
  derivative. `backward()` walks the graph in reverse topological order and
  applies the chain rule. When a value is used more than once, its gradients add
  up.
- **Files:**
  - `step3_autograd/engine.py`: the `Value` class.
  - `step3_autograd/test_engine.py`: compares your gradients with central finite
    differences.
  - `step3_autograd/demo.py`: a tiny neuron or MLP built from `Value` and fitted
    to a handful of points.
- **Done:** `uv run pytest step3_autograd` passes. `demo.py` prints a loss that
  drops to near zero.
- **Questions:**
  - Why `+=` on `.grad` rather than `=`?
  - Why central differences rather than forward differences? What goes wrong
    with a very small epsilon?
- **New syntax:** classes, `__init__`, operator methods (`__add__`, `__mul__`,
  `__radd__`, `__neg__`, `__pow__`, `__repr__`), closures stored as attributes,
  `set`, recursion, `isinstance`, `math.tanh` and `math.exp`. pytest: `test_*`
  functions, plain `assert`, `pytest.approx`.

## Step 4: MLP with character embeddings

- **Concepts:** each character maps to a learned embedding vector. The embeddings
  of the previous 3 characters are concatenated, passed through a tanh hidden
  layer and then a softmax. Training uses minibatch gradient descent. A
  train/val split shows whether the model is learning or memorising.
  Initialisation scale and learning rate start to matter. You write backprop by
  hand through every layer, in numpy only.
- **Files:**
  - `step4_mlp/mlp.py`
  - `step4_mlp/test_mlp_grads.py`: checks your hand-written gradients against
    finite differences on a tiny batch.
- **Done:** the gradient test passes. Train and val loss are printed during
  training. Val loss ends clearly below step 1 (reference: about 2.2 or lower).
  Samples look more like names than step 1's.
- **Questions:**
  - Why judge the model by val loss rather than train loss?
  - After training, what does a row of the embedding table mean?
  - How would you spot saturated tanh units?
- **New syntax:** gathering with an integer array, `C[X]`, which turns shape
  (B, 3) into (B, 3, d); `.reshape(B, -1)`; `np.tanh`; `np.add.at` for embedding
  gradients (`dC[X] += g` silently drops repeated indices);
  `rng.integers(0, n, size=B)`; `rng.permutation`; `if __name__ == "__main__":`,
  so the test can import functions without starting training.

## Step 5: tiny transformer (PyTorch)

- **Concepts:** self-attention lets each position weigh every earlier position,
  using queries, keys, values and a causal mask. Also: positional embeddings,
  multi-head attention, residual connections and layer norm. PyTorch's autograd
  replaces your hand-written backprop. The model trains on tiny Shakespeare.
- **Files:** `step5_transformer/transformer.py`. Run `uv add torch` first.
- **Done:** train and val loss are printed and falling. 500 generated characters
  show word-like tokens, line breaks and speaker names. Val loss ends above
  nanoGPT's ~1.47 (a much bigger model on a GPU) but well below a bigram model on
  the same text.
- **Questions:**
  - What breaks without the causal mask?
  - Why scale attention scores by 1/sqrt(head_size)?
  - What does attention give you that step 4's fixed 3-character window cannot?
- **New syntax and APIs:** `torch.nn.Module` with `forward`, `nn.Embedding`,
  `nn.Linear`, `nn.LayerNorm`, `F.cross_entropy`, `torch.tril` with
  `masked_fill`, `torch.optim.AdamW`, `loss.backward()`, `optimizer.zero_grad()`,
  `@torch.no_grad()`, `model.eval()` and `model.train()`, the `"mps"` device when
  it is available, `torch.manual_seed`.
