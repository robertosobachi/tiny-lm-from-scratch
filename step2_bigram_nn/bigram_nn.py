"""Step 2: the same bigram model as a single-layer neural net.

Goal: replace counting with a weight matrix trained by gradient descent. One-hot
inputs give logits, softmax gives probabilities, cross-entropy gives the loss. The
gradient is derived by hand. The final loss should land close to step 1's.

Run: uv run python step2_bigram_nn/bigram_nn.py
"""
