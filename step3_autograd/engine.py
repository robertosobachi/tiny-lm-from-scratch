"""Step 3: minimal scalar autograd engine.

Goal: a Value class that wraps a single number, records the operations that
produced it, and computes gradients of every upstream Value with backward() via
the chain rule.

Imported by demo.py and test_engine.py in this folder.
"""
