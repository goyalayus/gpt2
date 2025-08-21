#!/usr/bin/env python3
# nn.py – minimal character-level language model with a tiny autograd engine
# Fixed to eliminate RecursionError and streamline loss computation

import math
import os
import pickle
import random
import sys

# --------------------------------------------------------------------------
# GLOBAL SAFETY: lift the recursion limit once, before any backward passes
# --------------------------------------------------------------------------
sys.setrecursionlimit(50_000)

# --------------------------------------------------------------------------
# Part 1: The Autograd Engine (Value object)
# --------------------------------------------------------------------------
class Value:
    """A scalar value that supports automatic differentiation."""
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None          # populated by ops
        self._prev = set(_children)            # parents in the graph
        self._op = _op                         # operation label
        self.label = label                     # optional name

    # ---- magic methods ----------------------------------------------------
    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad  += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def __pow__(self, exp: float):
        assert isinstance(exp, (int, float)), "exp must be a number"
        out = Value(self.data ** exp, (self,), f'**{exp}')

        def _backward():
            self.grad += exp * (self.data ** (exp - 1)) * out.grad
        out._backward = _backward
        return out

    # activation functions
    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad
        out._backward = _backward
        return out

    # convenience wrappers
    def __neg__(self):             return self * -1
    def __radd__(self, other):     return self + other
    def __sub__(self, other):      return self + (-other)
    def __rsub__(self, other):     return other + (-self)
    def __rmul__(self, other):     return self * other
    def __truediv__(self, other):  return self * other ** -1
    def __rtruediv__(self, other): return other * self ** -1

    # ---- back-prop --------------------------------------------------------
    def backward(self):
        """Iterative topological sort to avoid Python recursion limits."""
        topo, visited, stack = [], set(), [self]
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                topo.append(v)
                stack.extend(v._prev)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

# --------------------------------------------------------------------------
# Part 2: Simple neural-net building blocks
# --------------------------------------------------------------------------
class Module:
    def parameters(self): return []
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

class Neuron(Module):
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# --------------------------------------------------------------------------
# Part 3: Data preparation
# --------------------------------------------------------------------------
print("\n--- Preparing Data ---")
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars       = sorted(set(text))
vocab_size  = len(chars)
print(f"Dataset has {len(text)} characters.")
print(f"Vocabulary size is {vocab_size}.")

stoi   = {ch: i for i, ch in enumerate(chars)}
itos   = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]

block_size = 8
X, Y = [], []
for i in range(len(text) - block_size):
    context = text[i : i + block_size]
    target  = text[i + block_size]
    X.append(encode(context))
    Y.append(stoi[target])
print(f"Created {len(X)} examples.")

# --------------------------------------------------------------------------
# Part 4: Model, training loop, save / load
# --------------------------------------------------------------------------
# hyper-parameters
n_embd        = 10
n_hidden      = 64
max_steps     = 2_000
batch_size    = 32
learning_rate = 0.1
parameter_file = "model_parameters.pkl"

# create character embeddings (list of lists of Value)
C = [
    [Value(random.uniform(-1, 1)) for _ in range(n_embd)]
    for _ in range(vocab_size)
]

model = MLP(n_embd * block_size, [n_hidden, vocab_size])
parameters = [p for row in C for p in row] + model.parameters()
print(f"Total number of parameters: {len(parameters)}")

# load saved parameters if available
if os.path.exists(parameter_file):
    print(f"\n--- Loading parameters from {parameter_file} ---")
    with open(parameter_file, "rb") as f:
        saved = pickle.load(f)
    if len(saved) == len(parameters):
        for p, val in zip(parameters, saved):
            p.data = val
        print("--- Parameters loaded successfully. ---")
    else:
        print("--- WARNING: Parameter count mismatch. Training from scratch. ---")
else:
    print("\n--- No saved parameters found. Training from scratch. ---")

# vectorised multiclass hinge loss (one-vs-all)
def multiclass_hinge(logits, target_idx):
    correct = logits[target_idx]
    losses = [
        (log - correct + 1).relu()
        for i, log in enumerate(logits) if i != target_idx
    ]
    return sum(losses) * (1.0 / (len(logits) - 1))

print("\n--- Starting Training ---")
for step in range(max_steps):
    # mini-batch sampling
    idx = [random.randint(0, len(X) - 1) for _ in range(batch_size)]
    Xb  = [X[i] for i in idx]
    Yb  = [Y[i] for i in idx]

    # forward pass
    logits_batch = []
    for context in Xb:
        # flatten embeddings
        emb = [v for ix in context for v in C[ix]]
        logits_batch.append(model(emb))

    # compute mean loss over batch
    loss = sum(
        multiclass_hinge(logits, y)
        for logits, y in zip(logits_batch, Yb)
    ) * (1.0 / batch_size)

    # backward pass
    for p in parameters:
        p.grad = 0.0
    loss.backward()

    # parameter update (simple SGD)
    lr = learning_rate if step <= max_steps // 2 else 0.01
    for p in parameters:
        p.data -= lr * p.grad

    if step % 200 == 0:
        print(f"Step {step:4d} | loss {loss.data:.4f}")

# save parameters
print("\n--- Training complete. Saving parameters... ---")
with open(parameter_file, "wb") as f:
    pickle.dump([p.data for p in parameters], f)
print(f"--- Parameters saved to {parameter_file} ---")

# --------------------------------------------------------------------------
# Part 5: Text generation
# --------------------------------------------------------------------------
print("\n--- Generating Text ---")
context = [0] * block_size   # initialise with start tokens
generated = []
for _ in range(500):
    emb    = [v for ix in context for v in C[ix]]
    logits = model(emb)
    scores = [l.data for l in logits]
    next_ch = scores.index(max(scores))        # greedy sampling
    generated.append(itos[next_ch])
    context = context[1:] + [next_ch]

print("\n--- Generated Sample ---")
print("".join(generated))
