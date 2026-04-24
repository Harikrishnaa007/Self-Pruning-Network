# Self-Pruning Neural Network — Report

## 1. Why L1 on Sigmoid Gates Encourages Sparsity

The total training loss is:

```
Total Loss = CrossEntropy(logits, labels)  +  λ × Σ sigmoid(gate_score_ij)
```

The second term is the **L1 norm of all gate values** — the sum of every
`sigmoid(gate_score)` across all layers.

### The key intuition

The gradient of the sparsity term with respect to a gate score is:

```
∂(SparsityLoss) / ∂(gate_score_ij)  =  gate_ij × (1 − gate_ij)
```

This gradient is largest around `gate ≈ 0.5` and shrinks toward 0 at
both extremes. This creates a natural decision process:

- If a weight is **useful**, the classification loss pushes back and the
  gate stabilises at some positive value.
- If a weight is **not useful**, nothing counteracts the sparsity penalty,
  so the gate keeps being pushed toward 0 until `sigmoid` saturates
  at essentially zero — the weight is pruned.

Compare this to an **L2 penalty** (`Σ gate²`), whose gradient `2·gate`
vanishes as `gate → 0`. L2 shrinks gates but never fully eliminates them.
L1's approximately constant pressure near zero is what makes it a
**sparsity-promoting** regulariser — it keeps pushing even when a gate
is already small.

The result is a **bimodal gate distribution**: a large spike at 0 (pruned
weights) and a smaller cluster at higher values (retained weights), with
very few gates in between.

---

## 2. Results Table

| λ (lambda) | Test Accuracy | Sparsity Level (%) |
|:----------:|:-------------:|:------------------:|
| 1e-5       | 53.56%        | 2.50%              |
| 1e-4       | 50.37%        | 56.67%             |
| 5e-4       | 47.45%        | 93.97%             |

### Analysis

- **λ = 1e-5 (Low)**: The sparsity penalty is very weak relative to the
  classification loss. Almost no gates are driven to zero (only 2.5%
  pruned). The network behaves like a standard unpruned MLP, achieving
  the highest accuracy of 53.56%.

- **λ = 1e-4 (Medium) ✅ Best model**: A well-balanced trade-off.
  Over half the weights (56.67%) are pruned while accuracy drops by only
  ~3 percentage points. This is the recommended operating point — the
  network is meaningfully compressed without significant quality loss.

- **λ = 5e-4 (High)**: Aggressive pruning — 93.97% of all weights are
  effectively removed. The network retains only ~6% of its connections
  yet still achieves 47.45% accuracy (versus 53.56% unpruned), a drop
  of just 6 points for a 16× compression. This demonstrates how
  robust the retained connections are.

### Key takeaway

A **6% accuracy drop** while removing **94% of all weights** confirms
that the self-pruning mechanism successfully identifies and eliminates
redundant connections during training.

---

## 3. Gate Distribution Plot

The plot below shows the gate value distribution for the best model
(λ = 1e-4).

![Gate Distribution](gate_distribution_plot.png)

### How to read it

- The **tall spike near 0** represents 967,059 pruned weights (56.67%)
  whose gates have been driven to essentially zero by the L1 penalty.
  These weights contribute nothing to the network's output.

- The **smaller tail from 0.01 to ~0.35** represents the 739,437 active
  weights (43.33%) that the network chose to retain because they were
  useful for classification.

- **No gates reach values above ~0.40** — unlike a standard network
  where weights are unconstrained. The sigmoid + L1 pressure keeps
  even active gates relatively small, acting as implicit regularisation.

This bimodal structure (spike at 0, cluster away from 0) is the hallmark
of a successfully trained self-pruning network.

---

## 4. Model Architecture

```
Input: CIFAR-10 image  (32×32×3 = 3,072 pixels, flattened)
  │
  ▼
PrunableLinear(3072 → 512)  +  ReLU     [1,572,864 gated weights]
  │
  ▼
PrunableLinear(512  → 256)  +  ReLU     [  131,072 gated weights]
  │
  ▼
PrunableLinear(256  →  10)              [    2,560 gated weights]
  │
  ▼
Output: 10 class logits

Total gated weights: 1,706,496
```

Each weight `w_ij` is paired with a gate score `s_ij` (also learned).
Forward pass: `output = (W ⊙ σ(S)) x + b`
