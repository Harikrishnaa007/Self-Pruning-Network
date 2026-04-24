import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — PrunableLinear
# ══════════════════════════════════════════════════════════════════════════════

class PrunableLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight and bias (same as nn.Linear)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Gate scores — one scalar per weight, same shape as weight.
        # Initialised to -2 so sigmoid(-2) ≈ 0.12, i.e. gates start mostly
        # closed, giving the sparsity penalty room to push them to zero.
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), -2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: map gate scores to [0, 1]
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: mask the weights — pruned weights are near zero
        pruned_weights = self.weight * gates

        # Step 3: standard linear transform using the masked weights
        return F.linear(x, pruned_weights, self.bias)


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Network definition + sparsity loss
# ══════════════════════════════════════════════════════════════════════════════

class PrunableMLP(nn.Module):
    """
    3-layer feed-forward network built entirely from PrunableLinear layers.
    Architecture: 3072 → 512 → 256 → 10
    """

    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)   # flatten: (B,3,32,32) → (B,3072)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def sparsity_loss(model: nn.Module) -> torch.Tensor:
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in model.modules():
        if isinstance(layer, PrunableLinear):
            gates = torch.sigmoid(layer.gate_scores)
            loss  = loss + gates.sum()   # L1 norm (gates are always positive)
    return loss


def compute_sparsity(model: nn.Module, threshold: float = 1e-2) -> float:
    total = pruned = 0
    for layer in model.modules():
        if isinstance(layer, PrunableLinear):
            gates   = torch.sigmoid(layer.gate_scores)
            total  += gates.numel()
            pruned += (gates < threshold).sum().item()
    return 100.0 * pruned / total if total > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Training loop, evaluation, results
# ══════════════════════════════════════════════════════════════════════════════

def get_dataloaders(batch_size: int = 128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = datasets.CIFAR10('./data', train=True,  download=True, transform=transform)
    test_set  = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=256,        shuffle=False)
    return train_loader, test_loader


def train(lambda_val: float = 1e-4, epochs: int = 20) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = PrunableMLP().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Cosine annealing decays LR from 1e-3 → 1e-5 over `epochs` steps
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    train_loader, _ = get_dataloaders()

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output  = model(data)
            cls_loss = F.cross_entropy(output, target)
            sp_loss  = sparsity_loss(model)

            # Total loss balances classification quality vs sparsity
            loss = cls_loss + lambda_val * sp_loss
            loss.backward()
            optimizer.step()

        scheduler.step()
        print(f"  Epoch {epoch+1:>2}/{epochs} done")

    sparsity = compute_sparsity(model)
    print(f"  → Sparsity after training: {sparsity:.2f}%")
    torch.save(model.state_dict(), f"model_lambda_{lambda_val}.pth")
    return model


@torch.no_grad()
def evaluate(model_path: str) -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = PrunableMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    model.eval()

    _, test_loader = get_dataloaders()
    correct = total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        pred     = model(data).argmax(dim=1)
        correct += (pred == target).sum().item()
        total   += target.size(0)

    acc      = 100.0 * correct / total
    sparsity = compute_sparsity(model)
    return acc, sparsity


def plot_gate_distribution(model_path: str, lambda_val: float,
                            save_path: str = "gate_distribution.png"):
    device = torch.device("cpu")
    model  = PrunableMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    model.eval()

    all_gates = []
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, PrunableLinear):
                gates = torch.sigmoid(layer.gate_scores).flatten()
                all_gates.append(gates.numpy())
    all_gates = np.concatenate(all_gates)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(all_gates, bins=120, range=(0, 0.45), color='#f1a84e',
            alpha=0.85, edgecolor='none')
    ax.axvline(0.01, color='red', linestyle='--', lw=1.8,
               label='Prune threshold (0.01)')

    pruned  = (all_gates < 0.01).sum()
    active  = len(all_gates) - pruned
    ymax    = ax.get_ylim()[1]
    ax.annotate(f'Pruned\n{pruned:,}\n({100*pruned/len(all_gates):.1f}%)',
                xy=(0.004, ymax * 0.5), xytext=(0.06, ymax * 0.72),
                fontsize=10, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))
    ax.annotate(f'Active\n{active:,}\n({100*active/len(all_gates):.1f}%)',
                xy=(0.08, ymax * 0.15), xytext=(0.18, ymax * 0.45),
                fontsize=10, color='#f1a84e',
                arrowprops=dict(arrowstyle='->', color='#f1a84e'))

    ax.set_title(f'Gate Value Distribution  (λ = {lambda_val}, best model)',
                 fontsize=13)
    ax.set_xlabel('Gate value  σ(gate_score)', fontsize=11)
    ax.set_ylabel('Count',                     fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {save_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    LAMBDAS = [1e-5, 1e-4, 5e-4]
    EPOCHS  = 20
    results = []

    # ── Train all three λ values ──────────────────────────────────────────────
    for lam in LAMBDAS:
        print(f"\n{'='*45}")
        print(f"  Training  λ = {lam}")
        print(f"{'='*45}")
        train(lambda_val=lam, epochs=EPOCHS)

    # ── Evaluate all checkpoints ──────────────────────────────────────────────
    print(f"\n{'='*45}")
    print("  Evaluation Results")
    print(f"{'='*45}")
    print(f"  {'Lambda':<10} {'Accuracy':>12} {'Sparsity':>12}")
    print(f"  {'-'*36}")

    for lam in LAMBDAS:
        acc, sparsity = evaluate(f"model_lambda_{lam}.pth")
        results.append({'lam': lam, 'acc': acc, 'sparsity': sparsity})
        print(f"  {str(lam):<10} {acc:>11.2f}% {sparsity:>11.2f}%")

    # ── Plot: best model = λ = 1e-4 (highest sparsity with reasonable accuracy)
    best = min(results, key=lambda r: abs(r['sparsity'] - 50))
    plot_gate_distribution(
        model_path=f"model_lambda_{best['lam']}.pth",
        lambda_val=best['lam'],
        save_path="gate_distribution.png"
    )
