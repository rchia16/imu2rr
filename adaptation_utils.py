'''
# Build SSA aligner from SOURCE train embeddings
ssa = SSAAligner(k=32)
Zs, _ = _collect_z(model, source_train_loader, device, max_batches=100)
ssa.fit_source(Zs.cpu())

# For each target subject:
Zt, _ = _collect_z(model, target_fewshot_loader, device, max_batches=50)
ssa.fit_target(Zt.cpu())

# During inference, align z before head:
@torch.no_grad()
def predict_with_ssa(model, x, ssa, device):
    z = model.z(x.to(device).float()).cpu()
    z_aligned = ssa.apply(z).to(device)
    return model.head(z_aligned)

# Unlabeled target loader (or just x from loader)
test_time_train_flow(
    model,
    test_loader=target_unlabeled_loader,
    device=device,
    ttt_lr=3e-5,
    steps_per_batch=1,
    style_k=4,
    beta_cons=0.1,
    use_style_views=True,
)

cmt_adapt_head(
    model,
    source_loader=source_train_loader,           # labeled
    target_fewshot_loader=target_fewshot_loader, # x only is enough
    device=device,
    top_k=16,
    n_aug=2048,
    steps=200,
    lr=1e-3,
)
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Flow bits (same idea as vit_experiment.py SimpleRealNVPFlow)
# ============================================================
class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=128, mask_even=True):
        super().__init__()
        mask = torch.zeros(dim)
        mask[::2] = 1.0 if mask_even else 0.0
        mask[1::2] = 0.0 if mask_even else 1.0
        self.register_buffer("mask", mask)

        self.net_s = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim))
        self.net_t = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, dim))

    def forward(self, z, reverse=False):
        m = self.mask
        z_masked = z * m
        s = torch.tanh(self.net_s(z_masked))
        t = self.net_t(z_masked)
        if not reverse:
            z_trans = z_masked + (1 - m) * (z * torch.exp(s) + t)
            log_det = ((1 - m) * s).sum(dim=1)
        else:
            z_trans = z_masked + (1 - m) * ((z - t) * torch.exp(-s))
            log_det = -((1 - m) * s).sum(dim=1)
        return z_trans, log_det

class SimpleRealNVPFlow(nn.Module):
    def __init__(self, dim, num_layers=4, hidden_dim=128):
        super().__init__()
        self.layers = nn.ModuleList([AffineCoupling(dim, hidden_dim, mask_even=(i % 2 == 0))
                                     for i in range(num_layers)])

    def forward(self, z):
        log_det_total = torch.zeros(z.size(0), device=z.device)
        u = z
        for layer in self.layers:
            u, log_det = layer(u, reverse=False)
            log_det_total += log_det
        return u, log_det_total

    def log_prob(self, z):
        u, log_det = self.forward(z)
        log_pu = -0.5 * (u ** 2).sum(dim=1)  # up to constant
        return log_pu + log_det


# ============================================================
# Style views + embedding consistency (ported from vit_experiment.py)
# ============================================================
def style_shift_imu(
    x,  # (B,T,C)
    gain_range=(0.7, 1.3),
    shift_max=24,
    drift_max=0.12,
    noise_max=0.05,
    leak_max=0.05,
):
    B, T, C = x.shape
    device, dtype = x.device, x.dtype
    y = x

    if shift_max > 0:
        shifts = torch.randint(-shift_max, shift_max + 1, (B, C), device=device)
        t = torch.arange(T, device=device).view(1, T, 1)
        s = shifts.view(B, 1, C)
        idx = (t - s) % T
        y = torch.gather(y, dim=1, index=idx.long())

    g = torch.empty((B, 1, C), device=device, dtype=dtype).uniform_(*gain_range)
    y = y * g

    std = y.std(dim=1, keepdim=True).clamp_min(1e-6)

    if drift_max > 0:
        drift_strength = torch.empty((B, 1, C), device=device, dtype=dtype).uniform_(0.0, drift_max) * std
        rw = torch.randn((B, T, C), device=device, dtype=dtype).cumsum(dim=1)
        rw = rw / rw.std(dim=1, keepdim=True).clamp_min(1e-6)
        rw = F.avg_pool1d(rw.transpose(1, 2), kernel_size=101, stride=1, padding=50).transpose(1, 2)
        rw = rw / rw.std(dim=1, keepdim=True).clamp_min(1e-6)
        y = y + rw * drift_strength

    if noise_max > 0:
        noise_strength = torch.empty((B, 1, C), device=device, dtype=dtype).uniform_(0.0, noise_max) * std
        n = torch.randn((B, T, C), device=device, dtype=dtype)
        n = F.avg_pool1d(n.transpose(1, 2), kernel_size=33, stride=1, padding=16).transpose(1, 2)
        n = n / n.std(dim=1, keepdim=True).clamp_min(1e-6)
        y = y + n * noise_strength

    if C > 1 and leak_max > 0:
        leak = torch.empty((B, C, C), device=device, dtype=dtype).uniform_(0.0, leak_max)
        leak = leak * (1.0 - torch.eye(C, device=device, dtype=dtype).view(1, C, C))
        leak = leak / leak.sum(dim=-1, keepdim=True).clamp_min(1.0)
        y = y + torch.einsum("bcc,btc->btc", leak, y)

    return y

def make_style_views(x, K=4, **kwargs):
    views = [x]
    for _ in range(K - 1):
        views.append(style_shift_imu(x, **kwargs))
    return torch.stack(views, dim=0)  # (K,B,T,C)

def emb_consistency_loss(z_list):
    # anchor at view0
    e0 = F.normalize(z_list[0], dim=-1)
    loss = 0.0
    for k in range(1, len(z_list)):
        ek = F.normalize(z_list[k], dim=-1)
        loss = loss + F.mse_loss(ek, e0)
    return loss / max(1, (len(z_list) - 1))


# ============================================================
# Attention pooling over time
# ============================================================
class AttnPool1D(nn.Module):
    """x: (B,T,D) -> (B,D)"""
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        scores = torch.einsum("btd,d->bt", x, self.q)
        w = F.softmax(scores, dim=1)
        w = self.drop(w)
        return torch.einsum("btd,bt->bd", x, w)


# ============================================================
# Freeze policy: "LayerNorm + proj" style (like vit_experiment.py)
# ============================================================
def freeze_all_but_layernorm_and_proj(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze LayerNorm
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad = True

    # unfreeze any module params that look like projection/adapters
    for name, m in model.named_modules():
        if any(k in name.lower() for k in ["proj", "adapter", "ln_proj", "spec_proj", "in_proj"]):
            for p in m.parameters():
                p.requires_grad = True


# ============================================================
# Unified wrapper: encoder -> pooled z -> head, with optional flow
# ============================================================
class AdaptableIMURR(nn.Module):
    """
    encoder(x) must return:
      - (B,T',D) sequence reps, OR
      - (B,D) pooled reps
    We always produce:
      - z (B,D) pooled embedding
      - y_hat (B,1)
    """
    def __init__(self, encoder: nn.Module, d_model: int, dropout=0.1, use_flow=False, flow_layers=4):
        super().__init__()
        self.encoder = encoder
        self.pool = AttnPool1D(d_model, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.flow = SimpleRealNVPFlow(d_model, num_layers=flow_layers) if use_flow else None

    def encode_seq(self, x):
        h = self.encoder(x)
        if isinstance(h, (tuple, list)):
            h = h[0]
        if h.dim() == 2:
            # make it "sequence-like" for pooling
            h = h.unsqueeze(1)  # (B,1,D)
        return h  # (B,T',D)

    def z(self, x):
        h = self.encode_seq(x)     # (B,T',D)
        return self.pool(h)        # (B,D)

    def forward(self, x):
        z = self.z(x)
        y_hat = self.head(z)
        return y_hat


# ============================================================
# TTT-Flow adaptation (unsupervised)
# Mirrors vit_experiment.py: flow NLL + style-view consistency, adapt LN/proj
# ============================================================
@torch.no_grad()
def _collect_z(model: AdaptableIMURR, loader, device, max_batches=None):
    model.eval()
    zs, ys = [], []
    nb = 0
    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            x, y = batch, None
        x = x.to(device).float()
        zs.append(model.z(x).detach().cpu())
        if y is not None:
            ys.append(y.detach().cpu())
        nb += 1
        if max_batches is not None and nb >= max_batches:
            break
    Z = torch.cat(zs, dim=0)
    Y = torch.cat(ys, dim=0) if len(ys) else None
    return Z, Y

def test_time_train_flow(
    model: AdaptableIMURR,
    test_loader,
    device,
    ttt_lr=3e-5,
    steps_per_batch=1,
    style_k=4,
    beta_cons=0.1,
    use_style_views=True,
):
    """
    Unsupervised TTT using flow likelihood on pooled embedding z.
    - freezes flow, head; adapts LayerNorm + proj-ish params in encoder/pool
    - loss = mean_k NLL(z_k) + beta_cons * consistency(z_k across views)
    """
    if model.flow is None:
        print("[TTTFlow] model.flow is None; skipping.")
        return

    # freeze everything except LN/proj, then explicitly freeze flow+head
    freeze_all_but_layernorm_and_proj(model)
    for p in model.flow.parameters(): p.requires_grad = False
    for p in model.head.parameters(): p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        print("[TTTFlow] No trainable params after freezing; skipping.")
        return

    opt = torch.optim.Adam(params, lr=ttt_lr)
    model.train()

    for batch in test_loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device).float()

        for _ in range(steps_per_batch):
            opt.zero_grad()

            if use_style_views:
                views = make_style_views(x, K=style_k, shift_max=24, leak_max=0.05)
            else:
                views = x.unsqueeze(0)

            z_list, nlls = [], []
            for k in range(views.size(0)):
                z_k = model.z(views[k])
                z_list.append(z_k)
                nlls.append((-model.flow.log_prob(z_k)).mean())

            loss = torch.stack(nlls).mean()

            if use_style_views and style_k > 1:
                loss = loss + beta_cons * emb_consistency_loss(z_list)

            if torch.isnan(loss) or torch.isinf(loss):
                print("[TTTFlow] NaN/Inf encountered; stopping.")
                return

            loss.backward()
            opt.step()


# ============================================================
# SSA (Significant Subspace Alignment) in embedding space
# ============================================================
def _pca_components(Z: torch.Tensor, k: int):
    """
    Z: (N,D) on CPU or GPU
    Returns (mu, U) where U: (D,k) orthonormal
    """
    mu = Z.mean(dim=0, keepdim=True)
    X = Z - mu
    # economical SVD: X = U_svd S Vh ; PCs are V (columns) = Vh^T
    # Use torch.linalg.svd for stability
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    U = Vh[:k].T.contiguous()
    return mu, U

class SSAAligner:
    """
    Fit on source embeddings; then given target fewshot embeddings, compute alignment.
    Apply to any z: (B,D).
    """
    def __init__(self, k: int = 32):
        self.k = k
        self.mu_s = None
        self.U_s = None
        self.mu_t = None
        self.U_t = None
        self.R = None  # (k,k)

    def fit_source(self, Zs: torch.Tensor):
        self.mu_s, self.U_s = _pca_components(Zs, self.k)

    def fit_target(self, Zt: torch.Tensor):
        self.mu_t, self.U_t = _pca_components(Zt, self.k)
        # map target PC coords -> source PC coords
        self.R = (self.U_t.T @ self.U_s)  # (k,k)

    def apply(self, z: torch.Tensor):
        """
        z: (B,D) on same device as stored tensors (recommend CPU for simplicity)
        returns aligned z: (B,D)
        """
        assert self.mu_s is not None and self.U_s is not None and self.mu_t is not None and self.U_t is not None and self.R is not None
        # coords in target subspace
        ct = (z - self.mu_t) @ self.U_t        # (B,k)
        cs = ct @ self.R                       # (B,k)
        z_aligned = cs @ self.U_s.T + self.mu_s
        return z_aligned


# ============================================================
# CMT (Causal Mechanism Transfer) in embedding space
# ============================================================
def pick_causal_dims_corr(Zs: torch.Tensor, ys: torch.Tensor, top_k: int = 16):
    """
    Zs: (N,D), ys: (N,1) or (N,)
    returns causal_idx: (top_k,)
    """
    y = ys.view(-1)
    Z = Zs
    # corr per dim (robust enough baseline)
    Zc = Z - Z.mean(dim=0, keepdim=True)
    yc = y - y.mean()
    num = (Zc * yc.unsqueeze(1)).sum(dim=0)
    den = torch.sqrt((Zc**2).sum(dim=0) * (yc**2).sum() + 1e-8)
    corr = (num / den).abs()
    return torch.topk(corr, k=min(top_k, Z.shape[1])).indices

@torch.no_grad()
def estimate_target_nuisance_stats(Zt: torch.Tensor, nuisance_mask: torch.Tensor):
    """
    Zt: (N,D)
    nuisance_mask: (D,) bool, True for nuisance dims
    """
    mu = Zt[:, nuisance_mask].mean(dim=0)
    std = Zt[:, nuisance_mask].std(dim=0).clamp_min(1e-6)
    return mu, std

def synthesize_cmt_pairs(Zs, ys, nuisance_mask, mu_t, std_t, n_aug=2048, seed=0):
    """
    Build synthetic pairs:
      - causal dims sampled from random source points
      - nuisance dims sampled from target nuisance Normal(mu_t, std_t)
    """
    g = torch.Generator(device=Zs.device)
    g.manual_seed(seed)
    N, D = Zs.shape
    idx = torch.randint(0, N, (n_aug,), generator=g, device=Zs.device)

    Zc = Zs[idx].clone()
    y_aug = ys[idx].clone()

    # sample nuisance
    nnz = nuisance_mask.sum().item()
    eps = torch.randn((n_aug, nnz), generator=g, device=Zs.device)
    Zc[:, nuisance_mask] = mu_t + eps * std_t
    return Zc, y_aug

def cmt_adapt_head(
    model: AdaptableIMURR,
    source_loader,          # labeled source/train loader
    target_fewshot_loader,  # unlabeled or labeled; we only need x to estimate nuisance stats
    device,
    top_k=16,
    n_aug=2048,
    steps=200,
    lr=1e-3,
    max_source_batches=50,
    max_target_batches=20,
):
    """
    CMT in z-space:
      1) compute source z and y
      2) pick causal dims
      3) compute target nuisance stats
      4) synthesize (z,y) pairs
      5) finetune ONLY the regression head on synthetic pairs
    """
    model.eval()
    Zs, Ys = _collect_z(model, source_loader, device, max_batches=max_source_batches)
    Zt, _  = _collect_z(model, target_fewshot_loader, device, max_batches=max_target_batches)

    Zs = Zs.to(device)
    Ys = Ys.to(device).float()
    Zt = Zt.to(device)

    causal_idx = pick_causal_dims_corr(Zs, Ys, top_k=top_k)
    nuisance_mask = torch.ones(Zs.shape[1], device=device, dtype=torch.bool)
    nuisance_mask[causal_idx] = False

    mu_t, std_t = estimate_target_nuisance_stats(Zt, nuisance_mask)

    Z_aug, Y_aug = synthesize_cmt_pairs(Zs, Ys, nuisance_mask, mu_t, std_t, n_aug=n_aug, seed=0)

    # finetune head only
    for p in model.parameters(): p.requires_grad = False
    for p in model.head.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(model.head.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    model.train()
    bs = 256
    for it in range(steps):
        j = torch.randint(0, Z_aug.size(0), (bs,), device=device)
        z = Z_aug[j]
        y = Y_aug[j]

        opt.zero_grad()
        y_hat = model.head(z)
        loss = loss_fn(y_hat, y)
        loss.backward()
        opt.step()

    model.eval()

