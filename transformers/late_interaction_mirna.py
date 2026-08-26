"""
Late-interaction model for miRNA:MRE binding prediction.

Design rationale
----------------
The miRBind-style CNN builds a (MAX_MIRNA x MRE_LEN) pairing matrix from a fixed
5x5 nucleotide-identity lookup, so complementarity is hard-coded into the input.
A pooled dual-encoder ("CLIP-style") would remove that hard-coding but also throw
away position-to-position correspondence -- which is exactly what defines the
3'-compensatory class (weak seed at miRNA 2-8, strong pairing at 13-16).

Late interaction keeps both properties:
  * pair features are LEARNED (token embeddings, not a nucleotide lookup)
  * position-to-position correspondence is PRESERVED (no pooling before scoring)

Scoring produces a per-miRNA-position MaxSim vector of length MAX_MIRNA, which is
register-aligned: index i is miRNA position i+1. The classifier head therefore sees
"how well did each miRNA position find a partner", which is the natural
representation for seed-vs-3'-supplementary compensation.

Encoders use RoPE within each molecule's OWN coordinate frame (never across a
concatenated pair, where relative distances would be biologically meaningless).

Swap in BiMamba2 by replacing `TransformerEncoder` -- see ENCODER SWAP note below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Match the existing dataset conventions
MAX_MIRNA = 30
MRE_LEN = 50
PAD_IDX = 4          # A=0 C=1 G=2 U=3 pad/N=4
VOCAB = 5

# miRNA positional roles (0-based index -> role id)
ROLE_ANCHOR = 0      # pos 1
ROLE_SEED = 1        # pos 2-8
ROLE_CENTRAL = 2     # pos 9-12
ROLE_SUPP = 3        # pos 13-17  (3' supplementary)
ROLE_TAIL = 4        # pos 18+
ROLE_PAD = 5
N_ROLES = 6


def build_role_ids(max_len: int = MAX_MIRNA) -> torch.Tensor:
    """Static map from miRNA index -> role id. Register prior for the encoder."""
    roles = torch.full((max_len,), ROLE_TAIL, dtype=torch.long)
    roles[0] = ROLE_ANCHOR
    roles[1:8] = ROLE_SEED       # positions 2-8
    roles[8:12] = ROLE_CENTRAL   # positions 9-12
    roles[12:17] = ROLE_SUPP     # positions 13-17
    return roles


# ---------------------------------------------------------------- RoPE
def rope_cache(seq_len: int, head_dim: int, device, base: float = 10_000.0):
    """Cos/sin tables for rotary embeddings. head_dim must be even."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)              # (L, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, L, Dh) -> rotated. cos/sin: (L, Dh/2)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None]
    sin = sin[None, None]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class RoPESelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                     # (B, H, L, Dh)

        cos, sin = rope_cache(L, self.head_dim, x.device)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:                      # True == pad
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = self.drop(attn.softmax(dim=-1))
        y = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(y)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPESelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        x = x + self.drop(self.attn(self.norm1(x), key_padding_mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """
    ENCODER SWAP: to use BiMamba2 instead, replace the EncoderBlock stack with your
    bidirectional Mamba2 blocks. Keep the signature -- (B, L) ids -> (B, L, D) tokens.
    RoPE is not needed for Mamba (the recurrence is inherently ordered); just drop it
    and keep the role embedding.
    """

    def __init__(self, d_model=128, n_layers=4, n_heads=8, dropout=0.1,
                 n_roles: int | None = None, max_len: int = MAX_MIRNA):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d_model, padding_idx=PAD_IDX)
        self.use_roles = n_roles is not None
        if self.use_roles:
            self.role_emb = nn.Embedding(n_roles, d_model)
            self.register_buffer("role_ids", build_role_ids(max_len), persistent=False)
            # small init: a prior, not a second positional encoding
            nn.init.normal_(self.role_emb.weight, std=0.02)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad = ids.eq(PAD_IDX)                      # (B, L) True == pad
        x = self.tok(ids)
        if self.use_roles:
            roles = self.role_ids[: ids.size(1)].clone()
            r = self.role_emb(roles)[None].expand(ids.size(0), -1, -1)
            r = torch.where(pad[..., None], torch.zeros_like(r), r)
            x = x + r
        for blk in self.blocks:
            x = blk(x, key_padding_mask=pad)
        return self.norm(x), pad


# ------------------------------------------------- late interaction scoring
@dataclass
class Scores:
    logit: torch.Tensor        # (B,)
    maxsim: torch.Tensor       # (B, MAX_MIRNA) register-aligned per-position score
    sim_map: torch.Tensor      # (B, MAX_MIRNA, MRE_LEN) learned soft pairing map
    partner: torch.Tensor      # (B, MAX_MIRNA) argmax target index per miRNA position
    z_mirna: torch.Tensor      # (B, D) pooled -- for InfoNCE if wanted
    z_mre: torch.Tensor        # (B, D)


class LateInteractionMiRNA(nn.Module):
    def __init__(self, d_model=128, d_interact=64, n_layers=4, n_heads=8,
                 dropout=0.1, head_hidden=128, temperature=1.0):
        super().__init__()
        self.enc_mirna = TransformerEncoder(
            d_model, n_layers, n_heads, dropout, n_roles=N_ROLES, max_len=MAX_MIRNA
        )
        self.enc_mre = TransformerEncoder(
            d_model, n_layers, n_heads, dropout, n_roles=None, max_len=MRE_LEN
        )
        # project into the interaction space where token-pair similarity is computed
        self.proj_mirna = nn.Linear(d_model, d_interact, bias=False)
        self.proj_mre = nn.Linear(d_model, d_interact, bias=False)
        self.temperature = temperature

        # head over the register-aligned MaxSim vector
        self.head = nn.Sequential(
            nn.LayerNorm(MAX_MIRNA),
            nn.Linear(MAX_MIRNA, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden // 2, 1),
        )

    @staticmethod
    def _masked_mean(x, pad):
        keep = (~pad).float().unsqueeze(-1)
        return (x * keep).sum(1) / keep.sum(1).clamp_min(1.0)

    def forward(self, mirna_ids: torch.Tensor, mre_ids: torch.Tensor) -> Scores:
        hm, pad_m = self.enc_mirna(mirna_ids)      # (B, Lm, D)
        ht, pad_t = self.enc_mre(mre_ids)          # (B, Lt, D)

        qm = F.normalize(self.proj_mirna(hm), dim=-1)
        kt = F.normalize(self.proj_mre(ht), dim=-1)

        sim = qm @ kt.transpose(1, 2) / self.temperature      # (B, Lm, Lt)
        sim = sim.masked_fill(pad_t[:, None, :], float("-inf"))

        maxsim, partner = sim.max(dim=-1)                      # (B, Lm)
        maxsim = maxsim.masked_fill(pad_m, 0.0)
        partner = partner.masked_fill(pad_m, -1)

        logit = self.head(maxsim).squeeze(-1)

        return Scores(
            logit=logit,
            maxsim=maxsim,
            sim_map=sim.masked_fill(pad_m[:, :, None], 0.0).nan_to_num(neginf=0.0),
            partner=partner,
            z_mirna=self._masked_mean(hm, pad_m),
            z_mre=self._masked_mean(ht, pad_t),
        )


# ------------------------------------------------------------------ losses
def bce_loss(scores: Scores, labels: torch.Tensor, weight: torch.Tensor | None = None):
    return F.binary_cross_entropy_with_logits(scores.logit, labels.float(), weight=weight)


def infonce_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.07):
    """
    Symmetric InfoNCE over pooled embeddings. Only meaningful if the batch is built
    so the in-batch negatives are HARD (e.g. dG-matched); with random in-batch
    negatives this term is close to free and will teach little.
    """
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = z_a @ z_b.t() / temperature
    tgt = torch.arange(z_a.size(0), device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))


def margin_loss(scores_pos: torch.Tensor, scores_neg: torch.Tensor, margin: float = 1.0):
    """Hinge between a weak positive and its dG-matched hard negative."""
    return F.relu(margin - (scores_pos - scores_neg)).mean()


# ----------------------------------------------------------- interpretability
def register_profile(scores: Scores) -> dict[str, torch.Tensor]:
    """
    Mean MaxSim within each miRNA register. Lets you read off directly whether the
    model is finding 3'-supplementary partners on the compensatory FNs -- the
    diagnostic the pairing-matrix CNN could not give you.
    """
    roles = build_role_ids(MAX_MIRNA).to(scores.maxsim.device)
    out = {}
    for name, rid in (("seed", ROLE_SEED), ("central", ROLE_CENTRAL),
                      ("supplementary", ROLE_SUPP), ("tail", ROLE_TAIL)):
        m = roles.eq(rid)
        out[name] = scores.maxsim[:, m].mean(dim=1)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 8
    mirna = torch.randint(0, 4, (B, MAX_MIRNA))
    mirna[:, 22:] = PAD_IDX                    # variable-length miRNAs
    mre = torch.randint(0, 4, (B, MRE_LEN))

    model = LateInteractionMiRNA()
    s = model(mirna, mre)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")
    print("logit    ", tuple(s.logit.shape))
    print("maxsim   ", tuple(s.maxsim.shape), "(register-aligned)")
    print("sim_map  ", tuple(s.sim_map.shape))
    print("partner  ", tuple(s.partner.shape))

    labels = torch.randint(0, 2, (B,))
    loss = bce_loss(s, labels) + 0.1 * infonce_loss(s.z_mirna, s.z_mre)
    loss.backward()
    print("loss     ", float(loss))

    prof = register_profile(s)
    print("register profile:", {k: round(float(v.mean()), 4) for k, v in prof.items()})
    print("backward OK")
