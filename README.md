
# nanoGPT with Iron RoPE
A test repo for Fourier Extended RoPE / Iron Rope
Below is a README‑ready, plain‑text/Markdown version you can paste directly.

---

## Fourier‑Extended RoPE (“iron RoPE”)

**Simple explanation.**
Rotate `q/k` channel pairs by multi‑dimensional Fourier phases `θ = W · p` (where `p` are token coordinates), so attention scores depend on **relative geometry** `Δp` as well as content—i.e., RoPE generalized from 1‑D to `d`‑D.

### Notation

* `T` = sequence length, `H` = #heads, `Dh` = head dim, `m` = #rotation pairs per head (must satisfy `2m ≤ Dh`)
* `d` = coordinate dimension (1 for text index, 2 for `(x,y)`, 3 for `(x,y,t)`, …)
* `p_i ∈ ℝ^d` = coordinates for token *i* (e.g., `p_i = [i]` for text)
* `W ∈ ℝ^{m×d}` = frequency bank (log‑spaced or Gaussian rows)
* `q_i, k_i, v_i ∈ ℝ^{Dh}` = head‑wise projections at token *i*

### Rotary map (per head)

Split `q_i = [q_i^rot ∈ ℝ^{2m}, q_i^pass ∈ ℝ^{Dh−2m}]` into `m` 2‑D pairs; same for `k_i`.
Compute phases: `Θ_i = W · p_i ∈ ℝ^m`. Let `c_i = cos(Θ_i)`, `s_i = sin(Θ_i)`.

For each pair `r = 1..m`, with `q_i^rot[r] = (q0, q1)` and `k_i^rot[r] = (k0, k1)`:

```
rot_θ(q0,q1) = ( q0 * c_i[r] − q1 * s_i[r],
                  q0 * s_i[r] + q1 * c_i[r] )
```

Apply to `q` and `k` independently:

```
q̃_i = R(Θ_i) q_i
k̃_i = R(Θ_i) k_i
```

where `R(Θ_i)` is block‑diag of the `m` planar `2×2` rotations and identity on pass‑through dims.

### Attention (unchanged otherwise)

```
logits:  ℓ_{ij} = (1/√Dh) · ⟨ q̃_i , k̃_j ⟩
weights: a_{ij} = softmax_j( ℓ_{ij} + causal_mask )
output:  y_i     = Σ_j a_{ij} · v_j
```

### Key property (relative geometry)

Because planar rotations are orthonormal: `R(Θ_i)^T R(Θ_j) = R(Θ_j − Θ_i)`.
Thus

```
ℓ_{ij} = (1/√Dh) · q_i^T R(Θ_j − Θ_i) k_j
Θ_j − Θ_i = W · (p_j − p_i)  ⇒  scores depend on Δp = p_j − p_i
```

This yields translation‑equivariance in the chosen coordinate space.

### Single‑pair expansion (intuition)

For one rotated pair `r`:

```
⟨ rot_{θ_i}(q), rot_{θ_j}(k) ⟩
= cos(θ_j−θ_i) · (q0 k0 + q1 k1)
+ sin(θ_j−θ_i) · (q0 k1 − q1 k0)
```

Each pair acts like a small Fourier filter over the relative offset with frequency `W[r,:]`.

### Algorithm (drop‑in for one attention head)

1. Choose `W ∈ ℝ^{m×d}` (log‑spaced per axis or Gaussian RFF; share across heads for simplicity).
2. For `i = 1..T`, set coordinates `p_i ∈ ℝ^d` (text: `p_i = [i]`).
3. Compute phases `Θ_i = W · p_i`; `c_i = cos(Θ_i)`, `s_i = sin(Θ_i)`.
4. Project tokens: `q_i, k_i, v_i = linear(x_i)` and reshape to `[B, H, T, Dh]`.
5. For each token `i`, rotate the first `2m` dims of `q_i` and `k_i` pairwise via `c_i, s_i`; leave remaining `Dh−2m` dims unchanged.
6. Compute standard causal attention with `q̃, k̃, v` (same mask/softmax/projection as baseline).

### Defaults / tips

* Start with `d = 1` (text), `m ≈ 32–64` (clipped so `2m ≤ Dh`), `W` log‑spaced like classic RoPE (base ≈ `10000`).
* Everything else stays identical to your Transformer; this adds only `O(B·H·T·m)` sin/cos/rotations.
* Extending to geometry: set `d = 2` and `p_i = (x_i, y_i)` (patch centers), or `d = 3` with time.
* Sanity: if `d = 1` and `W` uses the standard RoPE schedule, this reduces to vanilla RoPE; if `m = 0`, it’s the identity.

### One‑liner

> **Fourier‑extended RoPE (“iron RoPE”)** rotates `q/k` by multi‑D Fourier phases `θ = W · p` so attention directly keys on relative offsets `Δp` in any geometry (1D/2D/3D), extending RoPE’s relative inductive bias beyond sequences with minimal code change.

