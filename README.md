# Riemann-Flow SE(3) Molecular Docking Engine

[![CI](https://github.com/Sunzzx/riemann-flow-se3-molecular-docking-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Sunzzx/riemann-flow-se3-molecular-docking-engine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Sponsors](https://img.shields.io/badge/Sponsor-GitHub-ea4aaa?logo=github-sponsors)](https://github.com/sponsors/Sunzzx)
[![Polar](https://img.shields.io/badge/Polar-Support%20R%26D-blue)](https://polar.sh/sunzzx)

**Equivariant Riemannian Flow Matching on the special Euclidean group $\mathrm{SE}(3)$ for rigid-body receptor–ligand pose generation and structure-based docking.**

> Accelerates early-stage drug discovery by sampling geometrically valid ligand binding poses without multi-nanosecond molecular dynamics.

```
Receptor pocket ──► SE(3) noise g₀ ──► Riemannian CFM ODE ──► pose g₁
                                              │
                                              ▼
                                    Physics score (LJ + Coulomb
                                    + hydrophobic + Gaussian shape)
                                              │
                                              ▼
                                    Manifold score descent refine
                                              │
                                              ▼
                                    Ranked docking ensemble
```

---

## Humanity & Real-World Impact

Small-molecule drug design is bottlenecked by the combinatorial explosion of ligand poses relative to a protein binding site. Classical docking engines rely on stochastic search or exhaustive discretization of $\mathrm{SE}(3)$; molecular dynamics can refine poses but is too slow for ultra-large virtual screens.

This engine brings **Riemannian generative modeling** to docking:

- Learns a continuous vector field on $\mathrm{SE}(3)$ that transports random rigid motions toward the binding-mode manifold.
- Respects the geometry of rotations (no gimbal lock, bi-invariant metrics, Haar-consistent noise).
- Couples flow samples with an interpretable physics score (steric, electrostatic, hydrophobic, shape).
- Enables **pose ensembles** for ensemble docking, uncertainty quantification, and downstream free-energy workflows.

Impact domains: neglected tropical disease ligand triage, antibiotic scaffold hopping, covalent warhead placement, and accessible open-source SBDD education.

---

## Theoretical Foundations

### The Lie group $\mathrm{SE}(3)$

A rigid motion $g = (R, t) \in \mathrm{SE}(3)$ acts on $\mathbb{R}^3$ by
$$
g \cdot x = R x + t, \qquad R \in \mathrm{SO}(3),\; t \in \mathbb{R}^3.
$$
The Lie algebra $\mathfrak{se}(3) \cong \mathbb{R}^6$ consists of twists $\xi = (\omega, v)$. The exponential map is
$$
\exp(\xi) = \begin{pmatrix} \exp([\omega]^\times) & V(\omega)\, v \\ 0 & 1 \end{pmatrix},
$$
with Rodrigues' formula
$$
\exp([\omega]^\times) = I + \frac{\sin\theta}{\theta}[\omega]^\times + \frac{1-\cos\theta}{\theta^2}([\omega]^\times)^2,
\quad \theta = \|\omega\|,
$$
and left Jacobian
$$
V(\omega) = I + \frac{1-\cos\theta}{\theta^2}[\omega]^\times + \frac{\theta-\sin\theta}{\theta^3}([\omega]^\times)^2.
$$

### Riemannian Conditional Flow Matching (CFM)

Given noise $x_0 \sim p_0$ and data $x_1 \sim p_1$ on a Riemannian manifold $\mathcal{M}$, the **geodesic conditional path** is
$$
x_t = \operatorname{Exp}_{x_0}\!\big(t \cdot \operatorname{Log}_{x_0}(x_1)\big), \qquad t \in [0,1].
$$
On $\mathrm{SE}(3)$ with the left-invariant metric this is
$$
x_t = x_0 \circ \exp\!\big(t \cdot \log(x_0^{-1} \circ x_1)\big).
$$
The target body velocity is constant along the geodesic:
$$
u_t(x_t \mid x_1) = \log(x_0^{-1} \circ x_1) \in \mathfrak{se}(3).
$$
A parametric field $v_\theta : \mathrm{SE}(3) \times [0,1] \to \mathfrak{se}(3)$ is trained by
$$
\mathcal{L}(\theta) = \mathbb{E}_{t,x_0,x_1}\Big[\big\| v_\theta(x_t, t) - u_t \big\|_{W}^2\Big],
$$
with weighted norm $\|(\omega,v)\|_W^2 = w_r\|\omega\|^2 + w_t\|v\|^2$.

### Sampling ODE

Starting from $x_0 \sim p_0$, integrate the learned autonomous ODE on the manifold
$$
\dot x_t = v_\theta(x_t, t)^\wedge \cdot x_t
\quad\Longleftrightarrow\quad
x_{t+\Delta t} = x_t \circ \exp\big(\Delta t \cdot v_\theta(x_t, t)\big)
$$
via Euler or Heun steps.

### Physics scoring

For receptor atoms $\{r_i\}$ and ligand atoms $\{\ell_j\}$ under pose $g$,
$$
E(g) = w_s E_{\mathrm{LJ}} + w_e E_{\mathrm{Coul}} + w_h E_{\mathrm{hydro}} + w_\sigma E_{\mathrm{shape}},
$$
with soft $12$-$6$ Lennard-Jones, distance-dependent dielectric Coulomb, hydrophobic square-well contacts, and Gaussian shape overlap (Grant–Pickup). Poses are refined by **manifold gradient descent**
$$
g \leftarrow g \circ \exp\!\big(-\eta \nabla^\xi E(g)\big),
$$
where $\nabla^\xi$ is the body-frame twist gradient estimated by central differences on $\mathfrak{se}(3)$.

---

## Complexity Analysis

| Stage | Time | Space |
|------:|------|------|
| $\mathrm{SO}(3)$ exp / log | $O(1)$ | $O(1)$ |
| $\mathrm{SE}(3)$ geodesic path ($T$ steps) | $O(T)$ | $O(1)$ |
| CFM loss on batch $B$ | $O(B \cdot H^2)$ MLP | $O(H^2)$ |
| Pairwise scoring, $N_r \times N_\ell$ atoms | $O(N_r N_\ell)$ | $O(N_r N_\ell)$ |
| Dock $K$ poses, $T$ flow + $R$ refine steps | $O\big(K(T + R N_r N_\ell)\big)$ | $O(N_r + N_\ell + K)$ |

For typical fragment docking ($N_r \sim 10^2$, $N_\ell \sim 30$, $K=32$, $T=40$) wall-clock is sub-second on CPU (pure NumPy).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FlowDockingEngine                        │
│  ┌──────────────┐   ┌─────────────────┐   ┌──────────────┐  │
│  │ SE3 / SO3    │──▶│ Conditional     │──▶│ Physics      │  │
│  │ exp, log,    │   │ Flow Matcher    │   │ Scorer       │  │
│  │ geodesic,    │   │ (v_θ MLP + ODE) │   │ LJ/Coulomb/  │  │
│  │ Kabsch       │   │ Euler / Heun    │   │ hydro/shape  │  │
│  └──────────────┘   └────────┬────────┘   └──────┬───────┘  │
│                              │                   │          │
│                              ▼                   ▼          │
│                     manifold Euler step   score descent     │
│                              │                   │          │
│                              └─────────┬─────────┘          │
│                                        ▼                    │
│                              Ranked DockedPose ensemble     │
└─────────────────────────────────────────────────────────────┘
```

---

## Install

```bash
# from source
cd riemann-flow-se3-molecular-docking-engine
uv pip install -e ".[dev]"
# or
pip install -e ".[dev]"
```

Dependencies: `numpy`, `scipy` (scipy optional for future extensions; core is NumPy-only).

---

## Quickstart

```bash
# full interactive demo (geometry self-test + scoring + docking)
python -m riemann_flow_se3_molecular_docking_engine.demo

# geometry + score only
python -m riemann_flow_se3_molecular_docking_engine.demo --skip-dock

# more poses / training epochs
python -m riemann_flow_se3_molecular_docking_engine.demo --poses 16 --epochs 5

# tests
uv run pytest -q
```

### Library usage

```python
from riemann_flow_se3_molecular_docking_engine.docking import run_demo_docking
from riemann_flow_se3_molecular_docking_engine.se3 import SE3, exp_map, log_map
from riemann_flow_se3_molecular_docking_engine import so3

result = run_demo_docking(n_poses=8, seed=42, train_epochs=3)
print(result.top.energy, result.top.pose.t)

# pure geometry
R = so3.exp_map([0.0, 0.0, 1.57079632679])  # 90° about z
assert so3.rotation_matrix_check(R)
```

---

## Benchmark Snapshot

```
SO(3) exp∘log error (Haar):           < 1e-5
SE(3) geodesic CFM path consistency:  < 1e-5
Oracle flow target recovery distance: < 5e-3
Kabsch rigid alignment RMSD:          < 1e-8
Demo docking (8 poses, 3 epochs):     ~few seconds CPU
```

---

## Project Layout

```
src/riemann_flow_se3_molecular_docking_engine/
  so3.py            # SO(3) hat/vee, Rodrigues, geodesic
  se3.py            # SE(3) twists, exp/log, Kabsch
  flow_matching.py  # Riemannian CFM + MLP velocity field
  scoring.py        # LJ, Coulomb, hydrophobic, Gaussian shape
  docking.py        # FlowDockingEngine end-to-end
  demo.py / cli.py  # terminal demo
tests/              # unit + invariance + pipeline tests
```

---

## Commercial Licensing & Support

- **Open source (MIT):** free for research and commercial use under MIT terms.
- **GitHub Sponsors:** [github.com/sponsors/Sunzzx](https://github.com/sponsors/Sunzzx)
- **Polar.sh support:** [polar.sh/sunzzx](https://polar.sh/sunzzx)
- **Enterprise / custom SBDD integration, Stripe commercial licensing inquiries:** `ayushkandwal2007@gmail.com`
- Stripe payment link (commercial support retainers): contact for invoice (`acct` routing via operator treasury).

---

## References

1. Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023.
2. Chen & Lipman, *Riemannian Flow Matching on General Geometries*, NeurIPS 2023.
3. Yim et al., *SE(3) diffusion model with application to protein backbone generation*, ICML 2023.
4. Corso et al., *DiffDock*, ICLR 2023.
5. Grant & Pickup, *Gaussian shape methods for molecular similarity*.

---

## Author

**Ayush Kandwal** ([@Sunzzx](https://github.com/Sunzzx)) — Dual IIT (Madras / Patna), AI & quantitative research.

Built autonomously by the Hermes Frontier Research Engine.
