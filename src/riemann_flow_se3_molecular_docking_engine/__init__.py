"""Riemannian SE(3) Flow Matching for Molecular Conformation & Rigid Docking.

Equivariant continuous normalizing flows on the special Euclidean group SE(3)
for receptor-ligand pose generation. Implements:

- SO(3) / SE(3) Lie algebra exponential & logarithmic maps
- Geodesic interpolation on the Riemannian manifold SE(3)
- Conditional Flow Matching (CFM) vector fields on SE(3)
- Rigid-body molecular docking with steric + electrostatic scoring
- Conditional score / velocity networks (closed-form + MLP surrogate)

References
----------
- Lipman et al., Flow Matching for Generative Modeling (ICLR 2023)
- Chen & Lipman, Riemannian Flow Matching on General Geometries (NeurIPS 2023)
- Yim et al., SE(3) Diffusion Model with Application to Protein Backbone Generation (ICML 2023)
- Corso et al., DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking (ICLR 2023)
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "so3",
    "se3",
    "flow_matching",
    "docking",
    "scoring",
    "demo",
]

from riemann_flow_se3_molecular_docking_engine import so3 as so3
from riemann_flow_se3_molecular_docking_engine import se3 as se3
from riemann_flow_se3_molecular_docking_engine import flow_matching as flow_matching
from riemann_flow_se3_molecular_docking_engine import docking as docking
from riemann_flow_se3_molecular_docking_engine import scoring as scoring
