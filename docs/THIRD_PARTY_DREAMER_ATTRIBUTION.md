# Third-party Dreamer attribution

The report-aligned SimLingo Dreamer branch is a new PyTorch implementation for
the compact structured observation used by this repository. Its design adapts
algorithmic mechanisms found in the following vendored sources.

## DreamerV3

- Source: `external/cardreamer_upstream/dreamerv3/`
- Upstream author: Danijar Hafner
- License: MIT
- Adapted mechanisms: categorical recurrent state-space model, prior/posterior
  transition, straight-through stochastic state, balanced dynamics and
  representation KL, continuation-weighted latent imagination, and imagined
  actor/critic optimization.

The original copyright and MIT license are retained at
`external/cardreamer_upstream/dreamerv3/LICENSE`.

## CarDreamer

- Source: `external/cardreamer_upstream/`
- Pinned local commit: `160132436aeda1de54956f9910e56f3970a565aa`
- Copyright: The Regents of the University of California, Davis campus
- License: non-commercial use by nonprofit educational or research
  institutions, subject to the terms in
  `external/cardreamer_upstream/LICENSE`.

CarDreamer's CARLA environments, BEV observations, autonomous driving agent,
and task definitions are not imported into the SimLingo runtime. CarDreamer is
used as an inspected technical reference and as the repository carrying the
bundled DreamerV3 source. The resulting architecture remains SimLingo as the
reference driver with a compact Dreamer residual complement.
