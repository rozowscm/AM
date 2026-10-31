# Adaptive Management under Structural Uncertainty

MATLAB code for:

> Rozowski, C., & Fackler, P. L. (2025). Adaptive Management under Structural Uncertainty: A Linear Opinion Pool Approach to Expanding the Model Set. *Methods in Ecology and Evolution*, 16(9), 1992–2009. https://doi.org/10.1111/2041-210X.70137

The paper treats the true system as a weighted combination (a linear opinion pool) of the candidate models instead of assuming that one of them is correct. Beliefs about the model weights follow a Dirichlet distribution. The code implements approximate updating rules for those weights, compares them with exact updating and Bayesian model averaging, and solves the resulting adaptive management problem for a Gouldian finch example with three expert models.

## Contents

All code is in `Code_9_15_21/`.

**Core functions**
- `amdpdirichlet.m`: sets up the adaptive management problem with Dirichlet beliefs over model weights, using either Kullback–Leibler matching or the Cowell, Dawid and Sebastiani (1996) rule
- `modelweightsupdate.m`, `modelweightsupdatesim.m`: Kullback–Leibler updates of the Dirichlet parameters
- `cds96.m`: the Cowell, Dawid and Sebastiani (1996) updating rule
- `amdpWsim.m`: simulates a weighted adaptive management model
- Helpers: `pdfdirichlet.m`, `randdisc.m`, `normalizeP.m`, `b2xy.m`, `beliefplot.m`, `dirsolvec.c`

**Gouldian finch application**
- `finch.m`, `finch_new.m`, `finch_new_dots.m`, `finch_new_dots2.m`, `finch_decisions.m`, `finch_decisions2.m`, `FinchWSim.m`, `FinchWSim_speed.m`, `amdpWsimTest.m`

**Updating illustrations and examples**
- `modelweightsfigures.m`, `modelweightsfiguresReview3.m`, `TwoModelFigures.m`, `threemodelfigures.m`, `Markovsim.m`
- `cointossdemo.m`, `cointossdemo_extreme.m`: the coin-toss experiment from Yang and Zhu (2018), *Proceedings of the National Academy of Sciences*

**Saved output**
- `simulations.txt`, `results.mat`, `BMA_MM_KL.pdf`, `BMA_MM_KL.jpg`, `Jeffreys_MM_KL.jpg`

## Requirements

- MATLAB
- MDPSolve, Paul Fackler's MATLAB toolbox for Markov decision problems. The version used is included as `MDPSOLVE-9-2-15.zip`. The current version is at https://github.com/PaulFackler/MDPSolve.
- `dirsolvec` is a C MEX function. The included `dirsolvec.mexw64` is built for 64-bit Windows; on other systems, compile it in MATLAB with `mex dirsolvec.c`.

## Getting started

1. Unzip MDPSolve and add it to the MATLAB path with its subfolders.
2. Add `Code_9_15_21` to the MATLAB path.
3. Run `finch.m` for the Gouldian finch application, or the figure scripts for the updating illustrations.

## Citation

```bibtex
@article{rozowski2025adaptive,
  author  = {Rozowski, Casey and Fackler, Paul L.},
  title   = {Adaptive Management under Structural Uncertainty: A Linear Opinion Pool Approach to Expanding the Model Set},
  journal = {Methods in Ecology and Evolution},
  year    = {2025},
  volume  = {16},
  number  = {9},
  pages   = {1992--2009},
  doi     = {10.1111/2041-210X.70137}
}
```

## References

- Cowell, R. G., Dawid, A. P., & Sebastiani, P. (1996). A comparison of sequential learning methods for incomplete data. *Bayesian Statistics 5*.
- Yang, Z., & Zhu, T. (2018). Bayesian selection of misspecified models is overconfident and may cause spurious posterior probabilities for phylogenetic trees. *Proceedings of the National Academy of Sciences*, 115(8), 1854–1859. https://doi.org/10.1073/pnas.1712673115
