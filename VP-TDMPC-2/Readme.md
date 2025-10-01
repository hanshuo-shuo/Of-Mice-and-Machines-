## As a fork version

This repository is a fork of the original [TDMPC-2](https://github.com/nicklashansen/tdmpc2), based on the early release (via [tdmpc2-prey](https://github.com/hanshuo-shuo/tdmpc2-prey)).
 
Our experiments and paper results were conducted on the old version.
 
Note that the official TDMPC-2 repository was updated in April 2025 with an additional episodic RL feature, which was not available at the time of our work. Therefore, this feature is **not included** in our implementation.

Pretrain model can be downloaded at: https://drive.google.com/drive/folders/1CcC7jM2aGaG5PEd8iuSLZgtLR3cqZECU?dmr=1&ec=wgc-drive-globalnav-goto


## A Note on Reproducibility
As we state in the paper, this is a work of scientific exploration, not an engineering benchmark. The environment is inherently stochastic, and our model reflects this.

Performance can, and will, vary significantly across runs. Reproducing the exact results from the paper may be challenging.
This variability is an expected outcome and a core part of our scientific claim.

The source code for the simulation environment can be found at: [https://github.com/hanshuo-shuo/Mice] 