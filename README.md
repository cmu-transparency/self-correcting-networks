# self-correcting-networks

Library for training self-correcting neural networks that automatically revise their outputs to satisfy non-relational safety properties.

This repository implements the method described in [this paper](https://arxiv.org/pdf/2107.11445.pdf). If you use this code, please use the following citation:
```bibtex
@MISC{leino21self-correcting,
  title = {Self-Correcting Neural Networks for Safe Classification},
  author = {Klas Leino and Aymeric Fromherz and Ravi Mangal and Matt Fredrikson and Bryan Parno and Corina Păsăreanu},
  eprint = {2107.11445},
  archivePrefix = {arXiv},
  year = {2021}
}
```

## Installation

We recommend performing installation within a virtual environment.
The library can be installed via the following steps:
  1. Clone the repository and change into its root directory.
  2. Install from source via
```
pip install -e .
```

Code to run the experiments from the paper can be found in the `evaluation.py` files located in `scnet/acas` (for ACAS Xu experiments), `scnet/collision` (for collision avoidance experiments), and `scnet/misc` (for CIFAR-100 and synthetic experiments).
To run the experiments on synthetic data, you must first generate the synthetic datasets using `scnet/misc/synthetic_data.py`.
