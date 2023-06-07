## Installation

- We'll use `conda` to install dependencies and set up the environment.
We recommend using the [Python 3.9 Miniconda installer](https://docs.conda.io/en/latest/miniconda.html#linux-installers).
- After installing `conda`, install [`mamba`](https://mamba.readthedocs.io/en/latest/) to the base environment. Please refer to [`mamba Installation`](https://mamba.readthedocs.io/en/latest/installation.html) for details. `mamba` is a faster, drop-in replacement for `conda`.

- Next, create a new environment named `m2hub` and install dependencies. Instructions are for PyTorch 1.13.1, CUDA 11.6 specifically.
    ```bash
    mamba env create -f environment.yml
    ```

- Activate the conda environment with `conda activate m2hub`.
