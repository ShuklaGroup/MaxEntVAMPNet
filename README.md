# MaxEntVAMPNet

Codes for [*Active Learning of the Conformational Ensemble of Proteins using Maximum Entropy VAMPNets*](https://www.biorxiv.org/content/10.1101/2023.01.12.523801v1.full).

## Citation

If using the code in this repository, please include the following in your citations:

```bibtex
@article{kleiman2023active,
  title={Active Learning of the Conformational Ensemble of Proteins using Maximum Entropy VAMPNets},
  author={Kleiman, Diego E and Shukla, Diwakar},
  journal={bioRxiv},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Environment

The `clean_env.yml` file can be used with `conda` to recreate the environment used while conducting the research.

`conda env create -f clean_env.yml`

## Description

This repository contains codes implementing different adaptive sampling methods (`AdaptiveSampling.py`, `EntropyBased.py`, `Reap.py`, `VaeReap.py`, `VampNetReap.py`, `VampReap.py`) and supporting objects or methods (`Agent.py`, `Simulations.py`, `utils.py`).

The `examples` directory contains usage examples to run molecular dynamics simulations. See the README in each folder for more details.

## Methods

There are many different adaptive sampling methods implemented in this repository as Python classes. Check the table below for a quick summary of the main ones. The kinetic models correspond to those implemented in [deeptime](https://deeptime-ml.github.io/latest/index.html). 

| Name  | Selection strategy | Kinetic model | File | Main references |
| ----- | ------------------ | ------------- | ---- | --------- |
| LeastCounts  | LeastCounts  | None | `AdaptiveSampling.py` | Bowman et al. J. Chem. Theory Comput. 2010, 6, 3, 787–794.  |
| VampLeastCounts  | LeastCounts  | VAMP | `AdaptiveSampling.py` | Bowman et al. J. Chem. Theory Comput. 2010, 6, 3, 787–794.<br />Wu et al. J. Nonlinear Sci. 2019, 30, 23–66.|
| VampNetLeastCounts  | LeastCounts  | VAMPNet | `AdaptiveSampling.py` | Bowman et al. J. Chem. Theory Comput. 2010, 6, 3, 787–794.<br />Mardt et al. Nat. Commun. 2018, 9.|
| VaeLeastCounts  | LeastCounts  | TVAE | `AdaptiveSampling.py` | Bowman et al. J. Chem. Theory Comput. 2010, 6, 3, 787–794.<br />Wehmeyer et al. Chem. Phys. 2018, 148, 241703.|
| MultiagentReap  | MA REAP  | None | `Reap.py` | Kleiman et al. J. Chem. Theory Comput. 2022, 18, 9, 5422–5434.<br />|
| MultiagentVampReap  | MA REAP  | VAMP | `VampReap.py` | Kleiman et al. J. Chem. Theory Comput. 2022, 18, 9, 5422–5434.<br />Wu et al. J. Nonlinear Sci. 2019, 30, 23–66.|
| MultiagentVampNetReap  | MA REAP  | VAMPNet | `VampNetReap.py` | Kleiman et al. J. Chem. Theory Comput. 2022, 18, 9, 5422–5434.<br />Mardt et al. Nat. Commun. 2018, 9.|
| MultiagentVaeReap  | MA REAP  | TVAE | `VaeReap.py` | Kleiman et al. J. Chem. Theory Comput. 2022, 18, 9, 5422–5434.<br />Wehmeyer et al. Chem. Phys. 2018, 148, 241703.|
| EntropyBasedSampling  | MaxEnt  | VAMPNet | `EntropyBased.py` | Kleiman et al. bioRxiv. 2023, 10.1101/2023.01.12.523801.<br />Mardt et al. Nat. Commun. 2018, 9.|




