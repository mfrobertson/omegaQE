# omegaQE

Code focussing on the post-Born CMB curl (shear-B mode) field; used to produce results in 
https://arxiv.org/abs/2303.13313.

Also contains general tools for analytical modelling of large-scale structure power spectra, 
post-Born bispectra, Fisher-forecatsing, Gaussian flat sky simulations, quadratic estimator biases etc.

All code interacting with the [DEMNUni](https://arxiv.org/abs/1505.07148) and [AGORA](https://yomori.github.io/agora/index.html) N-body simulations to produce the results in 
https://arxiv.org/abs/2406.19998 can be found in `fullsky_sims`.


### Installation
Usual procedure - once inside the directory of the cloned repository run 
```
pip install -e . 
```
or if in conda environment
```
conda develop .
```
#### Requirements
Requires python packages
- [CAMB](https://camb.readthedocs.io/en/latest/) for CMB and matter power-spectra, and other useful functionality
- [LensIt](https://lensit.readthedocs.io/en/latest/) for flat-sky CMB lensing reconstruction
- [plancklens](https://plancklens.readthedocs.io/en/latest/) for CMB iterative reconstruction bias forecasts
- [vector](https://pypi.org/project/vector/) (version 0.8.5)

