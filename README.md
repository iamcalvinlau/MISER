# SPIDER_mod
Modified Sparse Identification based on SPIDER

These are some Python scripts and Jupyter notebooks hacking together pySINDy capabilities with some custom ugly modifications to attempt to recover the capability of "sparse physics-informed discovery of empirical relations" (SPIDER) shown by Gurevich, Reinbold, and Grigoriev [Learning fluid dynamics using sparse physics-informed discovery of empirical relations. arXiv preprint arXiv:2105.00048 (2021)] or that of Alves and Fiuza [Data-driven discovery of reduced plasma physics models from fully-kinetic simulations. https://arxiv.org/abs/2011.01927]. 

Mainly, I want to recover the noise-robust sparse identification without having to analytically calculate the weighting functions for the derivatives. In the AF2020 paper, it seems like a simple central finite difference was used with some effectiveness (and no fancy weighting was done to avoid possible biases). 

I assume that a smooth operator for the derivatives will help to raise that effectiveness to a similar level of GRG2021. In the current implementation, it looks like a smooth spline used to calculate the derivatives can achieve similar levels of robustness to noise.
