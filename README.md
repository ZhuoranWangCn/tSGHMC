# The Tamed Stochastic Gradient Hamiltonian Monte Carlo
This repository is the official implementation of "The Tamed Stochastic Gradient Hamiltonian Monte Carlo", which can be found at

Authors: Zhuoran Wang and Ying Zhang

Abstract: In this paper, we propose a novel tamed stochastic gradient Hamiltonian Monte Carlo (tSGHMC) algorithm for sampling and stochastic optimization with superlinearly growing stochastic gradients. Under a certain continuity in average condition and a strong convexity condition, we establish a non-asymptotic error bound in Wasserstein-2 distance for tSGHMC with the rate of convergence equal to $1/4$. We then derive an upper estimate for the associated expected excess risk, which provides a theoretical guarantee for tSGHMC to solve optimization problems under consideration. To illustrate the effectiveness of tSGHMC, we apply the algorithm to various examples, including a newsvendor problem and a Conditional Value-at-Risk (CVaR) minimization problem, using synthetic and real-world datasets. Numerical results support our theoretical findings. We also compare tSGHMC with its first-order counterpart, namely, the tamed unadjusted stochastic Langevin algorithm. Numerical experiments confirm the superiority of tSGHMC, which achieves lower root mean square error and expected excess risk.

## File Description
+ `Algorithms.py` contains the implementation of the **tSGHMC** algorithm used in our paper and the baseline **TUSLA** algorithm used for comparison.
+ `Example1_Sampling.ipynb` contains the code for Subsection 3.1 of the paper, *Posterior Sampling for Penalized Logistic Regression*.
+ `Example2_Artificial.ipynb` contains the code for Subsection 3.2 of the paper, *Artificial Example*.
+ `Example3_Newsvendor.ipynb` contains the code for Subsection 3.3 of the paper, *Newsvendor Problem*.
+ `Example4.1.1_CVaR.ipynb` contains the code for the first experiment in Subsection 3.4.1 of the paper, *One-dimensional VaR and CVaR Computation*.
+ `Example4.1.2_Portfolio.ipynb` contains the code for the second experiment in Subsection 3.4.1 of the paper, *Portfolio Allocation Problem*.
+ `Example4.2_NN.ipynb` contains the code for Subsection 3.4.2 of the paper, *Nonlinear Regression*.

