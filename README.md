# Bayesian_Optimization 
\begin{equation}
    \text{Cov}(f_s(x), f_{s'}(x')) = \sum_{q=1}^{Q} a_{s,q} a_{s',q} k_q(x, x')
\end{equation}

1.1.5 Hyper-parameter learning (marginal likelihood, MLE)
The kernel and noise hyper-parameters θ (e.g. length-scales, variances) are chosen by max-
imising the log marginal likelihood:
log p(y | X, θ) = −1
2 y⊤Kθ + σ2
ε I−1y − 1
2 log Kθ + σ2
ε I − n
2 log 2π.
Gradients w.r.t. each θj are
∂
∂θj
log p(y | X, θ) = 1
2 y⊤C−1 ∂C
∂θj
C−1y − 1
2 tr

C−1 ∂C
∂θj

, C = Kθ + σ2
ε I.
References:

https://jamesbrind.uk/posts/2d-gaussian-process-regression//

https://gaussianprocess.org/gpml/chapters/RW.pdf

https://arxiv.org/pdf/1807.02811

https://github.com/bayesian-optimization/BayesianOptimization
