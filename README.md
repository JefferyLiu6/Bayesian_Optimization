# Bayesian_Optimization 
\begin{equation}
    \text{Cov}(f_s(x), f_{s'}(x')) = \sum_{q=1}^{Q} a_{s,q} a_{s',q} k_q(x, x')
\end{equation}

<<<<<<< HEAD
https://jamesbrind.uk/posts/2d-gaussian-process-regression//


Reference Book: Bayesian Optimization in Action
=======
As we saw in the previous section, naïvely seeking to improve from the incumbent leads to over-exploitation from the PoI. This is because simply moving away from the incumbent by a small amount in the appropriate direction can achieve a high PoI. Therefore, optimizing this PoI is not what we want to do. In this section, we learn to further account for the magnitude of the possible improvements we may observe. In other words, we also care about how much improvement we can make from the incumbent. This leads us to one of the most popular BayesOpt policies: Expected Improvement (EI).
References:

https://jamesbrind.uk/posts/2d-gaussian-process-regression//

https://gaussianprocess.org/gpml/chapters/RW.pdf

https://arxiv.org/pdf/1807.02811

https://github.com/bayesian-optimization/BayesianOptimization
>>>>>>> 3f5a95f8d6b6f1f4752634179d4b9f8cfbb3819f
