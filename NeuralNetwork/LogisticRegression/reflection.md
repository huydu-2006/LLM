Some things i learned from this section :
1. MLE, MAP
$$
\begin{aligned}
MLE \text{ for } \theta: \theta = \argmax_{\theta} P(X_1, X_2, ..., X_N | \theta) \\
MAP \text{ for } \theta: \theta = \argmax_\theta P(\theta | X) = \argmax_{\theta} P(X | \theta).P(\theta)
  \end{aligned}
$$
2. How does the linear classifier classify between labels
3. Taking expectation over a window in noisy loss function for more clear observation
4. SGD wouldn't never make loss function reach zero. Hence, we control quality by number of epoch and norm of diff between two w
