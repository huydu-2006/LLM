# Mathematic model
$$
\begin{aligned}
L(w) = \frac{1}{2N}\|Y-XW\|_2^2\\
\rightarrow \nabla _w (L(w)) = \frac{1}{N}X^T(XW-Y)\\
\nabla _w(L(w)) = 0 \Leftrightarrow W = (X^TX)^{\dagger}X^TY
\end{aligned}
$$
# Convention
1. X $\in R^{N\times d}$ where N is the numbers points, d is the dimension of feature vector. This convention is required for properly working of sklearn library
2. Y $\in R^{N\times 1}$ which is target vector
# Discusstion
1. Linear regression is sensitive to noise. We can use Huber regressor to solve this problem
2. Ridge regression can control overfiting problem
