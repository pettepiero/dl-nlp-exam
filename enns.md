# Epistemic Neural Networks
https://arxiv.org/pdf/2107.08924

An ENN architecture is specified by a pair:
1. a parametrized function class $f$
2. a reference distribution $P_z$

The vector-valued output $f_\theta (x,z)$ of an ENN depends additionally on an *epistemic index z*, which takes values in the support of $P_z$. Concretely, the reference distribution could be a uniform distribution over a finite set or a standard Gaussian over a vector space. The index *z* is used to express epistemic uncertainty.


Given inputs $x_1, ... , x_\tau$, a joint prediction assigns a probability $\hat P_{1:\tau}(y_{1:\tau})$ to each class combination $y_1, ..., y_\tau$. Conventional neural networks are not designed to provide joint distributions, but we can obtain them with the product of independent outcomes $y_{1:\tau}$:
$$\hat P_{1:\tau}^{NN}(y_{1:\tau}) = \prod_{t=1}^\tau softmax(f_\theta(x_t))_{y_t}$$ 

However, this representation models each outcome $y_{1:\tau}$ as independent and so fails to distinguish ambiguity from insufficiency of data (remember the example in the following figure).

![Figure 1 from article](./figures/Screenshot%20from%202024-06-21%2010-02-02.png)

The idea behind ENNs is that of integrating over epistemic indices in order to introduce dependencies between predictions and in this way the joint predictions are not necessarily the product of marginals.