

## Joint predictions in a Combinatorial decision problem

Consider the problem of a customer interacting with a recommendation system that proposes a selection of K > 1 movies from an inventory of N movies X1, .., XN .

$phi$ represents the preferences of the user, and each $X_i$ is a vector containing the feature of movie $i$. Both $phi$ and $X_i$ have dimension $d$.
The goal is to create an agent which **maximizes the probability that the user will enjoy at least one of the K suggested movies**.

Consider the simple case where the user is drawn from two possible user types $\{\phi_1, \phi_2\}$, and the recommendation system should propose $K=2$ movies. 
In the table that you see we have these numerical values and the evaluated probabilities according to the formula on the top of the slide. What we can see is that An agent that optimizes the expected probability for each movie individually will end up recommending the pair $(X_3, X_4)$ to an unknown $phi \sim Unif(\phi_1, \phi_2)$.

By contrast, **an agent that considers the joint predictive distribution** for $\tau \ge 2$ can see that instead selecting the pair $(X_1, X_2)$ will give close to 100% certainty that the user will enjoy one of the movies.

These values were chosen for this specific example, but they show that *in combinatorial decision problems, where the outcome depends on the joint predictive distribution, optimization based on the marginal predictive distribution is insufficient to guar antee good decisions.*

Note that, **conditioned on the rue environment, the data generating process is actually i.i.d. The key point is that, when the true underlying environment is unknown, a coupling in future observations is introduced**.


The paper "From Predictions to Decisions: the Importance of Joint Predictive Distributions" continues to show that examining the joint predictive distributions can be essential for good performance also in **sequential decision problems**.



## Why Not BNNs
From what I understand, there seems to be a gap in machine learning in dealing with different types of uncertainty. On one side, we have the Bayesian methods where uncertainty is handled in a very principled and correct way, but very costly even for small problems. On the other side we have the "Deep learning" approach where we scale up and buy more compute to make better models. In this approach the measures of uncertainty are not as good as in the first case, but we don't really seem to be so worried.
We seems to care more about the end-to-end result rather than what's really happening inside the box. The direction of progress, from my understanding, is that of increasing the data and the compute rather than focus on the process itself. With this in mind, the ENNs were developed with the idea of making *something that's just really good at doing joint predictions*.
The effort Osband made was to bridge the gap between the two approaches and create something that allows us to make good decisions in the real world, using uncertainty.

Why not BNNs? They are often just impossible to compute, most of the time we can only write them down on a whiteboard. They *would* give us the optimal solution, but they are too expensive computationally. Another approach that people have is training ensembles, because it *feels a bit like being Bayesian*. They combine different models and if they disagree a lot then they say "I'm uncertain". But the problem with this techniques is that you've got to train a lot, say a 100, networks. The technique that Osband proposes only adds a little compute and obtains good results.

+every BNN can be expressed as ENN


## The epinet

The epinet is an architecture that can supplement any conventional neural network to make a new kind of ENN architecture. It is straightforward to add an epinet to an existing **pretrained** model.
