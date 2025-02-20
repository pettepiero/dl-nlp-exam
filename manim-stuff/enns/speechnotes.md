# Epistemic Neural Network slides
## Neural Networks and Uncertainty
Today I would like to talk about **uncertainty**.
The problem that Osband tackles and has tackled in lots of his articles is that of uncertainty in Neural Networks. In this presentation, I will walk through some of his latest work and try to expose his ideas in an ordered way. My goal is to try to understand what Epistemic Neural Networks are and why they have been created.

Let's start by recognizing that conventional NNs don't have the ability to dinstinguish between two types of uncertainty:
- **Aleatoric uncertainty**: given by the genuine data ambiguity
- **Epistemic uncertainty**: due to ignorance of the model

Consider the following example (which will be presented multiple times today): suppose you have a network which is trained on classifying images of ducks and rabbits. If the output of the model on a given image is 5050, it is unclear whether this is because of insufficiency of data or because the image is ambiguous. 

In this first case, if I asked anyone what the image is, we would all say rabbit and therefore we can all agree that the model has not been trained enough. In this case we are therefore dealing with epistemic uncertainty. 

Suppose now we feed this famous image to the model. The output will likely still be 5050, but in this case it is the image that is too ambiguous to tell. This is an example of aleatoric uncertainty. We should ask ourselves:

> *Would other networks trained for the same purpose give the same output or would the model learn a class if trained on more data?*

This problem should not be underestimated, especially for decision making systems. On this topic, Wen and Osband produced an interesting and **theoretical** paper justifying the importance of **joint predictions** for a broad class of decision problems. This is the basis that we will need to understand the *Epistemic Neural Networks* paper.

## Joint predictions in decision making
The core idea of the paper on the screen is that **joint predictions** allow the distinction between uncertainty due to genuine ambiguty and insufficiency of data. They claim that most of the supervised learning work has focused on producing accurate predictions for single inputs, and they use the term *marginal prediction* to denote that (because it's for single inputs).

The idea is shown in this figure. Suppose now we feed the network two identical images of the image we used earlier, and obtained what we call a *joint prediction* intended as a prediction for each possible combination of the output classes. We are giving two copies of the image and we are expecting two predictions. We show two possible joint predictions that are plausible with the 5050 marginal output. In this way we can distinguish the two scenarios, just by looking at the table.
The first case shows inevitable uncertainty in the model that cannot be resolved with training. This is the aleatoric case (random chance). The second case tells us that additional training should resolve uncertainty. This is the epistemic case, and it tells us that **they are either both rabbits or both ducks**.

**We can now tell apart the two cases**, and the network that produces directly the joint distribution is what we call an **epistemic neural network**.

We can make an example on a combinatorial decision problem to grasp the concept better. This type of problem consists of finding an optimal object from a finite set of objects, where the set of feasible solutions is discrete or can be reduced to a discrete set. 

## Joint predictions in a Combinatorial decision problem

Consider the problem of a customer interacting with a recommendation system that proposes a selection of $K > 1$ movies from an inventory of $N$ movies $X_1, .., X_N$.

$phi$ represents the preferences of the user, and each $X_i$ is a vector containing the feature of movie $i$. Both $phi$ and $X_i$ have dimension $d$. We model the probability that a user will enjoy movie $i$ by a logistic model $Y_i \sim logit(\phi^T X_i)$.
The goal is to create an agent which **maximizes the probability that the user will enjoy at least one of the K suggested movies**.

Consider the simple case where the user is drawn from two possible user types $\{\phi_1, \phi_2\}$, and the recommendation system should propose $K=2$ movies out of the four movies on the screen.
In the table that you see we have these numerical values and the evaluated probabilities according to the formula on the top of the slide. What we can see is that An agent that optimizes the expected probability for each movie individually will end up recommending the pair $(X_3, X_4)$ to an unknown $\phi \sim Unif(\phi_1, \phi_2)$.

By contrast, **an agent that considers the joint predictive distribution** for $\tau \ge 2$ can see that instead selecting the pair $(X_1, X_2)$ will give close to 100% certainty that the user will enjoy one of the movies.

These values were chosen for this specific example, but they show that *in combinatorial decision problems, where the outcome depends on the joint predictive distribution, optimization based on the marginal predictive distribution is insufficient to guar antee good decisions.*

Note that, **conditioned on the rue environment, the data generating process is actually i.i.d. The key point is that, when the true underlying environment is unknown, a coupling in future observations is introduced**.


The paper "From Predictions to Decisions: the Importance of Joint Predictive Distributions" continues to show that examining the joint predictive distributions can be essential for good performance also in **sequential decision problems** and **multi-armed bandits**.

## What are ENNs
Now that it's clearer to us why we're interested in joint predictions, we can try to understand ENNs.

So, conventional NNs are specified by their parameters $\theta$ and their parametrized function class $f_\theta$. In a typical multiclass classification problem we have to convert the output to a probability over the classes via softmax, and we can compute the joint prediction by taking the product of the single predictions. In this way we're assuming the independence of the single points and we cannot distinguish ambiguity from insufficiency of data.

ENNs, on the other hand, are specified by a reference distribution $P_Z$ and an epistemic index $z \sim P_Z$. The reference distribution is usually a multidimensional gaussian over a vector space or a uniform distribution over a finite set. The epistemic index $z$ can be seen as a latent variable that the nn can use to model its own uncertainty.
It is the index $z$ that is used to express epistemic uncertainty. In particular, **variation of the network output with z indicates uncertainty that might be resolved by future data.**
So in ENNs, when you make predictions, you use the same epistemic index $z$ across multiple predictions, therefore the joint is not equal to the product of the marginals anymore.

  
**Isn't this just training an ensemble? What is the benefit?**



## Why not BNNs
From what I understand, there seems to be a gap in machine learning in dealing with different types of uncertainty. On one side, we have the Bayesian methods where uncertainty is handled in a very principled and correct way, but very costly even for small problems. On the other side we have the "Deep learning" approach where we scale up and buy more compute to make better models. In this approach the measures of uncertainty are not as good as in the first case, but we don't really seem to be so worried.
We seems to care more about the end-to-end result rather than what's really happening inside the box. The direction of progress, from my understanding, is that of increasing the data and the compute rather than focus on the process itself. With this in mind, the ENNs were developed with the idea of making *something that's just really good at doing joint predictions*.
The effort Osband made was to bridge the gap between the two approaches and create something that allows us to make good decisions in the real world, using uncertainty.

Why not BNNs? They are often just impossible to compute, most of the time we can only write them down on a whiteboard. They *would* give us the optimal solution, but they are too expensive computationally. Another approach that people have is training ensembles, because it *feels a bit like being Bayesian*. They combine different models and if they disagree a lot then they say "I'm uncertain". But the problem with this techniques is that you've got to train a lot, say a 100, networks. The technique that Osband proposes only adds a little compute and obtains good results.

+every BNN can be expressed as ENN


## The epinet

The epinet is an architecture that can supplement any conventional neural network to make a new kind of ENN architecture. It is straightforward to add an epinet to an existing **pretrained** model. It is a NN with privileged access to inputs and outputs of activation units in the base network. The subset of these inputs and outputs that are taken as input to the epinet are called *features* $\phi_\zeta (x)$. The features can include the original input, but also hidden representations. Also the *epistemic indez* $z$ is an input to the epinet. For epinet parameters $\eta$, its output is $\sigma_\eta(sg[\phi_\zeta(x)],z)$, where $sg$ indicates "stop gradient". The stop gradient essentially means that we keep its argument fixed when computing the gradient.
**The idea is that if you've invested lots computation and learning, maybe those hidden units are useful in predicting your uncertainty.**

The epinet can be split in two separate pieces: $$\sigma_\eta (\tilde x ,z) = \sigma_\eta ^L (\tilde x, z) + \sigma^P(\tilde x, z)$$

The first one is learnable and in the paper takes the form of a simple MLP. The second one has no parameters and is reflects the prior uncertainty. After training, the resulting variation in $z$ is meant to be something like a posterior. We need to set up a prior if we want the network to use the epistemic index $z$. After the training of 

The idea comes from another paper from Osband, called *Randomized Prior Functions for Deep Reinforcement Learning*. Each plot is a particle of an ensemble of size 4. In each plot we show the effect of training a NN on the same datapoints (dots). The light yellow lines are the different prior functions for each particle. Then we add a trainable network (dashed lines) so that the resulting prediction (the sum) in blue, goes through all the points. The first plot would be $z_1$, the secondo $z_2$ and so on. **The different particles all agree on the training data, but they generalize differently depending on the effect of the learnt network and the prior.**

The idea is that if we change $z$, we can have different predictions and therefore we are uncertain. 

> How is the prior initialized? Not much info about that, but "such that the initial variation in $z$ reflects your uncertainty. A typical example is using the same architecture as the trainable $\sigma^P$ but with different parameters.
> 
> The design, initialization and scaling of the prior network $\sigma^P$ allows an algorithm designer to encode prior beliefs, and is essential for good performance. Typical choices might include $\sigma^P$ sampled from the same architecture as $\sigma^L$ but with different parameters.


## Training algorithm
Training is very similar to standard NNs training, you just have 


## Fine-Tuning LLMs via ENNs
Language models often pre-train on large unsupervised text corpora, then fine-tune on additional task-specific data. The focus of bidirectional encoders like BERT is on **computing contextualized representations of the input tokens**. They use self-attention to map sequences of input embeddings $(x_1, ... , x_n)$ to sequences of output embeddings the same length $(y_1, ... , y_n)$, where the output vectors have been contextualized using information from the entire input sequence. These output embeddings are **contextualized representations** and are generally useful across a range of downstream applications. These models are usually called **encoder only**.

Fine-tuning (in the transfer learning paradigm) is the process of taking the network learnt by these pretrained models and further train som additional, smaller network that takes the top layer of the network as input, to perform some **downstream task** like named entity tagging or question answering.
The intuition is that the pretraining phase learns a language model that instantiates **rich representations of word meaning**.

While it is much more efficient to fine-tune a LLM rather than re-training from scratch, it may still require substantial amounts of data.
In fact, **typical fine-tuning schemes do not prioritize the examples that they tune on**. The goal of this paper is to show that, if you can prioritize informative training data, you can achieve better performance while using fewer labels than training without prioritization. To do this, the LM is augmented with an epinet and the results are compared with respect to typical priority functions.


## Active Learning Framework
The field of active learning has studied the problem of optimizing which data to collect. Agents that succesfully prioritize their training labels to achieve lower losses on held-out data **with fewer training labels** are said to perform better. 

So we consider a training dataset $D$, with inputs $x_i$ and class labels $y_i$. Initially, at time step $t=0$, the agent can see only the training inputs $D_X$, but no associated labels. At each time step, the agent picks an index $a_t$ and reveals the corresponding class label $y_{a_t}$. Now the agent can use $D_X$ and the new datapoint to update its beliefs, then repeats.
**The way the index is chosen is specified by the proprity functions.**
As we said, the priority function is needed to choose which labels to reveal. It maps input $x$ and ENN parameters $\theta$ to a score $g(\theta, x) \in R$. Still focusing the problem on classification, these are the notations for class probabilities predicted by the ENN. Notice how the final class is obtained integrating out the conditioning random variable.

We can now talk about the priority functions that were considered in the paper. 

1. To start off, we have a simple **uniform prioritization**, meaning that **it does not distinguish between inputs**. 
2. Then we have what the paper calls **marginal priority functions**, which do not account for variability in the epistemic index $z$. They are simple conventional neural network predictions. We call these approaches marginal priority functions since they only depend on the marginal probability estimates for a single input x. **Entropy prioritization** can be interpreted from an *information theory* point of view as the **average level of surprise of the random variable's possible outcomes**.
Similarly, **margin prioritization** is based on the smallest difference between the highest and second highest class probabilities.

3. **Epistemic priority functions** use the ENN to prioritize uncertain, as opposed to ambiguous, data points. They use the joint predictions distribution. **Bald** prioritization is based on the mutual information gain -> EXPPLAINNSNSSAFNI

**Variance** prioritization uses the variation in predicted probabilities.



## Training algorithm and loss function

The training algorithm that is used is a variant of stochastic gradient descent, where at each step a large batch of the dataset is selected randomly without replacement. Then the agent selects a fixed number of elements of the candidate batch, using the priority function, to perform SGD on. A fixed number of epistemic indices $\tilde Z$ is selected and the loss is summed over these samples.

Then a **standard cross entropy loss** with regularization is used.


## Comparison of active learning agents
The active learning agents were compared using an open-source benchmark called "Neural Testbed" and developed still by Osband. It is a collection of neural-network-based, synthetic classification problems that evaluate the quality of an agent's predictive distributions.

We consider a generative model based on data generated by a random MLP. The inputs are generated from a standard normal and the labels are assigned with the application of softmax. In the formula on the screen, $h_{\theta^*}$ is a 2 layer MLP with ReLU activation functions and 50 hidden units at each layer. $\rho$ is a temperature parameter. What the researchers did was generate 200 training examples and evaluate the average test log-likelihood over $N=1000$ testing examples, averaging the results over 10 random seeds.

The **baseline agent** doesn't use active learning and on the screen there are some implementation details of the choices they made. In any case, what's important to know is that the goal was to create an agent that represented an upper bound on the performance for any agent that doesn't prioritize its training data.


## Language model experiment
The tasks that were considered on the paper are from the **General Language Understanding Evaluation (GLUE)** benchmark. These are diverse natural language understanding tasks that are widely accepted as a benchmark for performance in the field. Of these, **only the classification tasks were considered**, excluding so 3 out of the original 11 benchmark tasks.


+ General view of GLUE


A simple **per task fine-tuning** setting was considered, where each agent is trained on each GLUE task separately. In this scenario, **the agents are the different fine-tuning heads of the BERT model**. 
They are shown in the table in the slide. Each agent has a MLP fine-tuning head.


The baseline agent is meant to provide an upper bound on how well any agent that does not prioritize its data can do. On the screen there are some implementation details.

## Results
1. This first plot shows the performance of an epinet prioritized by variance vs methods that don't use prioritization. The benefit of using prioritization is only evident when using the epinet. In the other cases, their performance plateaus earlier as they prioritize potentially noisy examples, rather than informative training data. 

2. The second plot compares agent performance when prioritizing by variance, changing the ENN architecture. We see that epinet performs better than the competing approaches, and the ensemble performs better than dropout in this setting.