# Epistemic Neural Networks
## Introduction
- Who is Osband (paper from Google Deepmind, now researcher at OpenAI)
- Osband and uncertainty

## Neural Networks and Uncertainty
- example
- distinguish between uncertainty types
- aleatoric vs epistemic
- do not underestimate this problem for decision making systems
- theoretical paper by Wen and Osband

## Joint predictions in decision making
- Joint predictions allow distinction between uncertainty types
- **most of supervised ML has focused on accurate predictions for single inputs (marginal predictions)**
- we obtain the joint prediction, i.e. every possible combination
- epistemic case: **either both rabbits or both ducks**
- Epistemic Neural Network
- combinatorial decision problem

## Joint predictions in combinatorial decision problem
- customer, recommendation system, selection of K<1 movies, inventory of N elements
- preferences of the user, features of the movie, same dimension
- model probability through a **logistic sigmoid** 
- goal
- two possible user types, k=2
- agent that optimizes the expected probability for each move individually
- agent that considers the joint predictive distribution
- values are chosen, but in CDP optimization based on the marginal predictive distribution can be insufficient
- paper -> sequential decision problems and multi-armed bandit

## ENNs
- conventional nn
- classification problem
- softmax and joint
- enns
- reference distribution
- z is latent variable that the nn can use to model its uncertainty
- **variation with z indicates uncertainty that might be resolved with future data**
- for making joint predictions, you integrate over the reference distribution the product of the predictions for each index z
- when making joint predictions, you use the same z across multiple predictions
- **the joint prediction is not exactly the same as the product of marginals**
- In practice, either monte carlo over z or thompson sampling



## Why not Bayesian NN or Ensemble methods?
- from what i understand, there is gap in machine learning in dealing with different types of uncertainty
- bayesan -> principled and correct way, but costly
- deep learning approach -> scale up buy more compute
- we seem to care more about the end to end result than what's really happening inside the box
- direction of progress
- enn just want to be good at joint predictions
- bnn only on whiteboard
- ensembles feel bayesian 
- they combine different models and see if they disagree
- but you have to train 100 models, while enn only adds a little compute to one model
- BNNs are ENNs but not all ENNs are BNNs (BNNs unkown predictions, ENN focus on joint predictions)


## The Epinet
- can supplement any conventional nn (even already pretrained) to make a new kind of ENN
- nn with **privileged access** 
- features, can include original input but also hidden representations
- sg
- **invested lots in computation and learning -> maybe hidden units are useful in predicting uncertainty**
- given input, different ouputs from single forward pass of base net
- split epinet 
- idea comes from prior functions

### Training the epinet
- standard training algorithms
- standard log loss 
- What changes is that z is in data loss and regularization -> we say 'just integrate over z'
- in training we sample a batch of data and of indices z
- the idea is 'on average'
- epinet trained with standard log loss can approach optimal joint predictions as the epistemic index size grows
- for regression we use mean squared error instead of the log loss above
- 


## Fine tuning LLMs via ENNs
