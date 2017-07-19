
# Learning with priors

****************************************************

`What do I want to say?`<br>
That state representation learning can be done with prior

`What does the prior do?`<br>
The prior are a way to share knowledge we have about the wolrd with our deep learning algorithm to make it get the physics of the world 

`How to do it?`<br>
By constraining the learning process (by architecture or by optimization)


****************************************************

## Introduction

Contrarely to bayesian probability in the contexte of state representation learning a prior is not an a priori distribution but an a priori knowledge we have about the wolrd, in particular about the physics of the world. This knowledge help to learn a dense and efficient representation about the world. For human the acquisition of world's physics insight need years of interaction with it and a hard coded architecture of the brain which helps to aprehend it [citation]. A deep neural network only get tuned hyper-parameters, images or few hours of robots interaction to get how the world works. A human will probably have a lot of difficulty to understand the world just by looking at images and so does a neural network. That is why we must help him to understand the underlying factor of the world we already know with prior. In this part we present how to use priors to learn state representation and how to implement them into a neural network.

## Learned States

## A priori

Different priors have been proposed to adress the problem of sharing our knowledgte with neural network. There are almost always related to the time. For instance one of our best advantage against neural neural is our comprehension of the interaction between entities through time. The neural network have a lot of difficulties to make link between events if there are not neigboor. Explaining that information can easely be extract from time by looking at it the right way would help significantly a neural network to make decision. It could also reduce the complexity of the inference because there is a lot of coherence through time. Here is a list of the priors and the paper which formulate it or use it.
- Simplicity
- Time conherence/ continous
- Proportionnality
- Repeatability
- Causality
- Velocity/Acceleration/Inertia preservation (See PVE paper)
- [...]

## Methods

- architecture constraints
- siamese network + optimization constrainct
- metric learning

## Papers

- [ ] **PVEs: Position-Velocity Encoders for Unsupervised Learning of Structured State Representations**, Rico Jonschkowski, Roland Hafner, Jonathan Scholz, Martin Riedmiller, (2017), pdf, arXiv
- [ ] **Learning State Representations with Robotic Priors**, Rico Jonschkowski, Oliver Brock, (2015) , pdf <br>
- [ ] **A Physics-Based Model Prior for Object-Oriented MDPs** , *Jonathan Scholz, Martin Levihn, Charles L. Isbell, David Wingate*, (2014) [pdf](http://proceedings.mlr.press/v32/scholz14.pdf)  <br>
- [ ](hidden state representation) **The Curious Robot: Learning Visual Representations via Physical Interactions**,Lerrel Pinto, Dhiraj Gandhi, Yuanfeng Han, Yong-Lae Park, Abhinav Gupta,(2016) <br>
- [ ] **Label-Free Supervision of Neural Networks with Physics and Domain Knowledge** , *Russell Stewart , Stefano Ermon*, (2016) <br>
- [ ] **Slow Feature Analysis:Unsupervised Learning of Invariance**, *Laurenz Wiskott, Terrence J. Sejnowski*
