
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

- Simplicity <br>
The simplicity prior is implemented in most of state reprezsentation learning paper. This prior assume that it exits a low dimensional representationn of an high level input. This low dimensional representaiton is the one we want to learn.

- Time conherence/ continous
The time coherence or the prior of continuity assume that the state are fluctuating continuously throug time and that a radical change inside the environment have a low probability. Therefor it assume that the representation learned is continuous. It means that the manifold of the representation is continuous.

- Proportionnality
The proportionnality prior assume that for a same action the reactions of this actions will have proportionnal amplitude. The representation will then vary by the same ammount for two same actions in different situation. 

- Repeatability
The repeatability prior asssume that for two same actions the reaction will have proportionnal amplitude and have the same direction. 


- Causality
The causality is reward related. This prior assume that if we have two different rewards for two same action then the two starting should be differentiate in the representation manifold.

- Velocity/Acceleration/Inertia preservation (See PVE paper)
Those prior assume that the derivative of the position by the time fluactuate slowly throug time.

- Slowness principle
The slowness principle assume that feature we are interested in a probably fluctuating slowly. It constraint the representation to create a minfold with low variations.

- [...]

## Methods

There is different waf of implementation to integrate those priors to a training algorithm. The principe if that the representation's manifold respect the contrainste they impose. The constraint can be" implementated through the architecture of the neural network or thanks to a particcular cost function

- architecture constraints <br>
The architecture of tyhe neurla network can be spcialized in oder to implement a particular prior. For example  by imposing a botle neck at the output of the neural network impemen,t the prior of simplicity. Thze neural network is forced to find a loow dimensional way to answer its constraints.

- Markov chain <br>
A markov can make it possible to implement some of the priors. If each nodee of the Markov chain represent a state week can fiorce a certain dynamics between two consecutive nodes to iomplemennt a priors. The temporal cohetrebnce prior can be implemented by this principle by forcing to consecutive node to have close states.


- siamese network + optimization constrainct <br>
Siamese network are a generalization of the markov chain which make it possible to have more than two node connected to annother. It is particulary useful when more than two states are necessary toimpose a constraints. For example the implementation of the proportionnality prior need two states variation then four states. The siamese network can compute the for styate at the same time and compute a gradient which make the variation of states proportinnal. 

- metric learning
[???]

## Papers

- [ ] **PVEs: Position-Velocity Encoders for Unsupervised Learning of Structured State Representations**, Rico Jonschkowski, Roland Hafner, Jonathan Scholz, Martin Riedmiller, (2017), pdf, arXiv
- [ ] **Learning State Representations with Robotic Priors**, Rico Jonschkowski, Oliver Brock, (2015) , pdf <br>
- [ ] **A Physics-Based Model Prior for Object-Oriented MDPs** , *Jonathan Scholz, Martin Levihn, Charles L. Isbell, David Wingate*, (2014) [pdf](http://proceedings.mlr.press/v32/scholz14.pdf)  <br>
- [ ](hidden state representation) **The Curious Robot: Learning Visual Representations via Physical Interactions**,Lerrel Pinto, Dhiraj Gandhi, Yuanfeng Han, Yong-Lae Park, Abhinav Gupta,(2016) <br>
- [ ] **Label-Free Supervision of Neural Networks with Physics and Domain Knowledge** , *Russell Stewart , Stefano Ermon*, (2016) <br>
- [ ] **Slow Feature Analysis:Unsupervised Learning of Invariance**, *Laurenz Wiskott, Terrence J. Sejnowski*
