# State-Representation-Learning-An-Overview

# DEEP LEARNING IN ROBOTICS IS COMING  !!!!

## Abstract

Deep learning in robotics is coming. Very soon robot will be able to benefit from the deep learning framework to make complex choices and predictions in a robust and automous fashon. The interesting particularity of a robot is that it's input are , as a human, multimodale. The robot can use it's camera at the same time than a lidar, a radar, a microphone or all the tools you can imagine. For deep learning here come the classic problem of the curse of dimensionality. How to make an algortihm able to make prediction with several high dimension inputs and how make it find hidden dependencies between them online ? The solution is reduce the dimensionality by learning state representation. State representation learning means find the few hidden parameters of each of the input (or modularity). Once the hidden parameters are found the task of finding dependencies between modularities is no more bothered by the dimensionality. This paper aims to cover the state of the art about state representation learning. It presents the different methods used to disentangle the hidden parameters of an datasets and to validate the learned state representation. This overview is particularely focus on learning representation in low dimensionnality (<5) of known parameters like the state of an 3D object. This scope make it possible to assess the representation learned. 

## Learning with priors [Link](https://github.com/TLESORT/State-Representation-Learning-An-Overview/blob/master/LearningWithApriori.md)

- constraint
- siamese network

- **PVEs: Position-Velocity Encoders for Unsupervised Learning of Structured State Representations** <br> Rico Jonschkowski, Roland Hafner, Jonathan Scholz, Martin Riedmiller, (2017), pdf, arXiv

 - **Learning State Representations with Robotic Priors**<br> Rico Jonschkowski, Oliver Brock, (2015) , pdf <br>
 
 - **A Physics-Based Model Prior for Object-Oriented MDPs** <br> *Jonathan Scholz, Martin Levihn, Charles L. Isbell, David Wingate*, (2014) [pdf](http://proceedings.mlr.press/v32/scholz14.pdf)  

- **The Curious Robot: Learning Visual Representations via Physical Interactions**,Lerrel Pinto, Dhiraj Gandhi, Yuanfeng Han, Yong-Lae Park, Abhinav Gupta,(2016) (hidden state representation) <br>

- **Label-Free Supervision of Neural Networks with Physics and Domain Knowledge**<br> *Russell Stewart , Stefano Ermon*, (2016) 
 
- **Slow Feature Analysis:Unsupervised Learning of Invariance**<br> *Laurenz Wiskott, Terrence J. Sejnowski*
 - **Incremental Slow Feature Analysis** <br> *Varun Raj Kompella, Matthew Luciw, and Jurgen Schmidhuber* (2011)

## Autoencoder [Link](https://github.com/TLESORT/State-Representation-Learning-An-Overview/blob/master/Autoencoders.md)

- autoencoder
- denosing autoencoder


- **Stable reinforcement learning with autoencoders for tactile and visual data.**<br> *van Hoof, Herke, et al*, (2016)
- **Deep Spatial Autoencoders for Visuomotor Learning**<br> *Finn, Chelsea, et al.*, (2016)


## Variational learning

- variational autoencoder

- **Deep Variational Bayes Filters: Unsupervised Learning of State Space Models from Raw Data**  <br> Maximilian Karl, Maximilian Soelch, Justin Bayer, Patrick van der Smagt, (2017), pdf
- **Deep Kalman Filters** <br> Rahul G. Krishnan, Uri Shalit, David Sontag, (2015), pdf

## Reinforcement Learning

## Embedded (metric) Learning

## Multi modal learning

The inconvenient is the need to learn a dense representation before doing matching because it suffers from the curse of dimensionality 

## Against State Representation Learning

- **Learning to Filter with Predictive State Inference Machines** , *Wen Sun, Arun Venkatraman, Byron Boots, J. Andrew Bagnell*, (2016) [pdf](https://arxiv.org/pdf/1512.08836)

## Using Priors
2 papers of Jonchowsky PVE (Jonchosk)

## Against Priors
Learning Visual Reasoning Without Strong Priors
Ethan Perez, Harm de Vries, Florian Strub, Vincent Dumoulin, Aaron Courville, 2017

## Visual Relational Learning:
Learning Visual Reasoning Without Strong Priors
Ethan Perez, Harm de Vries, Florian Strub, Vincent Dumoulin, Aaron Courville, 2017

Relational Networks 

Visual Interaction Networks  (both DeepMind)


## non classified
- **Embed to control: A locally linear latent dynamics model for control from raw images** <br> Watter, Manuel, et al, (2015)
 
 - **Representation Learning: A Review and New Perspectives** <br> Yoshua Bengio, Aaron Courville, and Pascal Vincent, (2012), pdf
 
## Validation Methods


