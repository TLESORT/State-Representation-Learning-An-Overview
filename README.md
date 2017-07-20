# State-Representation-Learning(-In-Robotics): An-Overview

# DEEP LEARNING IN ROBOTICS IS COMING  !!!!

## Abstract

Deep learning in robotics is coming. Very soon robot will be able to benefit from the deep learning framework to make complex choices and predictions in a robust and automous fashon. The interesting particularity of a robot is that it's input are , as a human, multimodale. The robot can use it's camera at the same time than a lidar, a radar, a microphone or all the tools you can imagine. For deep learning here come the classic problem of the curse of dimensionality. How to make an algortihm able to make prediction with several high dimension inputs and how make it find hidden dependencies between them online ? The solution is reduce the dimensionality by learning state representation. State representation learning means find the few hidden parameters of each of the input (or modularity). Once the hidden parameters are found the task of finding dependencies between modularities is no more bothered by the dimensionality. This paper aims to cover the state of the art about state representation learning. It presents the different methods used to disentangle the hidden parameters of an datasets and to validate the learned state representation. This overview is particularely focus on learning representation in low dimensionnality (<5) of known parameters like the state of an 3D object. This scope make it possible to assess the representation learned. 


## Learning with priors [Link](https://github.com/TLESORT/State-Representation-Learning-An-Overview/blob/master/LearningWithApriori.md)

- constraint
- siamese network

- **PVEs: Position-Velocity Encoders for Unsupervised Learning of Structured State Representations** <br> Rico Jonschkowski, Roland Hafner, Jonathan Scholz, Martin Riedmiller, (2017), [pdf](https://arxiv.org/pdf/1705.09805.pdf)

 - **Learning State Representations with Robotic Priors**<br> Rico Jonschkowski, Oliver Brock, (2015) , pdf <br>
 
 - **A Physics-Based Model Prior for Object-Oriented MDPs** <br> *Jonathan Scholz, Martin Levihn, Charles L. Isbell, David Wingate*, (2014) [pdf](http://proceedings.mlr.press/v32/scholz14.pdf)  

- **The Curious Robot: Learning Visual Representations via Physical Interactions** <br> Lerrel Pinto, Dhiraj Gandhi, Yuanfeng Han, Yong-Lae Park, Abhinav Gupta,(2016) (hidden state representation) <br>

- **Label-Free Supervision of Neural Networks with Physics and Domain Knowledge**<br> *Russell Stewart , Stefano Ermon*, (2016) 
 
- **Slow Feature Analysis:Unsupervised Learning of Invariance**<br> *Laurenz Wiskott, Terrence J. Sejnowski* [pdf](https://papers.cnl.salk.edu/PDFs/Slow%20Feature%20Analysis_%20Unsupervised%20Learning%20of%20Invariances%202002-3430.pdf)

 - **Incremental Slow Feature Analysis** <br> *Varun Raj Kompella, Matthew Luciw, and Jurgen Schmidhuber* (2011) [pdf](https://www.ijcai.org/Proceedings/11/Papers/229.pdf)

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

- **Loss is its own Reward: Self-Supervision for Reinforcement Learning** (2016) <br>
 *Evan Shelhamer, Parsa Mahmoudieh, Max Argus, Trevor Darrell* [pdf](https://arxiv.org/pdf/1612.07307.pdf)

## Embedded (metric) Learning

- **A new embedding quality assessment method for manifold learning** (2016) <br>
*Yuanyuan Ren, and Bo Zhang*, [pdf](https://arxiv.org/pdf/1108.1636v1.pdf)

## Multi modal learning

The inconvenient is the need to learn a dense representation before doing matching because it suffers from the curse of dimensionality 

## Against State Representation Learning

- **Learning to Filter with Predictive State Inference Machines** , *Wen Sun, Arun Venkatraman, Byron Boots, J. Andrew Bagnell*, (2016) [pdf](https://arxiv.org/pdf/1512.08836)

## Using Priors
2 papers of Jonchowsky PVE (Jonchosk)
Conceptual abstraction. 2) Compositional structure. 3) Common sense priors: 

On predictive priors: http://www.jenskober.de/MunkCDC2016.pdf  Ils utilisent simplement le fait que l'état doit permettre de prédire efficacement le prochain état et la récompense. Ca pourrait être assez simple a tester dans notre cas j'imagine.



## Against Priors
Learning Visual Reasoning Without Strong Priors
Ethan Perez, Harm de Vries, Florian Strub, Vincent Dumoulin, Aaron Courville, 2017

## Relational and Symbolic Learning:
Learning Visual Reasoning Without Strong Priors
Ethan Perez, Harm de Vries, Florian Strub, Vincent Dumoulin, Aaron Courville, 2017

Relational Networks (Santoro’17) 

Towards Deep Symbolic Reinforcement Learning, Garnelo et al. NIPS 2016. Check if it uses priors? and tag it

Visual Interaction Networks (Watters’17)

Towards Deep Symbolic Reinforcement Learning, Garnelo et al. NIPS 2016

Reasoning about Time and Knowledge Neural-Symbolic Learning Systems, d’Avila Garcez et al., NIPS 2004.

A simple neural network module for relational reasoning, Santoro et al. 2017.

Graph convolutional Auto-Encoders. Thomas Kipf‏ @thomaskipf Graph auto-encoders (in TensorFlow) is now available on GitHub: https://github.com/tkipf/gae 


## Task-oriented state representation learning

Gated-attention architectures for Task-Oriented Language Grounding, Chaplot, 2017


## Policy learning- based approaches

 Black-Box Data-efficient Policy Search for Robotics. 2017. Konstantinos Chatzilygeroudis, Roberto Rama, Rituraj Kaushik, Dorian Goepp, Vassilis Vassiliades and Jean-Baptiste Mouret. Gaussian process regression for policy optimisation using model based policy search. The Black-DROPS algorithm learns a high-dimensional policy from scratch in 5 trials, which are enough to learn the whole dynamics of the arm from scratch. https://arxiv.org/abs/1703.07261  @IROS2017 #resibots


REINFORCEMENT LEARNING WITH UNSUPERVISED AUXILIARY TASKS (UNREAL). Max Jaderberg et al. 2016. UNREAL algorithm shows that augmenting a deep reinforcement learning agent with auxiliary control and reward prediction tasks can double improvement both in data efficiency and robustness to hyperparameter settings.  A successor in learning speed and the robustness to A3C (Over 87% of human scores). 


## One/Few-shot approaches

Siamese Neural Networks for One-shot Image Recognition, Gregory Koch Richard Zemel Ruslan Salakhutdinov

(REPEATED, USE TAGS?): Black-Box Data-efficient Policy Search for Robotics
Konstantinos Chatzilygeroudis, Roberto Rama, Rituraj Kaushik, Dorian Goepp,
Vassilis Vassiliades and Jean-Baptiste Mouret*

## GANS

Unsupervised Pixel–Level Domain Adaptation with Generative Adversarial Networks, Bousmalis 16  presents an unsupervised approach using a (GAN)–based architecture that is able to learn such a transformation
in an unsupervised manner, i.e. without using corresponding pairs from the two domains. 
It is the best state of the art approach on unsupervised learning for domain adaptation, improving over:
Decoupling from the Task-Specific Architecture, Generalization Across Label Spaces, achieve Training Stability and Data Augmentation.

BEGAN: Boundary Equilibrium Generative Adversarial Networks, David Berthelot et al. 17
In contrast to traditional GANS that require alternating training D and G, or pretraining D, \textit{BEGAN requires neither to train stably}. The discriminator has two competing goals: auto-encode real images and discriminate
real from generated images (balanced by gamma). They propose a new equilibrium enforcing method paired with a loss derived from the Wasserstein distance for training auto-encoder based GAN that balances the generator and discriminator during training. It provides a new approximate convergence measure, fast and stable training which controls the trade-off between image diversity and visual quality. 


## Validation Methods

On the Quantitative Evaluation of Deep Generative Models Russ Salakhutdinov  www.cs.cmu.edu/~rsalakhu/talk_Eval.pdf


## SURVEYS
- **Not peer reviewed? https://arxiv.org/pdf/1701.07274.pdf


- **Representation Learning: A Review and New Perspectives** <br> Yoshua Bengio, Aaron Courville, and Pascal Vincent, (2012), pdf

- **Survey paper on Geometry of Optimization & Implicit Regularization in Deep Learning with Neyshabur, Tomioka, Srebro 
https://arxiv.org/abs/1705.03071 


## non classified / Off topic: other interesting papers

- **Embed to control: A locally linear latent dynamics model for control from raw images** <br> Watter, Manuel, et al, (2015)


## Prediction-based learning/ forward models
Basic idea: The loss is based on prediction errors of next states.

Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection. The Google dataset release paper proposes a continuous servoing mechanism that uses \textit{the grasp prediction network to choose the motor commands for the robot that will maximize the probability of a success grasp}.  https://arxiv.org/pdf/1603.02199.pdf


On predictive priors: http://www.jenskober.de/MunkCDC2016.pdf  Ils utilisent simplement le fait que l'état doit permettre de prédire efficacement le prochain état et la récompense. Ca pourrait être assez simple a tester dans notre cas j'imagine.

TODO Add Pieter abbel 17 June17. Check if evolution of UNREAL (Mnih17 fits here too, I think so)


## Interpretability  methods for evaluating learned representations
Understanding intermediate layers using linear classifier probes. Alain and Bengio 16  https://arxiv.org/pdf/1610.01644.pdf

Explaining the Unexplained: A CLass-Enhanced Attentive Response (CLEAR)
Approach to Understanding Deep Neural Networks, Kumar et al 17  https://arxiv.org/pdf/1704.04133.pdf


# Category labels found that expand accross papers (to decide if they are columns to add to our paper table?)
Prediction-based
reward-less
Using priors
