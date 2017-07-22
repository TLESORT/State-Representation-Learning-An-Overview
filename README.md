# State-Representation-Learning(-In-Robotics): An-Overview

# [Overleaf](https://www.overleaf.com/10392879srrcsmhcgrkz)

# DEEP LEARNING IN ROBOTICS IS COMING  !!!!

## Abstract

Deep learning in robotics is coming. Very soon robot will be able to benefit from the deep learning framework to make complex choices and predictions in a robust and automous fashon. The interesting particularity of a robot is that it's input are , as a human, multimodale. The robot can use it's camera at the same time than a lidar, a radar, a microphone or all the tools you can imagine. For deep learning here come the classic problem of the curse of dimensionality. How to make an algortihm able to make prediction with several high dimension inputs and how make it find hidden dependencies between them online ? The solution is reduce the dimensionality by learning state representation. State representation learning means find the few hidden parameters of each of the input (or modularity). Once the hidden parameters are found the task of finding dependencies between modularities is no more bothered by the dimensionality. This paper aims to cover the state of the art about state representation learning. It presents the different methods used to disentangle the hidden parameters of an datasets and to validate the learned state representation. This overview is particularely focus on learning representation in low dimensionnality (<5) of known parameters like the state of an 3D object. This scope make it possible to assess the representation learned.


## Learning with priors [Link](https://github.com/TLESORT/State-Representation-Learning-An-Overview/blob/master/LearningWithApriori.md)

- constraint
- siamese networks
 See nice definition in:
* Siamese Neural Networks for One-shot Image Recognition, Gregory Koch Richard Zemel Ruslan Salakhutdinov


- **PVEs: Position-Velocity Encoders for Unsupervised Learning of Structured State Representations**, *Rico Jonschkowski, Roland Hafner, Jonathan Scholz, Martin Riedmiller*, (2017), [pdf](https://arxiv.org/pdf/1705.09805), [arXiv](https://arxiv.org/abs/1705.09805), [Arxiv](https://arxiv.org/abs/1705.09805) [bib](http://dblp.uni-trier.de/rec/bibtex/journals/corr/JonschkowskiHSR17)
PVE considers 1) Conceptual abstraction. 2) Compositional structure. 3) Common sense priors.

- **Learning State Representation for Deep Actor-Critic Control. Jelle Munk 2016. On predictive priors:  Ils utilisent simplement le fait que l'état doit permettre de prédire efficacement le prochain état et la récompense. Ca pourrait être assez simple a tester dans notre cas j'imagine.

 - **Learning State Representations with Robotic Priors**, *Rico Jonschkowski, Oliver Brock*, (2015) , [pdf](https://pdfs.semanticscholar.org/dc93/f6d1b704abf12bbbb296f4ec250467bcb882.pdf) [bib](http://dl.acm.org/citation.cfm?id=2825776)

 - **A Physics-Based Model Prior for Object-Oriented MDPs** (2014) <br>
*Jonathan Scholz, Martin Levihn, Charles L. Isbell, David Wingate*, [pdf](http://proceedings.mlr.press/v32/scholz14.pdf) [bib](http://dl.acm.org/citation.cfm?id=3045014)

- **The Curious Robot: Learning Visual Representations via Physical Interactions** <br> *Lerrel Pinto, Dhiraj Gandhi, Yuanfeng Han, Yong-Lae Park, Abhinav Gupta*,(2016), [pdf](https://arxiv.org/pdf/1604.01360.pdf) [bib](http://dblp.uni-trier.de/rec/bibtex/journals/corr/PintoGHPG16) [slides](https://pdfs.semanticscholar.org/a6ee/1a3d623daa2714f70232d4fa61cbd1b3cff3.pdf)

- **Label-Free Supervision of Neural Networks with Physics and Domain Knowledge**<br> *Russell Stewart , Stefano Ermon*, (2016)

- **Slow Feature Analysis:Unsupervised Learning of Invariance**<br> *Laurenz Wiskott, Terrence J. Sejnowski* [pdf](https://papers.cnl.salk.edu/PDFs/Slow%20Feature%20Analysis_%20Unsupervised%20Learning%20of%20Invariances%202002-3430.pdf) [bib](http://dl.acm.org/citation.cfm?id=638941)
 - **Incremental Slow Feature Analysis** <br> *Varun Raj Kompella, Matthew Luciw, and Jurgen Schmidhuber* (2011) [pdf](https://www.ijcai.org/Proceedings/11/Papers/229.pdf) [ArXiv](https://arxiv.org/abs/1112.2113) [bib](http://dblp.uni-trier.de/rec/bibtex/journals/corr/abs-1112-2113)

## Autoencoder [Link](https://github.com/TLESORT/State-Representation-Learning-An-Overview/blob/master/Autoencoders.md)

- autoencoder
- denosing autoencoder


- **Stable reinforcement learning with autoencoders for tactile and visual data.**<br> *van Hoof, Herke, et al*, (2016)
- **Deep Spatial Autoencoders for Visuomotor Learning**<br> *Finn, Chelsea, et al.*, (2015)


## Variational learning [Link](https://github.com/TLESORT/State-Representation-Learning-An-Overview/blob/master/Variational.md)

[NEED TO CONTINUE BIBB SEARCH FROM HERE]

- variational autoencoder

- **Deep Variational Bayes Filters: Unsupervised Learning of State Space Models from Raw Data**  <br> Maximilian Karl, Maximilian Soelch, Justin Bayer, Patrick van der Smagt, (2017), pdf
- **Deep Kalman Filters** <br> Rahul G. Krishnan, Uri Shalit, David Sontag, (2015), [pdf](https://arxiv.org/pdf/1511.05121.pdf)

- **Embed to control: A locally linear latent dynamics model for control from raw images** <br> Watter, Manuel, et al, (2015) [pdf](https://pdfs.semanticscholar.org/21c9/dd68b908825e2830b206659ae6dd5c5bfc02.pdf)

## Reinforcement Learning

- **Loss is its own Reward: Self-Supervision for Reinforcement Learning** (2016) <br>
 *Evan Shelhamer, Parsa Mahmoudieh, Max Argus, Trevor Darrell* [pdf](https://arxiv.org/pdf/1612.07307.pdf)


## Embedded (metric) Learning

- **A new embedding quality assessment method for manifold learning** (2016) <br>
*Yuanyuan Ren, and Bo Zhang*, [pdf](https://arxiv.org/pdf/1108.1636v1.pdf)

- **A Survey on Metric Learning for Feature Vectors and Structured Data** (2013) <br> *Aurélien Bellet, Amaury Habrard, Marc Sebban* [pdf](https://arxiv.org/pdf/1306.6709) [arXiv](https://arxiv.org/abs/1306.6709) [bib](http://dblp.uni-trier.de/rec/bibtex/journals/corr/BelletHS13)

## Multi modal learning

Their inconvenient is the need to learn a dense representation before doing matching because they suffers from the curse of dimensionality

## Against State Representation Learning

- **Learning to Filter with Predictive State Inference Machines** , *Wen Sun, Arun Venkatraman, Byron Boots, J. Andrew Bagnell*, (2016) [pdf](https://arxiv.org/pdf/1512.08836)




## Against Priors
* Learning Visual Reasoning Without Strong Priors
Ethan Perez, Harm de Vries, Florian Strub, Vincent Dumoulin, Aaron Courville, 2017

## Relational and Symbolic Learning:
* Learning Visual Reasoning Without Strong Priors
Ethan Perez, Harm de Vries, Florian Strub, Vincent Dumoulin, Aaron Courville, 2017

* Relational Networks (Santoro’17)

* Towards Deep Symbolic Reinforcement Learning, Garnelo et al. NIPS 2016. Check if it uses priors? and tag it

* Visual Interaction Networks (Watters’17)

* Towards Deep Symbolic Reinforcement Learning, Garnelo et al. NIPS 2016

* Reasoning about Time and Knowledge Neural-Symbolic Learning Systems, d’Avila Garcez et al., NIPS 2004.

* A simple neural network module for relational reasoning, Santoro et al. 2017. 
Proposes a reusable neural network module to reason about the relations between entities and their properties, where an MLP approximates object-to-object relation function and other MLP transforms summed pairwise object-to-object relations to some desired output (RN's operate on sets (due to summation in the formula) and thus are invariant to the order of objects in the input). -> Can we establish thus domain and range in relationships? future extension? In terms of architecture, RN module is used at the tail of a neural network taking input objects in form of CNN or LSTM embeddings. This work is evaluated on several tasks where it achieves reasonably good (even superhuman) performance (CLEVR and Sort-of-CLEVR - question answering about an image0

* Graph convolutional Auto-Encoders. Thomas Kipf‏ @thomaskipf Graph auto-encoders (in TensorFlow) is now available on GitHub: https://github.com/tkipf/gae


## Task-oriented state representation learning

* Gated-attention architectures for Task-Oriented Language Grounding, Chaplot, 2017


## Policy learning- based approaches

* Black-Box Data-efficient Policy Search for Robotics. 2017. Konstantinos Chatzilygeroudis, Roberto Rama, Rituraj Kaushik, Dorian Goepp, Vassilis Vassiliades and Jean-Baptiste Mouret. Gaussian process regression for policy optimisation using model based policy search. The Black-DROPS algorithm learns a high-dimensional policy from scratch in 5 trials, which are enough to learn the whole dynamics of the arm from scratch. https://arxiv.org/abs/1703.07261  @IROS2017 #resibots


* REINFORCEMENT LEARNING WITH UNSUPERVISED AUXILIARY TASKS (UNREAL). Max Jaderberg et al. 2016. UNREAL algorithm shows that augmenting a deep reinforcement learning agent with auxiliary control and reward prediction tasks can double improvement both in data efficiency and robustness to hyperparameter settings.  A successor in learning speed and the robustness to A3C (Over 87% of human scores).


## One/Few-shot approaches

* Siamese Neural Networks for One-shot Image Recognition, Gregory Koch Richard Zemel Ruslan Salakhutdinov

* Optimization as a model for few-shot learning. Ravi and Larochelle, 17




(REPEATED, USE TAGS?): Black-Box Data-efficient Policy Search for Robotics. Konstantinos Chatzilygeroudis, Roberto Rama, Rituraj Kaushik, Dorian Goepp, Vassilis Vassiliades and Jean-Baptiste Mouret*
They unsupervisedly learn perform an action with only 5 episodes and gaussian processes with a robot (Ergo Jr?).


## GANS

* Unsupervised Pixel–Level Domain Adaptation with Generative Adversarial Networks, Bousmalis 16  presents an unsupervised approach using a (GAN)–based architecture that is able to learn such a transformation
in an unsupervised manner, i.e. without using corresponding pairs from the two domains.
It is the best state of the art approach on unsupervised learning for domain adaptation, improving over:
Decoupling from the Task-Specific Architecture, Generalization Across Label Spaces, achieve Training Stability and Data Augmentation.

* BEGAN: Boundary Equilibrium Generative Adversarial Networks, David Berthelot et al. 17
In contrast to traditional GANS that require alternating training D and G, or pretraining D, \textit{BEGAN requires neither to train stably}. The discriminator has two competing goals: auto-encode real images and discriminate
real from generated images (balanced by gamma). They propose a new equilibrium enforcing method paired with a loss derived from the Wasserstein distance for training auto-encoder based GAN that balances the generator and discriminator during training. It provides a new approximate convergence measure, fast and stable training which controls the trade-off between image diversity and visual quality.


## Validation Methods

* On the Quantitative Evaluation of Deep Generative Models Russ Salakhutdinov  www.cs.cmu.edu/~rsalakhu/talk_Eval.pdf


* A new embedding quality assessment method for manifold learning, Zhang11. https://arxiv.org/pdf/1108.1636v1.pdf we have their matlab code
Interesting paper about evaluating embeddings through Normalization independent embedding quality (NIEAQA) ssessment, a normalization independent embedding quality criterion, for manifold learning purposes, based on the anisotropic scaling independent measure (ASIM), which compares the similarity between two configurations under motion and anisotropic coordinate scaling. NIEQA is based on ASIM, and consists of three assessments, a local one, a global one and a linear combination of the two. The local measure evaluates how well local neighborhood information is preserved under anisotropic coordinate scaling and rigid motion. NIEQA is valued between 0 and 1, where 0 represents a perfect preservation and its highlights is is the ability of being applicable to both normalized and isometric embeddings, it can provide both local and global assessments, and it can serve as a natural evaluation tool of learned embeddings \cite{Gracia14}.

* A methodology to compare Dimensionality Reduction algorithms in terms of loss of quality. Antonio Gracia, Santiago González, Víctor Robles, Ernestina Menasalvas, 2014

* Foolbox v0.8.0: A Python toolbox to benchmark the robustness of machine learning models. Rauber17 https://arxiv.org/pdf/1707.04131.pdf

## SURVEYS

- **Representation Learning: A Review and New Perspectives** <br> Yoshua Bengio, Aaron Courville, and Pascal Vincent, (2012), pdf

- **Survey paper on Geometry of Optimization & Implicit Regularization in Deep Learning with Neyshabur, Tomioka, Srebro
https://arxiv.org/abs/1705.03071

- ** A survey on metric learning for feature vectors and structured data. Aurélien Bellet, Amaury Habrard, and Marc Sebban.  2013

- ** Deep learning in neural networks: An overview. Schmidhuber, Jürgen Neural Networks - 2015 

- **Not peer reviewed? https://arxiv.org/pdf/1701.07274.pdf



## Prediction-based learning/ forward models:  http://realai.org/predictive-learning/

Basic idea: The loss is based on prediction errors of next states.

* Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection. The Google dataset release paper proposes a continuous servoing mechanism that uses \textit{the grasp prediction network to choose the motor commands for the robot that will maximize the probability of a success grasp}.  https://arxiv.org/pdf/1603.02199.pdf

* Learning State Representation for Deep Actor-Critic Control. Jelle Munk 2016. On predictive priors:  Ils utilisent simplement le fait que l'état doit permettre de prédire efficacement le prochain état et la récompense. Ca pourrait être assez simple a tester dans notre cas j'imagine.

* Learning a forward/inverse model to learn good representations : https://arxiv.org/pdf/1612.07307.pdf

* MatchNet and TempoNet: CortexNet: a robust predictive deep neural network trained on videos https://engineering.purdue.edu/elab/CortexNet/



## Interpretability  methods for evaluating learned representations
* Understanding intermediate layers using linear classifier probes. Alain and Bengio 16  https://arxiv.org/pdf/1610.01644.pdf

* Explaining the Unexplained: A CLass-Enhanced Attentive Response (CLEAR)
Approach to Understanding Deep Neural Networks, Kumar et al 17  https://arxiv.org/pdf/1704.04133.pdf





## Auxiliary tasks for improving learning: http://realai.org/auxiliary-tasks/

* Reinforcement Learning with Unsupervised Auxiliary Tasks 2016  https://arxiv.org/abs/1611.05397
* Continuous control with deep reinforcement learning, 2015.  https://arxiv.org/abs/1509.02971
* The Intentional Unintentional Agent: Learning to Solve Many Continuous Control Tasks Simultaneously, 2017  https://arxiv.org/pdf/1707.03300.pdf




## GANS

* State-of-the-art GANs for unsupervised representation learning: BEGAN (or BiGAN?), CycleGAN and pixel based GAN. See Allan Ma survey to appear soon.

* BEGAN: Boundary Equilibrium Generative Adversarial Networks, David Berthelot et al.
 In contrast to traditional GANS that require alternating training D and G, or pretraining D, \textit{BEGAN requires neither to train stably}. The discriminator has two competing goals: auto-encode real images and discriminate
 real from generated images (balanced by gamma). They propose a new equilibrium enforcing method paired with a loss derived from the Wasserstein distance for training auto-encoder based GAN that balances the generator and discriminator during training. It provides a new approximate convergence measure, fast and stable training which controls the trade-off between image diversity and visual quality.


* GAN to improve learning machine robustness in image segmentation, pose estimation and speech recognition using Houdini loss.  Houdini loss is a product of two terms. The first term is a stochastic margin, that is the
probability that the difference between the score of the actual target g(x; y) and that of the predicted
target g(x; ^y) is smaller than  N(0; 1). It reflects the confidence of the model in its predictions. Houdini is a lower bound of the task loss and considers the difference between the scores assigned by the network to the ground truth and the prediction, and it converges to the task loss.: Houdini: Fooling Deep Structured Prediction Models, Cisse17.



## Few-shot learning

* Few-Shot Learning Through an Information Retrieval Lens. Triantafillou, 2017
* Optimization as a model for few-shot learning. Ravi and Larochelle, 17




## Category labels found that expand across papers (to decide if they are columns to add to our paper table?)
Prediction-based/same as forward models?

reward-less
Using priors
0ne/few-shot
By demonstration, cloning, example



## non classified / Off topic: other interesting papers   ## TO CLASSIFY:


* Add Pieter abbel 17 June17. 

* Wang, X., Gupta, A.: Unsupervised learning of visual representations using videos.
ICCV (2015)

