# State-Representation-Learning(-In-Robotics): An-Overview

# [Overleaf](https://www.overleaf.com/10392879srrcsmhcgrkz)

# DEEP LEARNING IN ROBOTICS IS COMING  !!!!

## Abstract

Deep learning in robotics is coming. Very soon robot will be able to benefit from the deep learning framework to make complex choices and predictions in a robust and automous fashon. The interesting particularity of a robot is that it's input are , as a human, multimodale. The robot can use it's camera at the same time than a lidar, a radar, a microphone or all the tools you can imagine. For deep learning here come the classic problem of the curse of dimensionality. How to make an algortihm able to make prediction with several high dimension inputs and how make it find hidden dependencies between them online ? The solution is reduce the dimensionality by learning state representation. State representation learning means find the few hidden parameters of each of the input (or modularity). Once the hidden parameters are found the task of finding dependencies between modularities is no more bothered by the dimensionality. This paper aims to cover the state of the art about state representation learning. It presents the different methods used to disentangle the hidden parameters of an datasets and to validate the learned state representation. This overview is particularely focus on learning representation in low dimensionnality (<5) of known parameters like the state of an 3D object. This scope make it possible to assess the representation learned.

## Scope of the paper (I am still no sure about this)

This paper speak about learning low dimension representation easely interpretable which can be assess thanks to a ground truth. The groundtruth should nethertheless not be use for learning the representation. <br>
The representation can be task specific.<br>

Domain of aplication :<br>
- robotics<br>
- data compression<br>
- embbed learning<br>
- [...]<br>


## Learning with priors [Link](https://github.com/TLESORT/State-Representation-Learning-An-Overview/blob/master/LearningWithApriori.md)


- **PVEs: Position-Velocity Encoders for Unsupervised Learning of Structured State Representations**, *Rico Jonschkowski, Roland Hafner, Jonathan Scholz, Martin Riedmiller*, (2017), [pdf](https://arxiv.org/pdf/1705.09805), [arXiv](https://arxiv.org/abs/1705.09805), [Arxiv](https://arxiv.org/abs/1705.09805) [bib](http://dblp.uni-trier.de/rec/bibtex/journals/corr/JonschkowskiHSR17)

- **Learning State Representation for Deep Actor-Critic Control**. Jelle Munk 2016. 

 - **Learning State Representations with Robotic Priors**, *Rico Jonschkowski, Oliver Brock*, (2015) , [pdf](https://pdfs.semanticscholar.org/dc93/f6d1b704abf12bbbb296f4ec250467bcb882.pdf) [bib](http://dl.acm.org/citation.cfm?id=2825776)

 - **A Physics-Based Model Prior for Object-Oriented MDPs** (2014) <br>
*Jonathan Scholz, Martin Levihn, Charles L. Isbell, David Wingate*, [pdf](http://proceedings.mlr.press/v32/scholz14.pdf) [bib](http://dl.acm.org/citation.cfm?id=3045014)


- **Label-Free Supervision of Neural Networks with Physics and Domain Knowledge**<br> *Russell Stewart , Stefano Ermon*, (2016)


## Autoencoder [Link](https://github.com/TLESORT/State-Representation-Learning-An-Overview/blob/master/Autoencoders.md)


- **Stable reinforcement learning with autoencoders for tactile and visual data.**<br> *van Hoof, Herke, et al*, (2016)
- **Deep Spatial Autoencoders for Visuomotor Learning**<br> *Finn, Chelsea, et al.*, (2015)


## Variational autoencoder family [Link](https://github.com/TLESORT/State-Representation-Learning-An-Overview/blob/master/Variational.md)


- **Deep Variational Bayes Filters: Unsupervised Learning of State Space Models from Raw Data** , *Maximilian Karl, Maximilian Soelch, Justin Bayer, Patrick van der Smagt*, (2017),  [pdf](https://openreview.net/pdf?id=HyTqHL5xg) [arXiv](https://arxiv.org/abs/1605.06432) [bib](http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2016arXiv160506432K&data_type=BIBTEX&db_key=PRE&nocookieset=1)


- **Deep Kalman Filters**, *Rahul G. Krishnan, Uri Shalit, David Sontag*, (2015), [pdf](https://arxiv.org/abs/1511.05121) [arXiv](https://arxiv.org/abs/1511.05121) [bib](http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2015arXiv151105121K&data_type=BIBTEX&db_key=PRE&nocookieset=1)

- **Embed to control: A locally linear latent dynamics model for control from raw images** <br> *Watter, Manuel, et al*, (2015) [pdf](https://pdfs.semanticscholar.org/21c9/dd68b908825e2830b206659ae6dd5c5bfc02.pdf) [arXiv](https://arxiv.org/abs/1506.07365) [bib](http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2015arXiv150607365W&data_type=BIBTEX&db_key=PRE&nocookieset=1)


## Forward/ Predictive Models

- **Loss is its own Reward: Self-Supervision for Reinforcement Learning** (2016) <br>
 *Evan Shelhamer, Parsa Mahmoudieh, Max Argus, Trevor Darrell* [pdf](https://arxiv.org/pdf/1612.07307.pdf) :one:

### No reward learning
Once a loss on reward is defined, in end-to-end RL systems, the representation is delegated to backpropagation without further attention to other supervisory signals. Representation learning can thus be considered a bottleneck in current approaches bound by reward \cite{Shelhamer17}.  Next we describe some approaches in this line:

* Curiosity-driven Exploration by Self-supervised Prediction. Deepak Pathak et al. \cite{Pathak17} http://juxi.net/workshop/deep-learning-robotic-vision-cvpr-2017/papers/23.pdf
Self-supervised approach.



## Applications of state representaiton learning

### Reinforcement Learning

[CONTINUER LES CITATIONS A PARTIR D'ICI]

- **Loss is its own Reward: Self-Supervision for Reinforcement Learning** (2016) <br>
 *Evan Shelhamer, Parsa Mahmoudieh, Max Argus, Trevor Darrell* [pdf](https://arxiv.org/pdf/1612.07307.pdf) :two:



### Embedded (metric) Learning

- **A new embedding quality assessment method for manifold learning** (2016) <br>
*Yuanyuan Ren, and Bo Zhang*, [pdf](https://arxiv.org/pdf/1108.1636v1.pdf)

- **A Survey on Metric Learning for Feature Vectors and Structured Data** (2013) <br> *Aurélien Bellet, Amaury Habrard, Marc Sebban* [pdf](https://arxiv.org/pdf/1306.6709) [arXiv](https://arxiv.org/abs/1306.6709) [bib](http://dblp.uni-trier.de/rec/bibtex/journals/corr/BelletHS13)

### Multi modal learning

- **Gated-attention architectures for Task-Oriented Language Grounding** (2017) <br>
 *Devendra Singh Chaplot, Kanthashree Mysore Sathyendra, Rama Kumar Pasumarthi, Dheeraj Rajagopal, Ruslan Salakhutdinov* [pdf](https://arxiv.org/pdf/1706.07230) [arXiv](https://arxiv.org/abs/1706.07230) 

## Against State Representation Learning
- **Learning to Filter with Predictive State Inference Machines** , *Wen Sun, Arun Venkatraman, Byron Boots, J. Andrew Bagnell*, (2016) [pdf](https://arxiv.org/pdf/1512.08836)


## End-to-end approaches:
End-to-end reinforcement learning (RL) addresses representation learning at the same time as policy optimization, where current efforts tackle this problem normally from the point of view of stochastic optimization and exploration.



## Against Priors
* Learning Visual Reasoning Without Strong Priors
Ethan Perez, Harm de Vries, Florian Strub, Vincent Dumoulin, Aaron Courville, 2017



######
## Physics states and property learning:
- **Schema Networks: Zero-shot Transfer with a Generative Causal Model of Intuitive Physics**, *Kansky Tom Silver David A. Mely Mohamed Eldawy Miguel Lazaro-Gredilla Xinghua Lou, Nimrod Dorfman Szymon Sidor Scott Phoenix Dileep George*.
   Schema Networks are closely related to Object-Oriented MDPs (OO-MDPs) (Diuk et al., 2008) and Relational MDPs (R-MDPs) (Guestrin et al., 2003a).



## Policy learning- based approaches

* Black-Box Data-efficient Policy Search for Robotics. 2017. Konstantinos Chatzilygeroudis, Roberto Rama, Rituraj Kaushik, Dorian Goepp, Vassilis Vassiliades and Jean-Baptiste Mouret. Gaussian process regression for policy optimisation using model based policy search. The Black-DROPS algorithm learns a high-dimensional policy from scratch in 5 trials, which are enough to learn the whole dynamics of the arm from scratch. https://arxiv.org/abs/1703.07261  @IROS2017 #resibots


* REINFORCEMENT LEARNING WITH UNSUPERVISED AUXILIARY TASKS (UNREAL). Max Jaderberg et al. 2016. UNREAL algorithm shows that augmenting a deep reinforcement learning agent with auxiliary control and reward prediction tasks can double improvement both in data efficiency and robustness to hyperparameter settings.  A successor in learning speed and the robustness to A3C (Over 87% of human scores).




## Validation Methods and Frameworks 

Example ways of assessing CNN's performance is measuring networks receptive field [Cite online Tool], or attention maps. Other methods are described below.

* On the Quantitative Evaluation of Deep Generative Models Russ Salakhutdinov  www.cs.cmu.edu/~rsalakhu/talk_Eval.pdf

* A new embedding quality assessment method for manifold learning, Zhang11. https://arxiv.org/pdf/1108.1636v1.pdf we have their matlab code
Interesting paper about evaluating embeddings through Normalization independent embedding quality (NIEAQA) ssessment, a normalization independent embedding quality criterion, for manifold learning purposes, based on the anisotropic scaling independent measure (ASIM), which compares the similarity between two configurations under motion and anisotropic coordinate scaling. NIEQA is based on ASIM, and consists of three assessments, a local one, a global one and a linear combination of the two. The local measure evaluates how well local neighborhood information is preserved under anisotropic coordinate scaling and rigid motion. NIEQA is valued between 0 and 1, where 0 represents a perfect preservation and its highlights is is the ability of being applicable to both normalized and isometric embeddings, it can provide both local and global assessments, and it can serve as a natural evaluation tool of learned embeddings \cite{Gracia14}.

* A methodology to compare Dimensionality Reduction algorithms in terms of loss of quality. Antonio Gracia, Santiago González, Víctor Robles, Ernestina Menasalvas, 2014

* Foolbox v0.8.0: A Python toolbox to benchmark the robustness of machine learning models. Rauber17 https://arxiv.org/pdf/1707.04131.pdf


### Interpretability  methods for evaluating learned representations

[Assessment problematique]

The objective of WHAT? assessment?** is to give a quantitative value which estimates the quality of a representation. In the context of representation learning this can be harder than expected. The assessment should show if the representaiotn we learned is conform to what we expect.<br>
Do we expect something in particular?<br>
What can we assume other that the information hold by the representation? <- the neighborhood!<br>
and then what tool can we use?<br>
What learning method make us able to make stronger result assumption?<br><br>
Still thinking about it.....<br>


* Understanding intermediate layers using linear classifier probes. Alain and Bengio 16  https://arxiv.org/pdf/1610.01644.pdf

* Explaining the Unexplained: A CLass-Enhanced Attentive Response (CLEAR)
Approach to Understanding Deep Neural Networks, Kumar et al 17  https://arxiv.org/pdf/1704.04133.pdf

* Foolbox v0.8.0: A Python toolbox to benchmark the robustness of machine learning models. Rauber17 is a library that tests for adversarial attackes, different evaluation metrics and machine learning models.




### Qualitative method

### Quantitative method (METRICS)
Particular metrics of interest to assess quality of prediction go beyond the \textit{blurry} MSE (Mean Squared Error) loss function. Complementary feature learning strategies include multi-scale architectures, adversarial training methods, and image gradient difference loss functions as proposed in \cite{Mathieu15}. More concretely,  the Peak Signal to Noise Ratio, Structural Similarity Index Measure and image sharpness show to be better proxies for next frame prediction assessment \cite{Mathieu15}.



## Prediction-based learning/ forward models:  http://realai.org/predictive-learning/

Basic idea: The loss is based on prediction errors of next states.

* Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection. The Google dataset release paper proposes a continuous servoing mechanism that uses \textit{the grasp prediction network to choose the motor commands for the robot that will maximize the probability of a success grasp}.  https://arxiv.org/pdf/1603.02199.pdf

* Learning State Representation for Deep Actor-Critic Control. Jelle Munk 2016. On predictive priors:  Ils utilisent simplement le fait que l'état doit permettre de prédire efficacement le prochain état et la récompense. Ca pourrait être assez simple a tester dans notre cas j'imagine. See summary in fwd models.

* Learning a forward/inverse model to learn good representations : https://arxiv.org/pdf/1612.07307.pdf

* MatchNet and TempoNet: CortexNet: a robust predictive deep neural network trained on videos https://engineering.purdue.edu/elab/CortexNet/

* Deep multi-scale video prediction beyond mean square error, Mathieu15. Video Prediction can be done with more robust measures than MSE. In \cite{Mathieu15}, they propose several strategies for next frame prediction evaluation assessing the quality of the prediction in terms of Peak Signal to Noise Ratio, Structural Similarity Index Measure and image sharpness.  
IDEA: can be extended to be combined with optical flow prediction  and replace optical flow prediction algorithms with next frame prediciotion.

*  Value Prediction Networks (VPN) \cite{Oh17} (see summary in forward models)






 .
