# State-Representation-Learning(-In-Robotics): An-Overview

# [Overleaf](https://www.overleaf.com/10392879srrcsmhcgrkz)

# DEEP LEARNING IN ROBOTICS IS COMING  !!!!

## Abstract

Deep learning in robotics is coming. Very soon robot will be able to benefit from the deep learning framework to make complex choices and predictions in a robust and automous fashon. The interesting particularity of a robot is that it's input are , as a human, multimodale. The robot can use it's camera at the same time than a lidar, a radar, a microphone or all the tools you can imagine. For deep learning here come the classic problem of the curse of dimensionality. How to make an algortihm able to make prediction with several high dimension inputs and how make it find hidden dependencies between them online ? The solution is reduce the dimensionality by learning state representation. State representation learning means find the few hidden parameters of each of the input (or modularity). Once the hidden parameters are found the task of finding dependencies between modularities is no more bothered by the dimensionality. This paper aims to cover the state of the art about state representation learning. It presents the different methods used to disentangle the hidden parameters of an datasets and to validate the learned state representation. This overview is particularely focus on learning representation in low dimensionnality (<5) of known parameters like the state of an 3D object. This scope make it possible to assess the representation learned.

## Scope of the paper (I am still no sure about this)

This paper speak about low dimension representation easely interpretable which can be assess thanks to a ground truth. The groundtruth should nethertheless not be use for learning the representation. <br>
The representation can be task specific.<br>

Domain of aplication :<br>
- robotics<br>
- data compression<br>
- embbed learning<br>
- [...]<br>


## Assessment problematique

The objective of WHAT? assessment?** is to give a quantitative value which estimates the quality of a representation. In the context of representation learning this can be harder than expected. The assessment should show if the representaiotn we learned is conform to what we expect.<br>
Do we expect something in particular?<br>
What can we assume other that the information hold by the representation? <- the neighborhood!<br>
and then what tool can we use?<br>
What learning method make us able to make stronger result assumption?<br><br>
Still thinking about it.....<br>

## Learning with priors [Link](https://github.com/TLESORT/State-Representation-Learning-An-Overview/blob/master/LearningWithApriori.md)

- constraint
- siamese networks
 See nice definition in:
**Siamese Neural Networks for One-shot Image Recognition**, *Gregory Koch Richard Zemel Ruslan Salakhutdinov*


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

- variational autoencoder

- **Deep Variational Bayes Filters: Unsupervised Learning of State Space Models from Raw Data** , *Maximilian Karl, Maximilian Soelch, Justin Bayer, Patrick van der Smagt*, (2017),  [pdf](https://openreview.net/pdf?id=HyTqHL5xg) [arXiv](https://arxiv.org/abs/1605.06432) [bib](http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2016arXiv160506432K&data_type=BIBTEX&db_key=PRE&nocookieset=1)


- **Deep Kalman Filters**, *Rahul G. Krishnan, Uri Shalit, David Sontag*, (2015), [pdf](https://arxiv.org/abs/1511.05121) [arXiv](https://arxiv.org/abs/1511.05121) [bib](http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2015arXiv151105121K&data_type=BIBTEX&db_key=PRE&nocookieset=1)

- **Embed to control: A locally linear latent dynamics model for control from raw images** <br> *Watter, Manuel, et al*, (2015) [pdf](https://pdfs.semanticscholar.org/21c9/dd68b908825e2830b206659ae6dd5c5bfc02.pdf) [arXiv](https://arxiv.org/abs/1506.07365) [bib](http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2015arXiv150607365W&data_type=BIBTEX&db_key=PRE&nocookieset=1)

## Reinforcement Learning

[CONTINUER LES CITATIONS A PARTIR D'ICI]

- **Loss is its own Reward: Self-Supervision for Reinforcement Learning** (2016) <br>
 *Evan Shelhamer, Parsa Mahmoudieh, Max Argus, Trevor Darrell* [pdf](https://arxiv.org/pdf/1612.07307.pdf)
Self-supervised pre-training and joint optimization using auxiliary losses in the absence of rewards improve the
data efficiency and policy returns of end-to-end reinforcement learning \cite{Shelhamer17}.


## Embedded (metric) Learning

- **A new embedding quality assessment method for manifold learning** (2016) <br>
*Yuanyuan Ren, and Bo Zhang*, [pdf](https://arxiv.org/pdf/1108.1636v1.pdf)

- **A Survey on Metric Learning for Feature Vectors and Structured Data** (2013) <br> *Aurélien Bellet, Amaury Habrard, Marc Sebban* [pdf](https://arxiv.org/pdf/1306.6709) [arXiv](https://arxiv.org/abs/1306.6709) [bib](http://dblp.uni-trier.de/rec/bibtex/journals/corr/BelletHS13)

## Multi modal learning

Their inconvenient is the need to learn a dense representation before doing matching because they suffers from the curse of dimensionality

## Against State Representation Learning
- **Learning to Filter with Predictive State Inference Machines** , *Wen Sun, Arun Venkatraman, Byron Boots, J. Andrew Bagnell*, (2016) [pdf](https://arxiv.org/pdf/1512.08836)


## End-to-end approaches:
End-to-end reinforcement learning (RL) addresses representation learning at the same time as policy optimization, where current efforts tackle this problem normally from the point of view of stochastic optimization and exploration.

## Self-supervision
Self-supervised auxiliary losses extend the limitations of traditional reinforcement learning to learn from all experience, whether rewarded or not \cite{Shelhamer17}

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

######
## Physics states and property learning:
* Schema Networks: Zero-shot Transfer with a Generative Causal Model of Intuitive Physics, Kansky Tom Silver David A. Mely Mohamed Eldawy Miguel Lazaro-Gredilla Xinghua Lou, Nimrod Dorfman Szymon Sidor Scott Phoenix Dileep George.   Schema Networks are closely related to Object-Oriented MDPs (OO-MDPs) (Diuk et al., 2008) and Relational MDPs (R-MDPs) (Guestrin et al., 2003a).



## Policy learning- based approaches

* Black-Box Data-efficient Policy Search for Robotics. 2017. Konstantinos Chatzilygeroudis, Roberto Rama, Rituraj Kaushik, Dorian Goepp, Vassilis Vassiliades and Jean-Baptiste Mouret. Gaussian process regression for policy optimisation using model based policy search. The Black-DROPS algorithm learns a high-dimensional policy from scratch in 5 trials, which are enough to learn the whole dynamics of the arm from scratch. https://arxiv.org/abs/1703.07261  @IROS2017 #resibots


* REINFORCEMENT LEARNING WITH UNSUPERVISED AUXILIARY TASKS (UNREAL). Max Jaderberg et al. 2016. UNREAL algorithm shows that augmenting a deep reinforcement learning agent with auxiliary control and reward prediction tasks can double improvement both in data efficiency and robustness to hyperparameter settings.  A successor in learning speed and the robustness to A3C (Over 87% of human scores).




########
## Validation Methods

Example ways of assessing CNN's performance is measuring networks receptive field [Cite online Tool], or attention maps. Other methods are described below.

* On the Quantitative Evaluation of Deep Generative Models Russ Salakhutdinov  www.cs.cmu.edu/~rsalakhu/talk_Eval.pdf

* A new embedding quality assessment method for manifold learning, Zhang11. https://arxiv.org/pdf/1108.1636v1.pdf we have their matlab code
Interesting paper about evaluating embeddings through Normalization independent embedding quality (NIEAQA) ssessment, a normalization independent embedding quality criterion, for manifold learning purposes, based on the anisotropic scaling independent measure (ASIM), which compares the similarity between two configurations under motion and anisotropic coordinate scaling. NIEQA is based on ASIM, and consists of three assessments, a local one, a global one and a linear combination of the two. The local measure evaluates how well local neighborhood information is preserved under anisotropic coordinate scaling and rigid motion. NIEQA is valued between 0 and 1, where 0 represents a perfect preservation and its highlights is is the ability of being applicable to both normalized and isometric embeddings, it can provide both local and global assessments, and it can serve as a natural evaluation tool of learned embeddings \cite{Gracia14}.

* A methodology to compare Dimensionality Reduction algorithms in terms of loss of quality. Antonio Gracia, Santiago González, Víctor Robles, Ernestina Menasalvas, 2014

* Foolbox v0.8.0: A Python toolbox to benchmark the robustness of machine learning models. Rauber17 https://arxiv.org/pdf/1707.04131.pdf




#####
## METRICS
Particular metrics of interest to assess quality of prediction go beyond the \textit{blurry} MSE (Mean Squared Error) loss function. Complementary feature learning strategies include multi-scale architectures, adversarial training methods, and image gradient difference loss functions as proposed in \cite{Mathieu15}. More concretely,  the Peak Signal to Noise Ratio, Structural Similarity Index Measure and image sharpness show to be better proxies for next frame prediction assessment \cite{Mathieu15}.


#######
## SURVEYS
* Representation Learning: A Review and New Perspectives** <br> Yoshua Bengio, Aaron Courville, and Pascal Vincent, (2012), pdf

* Survey paper on Geometry of Optimization & Implicit Regularization in Deep Learning with Neyshabur, Tomioka, Srebro  https://arxiv.org/abs/1705.03071

* A survey on metric learning for feature vectors and structured data. Aurélien Bellet, Amaury Habrard, and Marc Sebban.  2013

* Deep learning in neural networks: An overview. Schmidhuber, Jürgen Neural Networks - 2015

* Not peer reviewed? https://arxiv.org/pdf/1701.07274.pdf

* Neuroscience-inspired AI: http://www.cell.com/neuron/pdf/S0896-6273(17)30509-3.pdf

* Survey: Flexible decision-making in recurrent neural networks trained with a biologically plausible rule [Miconi16] www.biorxiv.org/content/early/2016/07/26/057729

* X. Zhu, “Semi-Supervised Learning Literature Survey,” Technical Report 1530, Univ. of Wisconsin-Madison, 2006.




## Prediction-based learning/ forward models:  http://realai.org/predictive-learning/

Basic idea: The loss is based on prediction errors of next states.

* Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection. The Google dataset release paper proposes a continuous servoing mechanism that uses \textit{the grasp prediction network to choose the motor commands for the robot that will maximize the probability of a success grasp}.  https://arxiv.org/pdf/1603.02199.pdf

* Learning State Representation for Deep Actor-Critic Control. Jelle Munk 2016. On predictive priors:  Ils utilisent simplement le fait que l'état doit permettre de prédire efficacement le prochain état et la récompense. Ca pourrait être assez simple a tester dans notre cas j'imagine. See summary in fwd models.

* Learning a forward/inverse model to learn good representations : https://arxiv.org/pdf/1612.07307.pdf

* MatchNet and TempoNet: CortexNet: a robust predictive deep neural network trained on videos https://engineering.purdue.edu/elab/CortexNet/

* Deep multi-scale video prediction beyond mean square error, Mathieu15. Video Prediction can be done with more robust measures than MSE. In \cite{Mathieu15}, they propose several strategies for next frame prediction evaluation assessing the quality of the prediction in terms of Peak Signal to Noise Ratio, Structural Similarity Index Measure and image sharpness.  
IDEA: can be extended to be combined with optical flow prediction  and replace optical flow prediction algorithms with next frame prediciotion.

*  Value Prediction Networks (VPN) \cite{Oh17} (see summary in forward models)

########################
## Interpretability  methods for evaluating learned representations
* Understanding intermediate layers using linear classifier probes. Alain and Bengio 16  https://arxiv.org/pdf/1610.01644.pdf

* Explaining the Unexplained: A CLass-Enhanced Attentive Response (CLEAR)
Approach to Understanding Deep Neural Networks, Kumar et al 17  https://arxiv.org/pdf/1704.04133.pdf

* Foolbox v0.8.0: A Python toolbox to benchmark the robustness of machine learning models. Rauber17 is a library that tests for adversarial attackes, different evaluation metrics and machine learning models.



########################
## Auxiliary tasks for improving learning: http://realai.org/auxiliary-tasks/

* Reinforcement Learning with Unsupervised Auxiliary Tasks 2016  https://arxiv.org/abs/1611.05397
* Continuous control with deep reinforcement learning, 2015.  https://arxiv.org/abs/1509.02971
* The Intentional Unintentional Agent: Learning to Solve Many Continuous Control Tasks Simultaneously, 2017  https://arxiv.org/pdf/1707.03300.pdf





#################
## Few-shot learning

* Few-Shot Learning Through an Information Retrieval Lens. Triantafillou, 2017
* Optimization as a model for few-shot learning. Ravi and Larochelle, 17
* One-shot imitation learning, Duan17.
* Prototypical networks for few-shot learning. [web page] Snell17


## Category labels found that expand across papers (to decide if they are columns to add to our paper table?)
Prediction-based is same  or a subset of forward models?

reward-less
Using priors
0ne/few-shot
By demonstration, cloning, example
self-supervision

## Object disentanglement

* Understanding Visual Concepts with Continuation Learning https://arxiv.org/abs/1602.06822
* Early Visual Concept Learning with Unsupervised Deep Learning https://arxiv.org/abs/1606.05579
* Discovering objects and their relations from entangled scene representations https://arxiv.org/abs/1702.05068
* BetaVAE-Learning basic visual Concepts with a constrained variational framework https://openreview.net/pdf?id=Sy2fzU9gl
* Pixel Objectness, Jain17. https://arxiv.org/abs/1701.05349

##################
## non classified/other interesting papers / visual representation learning:

* Extending LSTMs with Neural Hawkes Process for event prediction based on timestamps, as opposed to using only the sequentiality of events: https://www.cs.colorado.edu/~mozer/Research/Selected%20Publications/talks/Mozer_NeuralHawkesProcessMemory_NIPS2016.pdf


#####################
### Unsupervised Imitation learning: https://sermanet.github.io/imitation/

*Unsupervised Perceptual Rewards for Imitation Learning was presented at RSS 2017 by Kelvin Xu

*Time-Contrastive Networks: Self-Supervised Learning from Multi-View Observation \cite{Sermanet17Time}

* Wang, X., Gupta, A.: Unsupervised learning of visual representations using videos.
ICCV (2015)

* CVPR17 http://juxi.net/workshop/deep-learning-robotic-vision-cvpr-2017/%20%20

* Unsupervised learning.: DEEP UNSUPERVISED LEARNING THROUGH SPATIAL
CONTRASTING Hoffer16 https://arxiv.org/pdf/1610.00243.pdf  Equivalent to DistanceRatioCriterion implemented in Torch for triple based comparison loss function.

#####################
##  One/Few-shot approaches

* Siamese Neural Networks for One-shot Image Recognition, Gregory Koch Richard Zemel Ruslan Salakhutdinov

* Optimization as a model for few-shot learning. Ravi and Larochelle, 17


(REPEATED, USE TAGS?): Black-Box Data-efficient Policy Search for Robotics. Konstantinos Chatzilygeroudis, Roberto Rama, Rituraj Kaushik, Dorian Goepp, Vassilis Vassiliades and Jean-Baptiste Mouret*
They unsupervisedly learn perform an action with only 5 episodes and gaussian processes with a robot (Ergo Jr?).



## No reward learning
Once a loss on reward is defined, in end-to-end RL systems, the representation is delegated to backpropagation without further attention to other supervisory signals. Representation learning can thus be considered a bottleneck in current approaches bound by reward \cite{Shelhamer17}.  Next we describe some approaches in this line:

* Curiosity-driven Exploration by Self-supervised Prediction. Deepak Pathak et al. \cite{Pathak17} http://juxi.net/workshop/deep-learning-robotic-vision-cvpr-2017/papers/23.pdf
Self-supervised approach.

* 


### CVPR 17 Best papers
Learning from Simulated and Unsupervised Images through Adversarial Training, Ashish Shrivastava



#####
## Miscelanea

* Neural Hawkes Process Memory: https://www.cs.colorado.edu/~mozer/Research/Selected%20Publications/talks/Mozer_NeuralHawkesProcessMemory_NIPS2016.pdf   The	neural	Hawkes	process	memory	belongs	to	two	new	classes	of	neural	net	models	that	are	emerging.
* Models	that	perform	dynamic	parameter	inference	as	a	sequence	is
processed	(vs.	stochastic	gradient	based	adaptation)
see	also	Fast	Weights paper	by	Ba,	Hinton,	Mnih,	Leibo,	&	Ionescu (2016),	Tau
Net paper	by	Nguyen	&	Cottrell	(1997)
* Models	that	operate	in	a	continuous	time	environment
see	also	Phased	LSTM	paper	by	Neil,	Pfeiffer,	Liu	(2016)


* Discrete-Event Continuous-Time Recurrent Nets Mozer17 Looks at different scales on timed events for prediction-based learning.


## RL avoiding continuous rewards: reducing the dimensionality of the action space by working with either binary inputs or defining some macro actions (an alternative to have continuous rewards)
*  Nicolas Heess, David Silver, and Yee Whye Teh. Actor-critic reinforcement learning with energy-based policies. In EWRL, pages 43–58. Citeseer, 2012.
* Yaakov Engel, Peter Szabo, and Dmitry Volkinshtein. Learning to control an octopus arm with gaussian process temporal difference methods. In Advances in neural information processing systems, pages 347– 354, 2005.



####
## Neuroscience inspired AI:

* Spacing effects in learning: a temporal ridgeline of optimal retention. Cepeda08



### FUTURE WORK IDEAS

* From Computational Neuroscience lab of Uni. of Tartu http://neuro.cs.ut.ee/lab/
Theory of mind is the ability to assign distinct mental states (beliefs, intents, knowledge,…) to other members. In this project, we aim to teach agents via reinforcement learning to solve a perspective-taking task that requires the agent to consider the perceptual state of another.

Partial information decomposition: estimating who knows what in complex sytems:
Mutual information quantifies the amount of information shared by two random variables. Such measure has been extensively applied to quantify information flowing in natural and man-made communication channels. However, it has been argued that an information theoretic description of computation (as opposed to simply communication) requires to quantify the distribution of information that a set of input variables has about an output variable. In particular, such information could be provided individually (unique), redundantly (shared), or exclusively jointly (synergetic). The partial information decomposition (PID) is an axiomatic framework to define such information distributions. In this project together with the group of Dirk Oliver Theis, we are developing a numerical estimator of PID and applying it to understand how information is distributed across parts of different complex systems.    

IDEA: could we use PID ideas to drive intrinsic motivation-based RL with multiagents?
 .
