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

The assessment objectif is to give a quantitative value which estimate the quality of a representation. In the context of representation learning this can be harder than expected. The assessment should show if the representaiotn we learned is conform to what we expect.<br>
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

* Learning State Representation for Deep Actor-Critic Control. Jelle Munk 2016. On predictive priors:  Ils utilisent simplement le fait que l'état doit permettre de prédire efficacement le prochain état et la récompense. Ca pourrait être assez simple a tester dans notre cas j'imagine.

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




#######################
## GANS and generative models for data augmentation
Latent spaces of GAN's generators captures semantic variations in the data distribution due to GANS' shown ability to learn generaltive models mapping simple latent distributions to arbitrarily comples ones \cite{Donahue17}. Some state of the art GANS useful in unsupervised learning are described below.

* BEGAN: Boundary Equilibrium Generative Adversarial Networks, David Berthelot et al.
 In contrast to traditional GANS that require alternating training D and G, or pretraining D, \textit{BEGAN requires neither to train stably}. The discriminator has two competing goals: auto-encode real images and discriminate  real from generated images (balanced by gamma). They propose a new equilibrium enforcing method paired with a loss derived from the Wasserstein distance for training auto-encoder based GAN that balances the generator and discriminator during training. It provides a new approximate convergence measure, fast and stable training which controls the trade-off between image diversity and visual quality.

* (BiGAN): Adversarial Feature Learning, Donahue17. Presents an extension of regular GANS to learn the inverse mapping: projecting data back into the latent space that allows the learned feature representation to be useful for auxiliary supervised discrimination tasks that is competitive with unsupervised and self-supervised feature learning.

* GAN to improve learning machine robustness in image segmentation, pose estimation and speech recognition using Houdini loss.  Houdini loss is a product of two terms. The first term is a stochastic margin, that is the probability that the difference between the score of the actual target g(x; y) and that of the predicted target g(x; ^y) is smaller than  N(0; 1). It reflects the confidence of the model in its predictions. Houdini is a lower bound of the task loss and considers the difference between the scores assigned by the network to the ground truth and the prediction, and it converges to the task loss.: Houdini: Fooling Deep Structured Prediction Models, Cisse17.

* Unsupervised Pixel–Level Domain Adaptation with Generative Adversarial Networks, Bousmalis 16  presents an unsupervised approach using a (GAN)–based architecture that is able to learn such a transformation in an unsupervised manner, i.e. without using corresponding pairs from the two domains. It is the best state of the art approach on unsupervised learning for domain adaptation, improving over: Decoupling from the Task-Specific Architecture, Generalization Across Label Spaces, achieve Training Stability and Data Augmentation.

* Soumith Chintala and Yann LeCun. A path to unsupervised learning through adversarial networks. In https://code.facebook.com/posts/1587249151575490/a-path-to-unsupervisedlearning-through-adversarial-networks/, 2016.

* State-of-the-art GANs for unsupervised representation learning: BEGAN,  BiGAN, CycleGAN and pixel based GAN. See Allan Ma survey to appear soon.

* Learning to generate images with perceptual similarity metrics. ICIP 2017. 

* GLO: "Optimizing the Latent Space of Generative Networks" Piotr Bojanowskii, Armand Joulin, David Lopez-Paz, and Arthur Szlam. Summary by LeCunn:
Short story: GLO model (Generative Latent Optimization) is a generative model in which a set of latent variables is optimized at training time to minimize a distance between a training sample and a reconstruction of it produced by the generator. This alleviates the need to train a discriminator as in GAN.
Slightly less short story: GLO, like GAN and VAE, is a way to train a generative model under uncertainty on the output.
A generative model must be able to generate a whole series of different outputs, for example, different faces, or different bedroom images.
Generally, a set of latent variables Z is drawn at random every time the model needs to generate an output. These latent variables are fed to a generator G that produces an output Y(e.g. an image) Y=G(Z).
Different drawings of the latent variable result in different images being produced, and the latent variable can be seen as parameterizing the set of outputs.
In GAN, the latent variable Z is drawn at random during training, and a discriminator is trained to tell if the generated output looks like it's been drawn from the same distribution as the training set.
In GLO, the latent variable Z is optimized during training so as to minimize some distance measure between the generated sample and the training sample Z* = min_z = Distance(Y,G(Z)). The parameters of the generator are adjusted after this minimization. The learning process is really a joint optimization of the distance with respect to Z and to the parameters of G, averaged on a training set of samples.
After training, Z can be sampled from their allowed set to produce new samples. Nice examples are shown in the paper.
GLO belongs to a wide category of energy-based latent variable models: define a parameterized energy function E(Y,Z), define a "free energy" F(Y) = min_z E(Y,Z). Then find the parameters that minimize F(Y) averaged over your training set, making sure to put some constraints on Z so that F(Y) doesn't become uniformly flat (and takes high values outside of the region of high data density). This basic model is at the basis of sparse modeling, sparse auto-encoders, and the "predictive sparse decomposition" model. In these models, the energy contains a term that forces Z to be sparse, and the reconstruction of Y from Z is linear. In GLO, the reconstruction is computed by a deconvolutional net.



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

* Curiosity-driven Exploration by Self-supervised Prediction. Deepak Pathak et al. \cite{Pathak17} http://juxi.net/workshop/deep-learning-robotic-vision-cvpr-2017/papers/23.pdf
Self-supervised approach. 


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