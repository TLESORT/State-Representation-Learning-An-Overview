- constraint
- siamese networks
 See nice definition in:
**Siamese Neural Networks for One-shot Image Recognition**, *Gregory Koch Richard Zemel Ruslan Salakhutdinov*


- **The Curious Robot: Learning Visual Representations via Physical Interactions** <br> *Lerrel Pinto, Dhiraj Gandhi, Yuanfeng Han, Yong-Lae Park, Abhinav Gupta*,(2016), [pdf](https://arxiv.org/pdf/1604.01360.pdf) [bib](http://dblp.uni-trier.de/rec/bibtex/journals/corr/PintoGHPG16) [slides](https://pdfs.semanticscholar.org/a6ee/1a3d623daa2714f70232d4fa61cbd1b3cff3.pdf)

- **Slow Feature Analysis:Unsupervised Learning of Invariance**<br> *Laurenz Wiskott, Terrence J. Sejnowski* [pdf](https://papers.cnl.salk.edu/PDFs/Slow%20Feature%20Analysis_%20Unsupervised%20Learning%20of%20Invariances%202002-3430.pdf) [bib](http://dl.acm.org/citation.cfm?id=638941)
 - **Incremental Slow Feature Analysis** <br> *Varun Raj Kompella, Matthew Luciw, and Jurgen Schmidhuber* (2011) [pdf](https://www.ijcai.org/Proceedings/11/Papers/229.pdf) [ArXiv](https://arxiv.org/abs/1112.2113) [bib](http://dblp.uni-trier.de/rec/bibtex/journals/corr/abs-1112-2113)


## Relational and Symbolic Learning:
* Learning Visual Reasoning Without Strong Priors
Ethan Perez, Harm de Vries, Florian Strub, Vincent Dumoulin, Aaron Courville, 2017

* Relational Networks (Santoro’17)

* Towards Deep Symbolic Reinforcement Learning, Garnelo et al. NIPS 2016. Check if it uses priors? and tag it
Handles three main components of neural-symbolic hybrid systems: 1)Conceptual abstraction. 2) Compositional structure. 3) Common sense priors, i.e., one of the first works bridging the gap among logics and neural models.

* Visual Interaction Networks (Watters’17)

* Reasoning about Time and Knowledge Neural-Symbolic Learning Systems, d’Avila Garcez et al., NIPS 2004.

* A simple neural network module for relational reasoning, Santoro et al. 2017.
Proposes a reusable neural network module to reason about the relations between entities and their properties, where an MLP approximates object-to-object relation function and other MLP transforms summed pairwise object-to-object relations to some desired output (RN's operate on sets (due to summation in the formula) and thus are invariant to the order of objects in the input). -> Can we establish thus domain and range in relationships? future extension? In terms of architecture, RN module is used at the tail of a neural network taking input objects in form of CNN or LSTM embeddings. This work is evaluated on several tasks where it achieves reasonably good (even superhuman) performance (CLEVR and Sort-of-CLEVR - question answering about an image0

* Graph convolutional Auto-Encoders. Thomas Kipf‏ @thomaskipf Graph auto-encoders (in TensorFlow) is now available on GitHub: https://github.com/tkipf/gae



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

#####################
##  One/Few-shot approaches

* Siamese Neural Networks for One-shot Image Recognition, Gregory Koch Richard Zemel Ruslan Salakhutdinov

* Optimization as a model for few-shot learning. Ravi and Larochelle, 17


(REPEATED, USE TAGS?): Black-Box Data-efficient Policy Search for Robotics. Konstantinos Chatzilygeroudis, Roberto Rama, Rituraj Kaushik, Dorian Goepp, Vassilis Vassiliades and Jean-Baptiste Mouret*
They unsupervisedly learn perform an action with only 5 episodes and gaussian processes with a robot (Ergo Jr?).



#################
## Few-shot learning

* Few-Shot Learning Through an Information Retrieval Lens. Triantafillou, 2017
* Optimization as a model for few-shot learning. Ravi and Larochelle, 17
* One-shot imitation learning, Duan17.
* Prototypical networks for few-shot learning. [web page] Snell17

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

* Unsupervised Perceptual Rewards for Imitation Learning was presented at RSS 2017 by Kelvin Xu

* Time-Contrastive Networks: Self-Supervised Learning from Multi-View Observation \cite{Sermanet17Time}

* Wang, X., Gupta, A.: Unsupervised learning of visual representations using videos.
ICCV (2015)

* CVPR17 http://juxi.net/workshop/deep-learning-robotic-vision-cvpr-2017/%20%20

* Unsupervised learning.: DEEP UNSUPERVISED LEARNING THROUGH SPATIAL
CONTRASTING Hoffer16 https://arxiv.org/pdf/1610.00243.pdf  Equivalent to DistanceRatioCriterion implemented in Torch for triple based comparison loss function.



### CVPR 17 Best papers
Learning from Simulated and Unsupervised Images through Adversarial Training, Ashish Shrivastava


## RL avoiding continuous rewards: reducing the dimensionality of the action space by working with either binary inputs or defining some macro actions (an alternative to have continuous rewards)
*  Nicolas Heess, David Silver, and Yee Whye Teh. Actor-critic reinforcement learning with energy-based policies. In EWRL, pages 43–58. Citeseer, 2012.
* Yaakov Engel, Peter Szabo, and Dmitry Volkinshtein. Learning to control an octopus arm with gaussian process temporal difference methods. In Advances in neural information processing systems, pages 347– 354, 2005.


########################
## Auxiliary tasks for improving learning: http://realai.org/auxiliary-tasks/

* Reinforcement Learning with Unsupervised Auxiliary Tasks 2016  https://arxiv.org/abs/1611.05397
* Continuous control with deep reinforcement learning, 2015.  https://arxiv.org/abs/1509.02971
* The Intentional Unintentional Agent: Learning to Solve Many Continuous Control Tasks Simultaneously, 2017  https://arxiv.org/pdf/1707.03300.pdf
