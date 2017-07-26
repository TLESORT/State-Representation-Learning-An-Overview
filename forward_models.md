## Extending priors and loss functions

* Few-Shot Learning Through an Information Retrieval Lens. Triantafillou17
Presents an information-retrieval based training objective that simultaneously optimizes all relative orderings of the points in each training batch. How to best exploit the information within each batch, and how to create training batches in order to best leverage the information in the training set, however, remains an open question.

mAP-DLM (max. average precision- Direct Loss Minimization) and mAP-SSVM are presented, and perform similarly. mAP-DLM, minimizes the direct loss directly while mAP-SSVM minimizes an upper bound of it. Comparing with a siamese network (where training batches are created in a way that enforces
that they have the same amount of information available for each update: each training batch B
is formed by sampling N classes uniformly at random and |B| examples from these classes. The
siamese network is then trained on all possible pairs from these sampled points), the mAP alues reached are similar, but the convergence is faster. Training objective: Each batch point is treated as a query and all (dicrete) ranks are computed. All rankins are opitmized simultaneously. 
While siamese training would consider all pairs in a query to perform similarity prediction (predict if the class of each pair of samples is the same or different), in their proposed mAP training, they consider all queries and their rankings,and move the points in the latent space to positions that simultaneously maximize the average precision (AP) of all rankings. 

Some details: Larger batch size implies larger ‘shot’. For example, for N = 8, |B| = 64 results to on average 8
examples of each class in each batch (‘8-shot’) whereas |B| = 16 results to on average 2-shot. When the ‘shot’ is smaller, there is a clear advantage in using their method over the all-pairs siamese.

They harness the power of neural networks for metric learning. These methods vary in terms of loss functions but have in common a mechanism for the parallel and identically-parameterized embedding of the points that will inform the loss function.
Siamese and triplet networks are commonly-used variants of this family that operate on pairs and triplets, respectively. Example applications include signature verification [8] and face verification [9, 10]. NCA and LMNN have also been extended to their deep variants [11] and [12], respectively.
These methods often employ hard-negative mining strategies for selecting informative constraints for training [10, 13]. A drawback of siamese and triplet networks is that they are local, in the sense that their loss function concerns pairs or triplets of training examples, guiding the learning process to optimize the desired relative positions of only two or three examples at a time. The myopia of these local methods introduces drawbacks that are reflected in their embedding spaces. [14] propose
a method to address this by using higher-order information. 
Relationship between DLM and SSVM: both yield a loss-informed weight update rule. The gradient computation differs from that of the direct loss minimization approach only in that, while SSVM considers the score of the ground-truth F(X; yGT;w), direct loss minimization considers the score of the current prediction F(X; yw;w).

Let f(x;w) be the embedding function, parameterized by a neural network and phi(x1; x2;w) the cosine similarity of points x1 and x2 in the embedding space given by w. phi(x1; x2;w) is typically referred in the literature as the score of a siamese network.




* Loss is its own Reward: self-supervision for RL:
Self-supervised pre-training and joint optimization improve the data efficiency and policy returns of end-to-end reinforcement learning. This paper shows the critical role of representation learning and confirms that re-training a decapitated agent, having destroyed the policy and value outputs while preserving the rest of the representation, is far faster than the initial training


* Deepmind Imagination-augmented agents.

The agents DeepMind introduce benefit from an ‘imagination encoder’- a neural network which learns to extract any information useful for the agent’s future decisions, but ignore what is not relevant. From their blog, the imagination-augmented agents outperform the imagination-less baselines considerably: they learn with less experience and are able to deal with the imperfections in modelling the environment. Because agents are able to extract more knowledge from internal simulations they can solve tasks more with fewer imagination steps than conventional search methods, like the Monte Carlo tree search.
When we add an additional ‘manager’ component, which helps to construct a plan, the agent learns to solve tasks even more efficiently with fewer steps. In the spaceship task it can distinguish between situations where the gravitational pull of its environment is strong or weak, meaning different numbers of these imagination steps are required. When an agent is presented with multiple models of an environment, each varying in quality and cost-benefit, it learns to make a meaningful trade-off. Finally, if the computational cost of imagination increases with each action taken, the agent imagines the effect of multiple chained actions early, and relies on this plan later without invoking imagination again.








# Auxiliary tasks

For the auxiliary task litterature, exploring the idea that self-generating multiple tasks can allow to learn efficiently single complex tasks, it is indeed a very interesting strand of work. However, the general idea is not new (what is new is to
do it with DL algorithm), and for example in the team we have studied many dimensions which are not yet integrated into these DL architectures (and so we have opportunities to make original contributions in DL by leveraging them).

For example, instead of self-generating/sampling random goals/reward functions (like in Intentional Unintentional Agent), one can do this actively using bandits that maximize learning progress over goals. We are right now finishing a paper with Sébastien and Yoan to summarize some of these ideas in a formalism that is quite close to these DL papers, we will soon be able to circulate it.

Classical papers of the team on them (outside DL yet) are:

* Active Learning of Inverse Models with Intrinsically Motivated Goal Exploration in Robots
Baranes, A., Oudeyer, P-Y. (2013)
Robotics and Autonomous Systems, 61(1), pp. 49-73.
http://www.pyoudeyer.com/ActiveGoalExploration-RAS-2013.pdf

* Intrinsically Motivated Learning of Real-World Sensorimotor Skills with Developmental Constraints
Oudeyer P-Y., Baranes A., Kaplan F. (2013)
in Intrinsically Motivated Learning in Natural and Artificial Systems, eds. Baldassarre G. and Mirolli M., Springer
http://www.pyoudeyer.com/OudeyerBaranesKaplan13.pdf

## More intrinsic motivation

* Curiosity-driven Exploration by Self-supervised Prediction 2017.


## Physics states and property learning:

* Schema Networks: Zero-shot Transfer with a Generative Causal Model of Intuitive Physics, Kansky Tom Silver David A. Mely Mohamed Eldawy Miguel Lazaro-Gredilla Xinghua Lou, Nimrod Dorfman Szymon Sidor Scott Phoenix Dileep George.
Schema Networks are closely related to Object-Oriented MDPs (OO-MDPs) (Diuk et al., 2008) and Relational MDPs (R-MDPs) (Guestrin et al., 2003a).
