
# Forward/ Predictive Models

Predicting context is critical for tackling unsupervised learning and being able to generate content as indicated by Richard Zemel \cite{}

* Loss is its own Reward: self-supervision for RL:
Self-supervised pre-training and joint optimization improve the data efficiency and policy returns of end-to-end reinforcement learning. This paper shows the critical role of representation learning and confirms that re-training a decapitated agent, having destroyed the policy and value outputs while preserving the rest of the representation, is far faster than the initial training.

Self-supervision defines losses via surrogate annotations synthesized from unlabeled data: "rewards capture the task while selfsupervision captures the environment". Since rewards might be delayed and sparse, the losses from self-supervision are instantaneous and ubiquitous: Augmenting RL with these auxiliary losses enriches
the representation through multi-task learning and improves policy optimization \cite{Shelhamer17}.

[Policy gradient methods iteratively optimize the policy return by estimating the gradient of the expected return with respect to the policy parameters where the expectation is sampled by executing the policy in the environment. To improve optimization, in an actor-critic method the policy gradient can be scaled not by the return itself but by an estimate of the advantage (Sutton & Barto, 1998).] The authors augment the policy gradient with auxiliary gradients from self-supervised tasks.

 supervised learning min E[Ldis(f(x); y)]
 unsupervised learning min E[Lgen(f(x); x)]
 self-supervised learning min E[Ldis(f(x); s(x))]   with surrogate annotation function s()

The actor-critic architecture is based on A3C (Mnih et al.,2015) but with capacity reduced for experimental efficiency. The self-supervised architectures share the same encoder as the actor-critic for transferability. Each self-supervised task augments the architecture with its own decoder and loss.

Adopting self-supervision for RL raises issues of multitask optimization and statistical dependence. RL setting
the distribution of transitions is neither i.i.d. nor stationary, so self-supervision should follow the policy distribution.

DOUBT?: Ie, same network? same transition data means same transition probability distribution?/how to assess it?

Reward can be cast into a proxy task as instantaneous prediction by regression or binning into positive, zero, and negative classes. Our selfsupervised reward task is to bin rt into r't in {0;+; -} with equal balancing of the classes as done independently by Jaderberg et al. (2016). WHAT DOES EQUAL BALANCING MEANS, not inbalance of dataset? how to correct for this?

Surrogate annotations for these tasks capture state, action, and successor (s; a; s0) relationships from transitions. A single transition suffices to define losses on dynamics (successors) and inverse dynamics (actions).

KEY POINT OF THE PAPER: Dynamics can be cast into a verification task by recognizing
whether state-successor (s; s0) pairs are drawn from the
environment or not. This can be made action conditional by
extending the data to (s; a; s0) and solving the same classification
task. Our self-supervised dynamics verification
task is to identify the corrupted observation otc in a history
from t0 to tk, where otc is corrupted by swapping it
with ot0 for t0 =2 ft0; : : : ; tkg. We synthesize negatives
by transplanting successors from other, nearby time steps.
While the transition function is not necessarily one-to-one,
and the synthetic negatives are noisy, in expectation these
surrogate annotations will match the transition statistics.

Inverse dynamics mapsSxS-> A is reduced to a classification problem for discrete actions or to regression if continuous ones.  Our self-supervised inverse dynamics task
is to infer the intervening actions of a history of observations.
When jAj << jSj, as is often the case, the selfsupervision
of inverse dynamics may be more statistically
and computationally tractable.

While a popular line of attack for unsupervised learning, the representations learned by reconstruction are relatively
poor for transfer (Donahue et al., 2016. They use it for comparison with their self-supervised auxiliary losses approach. 




* Value Prediction Networks (VPN) \cite{Oh17} integrate model-free and model-based RL methods into a single neural network. In contrast to typical model-based RL methods, VPN learns a dynamics model whose abstract states are trained to make option-conditional predictions of future values (discounted sum of rewards) rather than of future observations. VPN has several advantages over both model-free and model-based baselines in a stochastic environment where careful planning is required but building an accurate observation-prediction model is difficult. Because they outperform Deep Q-Network (DQN) on several Atari games with short-lookahead planning Atari games, can be a potential new way of learning state representations.

* Deepmind Imagination-augmented agents.

The agents DeepMind introduce benefit from an ‘imagination encoder’- a neural network which learns to extract any information useful for the agent’s future decisions, but ignore what is not relevant. From their blog, the imagination-augmented agents outperform the imagination-less baselines considerably: they learn with less experience and are able to deal with the imperfections in modelling the environment. Because agents are able to extract more knowledge from internal simulations they can solve tasks more with fewer imagination steps than conventional search methods, like the Monte Carlo tree search.
When we add an additional ‘manager’ component, which helps to construct a plan, the agent learns to solve tasks even more efficiently with fewer steps. In the spaceship task it can distinguish between situations where the gravitational pull of its environment is strong or weak, meaning different numbers of these imagination steps are required. When an agent is presented with multiple models of an environment, each varying in quality and cost-benefit, it learns to make a meaningful trade-off. Finally, if the computational cost of imagination increases with each action taken, the agent imagines the effect of multiple chained actions early, and relies on this plan later without invoking imagination again.


* Predictive priors paper: * Learning State Representation for Deep Actor-Critic Control. Jelle Munk 2016.
State representation learning is a form of unsupervised
learning, i.e., there are no training examples available since
it is not known a priori what the most suitable state representation
is to solve the problem. Learning an observation-to-state
mapping therefore involves either making assumptions
about the structure of the state representation or learning the
mapping as part of learning some other function.

For each of these priors, a loss function is defined. An observation-to-
state mapping is subsequently trained to minimize the
combined loss functions of the individual priors. The paper
then shows a performance increase when using the learned
state representation instead of the raw observations as input
to the Neural Fitted Q-iteration algorithm [1].

ML-DDPG architecture consist of three DNNs, a
model network, a critic network and an actor network. The
model network is trained by using a new concept that we call
predictive priors and is integrated with the actor and critic
networks by copying some of its weights

predictable transition prior which states
that, given a certain state st and an action at taken in that
state, one can predict1 the next state ^st+1. An important
difference with other methods like [22], [10], is that we do
not predict the next observation ^ot+1 but the next state ^st+1.
This becomes important if the observation ot contains task irrelevant
information. A state that needs to be able to predict
the next observation still has to contain this task-irrelevant
information to make the prediction, whereas in the proposed
case this information can be ignored altogether. The second
prior is the predictable reward prior which states that, given
a certain state st and an action at taken in that state, one
can predict the next reward ^rt+1. This prior enforces that
all information relevant to the task is available in the state,
which helps the predictable transition prior to converge to
a meaningful representation for the given task.

The point of using the predictive priors is to find a state representation
from which it is easier, i.e., fewer transformations
are necessary, to actually learn the transition and reward
function
The advantage of using the predictive priors is that state
representation learning is transformed from an unsupervised
learning problem to a supervised learning problem. Another advantage of this approach is that the state representation that is learned is goal directed. Observations that do
not correlate with the reward or are inherently unpredictable
will not be encoded in the state representation. This is in
contrast to methods like an auto-encoder or SFA since these
methods do not differentiate between observations that are
useful for solving a particular task and observations that are
not.
predictive priors are implemented by a model network
that learns to predict the next state and reward f^st+1; ^rt+1g
from the observation-action tuple fot; atg
Model network
The predictive priors are implemented by a model network
that learns to predict the next state and reward f^st+1; ^rt+1g
from the observation-action tuple fot; atg

data is collected by following a random policy based on the Ohrnstein-Uhlenbeck
process [25]


PROBLEM: One specific problem we encountered, with both the
DDPG and the ML-DDPG, was the fact that the actor
sometimes learned actions that lay outside the saturation
limits of the actuator. This is caused in part because all
the samples from which the networks learn are collected
prior to learning.   WHAT IS THE SATURATION LIMITS OF THE ACTUATOR?

How to tackle: to obtain the target st+1, the current approximation of the observation-to-state
mapping is used, to map the next observation ot+1 to the next state st+1. This could potentially lead to convergence problems, since the target depends on the current approximation. In practice, however, these problems did not occur.

-Loss function of the critique?
Can we clone only the first layer? See how to freeze first layer only (Mat)
-How to choose learning rate of actor and critique? Alpha_c and alpha_a
-The actions are scaled such that they have zero mean and a standard deviation of 1. We can scale our actions by sampling the actions : You take many pairs of images at time t and t+1, compute the actions (difference between 2 states) But, since we don't have a very big database, you can compute all actions available. This is how it is done, needs to be done in order to compute mean and std, which we use to normalize the actions right now.
-In all experiments, the saturation penalty described in Section III, was a necessary condition for the algorithms to converge.

KEY: 
Loss function of the actor correspond to the Q-function, with a saturation penalty added to the loss function to restrict the action space. In our case It has no saturation penalty, The q-function is a network, defined in baxter_school/rl/learningAlg.py  dqn in defined l248 and learning the Q-function  line 282
-Learning State Representation for Deep Actor-Critic Control
-experiments are repeated for different sizes of the ERD, to
compare how the algorithms perform when data is either
scarce or abundant
-In all experiments, the saturation penalty described in Section IIID,
was a necessary condition for the algorithms to converge.
-reward from the environment is based on the Euclidian
distance D between the reference position and the current
position of the tip of the second link and on the angular
velocities _ of the two links


lambda_a represents the trade-off between maximizing the reward and minimizing the saturation penalty.
-ERD of 30K samples.
-Both algorithms perform better, in terms of the final performance, if more
data is available. The advantage of the ML-DDPG over the DDPG seems relatively constant and does not degrade when data becomes abundant as was expected.
-the system is partially observable and that the reference is given in Cartesian coordinates whereas the position of the 2 links are given in joint angles.
-We also believe the performance can be further improved by tuning the reward function and/or the architecture of the DNNs which has not been done extensively to get these results
-reward from the environment is based on the Euclidean distance D between the food and the octopus segment
-Both algorithms have a similar rise and settling time and are able to learn the task in about
2000 learning steps. Both successfully learn to reach for the food, which takes them around 1:5s from their starting position. 
-??? In order to see if the learned policy also generalized to other initial positions, the octopus arm was randomly excited for 2s before testing the learned policy again. Also, in these cases, the octopus was successful in reaching the food. Perhaps in spite of the high dimensionality of the problem, the Octopus problem is relatively easy since it does not require a very precise control action, like in the case of the 2-link arm
-Whenever the goal is reached an extra bonus B is given. The reward function is given by
r(D,B) = (B -2) - D where B = 2 whenever the goal is reached and B = 0 otherwise. Shall we add distance to our reward function to make it continuous?
-big advantages of using the predictive priors is that it
does not require the agent to follow a specific policy and/or
exploration strategy. Hence the agent can learn from any
previously created ERD. Furthermore, the DNN is trained
using supervised learning as opposed to the approach used
in [7], where unsupervised learning was used.
-using DNNs in actorcritic
algorithms, is a very promising field of research,
especially for cases in which the state and action dimensions
of the problem are very high.
-More work is necessary to
visualise what kind of state representation the ML-DDPG is
actually learning and how it performs on other benchmarks
-The reward from the environment is based on the Euclidean
distance D between the food and the segment (efector)  in the Octopus. In 2-link benchmark, the reward from the environment is based on the Euclidian
distance D between the reference position and the current position of the tip of the second link and on the angular velocities of the two links. The reward function is given
By r(D, theta) = -(D + w*|theta|_2)
where w represent the trade-off between the two terms.

------





* DEEP MULTI-SCALE VIDEO PREDICTION BEYOND MEAN SQUARE ERROR. Mathieu15
Complementary feature learning strategies beyond MSE include multi-scale architectures, adversarial training methods, and image gradient difference loss functions as proposed in \cite{Mathieu15}. More concretely,  the Peak Signal to Noise Ratio, Structural Similarity Index Measure and image sharpness show to be better proxies for next frame prediction assessment \cite{Mathieu15}.  Architectures and losses may be used as building blocks for more sophisticated prediction models, involving memory and recurrence. Because unlike most optical flow algorithms, the model is fully differentiable, it can be fine-tuned for task transfer learning.


### Sequence-based Learning Predictive Models
IDEA: Can we translate these principles of unaligned (xi,yi) sample pairs to have all data non distributed into recording sequences, and use the loss functions as extra priors? 

*  Unsupervised Sequence Classification using  Sequential Output Statistics. Chen16
They propose solving an unsupervised learning problem of having unpaired xi, yi  samples by learning to predict without costly pairing of input data and corresonding labels. They assume that the probability distribution p(y1, . . . , yT ) of the output samples has a sequence structure, i.e., there is temporal dependency over y1, . . . , yT .
Furthermore, they assume that p(y1, . . . , yT ) is known a priori, which could be estimated from a different data source that has the same distribution of p(y1, . . . , yT).
Their objective is to learn the posterior probability p(yt|xt, Wd) (i.e., the predictor) from the input sequence {xt} by exploiting the distribution p(y1, . . . , yT ) on the output sequence, where p(y1, . . . , yT ) is learned from another totally unpaired sequence {y1, . . . , yT }.

* Unsupervised Sequence Classification using Sequential Output Statistics. Liu17. 
Proposes an unsupervised learning cost function based on sequential output statistics that is harder to optimize but drastically reduces to half the erros in fully supervised learning. It avoids the need for a strong generative model and proposes a stochastic primal-dual gradient method to solve the optimization problem .

* Andrew M Dai and Quoc V Le. Semi-supervised sequence learning. In Proceedings of the Advances in Neural Information Processing Systems (NIPS), pages 3079–3087, 2015.








# Auxiliary tasks

For the auxiliary task litterature, exploring the idea that self-generating multiple tasks can allow to learn efficiently single complex tasks, it is indeed a very interesting strand of work. However, the general idea is not new (what is new is to
do it with DL algorithm), and for example in the team we have studied many dimensions which are not yet integrated into these DL architectures (and so we have opportunities to make original contributions in DL by leveraging them).

For example, instead of self-generating/sampling random goals/reward functions (like in Intentional Unintentional Agent), one can do this actively using bandits that maximize learning progress over goals. We are right now finishing a paper with Sébastien and Yoan to summarize some of these ideas in a formalism that is quite close to these DL papers, we will soon be able to circulate it.

Classical papers of the team on them (outside DL yet) are:

* Active Learning of Inverse Models with Intrinsically Motivated Goal Exploration in Robots. Baranes, A., Oudeyer, P-Y. (2013) Robotics and Autonomous Systems, 61(1), pp. 49-73. http://www.pyoudeyer.com/ActiveGoalExploration-RAS-2013.pdf

* Intrinsically Motivated Learning of Real-World Sensorimotor Skills with Developmental Constraints
Oudeyer P-Y., Baranes A., Kaplan F. (2013)
in Intrinsically Motivated Learning in Natural and Artificial Systems, eds. Baldassarre G. and Mirolli M., Springer
http://www.pyoudeyer.com/OudeyerBaranesKaplan13.pdf
















## More intrinsic motivation

* Curiosity-driven Exploration by Self-supervised Prediction 2017.









## Extending priors and loss functions

* Few-Shot Learning Through an Information Retrieval Lens. Triantafillou17
Presents an information-retrieval based training objective that simultaneously optimizes all relative orderings of the points in each training batch. How to best exploit the information within each batch, and how to create training batches in order to best leverage the information in the training set, however, remains an open question.

mAP-DLM (max. average precision- Direct Loss Minimization) and mAP-SSVM are presented, and perform similarly. mAP-DLM, minimizes the direct loss directly while mAP-SSVM minimizes an upper bound of it. Comparing with a siamese network (where training batches are created in a way that enforces
that they have the same amount of information available for each update: each training batch B
is formed by sampling N classes uniformly at random and |B| examples from these classes. The
siamese network is then trained on all possible pairs from these sampled points), the mAP alues reached are similar, but the convergence is faster. Training objective: Each batch point is treated as a query and all (dicrete) ranks are computed. All rankins are opitmized simultaneously. 
While siamese training would consider all pairs in a query to perform similarity prediction (predict if the class of each pair of samples is the same or different), in their proposed mAP training, they consider all queries and their rankings,and move the points in the latent space to positions that simultaneously maximize the average precision (AP) of all rankings. 

Some details: Larger batch size implies larger ‘shot’. For example, for N = 8, |B| = 64 results to on average 8 examples of each class in each batch (‘8-shot’) whereas |B| = 16 results to on average 2-shot. When the ‘shot’ is smaller, there is a clear advantage in using their method over the all-pairs siamese.

They harness the power of neural networks for metric learning. These methods vary in terms of loss functions but have in common a mechanism for the parallel and identically-parameterized embedding of the points that will inform the loss function.
Siamese and triplet networks are commonly-used variants of this family that operate on pairs and triplets, respectively. Example applications include signature verification [8] and face verification [9, 10]. NCA and LMNN have also been extended to their deep variants [11] and [12], respectively.
These methods often employ hard-negative mining strategies for selecting informative constraints for training [10, 13]. A drawback of siamese and triplet networks is that they are local, in the sense that their loss function concerns pairs or triplets of training examples, guiding the learning process to optimize the desired relative positions of only two or three examples at a time. The myopia of these local methods introduces drawbacks that are reflected in their embedding spaces. [14] propose
a method to address this by using higher-order information. 
Relationship between DLM and SSVM: both yield a loss-informed weight update rule. The gradient computation differs from that of the direct loss minimization approach only in that, while SSVM considers the score of the ground-truth F(X; yGT;w), direct loss minimization considers the score of the current prediction F(X; yw;w).

Let f(x;w) be the embedding function, parameterized by a neural network and phi(x1; x2;w) the cosine similarity of points x1 and x2 in the embedding space given by w. phi(x1; x2;w) is typically referred in the literature as the score of a siamese network.


