Bible of key knowledge
Includes definitions and learnings:


Take home learnings: 
* In \cite{Yosinski14Understanding}\cite{JAIR17IReviewed} they study the transferability of features for the purpose of ne tuning. In that regard, authors nd that the distance between the source and target tasks is strongly related with the depth of the optimal layer to use for the transfer learning process.authors empirically evaluate several parameters that can aect . ->Daniela's idea of adapting learning process and rewards according to one of the 7 plots of movies and one of the x plots of tasks existing when learning.  Can we find a depth of network automatically or learning curriculum, i.e., syllabus for each of the tasks before teaching a robot to do it?

* Yosinksi14: How transferable are features in deep neural networks. 

* Carlos Mestre work: From continuous actions by demonstration, passing to discrete action instruction to a robot ("Push glass to right") and back to continuous actions of the robot to perform the task. First contribution in this field: Context-Based Generation of Continuous Actions to Reproduce Effects on Objects https://www.youtube.com/watch?v=Zhd-P3rouyc&feature=youtu.be


DEFINITIONS

* Instead of relying on an engineer to design a state estimator
to reconstruct the state vector from a set of observations,
ideally the machine should be able to learn this estimator as
well. Learning such an observation-to-state mapping, prior to
solving the RL problem, is known in the literature as state
representation learning \cite{Jonschowsky14}


* MDP: A stochastic process has the Markov property if the conditional probability distribution of future states of the process (conditional on both past and present states) depends only upon the present state, not on the sequence of events that preceded it. A process with this property is called a Markov process. 



* See paper for great RL definitions: Experience Replay DB and Soft for tacking unstable learning.  

A solution, proposed in [3], that reduces the coupling
between the target function and the actor and critic networks,
is to update the parameters of the target function using
“soft” updates. Instead of using  and  directly, a separate
set of weights Xi?minus and eta?minus are used, which slowly track
the parameters  and  of the actor and critic

like robotics, where the state and action spaces are continuous, function approximators have to be used to approximate both the action-value function Q and
the policy  [14]. Actor-critic algorithms are suitable in these situations since they allow both of these functions to be learned separately. This is in contrast with critic-only
methods, which require a complicated optimization at every time step to find the policy.
In actor-critic methods, the critic learns the action-value
function Q while the actor learns the policy . \cite{Munk16}
In order to
ensure that updates of the actor improve the expected discounted
return, the update should follow the policy gradient
[15]. The main idea behind actor-critic algorithms is that the
critic provides the actor with the policy gradient.

sequence R1..RN are normalized so that the minimum value of this sequence is -1.



# Recommended videos
GANS, Chintala: https://www.youtube.com/watch?v=QPkb5VcgXAM
