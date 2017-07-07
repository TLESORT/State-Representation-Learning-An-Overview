

**************************************************

# Learning State Representation for Deep Actor-Critic Control

**************************************************

Use of the "predictive prior to learn a task related state representation as a pretraining for the **ML-DDPG** algorithm (Model Learning Deterministic Policy Gradient).

the prior aim to predict the next state and reward. It is supervised. :
$L_m = ||  s_{t+1} - \hat{s_{t+1}}  ||^2_2    +  \lambda_m ||  r_{t+1} - \hat{r_{t+1}}  ||^2_2 $


it allows to train RL faster afterward. ( "“End-to-end  training of deep visuomotor policies," claim it is not possible?)
