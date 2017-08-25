Models        | Reference     | Supervision  | :fire: Keywords  | Priors | Objective function | rewards
 ------------- | ------------- | -----------  | ------  | ------ | ------------------ | ------  
 VAE           | Kingma'14     | No           |         | yes | Variational free energy objective function | no
 InfoGAN       |Chen'16        | No           | scalable| yes | Mutual information | no
 DE-IGN        | Kulkarni'15   | Semi         |         | yes |  | no
 Beta VAE      | Higgins'17    | Semi         | stable  | No | Variational free energy objective function {Jordan99]} with beta =1  | no
 PVE           | Jonschkowsky'17| No           |    | yes |  | no




 metrics        | Reference     | Objective  | :fire:
 ------------- | ------------- | -----------  | ------
 Disentanglement metric score | Higgins'17 | Unsupervised representation learning |
 Inception score |  | Generative models (GAN)  |
 Distortion |  | Unsupervised Representation learning  |
 Frechet Inception Distance (FID) | Heusel 17  | GANs  | Improved successor of inception


Potential column :


## Category labels found that expand across papers (to decide if they are columns to add to our paper table?)
Prediction-based is same  or a subset of forward models?

reward-less
Using priors
0ne/few-shot
By demonstration, cloning, example
self-supervision
