
#######################
## GANS and generative models for data augmentation [slides](https://www.slideshare.net/Artifacia/generative-adversarial-networks-and-their-applications)
Latent spaces of GAN's generators captures semantic variations in the data distribution due to GANS' shown ability to learn generaltive models mapping simple latent distributions to arbitrarily comples ones \cite{Donahue17}. Some state of the art GANS useful in unsupervised learning are described below.

* BEGAN: Boundary Equilibrium Generative Adversarial Networks, David Berthelot et al.
 In contrast to traditional GANS that require alternating training D and G, or pretraining D, \textit{BEGAN requires neither to train stably}. The discriminator has two competing goals: auto-encode real images and discriminate  real from generated images (balanced by gamma). They propose a new equilibrium enforcing method paired with a loss derived from the Wasserstein distance for training auto-encoder based GAN that balances the generator and discriminator during training. It provides a new approximate convergence measure, fast and stable training which controls the trade-off between image diversity and visual quality.

* (BiGAN): Adversarial Feature Learning, Donahue17. Presents an extension of regular GANS to learn the inverse mapping: projecting data back into the latent space that allows the learned feature representation to be useful for auxiliary supervised discrimination tasks that is competitive with unsupervised and self-supervised feature learning.

* GAN to improve learning machine robustness in image segmentation, pose estimation and speech recognition using Houdini loss.  Houdini loss is a product of two terms. The first term is a stochastic margin, that is the probability that the difference between the score of the actual target g(x; y) and that of the predicted target g(x; ^y) is smaller than  N(0; 1). It reflects the confidence of the model in its predictions. Houdini is a lower bound of the task loss and considers the difference between the scores assigned by the network to the ground truth and the prediction, and it converges to the task loss.: Houdini: Fooling Deep Structured Prediction Models, Cisse17.

* Unsupervised Pixel–Level Domain Adaptation with Generative Adversarial Networks, Bousmalis 16  presents an unsupervised approach using a (GAN)–based architecture that is able to learn such a transformation in an unsupervised manner, i.e. without using corresponding pairs from the two domains. It is the best state of the art approach on unsupervised learning for domain adaptation, improving over: Decoupling from the Task-Specific Architecture, Generalization Across Label Spaces, achieve Training Stability and Data Augmentation.

* Soumith Chintala and Yann LeCun. A path to unsupervised learning through adversarial networks. In https://code.facebook.com/posts/1587249151575490/a-path-to-unsupervisedlearning-through-adversarial-networks/, 2016.

* State-of-the-art GANs for unsupervised representation learning: BEGAN,  BiGAN, CycleGAN and pixel based GAN. See Allan Ma survey to appear soon.

* Learning to generate images with perceptual similarity metrics. ICIP 2017.

* GLO: **Optimizing the Latent Space of Generative Networks** Piotr Bojanowskii, Armand Joulin, David Lopez-Paz, and Arthur Szlam. Summary by LeCunn:<br>
Short story: GLO model (Generative Latent Optimization) is a generative model in which a set of latent variables is optimized at training time to minimize a distance between a training sample and a reconstruction of it produced by the generator. This alleviates the need to train a discriminator as in GAN.<br>
Slightly less short story: GLO, like GAN and VAE, is a way to train a generative model under uncertainty on the output.<br>
A generative model must be able to generate a whole series of different outputs, for example, different faces, or different bedroom images.<br>
Generally, a set of latent variables Z is drawn at random every time the model needs to generate an output. These latent variables are fed to a generator G that produces an output Y(e.g. an image) Y=G(Z).<br>
Different drawings of the latent variable result in different images being produced, and the latent variable can be seen as parameterizing the set of outputs.<br>
In GAN, the latent variable Z is drawn at random during training, and a discriminator is trained to tell if the generated output looks like it's been drawn from the same distribution as the training set.<br>
In GLO, the latent variable Z is optimized during training so as to minimize some distance measure between the generated sample and the training sample Z* = min_z = Distance(Y,G(Z)). The parameters of the generator are adjusted after this minimization. The learning process is really a joint optimization of the distance with respect to Z and to the parameters of G, averaged on a training set of samples.<br>
After training, Z can be sampled from their allowed set to produce new samples. Nice examples are shown in the paper.<br>
GLO belongs to a wide category of energy-based latent variable models: define a parameterized energy function E(Y,Z), define a "free energy" F(Y) = min_z E(Y,Z). Then find the parameters that minimize F(Y) averaged over your training set, making sure to put some constraints on Z so that F(Y) doesn't become uniformly flat (and takes high values outside of the region of high data density). This basic model is at the basis of sparse modeling, sparse auto-encoders, and the "predictive sparse decomposition" model. In these models, the energy contains a term that forces Z to be sparse, and the reconstruction of Y from Z is linear. In GLO, the reconstruction is computed by a deconvolutional net.
