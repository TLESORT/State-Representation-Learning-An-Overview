# Variational appraoch

(everything here is bad written, you don't have to read it :) )

The hidden parameters of a environenement can be interpreted as parameters of the distribution which generate our input data. If we can fit this distribution we can find the hidden parameteers which explains the best the distribution of the data. The variational approach aims to find this distributuion to exctract a representation. the problem is that we can only sample a low part of the distribution which produce the data. It means that we can not a probability for each data because the normalisation varaible is intractable and it make the minimization between a learn distribution and the true one difficult to realize. All the varaitional paper which learn state representaiton utilize a reparametrization trick to make the problem tractable.

# Variational autoencoders

# The reparametrization trick

# Papers

- **Deep Variational Bayes Filters: Unsupervised Learning of State Space Models from Raw Data**  <br> Maximilian Karl, Maximilian Soelch, Justin Bayer, Patrick van der Smagt, (2017), pdf

- **Deep Kalman Filters** <br> Rahul G. Krishnan, Uri Shalit, David Sontag, (2015), [pdf](https://arxiv.org/pdf/1511.05121.pdf)

- **Embed to control: A locally linear latent dynamics model for control from raw images** <br> Watter, Manuel, et al, (2015) [pdf](https://pdfs.semanticscholar.org/21c9/dd68b908825e2830b206659ae6dd5c5bfc02.pdf)
