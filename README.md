# Classification of cell images into parasitized and uninfected
## Study project to learn image classification with Keras

Data source: Tensorflow datasets https://www.tensorflow.org/datasets/catalog/malaria

All images have different sizes. Looks like they all have black background so padding is an option. But it would increase volume of data so I used convolution layers first followed by global pooling before fully connected layers.

In this version, I added residual connections with marginal improvement of accuracy from on the test set from 95.4% to 95.7%.

I also tried to select the optimum network architecture with automatic Bayesian, but due to usage limts at Google Colab had to do part of the search manually.
