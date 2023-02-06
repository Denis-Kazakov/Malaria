# Classification of cell images into parasitized and uninfected
## Study project to learn image classification with Keras

Data source: Tensorflow datasets https://www.tensorflow.org/datasets/catalog/malaria

All images have different sizes. Looks like they all have black background so padding is an option. But it would increase volume of data so I used convolution layers first followed by global pooling before fully connected layers.

I also tried to select the optimum network architecture with automatic Bayesian and Hyperband search, but Google Colab crashes when there is more then 7 convolution layers, so part of the optimization was manual.

Numbered notebooks are different attempts at optimization.

Maximum accuracy achieved on the test set: 95.4%.