# Classification of cell images into parasitized and uninfected
## Study project to learn image classification with Keras

Data source: Tensorflow datasets https://www.tensorflow.org/datasets/catalog/malaria

All images have different sizes. Looks like they all have black background so padding is an option. But it would increase volume of data so I used convolution layers first followed by global pooling before fully connected layers.

This is the first attempt on my local computer. Too long to wait as I did not have GPU.