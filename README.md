# Classification of cell images into parasitized and uninfected
## Study project to learn image classification with Keras

Data source: Tensorflow datasets https://www.tensorflow.org/datasets/catalog/malaria

All images have different sizes. Looks like they all have black background so padding is an option. But it would increase volume of data so I used convolution layers first followed by global pooling before fully connected layers.

In this version, I added residual connections which marginally improved accuracy on the test set from 95.4% to 95.7%.

The network architecture was partially automatic with the Bayesian tuner, but due to usage limts at Google Colab I had to do part of the search manually.

I also tried to add batch normalization to the convolution layers but accuracy got worse.

Notebooks:
- dataload_colab - loading the dataset.
- EDA - done on a local machine
- residual_connections_optimization... - automatic and manual hyperparameter selection
- final_train_predict_resid - training the model on the combined train and validation set and checking on the test set
- residual_connections_BN - same model with batch normalization
- residual_connections_BN_final_test - final training on the combined train and validation set and checking on the test set

Spreadshit results.xlsx was used to log accuracy during the model optimization stage.


