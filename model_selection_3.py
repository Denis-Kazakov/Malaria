import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

# Loading train, validation and test sets
ds = tfds.load('malaria', 
               split=('train[:60%]', 'train[60%:80%]', 'train[80%:]'), 
               shuffle_files=False,
              data_dir='./data',
               batch_size=8,
              download=False,
              as_supervised=True,
              with_info=False)


inputs = keras.Input(shape=(None, None, 3))
x = keras.layers.Rescaling(scale=1.0 / 255)(inputs)
kernel_size = 2
x = keras.layers.Conv2D(
    filters=256,
    kernel_size=kernel_size,
    strides=(1, 1),
    padding="valid",
    activation='relu')(x)
x = keras.layers.MaxPooling2D(
    pool_size=(2, 2), 
    strides=None, 
    padding="valid")(x)
x = keras.layers.Conv2D(
    filters=512,
    kernel_size=kernel_size,
    strides=(1, 1),
    padding="valid",
    activation='relu')(x)
x = keras.layers.GlobalMaxPooling2D()(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(units=512,
                        activation="relu",
                      kernel_regularizer=None)(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='Adam',
      loss='binary_crossentropy',
      metrics=['accuracy'])


model.summary()


callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        min_delta=0,
        patience=5,
        verbose=1,
        mode="max",
        baseline=None,
        restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(
        filepath='/content/gdrive/MyDrive/datasets/Malaria/checkpoints/model_{epoch}',
        save_freq='epoch')
    ]

history = model.fit(
    ds[0],
    epochs=50,
    verbose="auto",
    callbacks=callbacks,
    validation_data=ds[1]
)

print(max(history.history[val_accuracy]))

print(model.evaluate(ds[2]))





