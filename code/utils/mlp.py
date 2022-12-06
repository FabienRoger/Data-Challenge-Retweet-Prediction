import tensorflow as tf
from tensorflow import keras
from keras import layers
from attrs import define


@define
class print_every_n_epochs_Callback(tf.keras.callbacks.Callback):
    n: int = 10

    def on_epoch_end(self, epoch, logs=None):
        if (int(epoch) % self.n) == 0:
            if "val_loss" in logs:
                print(
                    "Epoch: {:>3} | Loss: ".format(epoch)
                    + f"{logs['loss']:.4e}"
                    + " | Valid loss: "
                    + f"{logs['val_loss']:.4e}"
                )
            else:
                print("Epoch: {:>3} | Loss: ".format(epoch) + f"{logs['loss']:.4e}")


lr0 = 0.003
decrease_start = 40


def schedule(epoch, lr):
    if epoch < decrease_start:
        return lr0
    return lr0 * decrease_start / epoch


def get_trained_model(X_train, y_train, units=64, epochs=2000):
    scheduler = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)
    my_callbacks = [print_every_n_epochs_Callback(10), scheduler]

    _, n_features = X_train.shape

    inputs = keras.Input(shape=(n_features,), name="digits")
    x = inputs
    x = layers.Dense(units, activation="relu", name="dense_1")(x)
    x = layers.Dense(1, activation="linear", name="dense_2")(x)
    x = tf.keras.activations.exponential(x)
    x = layers.Dense(
        1,
        kernel_initializer=tf.keras.initializers.Ones(),
        activation="linear",
        name="predictions",
    )(x)
    model = keras.Model(inputs=inputs, outputs=x)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr0),
        loss=tf.keras.losses.MeanAbsoluteError(
            reduction="auto", name="mean_absolute_error"
        ),
    )

    model.fit(
        X_train,
        y_train,
        batch_size=64,
        epochs=epochs,
        verbose=0,
        callbacks=my_callbacks,
    )

    return model
