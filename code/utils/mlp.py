import tensorflow as tf
from tensorflow import keras
from keras import layers
from attrs import define


@define
class PrintEveryNEpochCallback(tf.keras.callbacks.Callback):
    """Callback that prints the loss every n epochs"""

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


def get_scheduler(lr0, decrease_start):
    """Return a scheduler implementing inverse learning rate decay"""
    def schedule(epoch, lr):
        if epoch < decrease_start:
            return lr0
        return lr0 * decrease_start / epoch

    return schedule


def get_trained_model(
    X_train, y_train, units=64, epochs=2000, lr0=0.003, lr_decrease_start=40
):
    """Build and train a simple MLP model"""
    scheduler = tf.keras.callbacks.LearningRateScheduler(
        get_scheduler(lr0, lr_decrease_start), verbose=0
    )
    my_callbacks = [PrintEveryNEpochCallback(10), scheduler]

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
