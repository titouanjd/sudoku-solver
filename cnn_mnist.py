import keras
from keras import layers


def create_cnn(input_shape: tuple = (28, 28, 1),
               num_classes: int = 10) -> keras.Model:
    """
    Creates a simple CNN model.

    Parameters:
    - input_shape: tuple, the shape of the input images (default: 28x28 grayscale images).
    - num_classes: int, the number of output classes (default: 10 for MNIST).

    Returns:
    - model: a compiled Keras CNN model.
    """
    data_augmentation = keras.Sequential([
        # Randomly rotate image by max 0.05 * 360 = 18 degrees
        layers.RandomRotation(0.05, fill_mode="constant"),

        # Randomly zoom in or out image by max 10%
        layers.RandomZoom(0.1, fill_mode="constant"),

        # Randomly shift image vertically/horizontally by max 5%
        layers.RandomTranslation(0.05, 0.05, fill_mode="constant")
    ], name="data_augmentation")

    model = keras.Sequential([
        keras.Input(shape=input_shape),

        # Data augmentation (only computed in training)
        data_augmentation,

        # First set of CONV => RELU => POOL layers
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Second set of CONV => RELU => POOL layers
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # FC layer
        layers.Flatten(),
        layers.Dropout(0.5),

        # Softmax classifier
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    return model

def main():
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Reshape the images and scale them to the [0, 1] range
    x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255
    x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255

    # Build the model
    model = create_cnn(input_shape=(28, 28, 1), num_classes=10)
    model.summary()

    # Train the model
    model.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.1)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print("Test accuracy:", test_acc)

    # Save the model
    model.save("cnn_mnist_model.keras")


if __name__ == "__main__":
    main()
