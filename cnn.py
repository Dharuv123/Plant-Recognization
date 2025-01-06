import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax')
    ])
    return model

def train_and_evaluate():
    # Load MNIST data
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    # Preprocess the data: reshape and normalize
    train_images = train_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    test_images = test_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

    # Create the CNN model
    model = create_cnn_model()

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=10, batch_size=100)

    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc:.2f}")

    # Save the trained model
    model.save('1.keras')
    print("Model has been successfully saved as 1.h5")

if __name__ == "__main__":
    train_and_evaluate()
