import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def train_model(image_dir, output_graph, output_labels, epochs=10):
    # Image dimensions and batch size
    IMG_HEIGHT, IMG_WIDTH = 224, 224
    BATCH_SIZE = 32

    # ImageDataGenerator for training and validation
    train_datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
    train_data = train_datagen.flow_from_directory(
        image_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        subset='training',
        class_mode='categorical')

    val_data = train_datagen.flow_from_directory(
        image_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        subset='validation',
        class_mode='categorical')

    # Using a MobileNetV2 model for transfer learning
    base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                                   include_top=False,
                                                   weights='imagenet')

    base_model.trainable = False  # Freeze the base model
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(train_data.class_indices), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Training the model
    model.fit(train_data, validation_data=val_data, epochs=epochs)

    # Save the model and labels
    model.save(output_graph)
    with open(output_labels, 'w') as f:
        for label in train_data.class_indices:
            f.write(f"{label}\n")

if __name__ == "__main__":
    image_dir = "C:/Users/dharu/OneDrive/Desktop/project/Plant-Recognition-master/training_plant_images"
    output_graph = "tf_files/retrained_graph.h5"  # Save model in H5 format
    output_labels = "tf_files/retrained_labels.txt"

    # Create the output directory if it doesn't exist
    os.makedirs("tf_files", exist_ok=True)

    # Train the model
    train_model(image_dir, output_graph, output_labels)