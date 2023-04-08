import tensorflow as tf

# This model architecture is a relatively simple CNN with a series of convolutional layers followed by 
# fully connected layers. It includes max pooling layers to reduce the spatial dimensions of the feature maps, 
# and dropout layers to reduce overfitting. The final output layer uses softmax activation for multi-class classification. 
# The model is compiled with the Adam optimizer, categorical crossentropy loss function, and categorical accuracy metric.

def build_model(input_shape, num_classes, learning_rate):
    # Define the input layer
    input_layer = tf.keras.layers.Input(shape=input_shape)

    # Apply a series of convolutional layers with increasing filters and decreasing kernel sizes
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(input_layer)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)

    # Flatten the output of the last convolutional layer
    x = tf.keras.layers.Flatten()(x)

    # Add a fully connected layer with 512 neurons and ReLU activation
    x = tf.keras.layers.Dense(units=512, activation='relu')(x)

    # Add a dropout layer to reduce overfitting
    x = tf.keras.layers.Dropout(rate=0.5)(x)

    # Add the final output layer with softmax activation for multi-class classification
    output_layer = tf.keras.layers.Dense(units=num_classes, activation='softmax')(x)

    # Define the model with the input and output layers
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Define the optimizer, loss function, and metrics to be used during training
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = [tf.keras.metrics.CategoricalAccuracy()]

    # Compile the model with the optimizer, loss function, and metrics
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
