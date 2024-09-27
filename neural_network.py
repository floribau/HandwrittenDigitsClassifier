import tensorflow as tf

# load data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# normalize data with min-max scaling
x_train = x_train / 255
x_test = x_test / 255

model = tf.keras.models.Sequential()

# First convolutional layer
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# First pooling layer
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# Second convolutional layer
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# Second pooling layer
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# Randomly drop units to prevent overfitting
model.add(tf.keras.layers.Dropout(0.5))
# Flatten the output of the conv layers
model.add(tf.keras.layers.Flatten())
# First dense layer
model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
# Second dense layer
model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
# Output layer for 10 classes (digits 0-9)
model.add(tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001)))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)
accuracy, loss = model.evaluate(x_test, y_test)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('handwritten-digits-nn.keras')
