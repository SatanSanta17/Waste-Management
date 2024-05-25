import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from preprocess_data import load_and_preprocess_data


train_dir = 'DATASET/TRAIN'
test_dir = 'DATASET/TEST'


train_images ,train_labels = load_and_preprocess_data(train_dir)

test_images, test_labels = load_and_preprocess_data(test_dir)


base_model = VGG16(weights = 'imagenet', include_top = False, input_shape=(224,224,3))

x = Flatten()(base_model.output)
x = Dense(128, activation = 'relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs = base_model.input, outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_images,train_labels, epochs =10, batch_size=32)

test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print("test Accuracy:", test_accuracy)

# Save the model
model.save('waste_sorting_model.h5')