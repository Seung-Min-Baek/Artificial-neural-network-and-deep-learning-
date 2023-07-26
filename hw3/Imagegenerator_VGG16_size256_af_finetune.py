# loading the model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers
import matplotlib.pyplot as plt
# from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# set image generators
train_dir = './datasets/x_ray_images/chest_xray/train/'
test_dir = './datasets/x_ray_images/chest_xray/test/'
validation_dir = './datasets/x_ray_images/chest_xray/val/'

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=20,
    class_mode='binary')
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=20,
    class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(256, 256),
    batch_size=20,
    class_mode='binary')

# model definition
input_shape = [256, 256, 3] # as a shape of image
model = load_model('X_rayQ3_256V3bf.h5')

conv_base = model.layers[0]
for layer in conv_base.layers:
    if layer.name.startswith('block5'):
        layer.trainable = True

model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
              loss='binary_crossentropy', metrics=['accuracy'])

# main loop without cross-validation
num_epochs =100
history = model.fit_generator(train_generator,
                              epochs=num_epochs,
                              steps_per_epoch=100,
                              validation_data=validation_generator,
                              validation_steps=50)

# saving the model
model.save('X_rayQ3_256V3af.h5')

# evaluation
train_loss, train_acc = model.evaluate_generator(train_generator)
test_loss, test_acc = model.evaluate_generator(test_generator)
print('train_loss:', train_loss)
print('test_loss:',test_loss)
print('train_acc:', train_acc)
print('test_acc:', test_acc)

# visualization
def plot_acc(h, title="accuracy"):
    plt.plot(h.history['acc'])
    plt.plot(h.history['val_acc'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)

def plot_loss(h, title="loss"):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)

plot_loss(history)
plt.savefig('X_rayQ3_256V3af.loss.png')
plt.clf()
plot_acc(history)
plt.savefig('X_rayQ3_256V3af.accuracy.png')

import os
import numpy as np
import sklearn.metrics

class_labels = ['NORMAL', 'PNEUMONIA']

test_images = []
y_test = []

for label_idx, class_label in enumerate(class_labels):
    class_folder = os.path.join(test_dir, class_label)
    file_names = os.listdir(class_folder)

    for file_name in file_names:
        file_path = os.path.join(class_folder, file_name)
        test_images.append(file_path)
        y_test.append(label_idx)

y_test = np.array(y_test)

y_pred = model.predict_generator(test_generator)
matrix = sklearn.metrics.confusion_matrix(y_test, y_pred > 0.5)
auc = sklearn.metrics.roc_auc_score(y_test, y_pred)

print("matrix(Q3 256 V3 af):", matrix)
print("auc:", auc)
