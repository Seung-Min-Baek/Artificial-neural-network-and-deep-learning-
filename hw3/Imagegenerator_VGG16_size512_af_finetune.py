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
    target_size=(512, 512),
    batch_size=10,
    class_mode='binary')
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(512, 512),
    batch_size=10,
    class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(512, 512),
    batch_size=10,
    class_mode='binary')

# model definition
input_shape = [512, 512, 3] # as a shape of image
model = load_model('X_rayQ3_512bf.h5')

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
model.save('X_rayQ3_512af.h5')

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
plt.savefig('X_rayQ3_512af.loss.png')
plt.clf()
plot_acc(history)
plt.savefig('X_rayQ3_512af.accuracy.png')

