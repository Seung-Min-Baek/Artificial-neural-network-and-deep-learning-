### Q1
import pickle
import numpy as np

with open('/content/drive/MyDrive/Colab Notebooks/practice_data.txt','rb') as f:
  data = pickle.load(f)

num_days, num_provinces = data.shape

timestep_size = 10

preprocess_input_data = np.zeros((num_days-timestep_size, timestep_size, num_provinces))
preprocess_target_data = np.zeros((num_days - timestep_size, num_provinces))

### Q2
for i in range(num_days-timestep_size):
  end_index = i + timestep_size
  preprocess_input_data[i] = data[i: end_index]
  preprocess_target_data[i] = data[end_index: end_index+1]

train_input = preprocess_input_data[:-100]
train_target = preprocess_target_data[:-100]
test_input = preprocess_input_data[-100:]
test_target = preprocess_target_data[-100:]

max_train = np.max(data[:-100])
train_input /= max_train
train_target /= max_train
test_input /= max_train
test_target /= max_train

print("train_input.shape:",train_input.shape)
print("train_target.shape:",train_target.shape)
print("test_input.shape:",test_input.shape) 
print("test_target.shape:",test_target.shape)

### Q3
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit

batch_size = 20
epoch = 1000

validation_losses = []
train_losses=[]

# 모델 정의
tscv = TimeSeriesSplit(n_splits=4)
model = models.Sequential()
model.add(layers.GRU(52, activation = 'sigmoid'))
model.add(layers.Dense(52))
model.compile(optimizer=Adam(learning_rate=0.001), loss = 'mse')

# TimeSeriesSplit
for train_index, val_index in tscv.split(train_input):
  x_train, x_val = train_input[train_index], train_input[val_index]
  y_train, y_val = train_target[train_index], train_target[val_index]
  print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)

  history = model.fit(x_train, y_train, epochs = epoch, batch_size = batch_size, validation_data=(x_val, y_val))
  train_loss = history.history['loss']
  train_losses.append(train_loss)
  val_loss = history.history['val_loss']
  validation_losses.append(val_loss)

  train_losses = np.array(train_losses)
val_losses = np.array(validation_losses)
print(train_losses)
print(val_losses)

# validation loss 평균
avg_loss = np.min(val_losses,axis=1)

# optimal epoch 찾기(argmin)
optimal_epochs = np.argmin(val_losses,axis=1)
print(optimal_epochs)

optimal_epoch = int(np.mean(optimal_epochs))
print(optimal_epoch)

print("first fold train losses: ",np.mean(train_losses[0]))
print("second fold train losses: ",np.mean(train_losses[1]))
print("third fold train losses: ",np.mean(train_losses[2]))
print("fourth fold train losses: ",np.mean(train_losses[3]))

print("first fold val losses: ",np.mean(val_losses[0]))
print("second fold val losses: ",np.mean(val_losses[1]))
print("third fold val losses: ",np.mean(val_losses[2]))
print("fourth fold val losses: ",np.mean(val_losses[3]))


sec_train_losses = []

for i in range(len(optimal_epochs)):
  history2 = model.fit(train_input, train_target, epochs = optimal_epochs[i], batch_size = batch_size)

  sec_train_loss = history2.history['loss']
  sec_train_losses.append(sec_train_loss)

sec_train_losses = np.array(sec_train_losses)
print("train_losses : ",sec_train_losses)

# 각 fold별 train losses의 평균
print(np.mean(sec_train_losses[0]))
print(np.mean(sec_train_losses[1]))
print(np.mean(sec_train_losses[2]))
print(np.mean(sec_train_losses[3]))

third_train_losses=[]

# optimal_epoch에서의 train loss
history3 = model.fit(train_input, train_target, epochs = optimal_epoch, batch_size = batch_size)

third_train_loss = history3.history['loss']
third_train_losses.append(third_train_loss)

third_train_losses = np.array(third_train_losses)
print(type(third_train_losses))
print("train_losses : ",third_train_losses)

# optimal_epoch에서의 train loss 평균
print(np.mean(third_train_losses))

train_loss = model.evaluate(train_input,train_target)
print('train_loss_evaluate :',train_loss)
test_loss = model.evaluate(test_input,test_target)
print('test_loss_evaluate :',test_loss)

### Q4

import matplotlib.pyplot as plt

target_value = np.sum(np.concatenate((train_target,test_target),axis=0),axis=1)

input_value = np.sum(model.predict(np.concatenate((train_input,test_input),axis=0)),axis=1)

plt.plot(target_value, color='black',label='target')
plt.plot(input_value, color='red',label='predicted' )

plt.title('Overall Daily Confirmed Cases')
plt.xlabel('Time')
plt.ylabel('Confirmed Cases')

plt.legend()
