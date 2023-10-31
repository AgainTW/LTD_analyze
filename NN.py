import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
print(f'Tensorflow version: {tf.__version__}')

# Load data
train_data = pd.read_csv('PP_train_data.csv')
test_data = pd.read_csv('PP_test_data.csv')

# Split input and output
x = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]
print(f"Columns of input: {list(x.columns)}")
print(f"Columns of output: {y.name}")
print(f"Shape of input: {x.shape}")
print(f"Shape of output: {y.shape}")

# Split train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print(f"Shape of x_train: {x_train.shape}")
print(f"Shape of x_test: {x_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# normalize
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# build model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(units=256, input_shape=(x_train.shape[1],), activation='elu'),
  tf.keras.layers.Dense(units=64, activation='relu'),
  tf.keras.layers.Dense(units=2, activation='softmax')
])

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# early_stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

# fit
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=64,
    epochs=100,
    callbacks=[early_stopping], # put your callbacks in a list
)

# After training
model.save("good_ta_model.h5")

# Loss
fig, ax = plt.subplots(figsize=(8,4))
plt.title('loss')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss', linestyle='--')
plt.legend()
plt.show()

# Accuracy
fig, ax = plt.subplots(figsize=(8,4))
plt.title('accuracy')
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy', linestyle='--')
plt.legend()
plt.show()

# build confusion_matrix
predict = model.predict(x_test)
y_pred=[]
for i in range(len(predict)):
  y_pred.append(np.argmax(predict[i]))

cr = classification_report(y_test, y_pred)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print(cr)

p_test = predict.argmax(axis=1)
cm = tf.math.confusion_matrix(y_test, p_test)

sns.heatmap(cm, annot=True, cmap='Blues', square=True, linewidths=0.01, linecolor='grey', fmt="d")
plt.title('Confustion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()