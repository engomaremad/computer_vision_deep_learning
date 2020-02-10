from keras.datasets import  mnist
import  matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.metrics import accuracy_score
import  numpy as np
#download mnist data and split into train and test sets

def decode(datum):
    return np.argmax(datum,axis=1)
#reshape data to fit model

(X_train, y_train), (X_test, y_test) = mnist.load_data()

'''

plt.imshow(X_train[0],cmap='gray')
plt.show()
'''

#reshape data to fit model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

y_train =to_categorical(y_train)
y_test  = to_categorical(y_test)



#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20,validation_split=0.25)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#predict first 4 images in the test set
res = model.predict(X_test)
acc = accuracy_score(decode(res), decode(y_test))
x =0