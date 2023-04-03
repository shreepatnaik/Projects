from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# Create layers (Functional API)
model = Sequential() # We create a sequential in a variable called model
model.add(Dense(5, input_dim=1, activation='relu')) # This is going to be our input layer which has 5 node and 1 input dimension with a ReLu activation function
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1,activation = 'sigmoid'))

# Compile the model (binary_crossentropy if 2 classes)
# opt = (lr=0.01)

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.save('models\\class1d.h5')
