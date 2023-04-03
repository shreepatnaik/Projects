# Importing standard libraries necessary for creating the NN model
import math
import keras
import numpy as np
import pandas as pd
# Importing necessary functions and objects from the libraries imported above.
from keras.models import Sequential # To create a sequential model
from keras.layers import Dense, Flatten, Activation  # Functions used in the fitting the neural network
from numpy import asarray # To send data as an array object
from matplotlib import pyplot # This is to plot the functions outputs


layersCount = 5
nueron_count = 50
lr=1.0
functionDen=12

# print("these are x - ", x, "and these are Y - ", y)
plot1 = pyplot.subplot2grid((10, 10), (6, 2), rowspan=5, colspan=5)
plot2 = pyplot.subplot2grid((10, 10), (0, 0), rowspan=4, colspan=4)
plot3 = pyplot.subplot2grid((11, 10), (0, 6), rowspan=4, colspan=5)

file1 = open("L"+str(layersCount)+"_N"+str(nueron_count)+"_lr_"+str(lr)+"_SUMMARY_LOGS", "a")


model = Sequential() # We create a sequential in a variable called model
model.add(Dense(5, input_dim=1, activation='relu', kernel_initializer='he_uniform')) # This is going to be our input layer which has 5 node and 1 input dimension with a ReLu activation function
model.add(Dense(nueron_count, activation='sigmoid', kernel_initializer='he_uniform'))
model.add(Dense(50, activation='sigmoid', kernel_initializer='he_uniform'))
model.add(Dense(50, activation='sigmoid', kernel_initializer='he_uniform'))
model.add(Dense(50, activation='sigmoid', kernel_initializer='he_uniform'))
model.add(Dense(1))
# Defining the optimiser
optimizer = keras.optimizers.Adam(lr=lr)
model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy']) # This needs to be in a different block of code as compiling a model creates a new model everytime we run the compile line. To train the model indefenitely we need to compile it once and then fit it any number of times
model.summary(print_fn=lambda x: file1.write(x + '\n'))


def approx(functionDen):
    # Data Generation
    x = np.arange(-1,1,0.01)
    # x = asarray([i for i in range(-20, 20)]) # I have chosen this value range to simplify the computation, the more the input data, longer will the time be for each epoch to be computed
    y = asarray([100 + math.cos(2*(math.pi)*(i/functionDen))+math.sin(2*(math.pi)*(i/functionDen)) for i in x])

    # pyplot.scatter(x,y) # We use these x and y values to plot the actual function's graph
    x = x.reshape((len(x), 1))
    y = y.reshape((len(y), 1))
    model_history = model.fit(x,y, validation_split=0.2, epochs=1000, batch_size=10) # We can fit the model any number of times by running this bit of code to train it from the last epoch that we have left it at.
    yhat = model.predict(x)

    hist_df = pd.DataFrame(model_history.history)
    for k in hist_df:
        for i in range(0,9):
            line="Epoch: "+str(i)+" | Loss: " + str(hist_df['loss'][i]) + " | Valid loss: " + str(hist_df['val_loss'][i]) +" | Val Accuracy : " + str(hist_df['val_accuracy'][i])+" | Accuracy : " + str(hist_df['accuracy'][i])
            print(line,file=file1)
        break

    file1.close()

    plot1.scatter(x,y, label='Actual')
    plot1.scatter(x,yhat, label='Predicted')
    plot1.set_title('Input (x) versus Output (y)')
    plot1.set_xlabel('Input Variable (x)')
    plot1.set_ylabel('Output Variable (y)')
    plot1.legend()


    plot2.plot(model_history.history['accuracy'])
    plot2.plot(model_history.history['val_accuracy'])
    plot2.set_title('model accuracy')
    plot2.set_ylabel('accuracy')
    plot2.set_xlabel('epoch')
    plot2.legend(['train', 'val'], loc='upper left')
    # plot2.savefig("A_" + "L_" + str(layersCount) + "_D_"+ str(functionDen) + ".png" )
    # pyplot.show()

    plot3.plot(model_history.history['loss'])
    plot3.plot(model_history.history['val_loss'])
    plot3.set_title('model loss')
    plot3.set_ylabel('loss')
    plot3.set_xlabel('epoch')
    plot3.legend(['train', 'val'], loc='upper left')

approx(functionDen)
# pyplot.tight_layout()
pyplot.savefig("LAO_" + "_L_"+str(layersCount) + "_N_"+ str(nueron_count)+"_lr_"+str(lr) + ".png" )
# pyplot.show()

# learning_rates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# for i in range(len(learning_rates)):
#     # determine the plot number
#     # plot_no = 420 + (i+1)
#     # pyplot.subplot(plot_no)
#     # fit model and plot learning curves for a learning rate
#     plot1 = pyplot.subplot2grid((10, 10), (6, 2), rowspan=5, colspan=5)
#     plot2 = pyplot.subplot2grid((10, 10), (0, 0), rowspan=4, colspan=4)
#     plot3 = pyplot.subplot2grid((11, 10), (0, 6), rowspan=4, colspan=5)

# pyplot.show()






