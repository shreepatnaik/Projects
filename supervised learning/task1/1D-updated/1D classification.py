# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense


dim="1D"
layersCount=5
neuron_count=50
lr=0.1
file1 = open(dim+"_L"+str(layersCount)+"_N"+str(neuron_count)+"_SUMMARY_LOGS_lr"+str(lr), "a")

mean1=0.1
variance1=0.3
mean2=1.5
variance2=0.6

## #creates dataset 2D normal distribution array
# #normal continous rand variable mean=local,co-variance=scale
X1 = np.random.normal(loc=mean1,scale=variance1,size=(100,1))
X2 = np.random.normal(loc=mean2,scale=variance2,size=(120,1))
# print(X1)
# plt.title("Distribution 1")
# plt.scatter(range(len(X1)),X1)
# plt.show()
# print(X2)
# plt.title("Distribution 2")
# plt.scatter(range(len(X2)),X2)
# plt.show()


print("------------------- Inputs and Dataset ----------------")
print("CLASS 1:")
print("mean:",mean1)
print("variance:",variance1)
probability_class_1=len(X1)/(len(X1)+len(X2))
print("Prior:",probability_class_1)
print("\n")

print("CLASS 2:")
print("mean:",mean2)
print("variance:",variance2)
probability_class_2=len(X2)/(len(X1)+len(X2))
print("Prior:",probability_class_2)


# #generate labels for classes
Y1= np.zeros(len(X1))
Y2= np.ones(len(X2))
print("shape of disribution1 and disribution2",X1.shape, X2.shape)
X = np.concatenate([X1,X2])
Y=np.concatenate([Y1,Y2])
print("length of dataset and labels",len(X), len(Y))
print("shape of dataset and labels",X.shape,Y.shape)
classes=['class 1','class 2']
# plt.scatter(range(len(X)),X, c = Y, cmap = plt.cm.RdYlBu,label=classes)
# plt.show()

#tabular representation
table = pd.DataFrame({ 'distribution' : X[:,0], 'label' : Y})
table.head()

plot3 = plt.subplot2grid((8, 16), (0,0), rowspan=6, colspan=6)
plot4 = plt.subplot2grid((8, 16), (0,8), rowspan=6, colspan=6)

# Create layers (Functional API)
model = Sequential() # We create a sequential in a variable called model
model.add(Dense(5, input_dim=1, activation='relu')) # This is going to be our input layer which has 5 node and 1 input dimension with a ReLu activation function
model.add(Dense(neuron_count, activation='relu'))
model.add(Dense(neuron_count, activation='relu'))
model.add(Dense(neuron_count, activation='relu'))
model.add(Dense(neuron_count, activation='relu'))
model.add(Dense(1,activation = 'sigmoid'))

# Compile the model (binary_crossentropy if 2 classes)
# opt = (lr=0.01)
optimizer = keras.optimizers.Adam(lr=lr)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#Importing and training model
model.summary(print_fn=lambda x: file1.write(x + '\n'))
history=model.fit(X, Y,validation_split=0.2,epochs=1000)
hist_df = pd.DataFrame(history.history)
for k in hist_df:
    print("\n",file=file1)
    for i in range(0,9):
        line="Epoch: "+str(i)+" | Loss: " + str(hist_df['loss'][i]) + " | Valid loss: " + str(hist_df['val_loss'][i]) +" | Val Accuracy : " + str(hist_df['val_accuracy'][i])+" | Accuracy : " + str(hist_df['accuracy'][i])
        print(line,file=file1)
    print(".",file=file1)
    print(".",file=file1)
    print(".",file=file1)
    print(".",file=file1)
    for i in range(995,1000):
        line="Epoch: "+str(i)+" | Loss: " + str(hist_df['loss'][i]) + " | Valid loss: " + str(hist_df['val_loss'][i]) +" | Val Accuracy : " + str(hist_df['val_accuracy'][i])+" | Accuracy : " + str(hist_df['accuracy'][i])
        print(line,file=file1)
    break

file1.close()



plot4.plot(history.history['accuracy'])
plot4.plot(history.history['val_accuracy'])
plot4.set_title('model accuracy')
plot4.set_ylabel('accuracy')
plot4.set_xlabel('epoch')
plot4.legend(['train', 'val'], loc='upper left')


plot3.plot(history.history['loss'])
plot3.plot(history.history['val_loss'])
plot3.set_title('model loss')
plot3.set_ylabel('loss')
plot3.set_xlabel('epoch')
plot3.legend(['train', 'val'], loc='upper left')
plt.savefig(str(dim)+"_LAO_L"+str(layersCount)+"_N"+str(neuron_count)+".png")


loss, accuracy = model.evaluate(X, Y)
print(f' Model loss on the test set: {loss}')
print(f' Model accuracy on the test set: {100*accuracy}')

#Classify Dataset
def plot_decision_boundary():
    y_pred = model.predict(X)
    # Check for multi-class
    if len(y_pred[0]) > 1:
        print("doing multiclass classification...")
        # We have to reshape our predictions to get them ready for plotting
        y_pred = np.argmax(y_pred, axis=1).reshape(X.shape)
    else:
        print("doing binary classifcation...")
        y_pred = y_pred.reshape(X.shape)
    print("predicted labels:",y_pred)

    figure = plt.figure(figsize = (12, 8))
    figure.suptitle('Classification', fontsize=16)
    plt.clf()
    plt.scatter(range(len(X)),X , c=y_pred, s=40, cmap=plt.cm.RdYlBu)
    plt.ylabel('Probability Distribution')
    plt.xlabel('Input')
    plt.savefig(str(dim)+"_CLASS_L"+str(layersCount)+"_N"+str(neuron_count)+".png")

plot_decision_boundary()
