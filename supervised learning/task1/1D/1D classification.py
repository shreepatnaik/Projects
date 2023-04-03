# import required libraries
from scipy.stats import norm
import seaborn as sb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

dim="1D"
layersCount=5
neuron_count=50
file1 = open(dim+"_L"+str(layersCount)+"_N"+str(neuron_count)+"_SUMMARY_LOGS", "a")

#creates dataset of 1D array 1-20 ,with intervel 0.1
X = np.arange(1,20,0.1)
mean=5.3
variance=1
# Creating the dataset
#normal continous rand variable mean=5.3,co-variance=1
norm_func=norm(loc=mean,scale=variance)
probability_class_1=norm_func.cdf(4.5)
probability_class_2=1-probability_class_1
print("------------------- Inputs and Dataset ----------------")
print("Input 1D array X:",X)
print("mean:",mean)
print("variance:",variance)
print("Prior of Class 1:",probability_class_1)
print("Prior of Class 2:",probability_class_2)


#probability distribution function
pdf = norm_func.pdf(X)
print('PDF:',pdf)
sb.set_style('whitegrid')
plt.title('Generated Data')
plt.plot(X,pdf,color='red')
plt.xlabel('Input 1D array')
plt.ylabel('Probability Distribution')

#generate labels for classes
Y = np.array([ 0 if norm_func.pdf(i)<probability_class_1 else 1 for i in X])
print("labels:",Y)
print("shape of dataset and labels",X.shape, Y.shape)
print("length of dataset and labels",len(X), len(Y))

#tabular representation
table = pd.DataFrame({ 'X' : pdf, 'label' : Y})
table.head()
print(table)

#Visualizing the expected distribution
classes=['class 1','class 2']
plt.title("Expected Classification")
plt.scatter(X,pdf,c=Y,cmap = plt.cm.RdYlBu,label=classes)

plot3 = plt.subplot2grid((8, 16), (0,0), rowspan=6, colspan=6)
plot4 = plt.subplot2grid((8, 16), (0,8), rowspan=6, colspan=6)
#Importing and training model
model = keras.models.load_model('models\\class1d.h5')
model.summary(print_fn=lambda x: file1.write(x + '\n'))
history=model.fit(pdf, Y,validation_split=0.2,epochs=1000)
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
    model = keras.models.load_model('models\\class1d.h5')
    y_pred = model.predict(pdf)
    # Check for multi-class
    if len(y_pred[0]) > 1:
        print("doing multiclass classification...")
        # We have to reshape our predictions to get them ready for plotting
        y_pred = np.argmax(y_pred, axis=1).reshape(X.shape)
    else:
        print("doing binary classifcation...")
        y_pred = y_pred.reshape(X.shape)
    print("predicted labels:",y_pred)
    # Plot decision boundary
    table = pd.DataFrame({ 'X' : pdf, 'label' : y_pred})
    table.head()
    print(table)
    print('\n')
    figure = plt.figure(figsize = (12, 8))
    figure.suptitle('Classification', fontsize=16)
    plt.scatter(X, pdf, c=y_pred, s=40, cmap=plt.cm.RdYlBu)
    plt.ylabel('Probability Distribution')
    plt.xlabel('Input')
    plt.legend()
    plt.savefig(str(dim)+"_CLASS_L"+str(layersCount)+"_N"+str(neuron_count)+".png")

plot_decision_boundary()
