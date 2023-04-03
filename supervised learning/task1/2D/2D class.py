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
X1 = np.arange(1,20,0.1).reshape(2,95)
X2 = np.arange(1,10,0.1).reshape(2,45)
mean1=5.3
variance1=1
mean2=4
variance2=2
# Creating the dataset
#normal continous rand variable mean=5.3,co-variance=1
norm_func1=norm(loc=mean1,scale=variance1)
norm_func2=norm(loc=mean2,scale=variance2)


print("------------------- Inputs and Dataset ----------------")
# print("Input 1D array X:",X)
# print("mean:",mean)
# print("variance:",variance)
probability_class_1=len(X1)/(len(X1)+len(X2))
probability_class_2=len(X2)/(len(X1)+len(X2))
print("Prior of Class 1:",probability_class_1)
print("Prior of Class 2:",probability_class_2)


#probability distribution function
pdf1 = norm_func1.pdf(X1)
print('PDF:',pdf1)
sb.set_style('whitegrid')
plt.title('Generated Data')
plt.plot(X1,pdf1,color='red')
plt.xlabel('Input 1D array')
plt.ylabel('Probability Distribution')
# plt.show()
pdf2 = norm_func2.pdf(X2)
print('PDF:',pdf2)
sb.set_style('whitegrid')
plt.title('Generated Data')
plt.plot(X2,pdf2,color='red')
plt.xlabel('Input 1D array')
plt.ylabel('Probability Distribution')
# plt.show()
#generate labels for classes
# Y = np.array([ 0 if norm_func.pdf(i)<probability_class_1 else 1 for i in X])
Y1= np.zeros(len(pdf1))
Y2= np.ones(len(pdf2))
# print("labels:",Y1)
# print("shape of dataset and labels",X.shape, Y1.shape,Y2.shape())
# print("length of dataset and labels",len(X), len(Y1),len(Y2))
pdf = np.concatenate([pdf1,pdf2],axis=1)
#The axis along which the arrays will be joined. If axis is None, arrays are flattened before use. Default is 0.
Y=np.concatenate([Y1,Y2])
print(len(Y),len(pdf))

#tabular representation
# table1 = pd.DataFrame({ 'pdf1' : pdf1, 'label1' : Y1})
# table1.head()
# print(table1)
# table2 = pd.DataFrame({'pdf2':pdf2,'label2':Y2})
# table2.head()
# print(table2)
# table3 = pd.DataFrame({ 'pdf':pdf,"label":Y})
# table3.head()
# print(table3)

#Visualizing the expected distribution
classes=['class 1','class 2']
plt.title("Expected Classification")
plt.scatter(pdf[:0],pdf[:1],c=Y,cmap = plt.cm.RdYlBu,label=classes)
# plt.show

plot3 = plt.subplot2grid((8, 16), (0,0), rowspan=6, colspan=6)
plot4 = plt.subplot2grid((8, 16), (0,8), rowspan=6, colspan=6)
#Importing and training model
model = keras.models.load_model('models\\class1d.h5')
model.summary(print_fn=lambda x: file1.write(x + '\n'))
history=model.fit(pdf, Y,validation_split=0.2,epochs=25)
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
    # for i in range(995,1000):
    #      line="Epoch: "+str(i)+" | Loss: " + str(hist_df['loss'][i]) + " | Valid loss: " + str(hist_df['val_loss'][i]) +" | Val Accuracy : " + str(hist_df['val_accuracy'][i])+" | Accuracy : " + str(hist_df['accuracy'][i])
    #      print(line,file=file1)
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


loss, accuracy = model.evaluate(pdf, Y)
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
        y_pred = np.argmax(y_pred, axis=1).reshape(pdf.shape)
    else:
        print("doing binary classifcation...")
        y_pred = y_pred.reshape(pdf.shape)
    print("predicted labels:",y_pred)
    # Plot decision boundary
    table = pd.DataFrame({ 'X' : pdf, 'label' : y_pred})
    table.head()
    print(table)
    print('\n')
    figure = plt.figure(figsize = (12, 8))
    figure.suptitle('Classification', fontsize=16)
    # plt.contourf(pdf1, pdf2, , cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(range(0,len(pdf1)), pdf1, c=y_pred[:len(pdf1)], s=40, cmap=plt.cm.RdYlBu)
    plt.scatter(range(0,len(pdf2)), pdf2, c=y_pred[len(pdf1):], s=40, cmap=plt.cm.RdYlBu)
    plt.ylabel('Probability Distribution')
    plt.xlabel('Input')
    plt.show()
    # plt.savefig(str(dim)+"_CLASS_L"+str(layersCount)+"_N"+str(neuron_count)+".png")

plot_decision_boundary()
