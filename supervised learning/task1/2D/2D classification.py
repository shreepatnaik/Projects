import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf



dim="2D"
layersCount=5
neuron_count=50
lr=2
file1 = open(dim+"_L"+str(layersCount)+"_N"+str(neuron_count)+"_SUMMARY_LOGS_lr"+str(lr), "a")

mean1=0.1
variance1=0.3
mean2=1.5
variance2=0.6
# #creates dataset 2D normal distribution array
# #normal continous rand variable mean=local,co-variance=scale
X1 = np.random.normal(loc=mean1,scale=variance1,size=(100,2))
X2 = np.random.normal(loc=mean2,scale=variance2,size=(120,2))
# print(X1)
# plt.title("Distribution 1")
# plt.scatter(X1[:,0],X1[:,1])
# plt.show()
# print(X2)
# plt.title("Distribution 2")
# plt.scatter(X2[:,0],X2[:,1])
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
# plt.scatter(X[:, 0], X[:, 1], c = Y, cmap = plt.cm.RdYlBu,label=classes)
# plt.show()

#tabular representation
table = pd.DataFrame({ 'coordinate X' : X[:,0],'coordinate Y': X[:,1], 'label' : Y})
# pd.options.display.max_rows = 1000
table.head()
print(table)




#The seed() method is used to initialize the random number generator,needs a number to start with (a seed value)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation = 'relu'), #we may right it "tf.keras.activations.relu" too
    tf.keras.layers.Dense(50, activation = 'relu'),
    tf.keras.layers.Dense(50, activation = 'relu'),
    tf.keras.layers.Dense(50, activation = 'relu'),
    tf.keras.layers.Dense(50, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])
model.compile( loss= tf.keras.losses.binary_crossentropy,optimizer = tf.keras.optimizers.Adam(lr = lr),metrics = ['accuracy'])
history=model.fit(X, Y, validation_split=0.2,epochs = 1000)
model.summary(print_fn=lambda x: file1.write(x + '\n'))

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

plot3 = plt.subplot2grid((8, 16), (0,0), rowspan=6, colspan=6)
plot4 = plt.subplot2grid((8, 16), (0,8), rowspan=6, colspan=6)

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

def plot_decision_boundary(model, X, y):
    # Define the axis boundaries of the plot and create a meshgrid
    #x,0 is all x coordinates ... xaxis min and max
    #x,1 lly are all y coordianates
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),np.linspace(y_min, y_max, 100))
    #similar to flatten brings all as [x1,x2,x3,y1,y2,y3]
    x_in = np.c_[xx.ravel(), yy.ravel()]
    print(len(x_in))
    # Make predictions using the trained model
    y_pred = model.predict(x_in)
    # Check for multi-class
    if len(y_pred[0]) > 1:
        print("doing multiclass classification...")
        # We have to argmax [[],[]..] arrary or arrays and reshape to x length our predictions to get them ready for plotting
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("doing binary classifcation...")
        y_pred = np.round(y_pred).reshape(xx.shape)
    print(y_pred)
    print(len(y_pred))
    # Plot decision boundary
    #x,y coordinate with hieght y_pred,contour fills colours in classification areas
    classes=['class 1','class 2']
    plt.clf()
    plt.contourf(xx, yy, y_pred, cmap='ocean')
    plt.contour(xx, yy, y_pred, colors='w')
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='Spectral',label=classes)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.savefig(str(dim)+"_CLASS_L"+str(layersCount)+"_N"+str(neuron_count)+".png")

plot_decision_boundary(model, X, Y)


#The axis along which the arrays will be joined. If axis=None, arrays are flattened before use. Default is 0.
#plt(horizntal coordinates,vertical coordinates,colour of dots y,colour)
#axes formaed in bigger circle coordinates

