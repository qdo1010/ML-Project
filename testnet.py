import time
import matplotlib.pyplot as plt
import numpy as np
import model as km
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
#from keras_sequential_ascii import sequential_model_to_ascii_printout
from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")
 
# Import Tensorflow with multiprocessing
import tensorflow as tf
import multiprocessing as mp
 
# Loading the CIFAR-10 datasets
from keras.datasets import cifar10

batch_size = 32

num_classes = 10
epochs = 100

sgd = SGD(lr=0.1, decay=0.0002, momentum=0.9)
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 
# x_train - training data(images), y_train - labels(digits)

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def print_time(t0, s):
    """Print how much time has been spent

    @param t0: previous timestamp
    @param s: description of this step

    """

    print("%.5f seconds to %s" % ((time.time() - t0), s))
    return time.time()


#fig = plt.figure(figsize=(8,3))
#for i in range(num_classes):
#    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
#    idx = np.where(y_train[:]==i)[0]
#    features_idx = x_train[idx,::]
#    img_num = np.random.randint(features_idx.shape[0])
#    im = np.transpose(features_idx[img_num,::],(1,2,0))
#    ax.set_title(class_names[i])
#    plt.imshow(im)
#plt.show()


#normalize
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train  /= 255
x_test /= 255


t0 = time.time()
nb_class = 10
model = km.SqueezeNet(nb_class, inputs=(3, 32, 32))
        # dp.visualize_model(model)
t0 = print_time(t0, 'build the model')

model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
t0 = print_time(t0, 'compile model')

#model.fit_generator(train_generator,samples_per_epoch=nb_train_samples,nb_epoch=args.epochs,validation_data=validation_generator,nb_val_samples=nb_val_samples)
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test,y_test),shuffle=True)
t0 = print_time(t0, 'train model')

plt.figure(0)
plt.plot(cnn.history['acc'],'r')
plt.plot(cnn.history['val_acc'],'g')
plt.xticks(np.arange(0, 101, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train','validation'])
 
 
plt.figure(1)
plt.plot(cnn.history['loss'],'r')
plt.plot(cnn.history['val_loss'],'g')
plt.xticks(np.arange(0, 101, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train','validation'])
 
plt.show()

