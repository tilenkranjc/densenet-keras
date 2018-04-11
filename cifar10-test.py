
# coding: utf-8


# In[2]:


# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


# In[3]:

import numpy as np
import time
import keras.backend as K
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import Adam




# In[4]:


from densenet import DenseNet


# In[5]:


(X_train, y_train), (X_test,y_test) = cifar10.load_data()

nb_classes = len(np.unique(y_train))
img_dim = X_train.shape[1:]
n_channels = X_train.shape[-1]

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# In[6]:


# normalization
X = np.vstack((X_train, X_test))

for i in range(n_channels):
    mean = np.mean(X[:,:,:,i])
    std = np.std(X[:,:,:,i])
    X_train[:,:,:,i] = (X_train[:,:,:,i] -mean)/std
    X_test[:,:,:,i] = (X_test[:,:,:,i] -mean)/std


# In[7]:


# model
growth_rate = 12
nb_layers = [12,12,12]
nb_filters = 16
learning_rate = 1e-3

model = DenseNet(img_dim,growth_rate, nb_classes, nb_filters, nb_layers)
model.summary()


# In[8]:



opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(loss='categorical_crossentropy',optimizer=opt, 
                metrics=['accuracy'])


# In[ ]:


# train network

print("Starting training...")

list_train_loss = []
list_test_loss = []
list_learning_rate = []

nb_epoch = 7
batch_size = 64

for e in range(nb_epoch):
    if e == int(0.5 * nb_epoch):
        K.set_value(model.optimizer.lr, np.float32(learning_rate/10.))
    if e == int(0.75 * nb_epoch):
        K.set_value(model.optimizer.lr, np.float32(learning_rate/100.))
        
    split_size = batch_size
    num_splits = X_train.shape[0]/split_size
    arr_splits = np.array_split(np.arange(X_train.shape[0]),num_splits)
    
    l_train_loss = []
    start =time.time()
    
    for batch_idx in arr_splits:
        X_batch, Y_batch = X_train[batch_idx], Y_train[batch_idx]
        train_logloss, train_acc = model.train_on_batch(X_batch, Y_batch)
        
        l_train_loss.append([train_logloss, train_acc])
        
    test_logloss, test_acc = model.evaluate(X_test, Y_test, verbose=0, batch_size=64)
    
    list_train_loss.append(np.mean(np.array(l_train_loss), 0).tolist())
    list_test_loss.append([test_logloss, test_acc])
    list_learning_rate.append(float(K.get_value(model.optimizer.lr)))
    
    print("Epoch %s/%s, Time: %s" % (e + 1, nb_epoch, time.time() - start))
    

