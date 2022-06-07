from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from cv2 import threshold
from tensorflow import keras
from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.metrics import plot_roc_curve,roc_curve,auc,roc_auc_score
from sklearn.metrics import accuracy_score
#from sklearm.utils import shuffle
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import math
from math import isnan
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier
n_hidden_clf = 10
n_nodes_clf = 32
num = (10**4)
num_noi = num
def generate_pt(num: np.float64):
    np.random.seed(int(time.time()))
    pt_source = []
    for i in range(num):
        temp =  np.random.normal(loc=0,scale = 28,size = 1)
        if temp < 0:
            temp = -temp
        pt_source = np.append(pt_source,temp)
    pt_target = []
    for i in range(num):
        temp =  np.random.normal(loc=90,scale = 28,size = 1)
        if temp < 0:
            temp = -temp
        pt_target = np.append(pt_target,temp)
    pt_background = []
    for i in range(num_noi):
        temp =  np.random.exponential(scale = 50, size = 1)
        if temp < 0:
            temp = -temp
        pt_background = np.append(pt_background,temp)
    range_data = [0, 200]
    plt.hist(pt_source, bins=100, range=range_data,density = True ,color ='r',histtype='step',label = 'pt_source')
    plt.hist(pt_target, bins=100, range=range_data ,density = True,color ='b',histtype='step',label = 'pt_target')
    plt.hist(pt_background, bins=100, range=range_data ,density = True,color ='g',histtype='step',label = 'pt_background')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
    return pt_source, pt_target, pt_background

def generate_dataframe(pt_source:np.array, pt_background:np.array):
    id = np.ones([len(pt_source),1])
    id = id.astype(int)
    id = id.flatten()
    id = pd.DataFrame({'class_id':id})
    id_b = np.zeros([len(pt_background),1])
    id_b = id_b.astype(int)
    id_b = id_b.flatten()
    id_b = pd.DataFrame({'class_id':pd.Series(id_b)})
    class_id = pd.concat([id,id_b])
    class_id.reset_index(drop=True, inplace=True)

    #f1_source = pd.DataFrame({'x1': pd.Series(f1_source)})
    #f2_source = pd.DataFrame({'x2': pd.Series(f2_source)})
    pt_source = pd.DataFrame({'pt': pd.Series(pt_source)})
    pt_background = pd.DataFrame({'pt': pd.Series(pt_background)})
    #f1_source.insert(loc = 2, column = 'class_ID',value = 1)

    #f1_background.insert(loc = 2, column = 'class_ID',value = 0)
    pt = pd.concat([pt_source, pt_background])
    pt.reset_index(drop=True, inplace=True)
    #f2 = pd.concat([f2_source, f2_background])
    f = pd.DataFrame({'pt':pt['pt'],'class_id': class_id['class_id']}, index=range(len(class_id)))
    #f = f.sample(frac = 1).reset_index(drop=True)
    #f = f.sample(frac = 1).reset_index(drop=True)
    #f = f.sample(frac = 1).reset_index(drop=True)
    #f = f.sample(frac = 1).reset_index(drop=True)
    return f

pt_source, pt_target, pt_background = generate_pt(num)
f_train_sb = generate_dataframe(pt_source,pt_background)
f_train_tb = generate_dataframe(pt_target,pt_background)
pt_source, pt_target, pt_background = generate_pt(num)
f_test_sb = generate_dataframe(pt_source,pt_background)
f_test_tb = generate_dataframe(pt_target,pt_background)
f_test_sb.dropna()
f_train_tb.dropna()
train_y_sb = f_train_sb.pop('class_id')
#train_y = np.array(train_y)
test_y_sb = f_test_sb.pop('class_id')
train_y_tb = f_train_tb.pop('class_id')
#train_y = np.array(train_y)
test_y_tb = f_test_tb.pop('class_id')
############################################################################################
from keras.models import Sequential
from keras.layers import Dense
def build_model():
    model = Sequential()
    model.add(Dense(20, input_dim=1, activation='relu'))
    model.add(Dense(40, activation='relu'))
    #model.add(Dense(80, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

from keras.wrappers.scikit_learn import KerasClassifier
keras_model = build_model()
keras_model.fit(f_train_sb, train_y_sb, epochs=50, batch_size=500, verbose=1,shuffle=True)

from sklearn.metrics import roc_curve
y_pred_keras = keras_model.predict(f_test_sb)
weight_sb = np.divide(  1-y_pred_keras,y_pred_keras)
np.save('/mnt/d/DW_project/weight_sb.npy', weight_sb)
###########################################################################
keras_model_ = build_model()
keras_model_.fit(f_train_tb, train_y_tb, epochs=50, batch_size=500, verbose=1,shuffle=True)

from sklearn.metrics import roc_curve
y_pred_keras_ = keras_model_.predict(f_test_tb)
weight_tb = np.divide(  1-y_pred_keras_,y_pred_keras_)
np.save('/mnt/d/DW_project/weight_tb.npy', weight_tb)

plt.hist(pt_source, bins=100, range=[0,200],density = True ,color ='r',histtype='step',label = 'pt_source', weights= weight_sb[0:num])
plt.hist(pt_target, bins=100, range=[0,200] ,density = True,color ='b',histtype='step',label = 'pt_target', weights= weight_tb[0:num])
plt.hist(pt_background, bins=100, range=[0,200] ,density = True,color ='g',histtype='step',label = 'pt_background')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()