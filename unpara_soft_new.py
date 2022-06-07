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
num_noi = 1*num
#I’m glad you asked. Yes you have a good understanding of the context. One clarification, #for the third feature, Pt, we haven’t given you the distributions explicitly because the #exact choice doesn’t matter, but for the plots I used truncated normal (i.e. gaussian) #distribution for source signal, an exponential for the background and just a gaussian #distribution for the target signal. You can do the same. Since F1 and F2 partially #depend on Pt, you’ll need to generate the Pt distributions first. The theta is sampled #always from the same distribution (the gaussian as described in the slides) for all #three classes.

#Why is re-weighting important?
#When you train the parameterised classifier, the network will have three inputs(F1, F2 #and Pt), and it may simply use the difference between the Pt distribution of the source #signal and background to make a decision. However this decision function will no longer #work to separate target signal and background. This is why one needs to mask the #discriminating power of Pt itself.
#If you prefer, you can first try without the re-weighing and see the results you get.

def generate_pt(num: np.float64):
    np.random.seed(int(time.time()))
    pt_source = []
    for i in range(num):
        temp =  np.random.normal(loc=0,scale = 0.8,size = 1)
        if temp < 0:
            temp = -temp
        pt_source = np.append(pt_source,temp)
    pt_target = []
    for i in range(num):
        temp =  np.random.normal(loc=10,scale = 1.2,size = 1)
        if temp < 0:
            temp = -temp
        pt_target = np.append(pt_target,temp)
    pt_background = []
    for i in range(num_noi):
        temp =  np.random.exponential(scale = 6, size = 1)
        if temp < 0:
            temp = -temp
        pt_background = np.append(pt_background,temp)
    
    plt.hist(pt_source, bins=100, range=[0,25],density = True ,color ='r',histtype='step',label = 'pt_source')
    
    plt.hist(pt_target, bins=100, range=[0,25] ,density = True,color ='b',histtype='step',label = 'pt_target')
    
    plt.hist(pt_background, bins=100, range=[0,25] ,density = True,color ='g',histtype='step',label = 'pt_background')
    #print(dn)
    #print(ds)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
    return pt_source, pt_target, pt_background

def generate_f1(pt_source: np.array, pt_target: np.array , pt_background: np.array): 
    theta = np.random.normal(loc = 0, scale = math.pi/3, size = num_noi)
    f1_background = np.power(pt_background,0.2)
    f1_background = np.add(pt_background, (-1)*np.multiply(pt_background,(np.sin(theta)-0.5)))
    f1_background = f1_background +50

    theta = np.random.normal(loc = 0, scale = math.pi/3, size = num)
    f1_source = np.power(pt_source,0.2)
    f1_source = np.add(f1_source, (1)*np.multiply(pt_source,(np.sin(theta)-0.5)))
    f1_source = f1_source +50

    theta = np.random.normal(loc = 0, scale = math.pi/3, size = num)
    f1_target = np.power(pt_target,0.2)
    f1_target = np.add(f1_target, (1)*np.multiply(pt_target,(np.sin(theta)-0.5)))
    f1_target = f1_target + 50

    plt.hist(f1_source, bins=100, range=[0,80],density = True ,color ='r',histtype='step',label = 'f1_source')

    plt.hist(f1_target, bins=100, range=[0,80],density = True ,color ='g',histtype='step',label = 'f1_target')

    plt.hist(f1_background, bins=100, range=[0,80],density = True ,color ='b',histtype='step',label = 'f1_background')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    return  f1_source, f1_target,f1_background

def generate_f2(pt_source: np.array, pt_target: np.array , pt_background: np.array): 
    np.random.seed(int(time.time()))
    theta = np.random.normal(loc = 0, scale = math.pi/3, size = num_noi)
    f2_background = np.power(pt_background,1.1)
    f2_background = np.add(f2_background, (-1)*np.multiply(pt_background,(np.cos(theta)-0.5)))
    f2_background = f2_background +5

    theta = np.random.normal(loc = 0, scale = math.pi/3, size = num)
    f2_source = np.power(pt_source,1.1)
    f2_source = np.add(f2_source, (1)*np.multiply(pt_source,(np.cos(theta)-0.5)))
    f2_source = f2_source +5

    theta = np.random.normal(loc = 0, scale = math.pi/3, size = num)
    f2_target = np.power(pt_target,1.1)
    f2_target = np.add(f2_target, (1)*np.multiply(pt_target,(np.cos(theta)-0.5)))
    f2_target = f2_target + 5

    plt.hist(f2_source, bins=100, range=[0,80],density = True ,color ='r',histtype='step',label = 'f2_source')

    plt.hist(f2_target, bins=100, range=[0,80],density = True ,color ='g',histtype='step',label = 'f2_target')

    plt.hist(f2_background, bins=100, range=[0,80],density = True ,color ='b',histtype='step',label = 'f2_background')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    return  f2_source, f2_target,f2_background

def generate_dataframe(f1_source:np.array, f1_background:np.array, f2_source:np.array, f2_background:np.array):
    id = np.ones([num,1])
    id = id.astype(int)
    id = id.flatten()
    id = pd.DataFrame({'class_id':id})
    id_b = np.zeros([num_noi,1])
    id_b = id_b.astype(int)
    id_b = id_b.flatten()
    id_b = pd.DataFrame({'class_id':pd.Series(id_b)})
    class_id = pd.concat([id,id_b])

    f1_source = pd.DataFrame({'x1': pd.Series(f1_source)})
    f2_source = pd.DataFrame({'x2': pd.Series(f2_source)})
    #f1_source.insert(loc = 2, column = 'class_ID',value = 1)
    f1_background = pd.DataFrame({'x1': pd.Series(f1_background)})
    f2_background = pd.DataFrame({'x2': pd.Series(f2_background)})
    #f1_background.insert(loc = 2, column = 'class_ID',value = 0)
    f1 = pd.concat([f1_source, f1_background])
    f2 = pd.concat([f2_source, f2_background])
    #f2 = pd.concat([f2_source, f2_background])
    f = pd.DataFrame({'x1':f1['x1'],'x2':f2['x2'],'class_id': class_id['class_id']})
    f = f.sample(frac = 1).reset_index(drop=True)
    f = f.sample(frac = 1).reset_index(drop=True)
    f = f.sample(frac = 1).reset_index(drop=True)
    f = f.sample(frac = 1).reset_index(drop=True)
    return f

pt_source, pt_target, pt_background = generate_pt(num)
f1_source, f1_target,f1_background = generate_f1(pt_source, pt_target, pt_background)
f2_source, f2_target,f2_background = generate_f2(pt_source, pt_target, pt_background)
f_train = generate_dataframe(f1_target, f1_background, f2_target, f2_background)

pt_source, pt_target, pt_background = generate_pt(num)
f1_source, f1_target,f1_background = generate_f1(pt_source, pt_target, pt_background)
f2_source, f2_target,f2_background = generate_f2(pt_source, pt_target, pt_background)
f_test = generate_dataframe(f1_source, f1_background,f2_source,f2_background)

train_y = f_train.pop('class_id')
train_y = np.array(train_y).astype(np.double)
test_y = f_test.pop('class_id')
test_y = np.array(test_y).astype(np.double)
X = np.transpose([ f_train['x1'].astype(np.double) ,  f_train['x2'].astype(np.double) ])
X = np.array(X).astype(np.double)
X_test = np.transpose([ f_test['x1'].astype(np.double) ,  f_test['x2'].astype(np.double) ])
X_test = np.array(X_test).astype(np.double)

inputs = Input(shape=(X.shape[1],))
hidden = Dense(n_nodes_clf, activation='relu')(inputs)

for i in range(n_hidden_clf -1):
    hidden = Dense(n_nodes_clf, activation='relu')(hidden) 
predictions = Dense(1, activation='softmax')(hidden)

nn_model = Model(inputs=inputs, outputs=predictions)
nn_model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy())

nn_model.fit(X,train_y,epochs = 40,validation_data=(
        f_test, test_y),  batch_size = 200, verbose = 1, shuffle = True)
y_pred_keras = nn_model.predict(X_test,batch_size = 200, verbose = 1)

score = nn_model.evaluate(x =X_test, y = test_y )
fpr_f, tpr_f, thresholds_f = roc_curve(test_y, y_pred_keras)
auc_f = auc(fpr_f, tpr_f)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr_f,tpr_f,label = 'roc(area = {:.3f})'.format(auc_f))
plt.xlabel('False psitive rate')
plt.ylabel('True postive rate')
plt.xlim([0,1])
plt.ylim([0,1])
plt.title('Receiver operating characteristic curve')
plt.legend(loc = 'best')

