from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from cv2 import threshold
from tensorflow import keras
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
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow_decision_forests as tfdf

def generate_pt(num: int):
    def generate_pt_(num: int):
        np.random.seed(int(time.time() + np.random.rand(1)+3))
        pt_source = []
        for i in range(int(num)):
            temp =  np.random.normal(loc=0,scale = 40,size = 1)
            if temp < 0:
                temp = -temp
            pt_source = np.append(pt_source,temp)
        pt_target = []
        for i in range(int(num)):
            temp =  np.random.normal(loc=100,scale = 40,size = 1)
            if temp < 0:
                temp = -temp
            pt_target = np.append(pt_target,temp)
        pt_background = []
        for i in range(int(num)):
            temp =  np.random.exponential(scale = 50, size = 1)
            #temp =  np.random.normal(loc=0, scale = 8,size = 1)
            if temp < 0:
                temp = -temp
            pt_background = np.append(pt_background,temp)
        
        return pt_source, pt_target, pt_background
    range_data = [0, 200]
    
    def generate_dataframe_pt(pt_source:np.array, pt_background:np.array):
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

    
        pt_source = pd.DataFrame({'pt': pd.Series(pt_source)})
        pt_background = pd.DataFrame({'pt': pd.Series(pt_background)})
        
        pt = pd.concat([pt_source, pt_background])
        pt.reset_index(drop=True, inplace=True)
        #f2 = pd.concat([f2_source, f2_background])
        f = pd.DataFrame({'pt':pt['pt'],'class_id': class_id['class_id']}, index=range(len(class_id)))
    
        return f

    pt_source, pt_target, pt_background = generate_pt_(num)
    f_train_sb = generate_dataframe_pt(pt_source,pt_background)
    f_train_tb = generate_dataframe_pt(pt_target,pt_background)
    pt_source, pt_target, pt_background = generate_pt_(num)
    f_test_sb = generate_dataframe_pt(pt_source,pt_background)
    f_test_tb = generate_dataframe_pt(pt_target,pt_background)
    #plt.hist(pt_source, bins=100, range=range_data,density = True ,color ='r',histtype='step',label = 'pt_source')
    #plt.hist(pt_target, bins=100, range=range_data ,density = True,color ='b',histtype='step',label = 'pt_target')
    #plt.hist(pt_background, bins=100, range=range_data ,density = True,color ='g',histtype='step',label = 'pt_background')

    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.legend()
    #plt.show()
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
    weight_sb[0:num] = weight_sb[0:num] / np.sum(weight_sb[0:num]) * np.sum(weight_sb[num:2 * num])
    ###########################################################################
    keras_model_ = build_model()
    keras_model_.fit(f_train_tb, train_y_tb, epochs=50, batch_size=500, verbose=1,shuffle=True)

    from sklearn.metrics import roc_curve
    y_pred_keras_ = keras_model_.predict(f_test_tb)
    weight_tb = np.divide(  1-y_pred_keras_,y_pred_keras_)
    weight_tb[0:num] = weight_tb[0:num] / np.sum(weight_tb[0:num]) * np.sum(weight_tb[num:2 * num])

    #plt.hist(pt_source, bins=100, range=[0,200],density = True ,color ='r',histtype='step',label = 'pt_source', weights= weight_sb[0:num])
    #plt.hist(pt_target, bins=100, range=[0,200] ,density = True,color ='b',histtype='step',label = 'pt_target', weights= weight_tb[0:num])
    #plt.hist(pt_background, bins=100, range=[0,200] ,density = True,color ='g',histtype='step',label = 'pt_background')
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.legend()
    #plt.show()
    return pt_source, pt_target, pt_background, weight_sb, weight_tb

   
    ##################################################generate data#################################################
def build_model():
    tuner = tfdf.tuner.RandomSearch(num_trials=20)

    # Hyper-parameters to optimize.
    #tuner.discret("max_depth", [4, 5, 6, 7])

    model = tfdf.keras.RandomForestModel(tuner=tuner)
    # model = Sequential()
    # model.add(Dense(8, input_dim=3, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # #model.add(Dense(160, activation='relu'))
    # #model.add(Dense(20, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    # # Compile model
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    '''
    inputs = Input(shape=(3,))
    hidden = Dense(n_nodes_clf, activation='relu')(inputs)

    for i in range(n_hidden_clf -1):
        hidden = Dense(n_nodes_clf, activation='relu')(hidden) 
    predictions = Dense(1, activation='softmax')(hidden)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy())
    '''
    return model
   
def generate_f1(pt_source: np.array, pt_target: np.array , pt_background: np.array, num): 
    theta = np.random.normal(loc = 0, scale = math.pi/3, size = len(pt_background))
    f1_background = np.power(pt_background,0.2)
    f1_background = f1_background - np.multiply(pt_background,(np.sin(theta)-0.5))
    f1_background = f1_background +50

    theta = np.random.normal(loc = 0, scale = math.pi/3, size = len(pt_source))
    f1_source = np.power(pt_source,0.2)
    f1_source = np.add(f1_source, (1)*np.multiply(pt_source,(np.sin(theta)-0.5)))
    f1_source = f1_source +50

    theta = np.random.normal(loc = 0, scale = math.pi/3, size = len(pt_target))
    f1_target = np.power(pt_target,0.2)
    f1_target = np.add(f1_target, (1)*np.multiply(pt_target,(np.sin(theta)-0.5)))
    f1_target = f1_target + 50

    #plt.hist(f1_source, bins=100, range=[-100,200],density = True ,color ='r',histtype='step',label = 'f1_source',weights=weight_sb[0:num])

    #plt.hist(f1_target, bins=100, range=[-100,200],density = True ,color ='g',histtype='step',label = 'f1_target',weights=weight_tb[0:num])

    #plt.hist(f1_background, bins=100, range=[-100,200],density = True ,color ='b',histtype='step',label = 'f1_background')

    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.legend()
    #plt.show()

    return  f1_source, f1_target,f1_background

def generate_f2(pt_source: np.array, pt_target: np.array , pt_background: np.array, num): 
    np.random.seed(int(time.time()))
    theta = np.random.normal(loc = 0, scale = math.pi/3, size = len(pt_background))
    f2_background = np.power(pt_background,1.1)
    f2_background = np.add(f2_background, (-1)*np.multiply(pt_background,(np.cos(theta)-0.5)))
    f2_background = f2_background +5

    theta = np.random.normal(loc = 0, scale = math.pi/3, size = len(pt_source))
    f2_source = np.power(pt_source,1.1)
    f2_source = np.add(f2_source, (1)*np.multiply(pt_source,(np.cos(theta)-0.5)))
    f2_source = f2_source +5

    theta = np.random.normal(loc = 0, scale = math.pi/3, size = len(pt_target))
    f2_target = np.power(pt_target,1.1)
    f2_target = np.add(f2_target, (1)*np.multiply(pt_target,(np.cos(theta)-0.5)))
    f2_target = f2_target + 5
    range_plot = [0, 250]
    #plt.hist(f2_source, bins=100, range=range_plot,density = True ,color ='r',histtype='step',label = 'f2_source', weights=weight_sb[0:num])

    #plt.hist(f2_target, bins=100, range=range_plot,density = True ,color ='g',histtype='step',label = 'f2_target', weights=weight_tb[0:num])

    #plt.hist(f2_background, bins=100, range=range_plot,density = True ,color ='b',histtype='step',label = 'f2_background')

    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.legend()
    #plt.show()

    return  f2_source, f2_target,f2_background

def generate_dataframe(f1_source_:np.array, f1_background_:np.array, f2_source_:np.array, f2_background_:np.array, pt_source_:np.array, pt_background_):
    id = np.ones([len(pt_source_),1])
    id = id.astype(int)
    id = id.flatten()
    id = pd.DataFrame({'class_id':id})
    id_b = np.zeros([len(pt_background_),1])
    id_b = id_b.astype(int)
    id_b = id_b.flatten()
    id_b = pd.DataFrame({'class_id':pd.Series(id_b)})
    class_id = pd.concat([id,id_b])
    class_id.reset_index(drop=True, inplace=True)

    f1_source_ = pd.DataFrame({'x1': pd.Series(f1_source_)})
    f2_source_ = pd.DataFrame({'x2': pd.Series(f2_source_)})
    pt_source_ = pd.DataFrame({'pt': pd.Series(pt_source_)})
    #f1_source.insert(loc = 2, column = 'class_ID',value = 1)
    f1_background_ = pd.DataFrame({'x1': pd.Series(f1_background_)})
    f2_background_ = pd.DataFrame({'x2': pd.Series(f2_background_)})
    pt_background_ = pd.DataFrame({'pt': pd.Series(pt_background_)})
    #f1_background.insert(loc = 2, column = 'class_ID',value = 0)
    f1 = pd.concat([f1_source_, f1_background_])
    f1.reset_index(drop = True, inplace=True)
    f2 = pd.concat([f2_source_, f2_background_])
    f2.reset_index(drop = True, inplace=True)
    pt = pd.concat([pt_source_, pt_background_])
    pt.reset_index(drop=True, inplace=True)
    #f2 = pd.concat([f2_source, f2_background])
    f = pd.DataFrame({'x1':f1['x1'],'x2':f2['x2'],'pt':pt['pt'],'class_id': class_id['class_id']}, index=range(len(class_id)))
    return f

if __name__ == "__main__":

    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=3,verbose=0,mode="auto",baseline=None,restore_best_weights=False)

    for i_num in  [0.1, 0.5, 1, 2, 4, 8, 20, 50, 100, 500]:
        n_hidden_clf = 10
        n_nodes_clf = 32
        num = i_num*10**3
        num = int(num)
        num_noi = num

        
        pt_source, pt_target, pt_background, weight_sb, weight_tb = generate_pt(num)
        f1_source, f1_target,f1_background = generate_f1(pt_source, pt_target, pt_background, num)
        f2_source, f2_target,f2_background = generate_f2(pt_source, pt_target, pt_background, num)
        f_train = generate_dataframe(f1_source, f1_background, f2_source, f2_background,pt_source,pt_background)
        keras_model = build_model()
        #train_y = f_train.pop('class_id')
        #keras_model.fit(f_train, train_y, epochs=400, batch_size=500, verbose=1,shuffle=True,sample_weight=weight_sb,validation_split = 0.2,callbacks=[callback])
        f_train.insert(loc=0, column='weight', value=weight_sb)
        keras_model.fit(tfdf.keras.pd_dataframe_to_tf_dataset(f_train, label="class_id",weight='weight'),shuffle=True)

        pt_source, pt_target, pt_background, weight_sb, weight_tb = generate_pt(500000)
        f1_source, f1_target,f1_background = generate_f1(pt_source, pt_target, pt_background, 500000)
        f2_source, f2_target,f2_background = generate_f2(pt_source, pt_target, pt_background, 500000)
        f_test = generate_dataframe(f1_target, f1_background, f2_target, f2_background,pt_target,pt_background)
        f_test.dropna()
        f_train.dropna()

        #train_y = np.array(train_y)
        

        from sklearn.metrics import roc_curve
        #y_pred_keras = keras_model.predict(f_test)
        y_pred_keras = keras_model.predict(tfdf.keras.pd_dataframe_to_tf_dataset(f_test, label="class_id"))
        test_y = f_test.pop('class_id')
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_y, y_pred_keras)

        from sklearn.metrics import auc
        auc_st = auc(fpr_keras, tpr_keras)

        from sklearn.ensemble import RandomForestClassifier
        # Supervised transformation based on random forests

        ########################################################################
        pt_source, pt_target, pt_background, weight_sb, weight_tb = generate_pt(num)
        f1_source, f1_target,f1_background = generate_f1(pt_source, pt_target, pt_background, num)
        f2_source, f2_target,f2_background = generate_f2(pt_source, pt_target, pt_background, num)
        f_train = generate_dataframe(f1_target, f1_background, f2_target, f2_background,pt_target,pt_background)
        #train_y = f_train.pop('class_id')
        keras_model_tt = build_model()
        f_train.insert(loc=0, column='weight', value=weight_tb)
        keras_model_tt.fit(tfdf.keras.pd_dataframe_to_tf_dataset(f_train, label="class_id",weight='weight'),shuffle=True)
        #keras_model_tt.fit(f_train, train_y, epochs=400, batch_size=500, verbose=1,sample_weight=weight_tb,shuffle=True)#,validation_split = 0.2)#,callbacks=[callback])
        pt_source, pt_target, pt_background, weight_sb, weight_tb = generate_pt(500000)
        f1_source, f1_target,f1_background = generate_f1(pt_source, pt_target, pt_background, 500000)
        f2_source, f2_target,f2_background = generate_f2(pt_source, pt_target, pt_background, 500000)
        f_test = generate_dataframe(f1_target, f1_background, f2_target, f2_background,pt_target,pt_background)
        f_test.dropna()
        f_train.dropna()
        
        from sklearn.metrics import roc_curve
        #y_pred_keras = keras_model_tt.predict(f_test)
        y_pred_keras = keras_model_tt.predict(tfdf.keras.pd_dataframe_to_tf_dataset(f_test, label="class_id"))
        test_y = f_test.pop('class_id')
        fpr_tt, tpr_tt, thresholds_tt = roc_curve(test_y, y_pred_keras)

        from sklearn.metrics import auc
        auc_tt = auc(fpr_tt, tpr_tt)



        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras, tpr_keras, label='source-target (area = {:.3f})'.format(auc_st))
        plt.plot(fpr_tt, tpr_tt, label='target-target (area = {:.3f})'.format(auc_tt),linestyle = 'dashed')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig('/mnt/d/DW_project/num/0530/' + str(num) + 'para' + '.png')
        plt.close()
        # Zoom in view of the upper left corner.