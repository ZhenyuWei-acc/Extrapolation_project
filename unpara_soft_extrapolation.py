#The value of ‘l’ in the equations for F1 and F2 is +1 for signal and -1 for background. Can you try this toy problem described in the slides?
#F1 = pt**(0.2)+l*pt*(sintheta-0.5)+50
#F2 = pt**(1.1)+l*pt*(costheta-0.5)+5
#theta ~ normal distribution (miu = 0, sigma = pi/3)
#truncated normal (i.e. gaussian) distribution for source signal
#an exponential for the background 
#a gaussian distribution for the target signal
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
num = (10**3)
num_noi = 110*num
#I’m glad you asked. Yes you have a good understanding of the context. One clarification, #for the third feature, Pt, we haven’t given you the distributions explicitly because the #exact choice doesn’t matter, but for the plots I used truncated normal (i.e. gaussian) #distribution for source signal, an exponential for the background and just a gaussian #distribution for the target signal. You can do the same. Since F1 and F2 partially #depend on Pt, you’ll need to generate the Pt distributions first. The theta is sampled #always from the same distribution (the gaussian as described in the slides) for all #three classes.

#Why is re-weighting important?
#When you train the parameterised classifier, the network will have three inputs(F1, F2 #and Pt), and it may simply use the difference between the Pt distribution of the source #signal and background to make a decision. However this decision function will no longer #work to separate target signal and background. This is why one needs to mask the #discriminating power of Pt itself.
#If you prefer, you can first try without the re-weighing and see the results you get.

def generate_pt(num: np.float64):
    np.random.seed(int(time.time()))
    pt_source = []
    for i in range(num):
        temp =  np.random.normal(loc=0,scale = 4.8,size = 1)
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

    #plt.hist(f2_source, bins=100, range=[0,80],density = True ,color ='r',histtype='step',label = 'f2_source')

    #plt.hist(f2_target, bins=100, range=[0,80],density = True ,color ='g',histtype='step',label = 'f2_target')

    #plt.hist(f2_background, bins=100, range=[0,80],density = True ,color ='b',histtype='step',label = 'f2_background')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    return  f2_source, f2_target,f2_background

def generate_dataframe(f_source:np.array, f_background:np.array):
    id = np.ones([num,1])
    id = id.astype(int)
    id = id.flatten()
    id = pd.DataFrame({'class_id':id})
    id_b = np.zeros([num_noi,1])
    id_b = id_b.astype(int)
    id_b = id_b.flatten()
    id_b = pd.DataFrame({'class_id':pd.Series(id_b)})
    class_id = pd.concat([id,id_b])

    f_source = pd.DataFrame({'x': pd.Series(f1_source)})
    #f2_source = pd.DataFrame({'x2': pd.Series(f2_source)})
    #f1_source.insert(loc = 2, column = 'class_ID',value = 1)
    f_background = pd.DataFrame({'x': pd.Series(f1_background)})
    #f2_background = pd.DataFrame({'x2': pd.Series(f2_background)})
    #f1_background.insert(loc = 2, column = 'class_ID',value = 0)
    f = pd.concat([f_source, f_background])
    #f2 = pd.concat([f2_source, f2_background])
    f = pd.DataFrame({'x':f['x'],'class_id': class_id['class_id']})
    f = f.sample(frac = 1).reset_index(drop=True)
    f = f.sample(frac = 1).reset_index(drop=True)
    f = f.sample(frac = 1).reset_index(drop=True)
    f = f.sample(frac = 1).reset_index(drop=True)
    return f


pt_source, pt_target, pt_background = generate_pt(num)
f1_source, f1_target,f1_background = generate_f1(pt_source, pt_target, pt_background)
f2_source, f2_target,f2_background = generate_f2(pt_source, pt_target, pt_background)
f_train = generate_dataframe(f2_target, f2_background)
f_test = generate_dataframe(f2_target,f2_background)

def plot_norm_hist(data_plot:pd.DataFrame):
    plt.figure()
   
    data_noise=data_plot[data_plot['class_id']==0].reset_index(drop=True)
    data_signal=data_plot[data_plot['class_id']==1].reset_index(drop=True)
    ds = plt.hist(data_signal['x'], bins=100, range=[0,80],density = True ,color ='r',histtype='step',label = 'signal')
    
    dn = plt.hist(data_noise['x'], bins=100, range=[0,80] ,density = True,color ='b',histtype='step',label = 'noise')
    #print(dn)
    #print(ds)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
    return 0
##plot_norm_hist(f1_train)
##plot_norm_hist(f1_test)
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

def input_fn_predic(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

COLUMN_NAMES = ['class_id']
SPECIES = [0, 1]

#step two: input data
train_y = f_train.pop('class_id')
train_y = train_y.astype(int)
test_y = f_test.pop('class_id')
test_y = test_y.astype(int)
my_feature_columns = []

for key in f_train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)

classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    #n_batches_per_layer=5,
    # Two hidden layers of 30 and 10 nodes respectively.
    #linear_optimizer='SGD',
    hidden_units= [30,10],
    # The model must choose between 2 classes.
    n_classes=2)

classifier.train(
    input_fn=lambda: input_fn(f_train, train_y, training=True),
    steps=5000)

eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(f_test, test_y, training=False))
#accu = np.append(accu,eval_result['accuracy'])
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

features = ['x']
n=10000
predict = f_test.head(n)
fact = test_y.head(n)
class_ID = np.zeros(n).astype(int)
probability = np.zeros(n).astype(int)
predict = pd.DataFrame({'x': predict['x'],'class_ID':class_ID,'probability':probability,'fact':fact})
'''
predict = f2_test
fact = test_y
class_ID = np.zeros(len(predict)).astype(int)
probability = np.zeros(len(predict)).astype(int)
predict = pd.DataFrame({'x': predict['x'],'class_ID':class_ID,'probability':probability,'fact':fact})
'''
#print("Please type numeric values as prompted.")
#for feature in features:
#    valid = True
#    while valid:
#        val = input(feature + ": ")
#        if not val.isdigit():
#            valid = False
#    predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn_predic(predict))
#plt.figure()
i = 0

for pred_dict in predictions:
    
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    predict['class_ID'][i] = SPECIES[class_id]
    predict['probability'][i] =  100*probability

    print('Prediction is "{}" ({:.1f}%)'.format(
    SPECIES[class_id], 100*probability))
    i = i+1

def plot_predict(prediction: pd.DataFrame):
    bins_center = []
    range_data = [math.floor(min(prediction['x'])),math.ceil(max(prediction['x']))]
    bin_num = range_data[1] - range_data[0]
    delta = np.abs(range_data[1]-range_data[0])/(bin_num)
    print(bin_num)
    for j in range(bin_num):
        bins_center = np.append(bins_center, range_data[0]+delta/2+j*delta)
    
    predict_plot = pd.DataFrame({'x': prediction['x'],'class_ID':prediction['class_ID'],'probability':prediction['probability'],'batch_ID':pd.Series( np.zeros(len(prediction['x'])) ) } )
    predict_plot.sort_values(by=['x'],ascending=True,inplace=True)
    y_signal = []
    y_noise = []
    for m in range(bin_num):
        range_batch = [bins_center[m]-delta/2,bins_center[m]+delta/2]    
        
        for k in range(len(prediction['x'])):
            temp_signal = 0
            temp_noise = 0
            if prediction['class_ID'][k] == 0:
                prediction['class_ID'][k] = 1
                prediction['probability'][k] = 100-prediction['probability'][k]

            if prediction['x'][k] > range_batch[0] and prediction['x'][k] <= range_batch[1]:
                #print(prediction['x'][k])
                predict_plot['batch_ID'][k] = m
    predict_plot = predict_plot.groupby(['batch_ID']).sum()
    x = np.linspace(range_data[0],range_data[1],len(predict_plot['probability']))
    y_signal = predict_plot['probability'] / sum(predict_plot['probability'])
    y_noise = max(y_signal)-y_signal / sum(max(y_signal)-y_signal)
    plt.bar(x-0.5,y_signal, label = 'signal distribution', fill = False, edgecolor = 'b')
    plt.bar(x+0.5,y_noise, label = 'nose distribution',fill = False, edgecolor = 'r')
    plt.legend()
    plt.ylabel('probability distribution')
    plt.xlabel('x')
    return 0

def plot_predict_roc(prediction: pd.DataFrame):
    thres = np.linspace(0,100,400)
    FPR = []
    TPR = []
    temp = prediction.groupby(['fact']).count()
    True_pos_num =temp['class_ID'][1]
    True_neg_num = len(prediction['x'])-True_pos_num
    for T in thres:
        fpr = 0
        tpr = 0
        for i in range(len(prediction['x'])):
            if ((prediction['class_ID'][i]==1 and prediction['probability'][i] > T) or (prediction['class_ID'][i]==0 and 100-prediction['probability'][i] > T)) and prediction['fact'][i]==0:
                fpr = fpr+1
            if ((prediction['class_ID'][i]==1 and prediction['probability'][i] > T) or (prediction['class_ID'][i]==0 and 100-prediction['probability'][i] > T)) and prediction['fact'][i]==1:
                tpr = tpr+1
        FPR = np.append(FPR,fpr/True_neg_num)
        TPR = np.append(TPR,tpr/True_pos_num)
    AUC = 0
    auc = 0
    for i in range(len(FPR)-1):
        AUC = AUC + abs((TPR[i]+TPR[i+1])*(FPR[i]-FPR[i+1])/2)
    print(TPR)
    print(FPR)
    fpr_, tpr_, _ = roc_curve(predict['class_ID'], predict['probability'])
    for i in range(len(fpr_)-1):
        auc = auc + abs((fpr_[i]+fpr_[i+1])*(tpr_[i]-tpr_[i+1])/2)
    print(fpr_)
    print(tpr_)
    print(_)
    print(auc)
    #plt.plot(tpr_,fpr_,'xb-',label = 'roc')
    plt.plot(FPR,TPR,'xb-',label = 'ROC')
    plt.plot(np.linspace(0,1,10),np.linspace(0,1,10),c='r',linestyle = '--',label = 'random gauss')
    plt.title("AUC = "+str(AUC))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.xlim = ([0,1])
    plt.ylim=([0,1])
    return TPR, FPR
TPR, FPR = plot_predict_roc(predict)
fpr, tpr, _ = roc_curve(predict['class_ID'], predict['probability'])
