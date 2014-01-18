from __future__ import division
from Randomforestlearner import Randomforestlearner
import numpy
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import math

def get_graph_two_plots(x_series,y_series,y1_series,xlabel,ylabel,name_file):
    plt.clf()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    first = plt.plot(x_series,y_series,color='r')
    second = plt.plot(x_series,y1_series,color='b')
    plt.legend(["Ypredict","Yactual"])
    plt.savefig(name_file)

def get_correlation(Y_return,Y_test):
    covariance_matrix = np.corrcoef(Y_return, np.squeeze(np.asarray(Y_test)))
    return covariance_matrix[0, 1]

def get_rmse(Y_return,Y_Test):
    return math.sqrt(np.mean(np.square(Y_return-Y_Test)))

def scatter(Y_return,Y_test,name_file):
    plt.clf()
    fig = plt.figure(figsize=(6, 5))
    graph = fig.add_subplot(1,1,1)
    graph.scatter(Y_return,Y_test)
    graph.set_title("predicted Y vs actual Y")
    graph.set_xlabel="Days"
    graph.set_ylabel="Y"
    fig.savefig(name_file)

def all_feature_graph(y1,y2,y3,y4,y5,x,xlabel,ylabel,name_file):
    plt.clf()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    first = plt.plot(x,y1)
    second = plt.plot(x,y2)
    third = plt.plot(x,y3)
    fourth = plt.plot(x,y4)
    fifth = plt.plot(x,y5)
    plt.legend(["mean","stddev","rsi","roc","slope"])
    plt.savefig(name_file)

def read_file(filename):
    required_all=list()
    inf = open(filename)
    for s in reversed(inf.readlines()):
        all_needed = s.split(',')
        required = all_needed[:2]
        if(all_needed[0]!='Date'):
            required_all.append(float(required[1]))
    return required_all

def get_mean(dataset_21):
    return numpy.mean(dataset_21)

def get_stddev(dataset_21):
    return numpy.std(dataset_21)

def get_relative_strength_idx(dataset_21):
    comparer = dataset_21[0]
    gain =0.0
    loss =0.0
    new_dataset= dataset_21[0:]
    for each in new_dataset:
        if each > comparer:
            gain = gain + (each-comparer)
        elif each < comparer:
            loss = loss + (comparer-each)
    if loss ==0.0:
        return 100.0
    elif gain ==0.0:
        return 0.0
    else:
        rs = (gain/len(new_dataset))/(loss/len(new_dataset))
        rsi = float(100 - (100/(1+rs)))
        return rsi

def roc(dataset_21):
    latest = dataset_21[len(dataset_21)-1]
    oldest = dataset_21[0]
    ROC = (latest - oldest) / (oldest) * 100  
    return ROC


def slope(dataset_21):
    x = numpy.array([i for i in range(1,22)])
    y = numpy.array([data for data in dataset_21])
    A = numpy.vstack([x, numpy.ones(len(x))]).T
    m, c = numpy.linalg.lstsq(A, y)[0]
    return m

def createX(datesandY):
    totalX =[]
    index = 21
    start = 0
    while index <=len(datesandY)-5:
        dataset_21 = datesandY[start:index]
        totalX.append([get_mean(dataset_21),get_stddev(dataset_21),get_relative_strength_idx(dataset_21),roc(dataset_21),slope(dataset_21),datesandY[index+4]-datesandY[index]])
        start = start+1
        index= index+1
    return numpy.array(totalX)
def main():
    curr_dirr = os.getcwd()
    os.chdir('proj3-data-fixed')
    datesandY = read_file('ML4T-000.csv')
    stack = createX(datesandY)
    # print len(stack)
    for i in range(1,10):
        datesandY = read_file('ML4T-00'+str(i)+'.csv')
        learnedvalues = createX(datesandY)
        stack = numpy.vstack((stack,learnedvalues))
    for i in range(11,100):
        datesandY = read_file('ML4T-0'+str(i)+'.csv')
        learnedvalues = createX(datesandY)
        stack = numpy.vstack((stack,learnedvalues))
    print len(stack)
    testdatesandY = read_file('ML4T-292.csv')
    test = createX(testdatesandY)
    (XTrain,YTrain) = numpy.split(stack,[5],axis=1)
    (XTest,YTest) = numpy.split(test,[5],axis=1)
    # print XTest
    randomforestlearner = Randomforestlearner(k=50)
    randomforestlearner.addEvidence(XTrain,YTrain)
    Y_Return = numpy.multiply(numpy.array(randomforestlearner.query(XTest)),-1)
    Y_Test = np.squeeze(np.asarray(YTest))
    
    start=0
    index=5
    print len(Y_Return)
    print len(Y_Test)
    print len(testdatesandY)

    while index<len(testdatesandY)-26:
        Y_Return[start]=Y_Return[start]+testdatesandY[index]
        Y_Test[start]=Y_Test[start]+testdatesandY[index]
        start = start+1
        index=index+1
    os.chdir(curr_dirr)
    get_graph_two_plots(numpy.arange(1,101),Y_Return[:100],Y_Test[:100],"Days","Y","YpredictvsYactual_292_first100.jpg")
    last126_test = Y_Test[-126:]
    last126_return = Y_Return[-126:]
    get_graph_two_plots(numpy.arange(1,101),last126_return[:100],last126_test[:100],"Days","Y","YpredictvsYactual_292_last100.jpg")
    scatter(Y_Return,Y_Test,"scatterplot_292.jpg")
    mean_series = XTest[:,0]
    std_series =XTest[:,1]
    rsi_series = XTest[:,2]
    roc_series = XTest[:,3]
    slope_series = XTest[:,4]
    all_feature_graph(mean_series[:100],std_series[:100],rsi_series[:100],roc_series[:100],slope_series[:100],numpy.arange(1,101),"Days","Features","Allfeature_292.jpg")
    print "Correlation 292 is {0}".format(get_correlation(Y_Test,Y_Return))
    print "RMSE 292 is {0}".format(get_rmse(Y_Test,Y_Return))
    os.chdir('proj3-data-fixed')
    testdatesandY = read_file('ML4T-132.csv')
    test = createX(testdatesandY)
    (XTrain,YTrain) = numpy.split(stack,[5],axis=1)
    (XTest,YTest) = numpy.split(test,[5],axis=1)
    # print XTest
    randomforestlearner = Randomforestlearner(k=50)
    randomforestlearner.addEvidence(XTrain,YTrain)
    Y_Return = numpy.multiply(numpy.array(randomforestlearner.query(XTest)),-1)
    Y_Test = np.squeeze(np.asarray(YTest))
    
    start=0
    index=5
    print len(Y_Return)
    print len(Y_Test)
    print len(testdatesandY)

    while index<len(testdatesandY)-26:
        Y_Return[start]=Y_Return[start]+testdatesandY[index]
        Y_Test[start]=Y_Test[start]+testdatesandY[index]
        start = start+1
        index=index+1
    os.chdir(curr_dirr)
    get_graph_two_plots(numpy.arange(1,101),Y_Return[:100],Y_Test[:100],"Days","Y","YpredictvsYactual_132_first100.jpg")
    last126_test = Y_Test[-126:]
    last126_return = Y_Return[-126:]
    get_graph_two_plots(numpy.arange(1,101),last126_return[:100],last126_test[:100],"Days","Y","YpredictvsYactual_132_last100.jpg")
    scatter(Y_Return,Y_Test,"scatterplot_132.jpg")
    mean_series = XTest[:,0]
    std_series =XTest[:,1]
    rsi_series = XTest[:,2]
    roc_series = XTest[:,3]
    slope_series = XTest[:,4]
    all_feature_graph(mean_series[:100],std_series[:100],rsi_series[:100],roc_series[:100],slope_series[:100],numpy.arange(1,101),"Days","Features","Allfeature_132.jpg")
    print "Correlation 132 is {0}".format(get_correlation(Y_Test,Y_Return))
    print "RMSE 132 is {0}".format(get_rmse(Y_Test,Y_Return))
if __name__ == '__main__':
    main()

