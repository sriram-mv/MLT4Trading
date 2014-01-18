from KNNLearner import KNNLearner
from Randomforestlearner import Randomforestlearner
import numpy
import numpy as np
import io,sys,time
import math
import matplotlib.pyplot as plt

def get_graph_two_plots_randomvsknn(x_series,y_series,y1_series,xlabel,ylabel,name_file):
    plt.clf()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    first = plt.plot(x_series,y_series)
    second = plt.plot(x_series,y1_series)
    plt.legend(["RF","KNN"])
    plt.savefig(name_file)

def get_rmse(Y_return,Y_Test):
    return math.sqrt(np.mean(np.square(Y_return-Y_Test)))

def get_correlation(Y_return,Y_test):
    covariance_matrix = np.corrcoef(Y_return, np.squeeze(np.asarray(Y_test)))
    return covariance_matrix[0, 1]

def knnlearner_test(filenames):
    for filename in filenames:
        rmse_series=[]
        covariance_series=[]
        for i in xrange(1,101):
            knnlearner=KNNLearner(k=i)
            get_set = knnlearner.getflatcsv(filename)
            get_set_60pr,get_set_40pr = numpy.split(get_set,[600])
            (X,Y) = numpy.split(get_set,[2],axis=1)
            (XTrain,XTest) = numpy.split(X,[600])
            (Ytrain,YTest) = numpy.split(Y,[600])
            knnlearner.build_hash(get_set_60pr)
            knnlearner.addEvidence(XTrain,Ytrain)
            query_X = numpy.array(XTest)
            (XY_return,Y_return) = knnlearner.query(XTest)
            Y_Test = np.squeeze(np.asarray(YTest))
            Y_Return = numpy.array(Y_return)
            rmse_series.append(get_rmse(Y_Test,Y_Return))
            covariance_series.append(get_correlation(Y_Test,Y_Return))
    return (rmse_series,covariance_series)

def randomforest_test(filenames):
    for filename in filenames:
        rmse_series_randomforest=[]
        covariance_series_randomforest=[]
        for k in range (1,101):
            randomforestlearner = Randomforestlearner(k=k)
            get_set = randomforestlearner.getflatcsv(filename)
            get_set_60pr,get_set_40pr = numpy.split(get_set,[600])
            (X,Y) = numpy.split(get_set,[2],axis=1)
            (XTrain,XTest) = numpy.split(X,[600])
            (Ytrain,YTest) = numpy.split(Y,[600])
            Y_Test = np.squeeze(np.asarray(YTest))
            randomforestlearner.addEvidence(XTrain,Ytrain)
            Y_Return = numpy.array(randomforestlearner.query(XTest))
            rmse_series_randomforest.append(get_rmse(Y_Test,Y_Return))
            covariance_series_randomforest.append(get_correlation(Y_Test,Y_Return))
    return (rmse_series_randomforest,covariance_series_randomforest)

def main():
    (rmse_series_randomforest,covariance_series_randomforest) = randomforest_test(['data-classification-prob.csv'])
    (rmse_series,covariance_series) = knnlearner_test(['data-classification-prob.csv'])
    get_graph_two_plots_randomvsknn(numpy.arange(1,101),rmse_series_randomforest,rmse_series,"K","RMSE","RMSEvsk(data-classification-prob)RFvsKNN.jpg")
    get_graph_two_plots_randomvsknn(numpy.arange(1,101),covariance_series_randomforest,covariance_series,"K","Correlation","Correlationvsk(data-classification-prob)RFvsKNN.jpg")
    (rmse_series_randomforest,covariance_series_randomforest) = randomforest_test(['data-ripple-prob.csv'])
    (rmse_series,covariance_series) = knnlearner_test(['data-ripple-prob.csv'])
    get_graph_two_plots_randomvsknn(numpy.arange(1,101),rmse_series_randomforest,rmse_series,"K","RMSE","RMSEvsk(data-ripple-prob)RFvsKNN.jpg")
    get_graph_two_plots_randomvsknn(numpy.arange(1,101),covariance_series_randomforest,covariance_series,"K","Correlation","Correlationvsk(data-ripple-prob)RFvsKNN.jpg")


    

if __name__ == '__main__':
    main()