from KNNLearner import KNNLearner
from LinRegLearner import LinRegLearner
import numpy
import numpy as np
import io,sys,time
import math
import matplotlib.pyplot as plt
class Timer: 
    def __enter__(self):
        self.start = time.clock()
        return self


    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

def get_graph(x_series,y_series,xlabel,ylabel,name_file,ylim=None):
    plt.clf()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim:
        plt.ylim(0,ylim)
    plt.plot(x_series,y_series)
    plt.savefig(name_file)

def get_graph_two_plots(x_series,y_series,y1_series,xlabel,ylabel,name_file):
    plt.clf()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    first = plt.plot(x_series,y_series)
    second = plt.plot(x_series,y1_series)
    plt.legend(["insample","outsample"])
    plt.savefig(name_file)

def scatter(Y_return,Y_test,name_file):
    plt.clf()
    fig = plt.figure(figsize=(6, 5))
    graph = fig.add_subplot(1,1,1)
    graph.scatter(Y_return,Y_test)
    graph.set_title("predicted Y versus actual Y for for the best K")
    graph.set_xlabel="K"
    graph.set_ylabel="Y"
    fig.savefig(name_file)

def get_rmse(Y_return,Y_Test):
    return math.sqrt(np.mean(np.square(Y_return-Y_Test)))

def get_correlation(Y_return,Y_test):
    covariance_matrix = np.corrcoef(Y_return, np.squeeze(np.asarray(Y_test)))
    return covariance_matrix[0, 1]

def linreglearner_test(filenames):
    for filename in filenames:
        linreglearner = LinRegLearner()
        get_set = linreglearner.getflatcsv(filename)
        get_set_60pr,get_set_40pr = numpy.split(get_set,[600])
        (X,Y) = numpy.split(get_set,[2],axis=1)
        (XTrain,XTest) = numpy.split(X,[600])
        (Ytrain,YTest) = numpy.split(Y,[600])
        with Timer() as t:
            linreglearner.addEvidence(XTrain,Ytrain)
        print 'LinRegLearner Training Time {0} sec for {1}'.format(t.interval,filename)
        with Timer() as t:
            Y_return = linreglearner.query(XTest)
        print 'LinRegLearner Query Time {0} sec for {1}'.format(t.interval,filename)

def knnlearner_test(filenames):
    for filename in filenames:
        train_time =[]
        query_time =[]
        rmse_series=[]
        rmse_series_insample=[]
        covariance_series=[]
        for i in xrange(1,51):
            knnlearner=KNNLearner(k=i)
            get_set = knnlearner.getflatcsv(filename)
            get_set_60pr,get_set_40pr = numpy.split(get_set,[600])
            (X,Y) = numpy.split(get_set,[2],axis=1)
            (XTrain,XTest) = numpy.split(X,[600])
            (Ytrain,YTest) = numpy.split(Y,[600])
            knnlearner.build_hash(get_set_60pr)
            with Timer() as t:
                knnlearner.addEvidence(XTrain,Ytrain)
            train_time.append(t.interval)
            query_X = numpy.array(XTest)
            with Timer() as t:
                (XY_return,Y_return) = knnlearner.query(XTest)
            query_time.append(t.interval)
            Y_Test = np.squeeze(np.asarray(YTest))
            Y_Return = numpy.array(Y_return)
            rmse_series.append(get_rmse(Y_Test,Y_Return))
            (XY_return_insample,Y_return_insample) = knnlearner.query(XTrain)
            Y_Train = np.squeeze(np.asarray(Ytrain))
            Y_return_insample = numpy.array(Y_return_insample)
            rmse_series_insample.append(get_rmse(Y_Train,Y_return_insample))
            covariance_series.append(get_correlation(Y_Test,Y_Return))
        min_rmse = min(float(i) for i in rmse_series)
        k_index = rmse_series.index(min_rmse)
        print "best k = ",k_index+1," for ",filename
        knnlearner_scatter = KNNLearner(k=k_index+1)
        get_set = knnlearner_scatter.getflatcsv(filename)
        get_set_60pr,get_set_40pr = numpy.split(get_set,[600])
        (X,Y) = numpy.split(get_set,[2],axis=1)
        (XTrain,XTest) = numpy.split(X,[600])
        (Ytrain,YTest) = numpy.split(Y,[600])
        knnlearner_scatter.build_hash(get_set_60pr)
        knnlearner_scatter.addEvidence(XTrain,Ytrain)
        (XY_return,Y_return) = knnlearner_scatter.query(XTest)
        Y_Test = np.squeeze(np.asarray(YTest))
        Y_Return = numpy.array(Y_return)
        scatter(Y_Return,Y_Test,"scatterplot("+filename+")(for bestk).pdf")
        get_graph(numpy.arange(1,51),train_time,"K","Train time in seconds","KNN_Train_time("+filename+").pdf",4)
        get_graph(numpy.arange(1,51),query_time,"K","Query time in seconds","KNN_Query_time("+filename+").pdf",4)
        get_graph(numpy.arange(1,51),rmse_series,"K","RMSE Error","RMSEvsk("+filename+").pdf")
        get_graph(numpy.arange(1,51),covariance_series,"K","Covariance Coefficeint","Covariance Coeff vs K("+filename+").pdf")
        get_graph_two_plots(numpy.arange(1,51),rmse_series_insample,rmse_series,"K","RMSE","insample_error_vs_outsample_error("+filename+").pdf")

def main():
    knnlearner_test(['data-classification-prob.csv','data-ripple-prob.csv'])
    linreglearner_test(['data-classification-prob.csv','data-ripple-prob.csv'])

    

if __name__ == '__main__':
    main()