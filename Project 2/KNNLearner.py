import math,random,sys,bisect,time
import numpy
import scipy
import math
import matplotlib
class KNNLearner(object):
    """The Main KNNLearner class"""
    def __init__(self, k):
        super(KNNLearner, self).__init__()
        self.k = k

    def addEvidence(self, XTrain, Ytrain):
        self.XTrain = XTrain
        self.Ytrain = Ytrain

    def query(self, XTes):
            final_Yresult = []
            for XTest in XTes:
                seconday_hash={}
                k_compare_list=[]
                final = numpy.absolute(numpy.array([XTest.__sub__(row) for row in self.XTrain]))
                final_distance = numpy.array([math.sqrt(math.pow(row[0],2)+(math.pow(row[1],2))) for row in final])
                for key,value in zip(final_distance,self.XTrain):
                    seconday_hash[key]=value
                sorted_final_distance = numpy.sort(final_distance)
                for i in xrange(0,self.k):
                    k_compare_list.append(seconday_hash[sorted_final_distance[i]])            
                k_neighbour = numpy.array([self.XtoYMap[tuple(value)] for value in k_compare_list])
                final_Yresult.append(k_neighbour.mean())
            final_XYresult = numpy.insert(XTes, 2, values=final_Yresult, axis=1)
            return (numpy.squeeze(numpy.asarray(final_XYresult)),final_Yresult)

    def getflatcsv(self,fname):
        inf = open(fname)
        return numpy.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    def build_hash(self, Total):
        new_dict ={}
        for element in Total:
            element_key = tuple(element[:2])
            element_value = tuple(element[2:])
            new_dict[element_key]=element_value
        self.XtoYMap = new_dict
