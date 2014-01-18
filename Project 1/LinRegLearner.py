import numpy as np
import numpy

class LinRegLearner(object):
    
    def __init__(self):
        super(LinRegLearner, self).__init__()

    def addEvidence(self, XTrain, Ytrain):
        self.XTrain=XTrain
        self.Ytrain=Ytrain
        final_XTrain = numpy.insert(XTrain, 0, values=np.ones(XTrain.shape[0]), axis=1)
        model = np.linalg.lstsq(final_XTrain,Ytrain)
        self.model = model[0]

    def query(self, XTest):
        final_XTest = numpy.insert(XTest, 0, values=np.ones(XTest.shape[0]), axis=1)
        final_result = np.dot(final_XTest,self.model)
        return final_result
                
    def getflatcsv(self,fname):
        inf = open(fname)
        return numpy.array([map(float,s.strip().split(',')) for s in inf.readlines()])