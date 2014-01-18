import numpy as np
import numpy
import random

FEATURE_NUMBER_step = 1
SPLIT_VAL_step = 2
LEFT_POINTER = 3
RIGHT_POINTER = 4
serial_no = 0
count =0

class Randomforestlearner(object):
    
    def __init__(self,k):
        super(Randomforestlearner, self).__init__()
        self.k = k
    def split(self,arr, cond):
        return [arr[cond], arr[~cond]]

    def combine_tree(self,current,left_subtree,right_subtree):

        if type(left_subtree[0]) == numpy.float64 and type(right_subtree[0]) == numpy.float64:
            current[0][3] = left_subtree[0]
            current[0][4] = right_subtree[0]
        
        elif type(left_subtree[0]) == numpy.float64 and type(right_subtree[0]) == numpy.ndarray:
            current[0][3] = left_subtree[0]
            current[0][4] = right_subtree[0][0]
        elif type(left_subtree[0]) == numpy.ndarray and type(right_subtree[0]) == numpy.float64:
            current[0][3] = left_subtree[0][0]
            current[0][4] = right_subtree[0]
        else:
                current[0][3] = left_subtree[0][0]
                current[0][4] = right_subtree[0][0]
        return numpy.vstack((current,numpy.vstack((left_subtree,right_subtree))))

    def buildtree(self,Trainset):
        global serial_no
        serial_no = serial_no+1
        if len(Trainset)==1:
            return numpy.array([serial_no,-1,Trainset[:,5],-1,-1])
        elif len(Trainset)>1:
            chosen_feature = random.randrange(0,5)
            # print len(Trainset)
            while True:
                chosen_step_1 = random.randrange(0,len(Trainset))
                chosen_step_2 = random.randrange(0,len(Trainset))
                if chosen_step_2!=chosen_step_1:
                    break
            splitval = numpy.mean([Trainset[chosen_step_1][chosen_feature],Trainset[chosen_step_2][chosen_feature]]) 
            (left_data,right_data) = self.split(Trainset, Trainset[:,chosen_feature]<=splitval)
            if(len(left_data)==0 or len(right_data)==0):
                # print "DANGER! being averted"
                return numpy.array([serial_no,-1,numpy.mean(Trainset[:,chosen_feature]),-1,-1])
            current = numpy.array([[serial_no,chosen_feature,splitval,0,0]])
            left_subtree = self.buildtree(left_data)
            right_subtree = self.buildtree(right_data)
            self.tree = self.combine_tree(current,left_subtree,right_subtree)
            
        return self.tree

    def treechecker(self,Tree):
        while len(Tree)<200000:
           Tree = self.buildtree(self.Trainset)
        return Tree 


    def addEvidence(self, XTrain,Ytrain):
        self.XTrain=XTrain
        self.Ytrain=Ytrain
        self.Trainset = numpy.concatenate((XTrain,Ytrain),axis=1)
        self.forest=[]
        for i in range(0,self.k):
            Tree = self.treechecker(self.buildtree(self.Trainset))
            print len(Tree)," ",i
            self.forest.append(Tree)
            # This resets the tree's serial no to ensure that the new trees created have same serial nos.
            global serial_no
            serial_no = 0

    def query(self,XTest):
        return self.queryforest(XTest)

    def queryforest(self,XTest):
        mean_val_list=[]
        for x_values in XTest:
            Values =[]
            Value = []
            for i in range(0,self.k):
                Value = self.queryone(x_values,self.forest[i])
                Values.append(Value)
            mean_val = sum(Values) / float(len(Values))
            mean_val_list.append(mean_val)    
        return mean_val_list

    def queryone(self, XTest_single,Tree):
        step=0
        value = 0
        # import pdb
        # pdb.set_trace()
        while step<=len(Tree)-1:
            if(Tree[step][FEATURE_NUMBER_step] ==1):
                if(XTest_single[1]<=Tree[step][SPLIT_VAL_step]):
                    step = Tree[step][LEFT_POINTER]
                    continue
                else:
                    step = Tree[step][RIGHT_POINTER]
                    continue
            elif(Tree[step][FEATURE_NUMBER_step]==-1):
                value = Tree[step][SPLIT_VAL_step]
                break
            elif(Tree[step][FEATURE_NUMBER_step] ==0):
                if(XTest_single[0]<=Tree[step][SPLIT_VAL_step]):
                    step = Tree[step][LEFT_POINTER]
                    continue
                else:
                    step = Tree[step][RIGHT_POINTER]
                    continue
            elif(Tree[step][FEATURE_NUMBER_step] ==2):
                if(XTest_single[2]<=Tree[step][SPLIT_VAL_step]):
                    step = Tree[step][LEFT_POINTER]
                    continue
                else:
                    step = Tree[step][RIGHT_POINTER]
                    continue
            elif(Tree[step][FEATURE_NUMBER_step] ==3):
                if(XTest_single[3]<=Tree[step][SPLIT_VAL_step]):
                    step = Tree[step][LEFT_POINTER]
                    continue
                else:
                    step = Tree[step][RIGHT_POINTER]
                    continue
            elif(Tree[step][FEATURE_NUMBER_step] ==4):
                if(XTest_single[4]<=Tree[step][SPLIT_VAL_step]):
                    step = Tree[step][LEFT_POINTER]
                    continue
                else:
                    step = Tree[step][RIGHT_POINTER]
                    continue
        return value    
              
    def getflatcsv(self,fname):
        inf = open(fname)
        return numpy.array([map(float,s.strip().split(',')) for s in inf.readlines()])