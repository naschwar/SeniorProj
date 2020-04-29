import ASR
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import os
import numpy
import operator
from sklearn import mixture
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


class HMM(object): 
    def __init__(self, phones, gmms, dataPath):
        self.phones = phones
        self.models = gmms
        self.dataPath = dataPath
        self.transitions = numpy.zeros((61,61)) 
        

    def train(self, sampPerWin): 
        path = self.dataPath + "TRAIN"
        files = ASR.getFiles(path)
        total =0
        for name in files:
            phnFile = name + '.phn'
            fp = open(phnFile)
            lines = fp.readlines()
            numLines = len(lines)
            for i in range(numLines -2):
                fromVals = lines[i].strip().split(' ')
                toVals =lines[i+1].strip().split(' ')
                self_tran = (int(fromVals[1])- int(fromVals[0]))/sampPerWin #find the number of transitions the phone makes back onto itself
                fromPhone = fromVals[2]
                toPhone = toVals[2]
                if ((toPhone in self.phones.keys()) and (fromPhone in self.phones.keys())):
                    total += (1 + self_tran)
                    fromI = self.phones[fromPhone] #index into transition matrix corresponds to index into phones array
                    toI = self.phones[toPhone] 
                    self.transitions[fromI][toI] +=1
                    self.transitions[fromI][fromI] += self_tran
        self.transitions = (self.transitions + 1)/total

    def decode(self, observations):
    #this function implements the viterbi algorithm to decode the path of phone states
        pathLen = len(observations) 
        path_probs = numpy.zeros((len(self.phones.keys()), pathLen)) 
        paths = {key: [] for key in self.phones.keys()} 
        for t in range(pathLen):
            obs = [observations[t]] 
            i =0
            for phn in self.phones.keys(): 
                emission = self.models[phn].score(obs) 
                if(t == 0): 
                    path_probs[i,0] = emission 
                else:   
                    transitions = numpy.add(numpy.log(self.transitions[:,i]), path_probs[:,t-1]) 
                    max = numpy.argmax(transitions) 
                    if(paths[phn] == []):
                        paths[phn]=[self.phones.keys()[max]]
                    else:
                        paths[phn].append(self.phones.keys()[max]) 
                    path_probs[i, t] = emission+transitions[max] 
                i +=1
        for phn in self.phones.keys():
            paths[phn].append(phn)
        pred = numpy.argmax(path_probs[:,-1]) 
        final = self.phones.keys()[pred] 
        path = paths[final] 
        return path


    def test(self):
        #this function tests the HMM model on test data using trained GMMs and a transition matrix from the training data
        testPath = self.dataPath + "TEST"
        testFiles = ASR.getFiles(testPath)
        fp = open("paths.txt", "w")
        mfcc_feat =[]
        trueP = {key: 0 for key in self.phones.keys()} #increment a phone's trueP score when HMM predicts that phone correctly
        trueN = {key: 0 for key in self.phones.keys()} #increment a phone's trueP when HMM correctly did not predict the phone from the observation
        falseP ={key: 0 for key in self.phones.keys()} #increment a phone's falseP when HMM predicts that phone, but actually was a different phone
        falseN = {key: 0 for key in self.phones.keys()} #increment a phone's falseN when HMM predicts a different phone, but actually was this phone
        for f in testFiles:
            obs ={}
            phnFile = f + '.phn'
            lengths = []
            mfcc_feat = ASR.featureExtract(self.phones.keys(), f, obs, lengths)
            path = self.decode(mfcc_feat)#get predicted sequence
            fp.write("\nPredicted Path:\n")
            fp.write(str(path))
            correctPath = ASR.getPath(phnFile, self.phones.keys(), lengths)#get correct sequence
            fp.write("\nCorrect Path:\n")
            fp.write(str(correctPath))
            fp.write("\n")
            ASR.compareResults(correctPath, path, self.phones.keys(), trueP, falseP, falseN, trueN)
        fp.close()
        ASR.get_PrecisionRecall(trueP, falseP, falseN, trueN, 1)#get precision and recall for each state after testing is complete
        return 

