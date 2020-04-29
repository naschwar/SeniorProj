import ASR
import numpy
import operator
from sklearn import mixture

class GMMs():
    def __init__(self, phones, dataPath):
        self.phones = phones
        self.dataPath = dataPath
        self.models = dict((p, []) for p in self.phones) 

    def train(self):
        trainPath = self.dataPath + "TRAIN"
        files = ASR.getFiles(trainPath)
        features ={}
        for name in files:
            ASR.featureExtract(self.phones.keys(), name, features)
        for phone, obs in features.items():
            gmm = mixture.GaussianMixture(n_components=3, covariance_type='full', max_iter = 100)
            gmm.fit(obs)
            self.models[phone] = gmm
        return features
    
    def test(self):
    #this function tests the GMM alone on test data
        testPath = self.dataPath + "TEST"
        testFiles = ASR.getFiles(testPath)
        num =0
        denom =0
        trueP = {key: 0 for key in self.phones.keys()} #increment a phone's trueP score when GMM predicts that phone correctly
        falseP ={key: 0 for key in self.phones.keys()} #increment a phone's falseP when GMM predicts that phone, but actually was a different phone
        falseN = {key: 0 for key in self.phones.keys()} #increment a phone's falseN when GMM predicts a different phone, but actually was this phone
        trueN = {key: 0 for key in self.phones.keys()} #increment a phone's trueP when GMM correctly did not predict the phone from the observation
        mKeys = self.models.keys()
        for f in testFiles:
            X_test = {}
            ASR.featureExtract(self.phones.keys(), f, X_test) #get features of a test audio file and sort them by phone into Xtest dictionary
            preds =[]
            for key,vals in X_test.items(): #key is the phone label and val is an array of extracted feature vectors 
                for val in vals:
                    logprobs = dict((p, []) for p in self.phones.keys()) 
                    for phone in mKeys:#test each model with the observations
                        gmm = self.models[phone] 
                        scores = numpy.asarray(gmm.score([val])) #array of logprob of each feature vector in array of vectors
                        logprobs[phone]= numpy.sum(scores) #logprob of that phone given all the observations
                    prediction = max(logprobs.items(), key=operator.itemgetter(1))[0] #phone is model that yields highest logpro from the observations
                    if prediction == key:
                        num +=1
                        trueP[prediction] +=1
                    else:
                        falseP[prediction] +=1
                        falseN[key] +=1
                    for i in range(len(self.phones.keys())):
                        phn = self.phones.keys()[i] 
                        if((phn != prediction) & (phn != key)):
                            trueN[phn] +=1
                    denom +=1
                    preds.append(prediction)
        ASR.get_PrecisionRecall(trueP, falseP, falseN, trueN,0) 
        accuracy = num/float(denom)
        return accuracy

    
