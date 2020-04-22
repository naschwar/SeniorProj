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

winstep =.0125
windowLen = .025 
fs = 16000
offset = 200 #fs *winstep
sampPerWin = 400 #windowLen * fs

def getFiles(path):
#this function returns a list of all the filenames in a folder
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if file.count('.') ==1: 
                name, ext = file.split('.')    
            else:
                name, ext, ext2 = file.split('.')   
            rootName = r + '\\' + name
            if rootName not in files:
                files.append(rootName)
    return files

def parseLines(lines):
#this function parses the "phn" files into the start frame, end frame and phone label
    filePhones =[]
    for line in lines:
        vals = line.strip().split(' ')
        start = int(vals[0])
        end = int(vals[1])
        phone = vals[2]
        filePhones.append([start, end, phone])
    return filePhones

def featureExtract(phones, name, features, lengths=[]):
#this function extracts the feature vectors that represent each frame of observation and sorts the observations into 
#the corresponding phone to help with GMM training/testing
    wavFile = name + '.WAV.wav'
    phnFile = name + '.phn'
    (rate,sig) = wav.read(wavFile) 
    mfcc_feat = mfcc(sig,rate, winlen = windowLen, winstep =winstep, winfunc = numpy.hanning) #get the array of feature vectors from the audio file
    fp = open(phnFile)
    lines = fp.readlines()
    filePhones = parseLines(lines)
    phoneNum = 0
    n =0
    length = 0
    for frame in mfcc_feat: #iterate through each vector and sort it into the phone it is labeled as in the phn file
        startSample = int(n * offset) #start sample in audio file
        finalSample = int(startSample + sampPerWin) -1 #end sample in audio file
        phnStart = filePhones[phoneNum][0] #labeled start sample of phone according to phn file
        phnEnd = filePhones[phoneNum][1] #labeled end sample of phone according to phn file
        if (startSample >= phnStart) and (finalSample < phnEnd):
            phoneName =filePhones[phoneNum][2]
            if phoneName in phones:
                if phoneName not in features.keys():
                    features[phoneName] = [] 
                features[phoneName].append(frame)#features is a dictionary that stores the observations by phone from all of the audio files
            length +=1 
        elif ((startSample >= phnStart) and (finalSample >= phnEnd)):
            length+=1
            if finalSample < filePhones[-1][1]: #check that frame is not outside full range of file
                phoneNum +=1 
                lengths.append(length) #lengths records the number of frames (feature vectors) for a single phone
                length =0
        else: 
            length +=1
        n +=1
    lengths.append(length)
    fp.close()
    return mfcc_feat #return the unsorted time-ordered feature vectors
    
               
def trainGMM(obs):
#this function fits the gaussian mixture models based on the given feature vectors for that model
    gmm = mixture.GaussianMixture(n_components=3, covariance_type='full', max_iter = 100)
    gmm.fit(obs)
    return gmm


def getTransitions(path, phones): 
#this function finds transition matrix probability by counting the transitions between frames seen in the train files
    B = numpy.zeros((61,61)) 
    files = getFiles(path)
    total =0
    for name in files:
        phnFile = name + '.phn'
        fp = open(phnFile)
        lines = fp.readlines()
        numLines = len(lines)
        for i in range(numLines -1):
            fromVals = lines[i].strip().split(' ')
            toVals =lines[i+1].strip().split(' ')
            self_tran = (int(fromVals[1])- int(fromVals[0]))/sampPerWin #find the number of transitions the phone makes back onto itself
            fromPhone = fromVals[2]
            toPhone = toVals[2]
            if (toPhone in phones) and (fromPhone in phones):
                total += (1 + self_tran)
                fromI = phones.index(fromPhone) #index into transition matrix corresponds to index into phones array
                toI = phones.index(toPhone) 
                B[fromI][toI] +=1
                B[fromI][fromI] += self_tran
    for i in range(61):
        for j in range(61):
            B[i][j]= ((B[i][j] + 1)/total)*100 #turn counts  into probabilities
    return B

def train(phones, trainPath):
#this function returns the training data (feature vectors)
    files = getFiles(trainPath)
    features ={}
    for name in files:
        featureExtract(phones, name, features)
    return features

def get_PrecisionRecall(trueP, falseP, falseN, trueN, model):
#this function calculates and records the precision, recall and accuracy of each phone 
#trueP, falseP, falseN, trueN are arrays with values for each phone
    keys = trueP.keys() 
    precision = {key: 0 for key in keys}
    recall = {key: 0 for key in keys}
    accuracy = {key: 0 for key in keys}
    if (model ==0):
        fp = open("GMM_results.txt", "w")
    else:
        fp = open("HMM_results.txt", "w")
    for key, value in trueP.items():
        TP = value
        FP = falseP[key]
        FN = falseN[key]
        ph_precision = (float(TP))/(TP + FP)
        precision[key] = ph_precision
        ph_recall = (float(TP))/(TP + FN)
        recall[key] = ph_recall
        accuracy[key] = float(trueP[key] + trueN[key])/(trueP[key] + trueN[key] + falseP[key] + falseN[key])
    for key, value in precision.items():
        fp.write(key + "\n")
        fp.write("   Precision: " + str(round(value, 3)) + "\n")
        r = recall[key]
        fp.write("   Recall:    " + str(round(r, 3)) + "\n")
        acc = accuracy[key]
        fp.write("   Accuracy:  " + str(round(acc, 3)) + "\n")
        fp.write("\n")
    fp.close()

def testGMMs(phones, testPath, models, X_train):
#this function tests the GMM alone on test data
    testFiles = getFiles(testPath)
    num =0
    denom =0
    trueP = {key: 0 for key in phones} #increment a phone's trueP score when GMM predicts that phone correctly
    falseP ={key: 0 for key in phones} #increment a phone's falseP when GMM predicts that phone, but actually was a different phone
    falseN = {key: 0 for key in phones} #increment a phone's falseN when GMM predicts a different phone, but actually was this phone
    trueN = {key: 0 for key in phones} #increment a phone's trueP when GMM correctly did not predict the phone from the observation
    mKeys = models.keys()
    for f in testFiles:
        X_test = {}
        featureExtract(phones, f, X_test) #get features of a test audio file and sort them by phone into Xtest dictionary
        preds =[]
        for key,vals in X_test.items(): #key is the phone label and val is an array of extracted feature vectors 
            for val in vals:
                logprobs = dict((p, []) for p in phones) 
                for phone in mKeys:#test each model with the observations
                    gmm = models[phone] 
                    scores = numpy.asarray(gmm.score([val])) #array of logprob of each feature vector in array of vectors
                    logprobs[phone]= numpy.sum(scores) #logprob of that phone given all the observations
                prediction = max(logprobs.items(), key=operator.itemgetter(1))[0] #phone is model that yields highest logpro from the observations
                if prediction == key:
                    num +=1
                    trueP[prediction] +=1
                else:
                    falseP[prediction] +=1
                    falseN[key] +=1
                for i in range(len(phones)):
                    phn = phones[i] 
                    if((phn != prediction) & (phn != key)):
                        trueN[phn] +=1
                denom +=1
                preds.append(prediction)
    get_PrecisionRecall(trueP, falseP, falseN, trueN,0) 
    accuracy = num/float(denom)
    return accuracy

def decode(observations, models, B, phones):
#this function implements the viterbi algorithm to decode the path of phone states
    pathLen = len(observations) #number of frames
    path_probs = numpy.zeros((len(phones), pathLen)) #matrix of probabilities for each phone at each time step
    paths = {key: [] for key in phones} #stores the most likely path that would end in each phone 
    for t in range(pathLen):
        obs = [observations[t]] #get a single feature vector
        i =0
        for phn in phones: #iterate through each state
            emission = models[phn].score(obs) #prob of the observations given the gmm
            if(t == 0): #first observation in sequence
                path_probs[i,0] = emission # prediction determined only by emission probability (transitions for all phones are the same)
            else:   
                transitions = numpy.add(numpy.log(B[:,i]), path_probs[:,t-1]) #multiply the prob of each phone at time t-1 with the prob of transitioning from that phone to this state 
                max = numpy.argmax(transitions) #find the max probability
                if(paths[phn] == []):
                    paths[phn]=[phones[max]]
                else:
                    paths[phn].append(phones[max]) #find the phone that yields the max probability
                path_probs[i, t] = emission+transitions[max] #update the state probability at time t to be the max probability possible
            i +=1
    for phn in phones:
        paths[phn].append(phn)
    pred = numpy.argmax(path_probs[:,-1]) #find which phone path ended with the greatest probability
    final = phones[pred] #get the phone yielding to the greatest probability
    path = paths[final] #get the path yielding the greatest probability
    return path


def getPath(phnFile, phones, lengths):
#this function returns an array with the correct order of phn frame as transcribed in a phn file
    correctPath =[]
    fp = open(phnFile)
    lines = fp.readlines()
    numFrames =0
    i =0
    for line in lines:
        vals = line.strip().split(' ')
        phn = vals[2]
        numFrames= lengths[i]
        i +=1
        for x in range(numFrames):
            if(phn in phones):
                correctPath.append(phn)
            else:
                correctPath.append("pass")
    return correctPath

def compareResults(correctPath, path, phones, trueP, falseP, falseN, trueN):
#this function compares the correct path(from the phn file) with the model's predictions
    if (len(correctPath) != len(path)):
        print("Path Length Error", len(correctPath), len(path))
        return 1,1,1
    for x in range(len(correctPath)):
        if(correctPath[x] != "pass"):
            if(correctPath[x] == path[x]):
                phn = correctPath[x]
                trueP[phn]+=1
            else:
                falsePred = path[x]
                actual = correctPath[x]
                falseP[falsePred] +=1
                falseN[actual] +=1
            for i in range(len(phones)):
                phn = phones[i] 
                if((phn != correctPath[i]) & (phn != path[i])):
                    trueN[phn] +=1
    #return trueP, falseP, falseN, trueN

def testHMM(phones, testPath, B, models):
#this function tests the HMM model on test data using trained GMMs and a transition matrix from the training data
    testFiles = getFiles(testPath)
    fp = open("paths.txt", "w")
    mfcc_feat =[]
    trueP = {key: 0 for key in phones} #increment a phone's trueP score when HMM predicts that phone correctly
    trueN = {key: 0 for key in phones} #increment a phone's trueP when HMM correctly did not predict the phone from the observation
    falseP ={key: 0 for key in phones} #increment a phone's falseP when HMM predicts that phone, but actually was a different phone
    falseN = {key: 0 for key in phones} #increment a phone's falseN when HMM predicts a different phone, but actually was this phone
    for f in testFiles:
        obs ={}
        phnFile = f + '.phn'
        lengths = []
        mfcc_feat = featureExtract(phones, f, obs, lengths)
#        print(numpy.sum(lengths))
#        print(len(mfcc_feat))
        path = decode(mfcc_feat, models, B, phones)#get predicted sequence
        fp.write("\nPredicted Path:\n")
        fp.write(str(path))
        correctPath = getPath(phnFile, phones, lengths)#get correct sequence
        fp.write("\nCorrect Path:\n")
        fp.write(str(correctPath))
        fp.write("\n")
        compareResults(correctPath, path, phones, trueP, falseP, falseN, trueN)
    fp.close()
    get_PrecisionRecall(trueP, falseP, falseN, trueN, 1)#get precision and recall for each state after testing is complete
    return 


def main():
    path = r"C:\Users\Nicole Schwartz\Anaconda3\seniorProject\new\darpa-timit-acousticphonetic-continuous-speech\data\\" 
    phones = ['b', 'd', 'g', 'p', 't', 'k', 'dx', 'q',  'jh', 'ch', 's', 'sh', 'z', 'zh',\
    'f', 'th', 'v', 'dh', 'm', 'n', 'ng', 'em', 'en', 'eng', 'nx', 'l', 'r', 'w','y', 'hh',\
    'hv', 'el', 'iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh',\
    'uw', 'ux', 'er', 'ax', 'ix', 'axr', 'ax-h', 'bcl','dcl','gcl','pcl','tcl', 'kcl', 'epi',\
    'pau','h#']
    models = dict((p, []) for p in phones) 
    trainPath = path + "TRAIN"
    X_train = train(phones, trainPath) #X_train is a dictionary with the observation vectors for each phone from the training data
    for phone, obs in X_train.items():
        models[phone] = trainGMM(obs)

    B = getTransitions(trainPath, phones)
    testPath = path + "TEST"
    accuracyGMM = testGMMs(phones, testPath, models, X_train)
    print("Accuracy of GMMs alone= ", round(accuracyGMM*100, 3))
    testHMM(phones, testPath,B, models)


if __name__ == "__main__":
    main()
