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
    filePhones =[]
    for line in lines:
        vals = line.strip().split(' ')
        start = int(vals[0])
        end = int(vals[1])
        phone = vals[2]
        filePhones.append([start, end, phone])
    return filePhones

def featureExtract(phones, name, features, lengths=[]):
    wavFile = name + '.WAV.wav'
    phnFile = name + '.phn'
    (rate,sig) = wav.read(wavFile) 
    mfcc_feat = mfcc(sig,rate, winlen = windowLen, winstep =winstep, winfunc = numpy.hanning)
    fp = open(phnFile)
    lines = fp.readlines()
    filePhones = parseLines(lines)
    phoneNum = 0
    n =0
    length = 0
    for frame in mfcc_feat:
        startSample = int(n * offset)
        finalSample = int(startSample + sampPerWin) -1
        phnStart = filePhones[phoneNum][0]
        phnEnd = filePhones[phoneNum][1]
        if (startSample >= phnStart) and (finalSample < phnEnd):
            phoneName =filePhones[phoneNum][2]
            if phoneName in phones:
                if phoneName not in features.keys():
                    features[phoneName] = [] 
                features[phoneName].append(frame)
            length +=1
        elif ((startSample >= phnStart) and (finalSample >= phnEnd)):
            length+=1
            if finalSample < filePhones[-1][1]: #check that frame is not outside full range of file
                phoneNum +=1 
                lengths.append(length)
                length =0
        else: 
            length +=1
        n +=1
    lengths.append(length)
    fp.close()
    return mfcc_feat
    
               
def trainGMM(obs):
    gmm = mixture.GaussianMixture(n_components=3, covariance_type='full', max_iter = 100)
    gmm.fit(obs)
    return gmm

#find transition matrix probability
#need to modify to do frame transitions not just phones (phones transition back to self)
def getTransitions(path, phones): 
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
            self_tran = (int(fromVals[1])- int(fromVals[0]))/sampPerWin
            fromPhone = fromVals[2]
            toPhone = toVals[2]
            if (toPhone in phones) and (fromPhone in phones):
                total +=1 + self_tran
                fromI = phones.index(fromPhone)
                toI = phones.index(toPhone)
                B[fromI][toI] +=1
                B[fromI][fromI] += self_tran
    for i in range(61):
        for j in range(61):
            B[i][j]= ((B[i][j] + 1)/total)*100
    return B

def train(phones, trainPath):
    files = getFiles(trainPath)
    features ={}
    for name in files:
        featureExtract(phones, name, features)
    return features

def get_PrecisionRecall(trueP, falseP, falseN, trueN, model):
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
    testFiles = getFiles(testPath)
    num =0
    denom =0
    trueP = {key: 0 for key in phones} #add to when guess phone correctly
    falseP ={key: 0 for key in phones} #add to when guess the phone, but actually different phone
    falseN = {key: 0 for key in phones} #add to when guess different phone, but actually the phone
    trueN = {key: 0 for key in phones} #add to when correctly predicted as not being the phone
    mKeys = models.keys()
    for f in testFiles: #get logprob of each phone within file
        X_test = {}
        featureExtract(phones, f, X_test)
        preds =[]
        for key,val in X_test.items():
            if val != []:
                logprobs = dict((p, []) for p in phones) 
                for phone in mKeys:
                    gmm = models[phone] #get each model of phone features
                    scores = numpy.asarray(gmm.score(val)) #gives log probability of each data point(feature value) in the multi-dimen mfcc obs vector
                    logprobs[phone]= numpy.sum(scores) #logprob of that phone given all the feature probabilities
                prediction = max(logprobs.items(), key=operator.itemgetter(1))[0]
                #print("prediction: ",prediction, " actual: ", key)
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
        #print(num/float(denom))
    get_PrecisionRecall(trueP, falseP, falseN, trueN,0) 
    accuracy = num/float(denom)
    return accuracy

def decode(observations, models, B, phones):
    pathLen = len(observations) #number of frames
    path_probs = numpy.zeros((len(phones), pathLen))
    paths = {key: [] for key in phones}
    #paths = numpy.arange([len(phones), pathLen], dtype=object)
    for t in range(pathLen):
        obs = [observations[t]]
        i =0
        for phn in phones:
            emission = models[phn].score(obs) #prob of the obsvations given the model 
            if(t == 0):
                path_probs[i,0] = emission
            else:   
                #print(B[:,i])
                transitions = numpy.add(numpy.log(B[:,i]), path_probs[:,t-1])
                max = numpy.argmax(transitions)
                if(paths[phn] == []):
                    paths[phn]=[phones[max]]
                else:
                    paths[phn].append(phones[max]) #corresponding phone
                    #print(phones[max])
                path_probs[i, t] = emission+transitions[max] 
            i +=1
    for phn in phones:
        paths[phn].append(phn)
    pred = numpy.argmax(path_probs[:,-1])
    final = phones[pred]
    path = paths[final]
    return path


def getPath(phnFile, phones, lengths):
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
    testFiles = getFiles(testPath)
    fp = open("paths.txt", "w")
    mfcc_feat =[]
    trueP = {key: 0 for key in phones} #add to when guess phone correctly
    trueN = {key: 0 for key in phones} #add to when correctly predicted as not being the phone
    falseP ={key: 0 for key in phones} #add to when guess the phone, but actually different phone
    falseN = {key: 0 for key in phones} #add to when guess different phone, but actually the phone
    for f in testFiles:
        obs ={}
        phnFile = f + '.phn'
        lengths = []
        mfcc_feat = featureExtract(phones, f, obs, lengths)
#        print(numpy.sum(lengths))
#        print(len(mfcc_feat))
        path = decode(mfcc_feat, models, B, phones)
        fp.write("\nPredicted Path:\n")
        fp.write(str(path))
        correctPath = getPath(phnFile, phones, lengths)
        fp.write("\nCorrect Path:\n")
        fp.write(str(correctPath))
        fp.write("\n")
        compareResults(correctPath, path, phones, trueP, falseP, falseN, trueN)
    fp.close()
    get_PrecisionRecall(trueP, falseP, falseN, trueN, 1)
    return 


def main():
    path = r"C:\Users\Nicole Schwartz\Anaconda3\seniorProject\new\darpa-timit-acousticphonetic-continuous-speech\data\\" 
    phones = ['b', 'd', 'g', 'p', 't', 'k', 'dx', 'q',  'jh', 'ch', 's', 'sh', 'z', 'zh',\
    'f', 'th', 'v', 'dh', 'm', 'n', 'ng', 'em', 'en', 'eng', 'nx', 'l', 'r', 'w','y', 'hh',\
    'hv', 'el', 'iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh',\
    'uw', 'ux', 'er', 'ax', 'ix', 'axr', 'ax-h', 'bcl','dcl','gcl','pcl','tcl', 'kcl', 'epi',\
    'pau','h#']
    models = dict((p, []) for p in phones) 
    accuracy =1
    trainPath = path + "TRAIN"
    X_train = train(phones, trainPath)
    for phone, obs in X_train.items():
        models[phone] = trainGMM(obs)
    B = getTransitions(trainPath, phones)
    testPath = path + "TEST"
    accuracyGMM = testGMMs(phones, testPath, models, X_train)
    print("Accuracy of GMMs alone= ", round(accuracyGMM*100, 3))
    testHMM(phones, testPath,B, models)
    # hmm = myGMMHMM(models)
    # hmm.n_components = 52
    # hmm.transmat_ = B
    # hmm.algorithm = 'viterbi'




    # accuracy = testHMM(testPath, hmm, phones)

if __name__ == "__main__":
    main()

#Notes:
#Do not need to train HMM because already trained emission and transition matrices