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
    