from HMM import HMM
from GMMs import GMMs
import ASR
import timeit
from hmmlearn.hmm import GMMHMM
import numpy as np

def program1(phones):
    path = r"C:\Users\Nicole Schwartz\Anaconda3\seniorProject\new\darpa-timit-acousticphonetic-continuous-speech\data\\" 

    gmms = GMMs(phones, path)
    start = timeit.default_timer()
    gmms.train()
    stop = timeit.default_timer()
    elapsed = stop - start 
    print("GMM training time: " + str(int(elapsed)/60) + "m  " + str(int(elapsed)%60) + "s" )
    start = timeit.default_timer()
    accuracyGMM = gmms.test()
    stop = timeit.default_timer()
    elapsed = stop - start 
    print("GMM testing time: " + str(int(elapsed)/60) + "m  " + str(int(elapsed)%60) + "s") 
    print("Accuracy of GMMs alone= ", round(accuracyGMM*100, 3))
    hmm = HMM(phones, gmms.models, path)
    start = timeit.default_timer()
    hmm.train(400)
    stop = timeit.default_timer()
    elapsed = stop - start 
    print("HMM training time: " + str(int(elapsed)/60) + "m  " + str(int(elapsed)%60) + "s")
    start = timeit.default_timer()
    hmm.test()
    stop = timeit.default_timer()
    elapsed = stop - start 
    print("HMM testing time: " + str(int(elapsed)/60) + "m  " + str(int(elapsed)%60) + "s")

def initialize(phones):
    models = {}
    start_prob = np.array([1.0, 0.0, 0.0])
    transmat = np.zeros((3, 3))
    for i in range(3):
        for j in range(i,3):
            trans = 1/(3-i)
            transmat[i][j] = trans
    for i in phones.keys():
        new_hmm = GMMHMM(n_components = 3, n_mix = 5, params = 'tmc', init_params = 'mc')
        new_hmm.startprob_ = start_prob
        new_hmm.transmat_ = transmat
        models[i] = new_hmm
    return models

def trainPhonemes(trainPath, phones, models):
    trainFiles = ASR.getFiles(trainPath)
    lengths = {key: [] for key in phones}
    obs = {key: [] for key in phones}
    for file in trainFiles:
        mfcc_feat = ASR.featureExtract(file)
        ASR.sort_phones(mfcc_feat, obs, lengths, file)
    for phn in phones:
        data = obs[phn]
        phn_lens = lengths[phn]
        num_comp = len(data)
        new_data = np.array(data)
        new_data = new_data.reshape(num_comp,13)
        models[phn].fit(new_data,phn_lens)
    return models


def testPhonemes(testPath, phones, models, trueP, falseP, falseN, trueN):
    testFiles = ASR.getFiles(testPath)
    for file in testFiles:
        obs = {key: [] for key in phones}
        lengths = {key: [] for key in phones}
        mfcc_feat = ASR.featureExtract(file)
        ASR.sort_phones(mfcc_feat, obs, lengths, file)
        for phone in obs.keys():
            start = 0
            phoneme_obs = obs[phone]
            length = lengths[phone]
            for i in range(len(length)):    
                first = 1
                best = 0
                new_data = phoneme_obs[start: start + length[i]]
                start += length[i]
                for phn, hmm in models.items():  
                    log_prob = hmm.score(new_data, [length[i]])
                    if (log_prob > best) or (first == 1):
                        best = log_prob
                        prediction = phn
                        first = 0
                if phone == prediction:
                    trueP[phone] += 1
                else:
                    falseP[prediction] += 1
                    falseN[phone] += 1
                for j in phones.keys():
                    if (j != phone) and (j != prediction):
                        trueN[j] += 1
   



def program2(phones):
    path = r"C:\Users\Nicole Schwartz\Anaconda3\seniorProject\new\darpa-timit-acousticphonetic-continuous-speech\data\\" 
    fp = open("paths2.txt", 'w')
    trueP = {key: 0 for key in phones.keys()} 
    trueN = {key: 0 for key in phones.keys()} 
    falseP ={key: 0 for key in phones.keys()} 
    falseN = {key: 0 for key in phones.keys()}

    models = initialize(phones)
    
    trainPath = path + "TRAIN"
    models = trainPhonemes(trainPath, phones, models)
    #hmm = HMM(phones, gmms.models, path)
    #hmm.train(64)

    testPath = path + "TRAIN\DR1"
    testPhonemes(testPath, phones, models, trueP, trueN, falseP, falseN)
    ASR.get_PrecisionRecall(trueP, falseP, falseN, trueN, 1)#get precision and recall for each state after testing is complete


def main():
    phones = {'b': 0, 'd': 1, 'g': 2, 'p': 3, 't': 4, 'k': 5, 'dx': 6, 'q': 7, 'jh': 8, 'ch': 9, 's': 10, 'sh': 11, 'z': 12, 'zh': 13,\
    'f': 14, 'th': 15, 'v': 16, 'dh': 17, 'm': 18, 'n': 19, 'ng': 20, 'em': 21, 'en': 22, 'eng': 23, 'nx': 24, 'l': 25, 'r': 26, 'w': 27, \
    'y': 28, 'hh': 29, 'hv': 30, 'el': 31, 'iy': 32, 'ih': 33, 'eh': 34, 'ey': 35, 'ae': 36, 'aa': 37, 'aw': 38, 'ay': 39, 'ah': 40, 'ao': 41,\
    'oy': 42, 'ow': 43, 'uh': 44, 'uw': 45, 'ux': 46, 'er': 47, 'ax': 48, 'ix': 49, 'axr': 50, 'ax-h': 51, 'bcl': 52, 'dcl': 53, 'gcl': 54,\
    'pcl': 55, 'tcl': 56, 'kcl': 57, 'epi': 58, 'pau': 59, 'h#': 60} 

    #program1(phones)
    program2(phones)



if __name__ == "__main__":
    main()
