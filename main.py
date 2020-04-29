from HMM import HMM
from GMMs import GMMs
from ASR import *
import timeit


def main():
    path = r"C:\Users\Nicole Schwartz\Anaconda3\seniorProject\new\darpa-timit-acousticphonetic-continuous-speech\data\\" 

    phones = {'b': 0, 'd': 1, 'g': 2, 'p': 3, 't': 4, 'k': 5, 'dx': 6, 'q': 7, 'jh': 8, 'ch': 9, 's': 10, 'sh': 11, 'z': 12, 'zh': 13,\
    'f': 14, 'th': 15, 'v': 16, 'dh': 17, 'm': 18, 'n': 19, 'ng': 20, 'em': 21, 'en': 22, 'eng': 23, 'nx': 24, 'l': 25, 'r': 26, 'w': 27, \
    'y': 28, 'hh': 29, 'hv': 30, 'el': 31, 'iy': 32, 'ih': 33, 'eh': 34, 'ey': 35, 'ae': 36, 'aa': 37, 'aw': 38, 'ay': 39, 'ah': 40, 'ao': 41,\
    'oy': 42, 'ow': 43, 'uh': 44, 'uw': 45, 'ux': 46, 'er': 47, 'ax': 48, 'ix': 49, 'axr': 50, 'ax-h': 51, 'bcl': 52, 'dcl': 53, 'gcl': 54,\
    'pcl': 55, 'tcl': 56, 'kcl': 57, 'epi': 58, 'pau': 59, 'h#': 60}
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
    


if __name__ == "__main__":
    main()

