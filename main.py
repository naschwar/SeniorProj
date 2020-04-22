from HMM import HMM
from GMMs import GMMs
from ASR import *


def main():
    path = r"C:\Users\Nicole Schwartz\Anaconda3\seniorProject\new\darpa-timit-acousticphonetic-continuous-speech\data\\" 
    phones = ['b', 'd', 'g', 'p', 't', 'k', 'dx', 'q',  'jh', 'ch', 's', 'sh', 'z', 'zh',\
    'f', 'th', 'v', 'dh', 'm', 'n', 'ng', 'em', 'en', 'eng', 'nx', 'l', 'r', 'w','y', 'hh',\
    'hv', 'el', 'iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh',\
    'uw', 'ux', 'er', 'ax', 'ix', 'axr', 'ax-h', 'bcl','dcl','gcl','pcl','tcl', 'kcl', 'epi',\
    'pau','h#']
    gmms = GMMs(phones, path)
    gmms.train()
    accuracyGMM = gmms.test()
    print("Accuracy of GMMs alone= ", round(accuracyGMM*100, 3))
    hmm = HMM(phones, gmms.models, path)
    hmm.transitions = hmm.train(400)
    hmm.test()



if __name__ == "__main__":
    main()
