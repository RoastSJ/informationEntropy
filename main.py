import math
import numpy as np
import pandas as pd

def informationEntropy(DistributionMatrix):
    def conditionalInformationEntropy(DistributionMatrix):
        m = np.zeros(DistributionMatrix.shape[1])
        for j in range(DistributionMatrix.shape[1]):
            for i in range(DistributionMatrix.shape[0]):
                pxy = DistributionMatrix[i,j]
                py = np.sum(DistributionMatrix[:,j])
                px_y = pxy/py
                if px_y > 0:
                    m[j] = m[j] - pxy*math.log2(px_y)
        return np.sum(m)
    def NormalInformationEntropy(DistributionMatrix):
        X = np.zeros(DistributionMatrix.shape[0])
        for i in range(DistributionMatrix.shape[0]):
            X[i] = np.sum(DistributionMatrix[i,:])
            X[i] = -X[i]*math.log2(X[i])
        return np.sum(X)
    H1 = NormalInformationEntropy(DistributionMatrix)
    H2 = NormalInformationEntropy(DistributionMatrix.T)
    H3 = conditionalInformationEntropy(DistributionMatrix)
    H4 = conditionalInformationEntropy(DistributionMatrix.T)
    Entropy = {
        "H(X)":H1,
        "H(Y)":H2,
        "H(X|Y)":H3,
        "H(Y|X)":H4,
        "H(XY)":H1 + H4,
        "H(X;Y)":H1-H3
        }
    return Entropy

Entropy1 = informationEntropy((pd.read_excel("data1.xls",header=None)).to_numpy())
Entropy2 = informationEntropy((pd.read_excel("data2.xls",header=None)).to_numpy())
print("data1:")
for k,v in Entropy1.items():
    print("\t"+k +":"+str(round(v,3)))
print("data2:")
for k,v in Entropy2.items():
    print("\t"+k +":"+str(round(v,3)))
