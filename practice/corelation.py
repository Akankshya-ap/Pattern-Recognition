import numpy as np
import matplotlib.pyplot as plt


def d_mean(x) :
    xmean=np.mean(x)
    return[xi-xmean for xi in x]
def covariance(x,y):
    n=len(x)
    return np.dot(d_mean(x),d_mean(y))/(n-1)
pagespeeds=np.random.normal(3.0,1.0,1000)


purchaseamount=np.random.normal(50.0,10.0,1000)

def corelation(x,y):
    stddevx=x.std()
    stddevy=y.std()
    return covariance(x,y)/stddevx/stddevy

print corelation(pagespeeds,purchaseamount)
