import numpy as np
import matplotlib.pyplot as plt
x=np.random.normal(3.0,1.0,1000)
y=100-(x+np.random.normal(0,0.1,1000))*3

plt.scatter(x,y)
#plt.show()

from scipy import stats

slope,intercept,r_value,p_value,std_err=stats.linregress(x,y)

print r_value**2


def predict(p) :
    return slope*p+intercept

line=predict(x)

plt.plot(x,line,c='r')
plt.show()
