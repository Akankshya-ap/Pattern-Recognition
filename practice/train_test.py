import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

np.random.seed(2)

x=np.random.normal(3.0,1.0,1000)
y=np.random.normal(50.0,30.0,1000)/x

#plt.scatter(x,y)
#plt.show()

trainx=x[80:]
testx=x[:80]

trainy=y[80:]
testy=y[:80]
'''
plt.scatter(trainx,trainy)
plt.show()
'''

"""
plt.scatter(testx,testy)
plt.show() 
"""

xa=np.array(trainx)
ya=np.array(trainy)

p=np.poly1d(np.polyfit(xa,ya,8))

xp=np.linspace(0,7,100)
axes=plt.axes()
axes.set_xlim([0,7])
axes.set_ylim([0,200])
#plt.scatter(xa,ya)
#plt.plot(xp,p(xp),c='r')
#plt.show()

xab=np.array(testx)
yab=np.array(testy)

r2=r2_score(testy,p(testx))
#r2=r2_score(trainy,p(trainx))
print r2

plt.scatter(xab,yab)
plt.plot(xp,p(xp),c='r')
plt.show()


r2=r2_score(testy,p(testx))
#r2=r2_score(trainy,p(trainx))
print r2
