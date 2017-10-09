import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)
x=np.random.normal(3.0,1.0,1000)
y=np.random.normal(50.0,10.0,1000)/x
#plt.scatter(x,y)

#plt.show()

xa=np.array(x)
ya=np.array(y)

p4=np.poly1d(np.polyfit(xa,ya,9))

xp=np.linspace(0,7,100)
plt.scatter(xa,ya)
plt.plot(xp,p4(xp),c='r')
plt.show()
