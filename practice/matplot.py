from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
x=np.arange(-3,3,.001)
#plt.plot(x,norm.pdf(x))
#plt.show()

"""
axes=plt.axes()
axes.set_xlim([-5,5])
axes.set_ylim([0,1])
axes.set_xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
axes.grid()
plt.plot(x,norm.pdf(x),'b-')
plt.plot(x,norm.pdf(x,1.0,0.5),'r--')
plt.show()
"""

plt.xkcd()

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.xticks([])
plt.yticks([])
ax.set_ylim([-30,10])

data=np.ones(100)
data[70:]-=np.arange(30)
