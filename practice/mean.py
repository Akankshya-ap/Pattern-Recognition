import numpy as np
incomes=np.random.normal(27000,15000,10000)
#print np.mean(incomes)

import matplotlib.pyplot as plt
plt.hist(incomes,50)

#plt.show()
#print np.median(incomes)
incomes=np.append(incomes,[100000000])
#print np.median(incomes)
#print np.mean(incomes)

ages=np.random.randint(18,high=90,size=500)
#print ages
from scipy import stats
print stats.mode(ages)
