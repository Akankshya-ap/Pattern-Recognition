import numpy as np
import matplotlib.pyplot as plt
incomes=np.random.normal(100.0,30.0,10000)
plt.hist(incomes,50)
plt.show()

#print np.mean(incomes)
#print np.median(incomes)
#incomes=np.append(incomes,[100000])
#print np.mean(incomes)
#print np.median(incomes)

print incomes.std()
print incomes.var()
