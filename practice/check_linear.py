import numpy as np
import matplotlib.pyplot as plt

pagespeeds=np.random.normal(3.0,50.0,1000)
purchaseamount=100-pagespeeds*3
plt.scatter(pagespeeds,purchaseamount)
plt.show()

