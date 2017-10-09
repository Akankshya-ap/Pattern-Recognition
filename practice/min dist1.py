import numpy as np
import matplotlib.pyplot as plt
import pandas

ds=pandas.read_csv('C:\Users\Akankshya\Desktop\pattern recog\pyt\iris_ds.csv')

datay=ds['flower']

datax=ds[['sw','sl','pw','pl']]

X_train=datax[9:]
Y_train=np.array(datay[9:])

X_test=np.array(datax[:9])
Y_test=np.array(datay[:9])

X_test_t=np.transpose(X_test)



print X_train.groupby(ds.flower).count()


print X_test
