import numpy as np
import matplotlib.pyplot as plt
import pandas

ds=pandas.read_csv('C:\Users\Akankshya\Desktop\pattern recog\pyt\iris_ds.csv')

data_y=ds['flower']

data_x=ds[['sw','sl','pw','pl']]

X_test=np.array(data_x)
Y_test=np.array(data_y)

X_test_t=np.transpose(X_test)

#print list(data_y.target_names)

print data_x.groupby(ds.flower).mean()
