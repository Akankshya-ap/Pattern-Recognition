import numpy as np
import matplotlib.pyplot as plt
import pandas

ds=pandas.read_csv('C:\Users\Akankshya\Desktop\pattern recog\pyt\iris_ds.csv')

datay=ds['flower']

datax=ds[['sw','sl','pw','pl']]


####Train_test####
X_train=datax.sample(frac=0.8,random_state=1)
Y_train=datay.loc[ds.index.isin(X_train.index)]

X_test=datax.loc[~ds.index.isin(X_train.index)]
Y_test=datay.loc[~ds.index.isin(Y_train.index)]

x=np.array(X_test,dtype=float)
y=np.array(Y_test)

#print X_test
#print x
#print np.transpose(x)

#########mean#######
mean=X_train.groupby(ds.flower).mean()
print mean
means= np.array(mean,dtype=float)


####fitting###
def discr(value):
    a=np.transpose(value)
    #print a
    #print np.transpose(a)
    #values=np.transpose(np.array(value))
    #print values
    #for k in range(0,len(means),1):
    meant= np.transpose(means)
       # print means[k]
        #print meant
       # print a
    g=np.dot(a,means)
    h=.5*np.matmul(meant,means)
    t=np.subtract(g,h)
    return h

print len(x)
for i in range(0,len(x),1):
    print x[i]
    m=0
    for j in range (0,len(means),1):
        if(discr(x)>m):
            m=discr(x)
            n=j
    #print Y_train.lo         

        

#X_test_t=np.transpose(X_test)



#print X_train.groupby(ds.flower).count()
#print Y_train

#print X_test
