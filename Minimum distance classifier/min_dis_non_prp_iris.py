import numpy as np
import matplotlib.pyplot as plt
import pandas
#from sklearn.datasets import make_classifiaction
#from sklearn.dtatsets import make_blobs

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
mean1= np.unique(np.array([y])).reshape(3,1)
#print mean1[1]
meanr=np.array(mean)
#print meanr
means= np.array(mean,dtype=float)

####fitting###
def discr(value,k):
    a=np.transpose(np.array([value]))
    #print a
    #print np.transpose(a)
    #values=np.transpose(np.array(value))
    #print values
    #for k in range(0,len(means),1):
    meant= np.transpose([means[k]])
    #print means[k]
        #print meant
       # print a
    #print a.shape
    g=np.dot(means[k],a)
    h=.5*np.matmul(means[k],meant)
    t=np.subtract(g,h)
    return t

l=[]
print len(x)
for i in range(0,len(x),1):
    print x[i]
    m=0
    for j in range (0,len(means),1):
        p=discr(x[i],j)
        if(p>m):
            m=p
            n=j
    print mean1[n]
    l.append(mean1[n])
    
r=np.array(l)

#######printing fitted values####    
print  r

#####printing actual value###
print y


#####getting accuracy#####
j=0
k=0
for i in range(0,len(x),1):
    if r[j]==y[i]:
        j=j+1
    else :
        j=j+1
        k=k+1
acc=(len(x)-k)*100/len(x)
print acc


######plotting#####
#x1,y1=make_classificiation(n_features=2,n_redundant=0,n_informative=2,n_clustres_per_class=1)
#plt.scatter(X_train[:,0],X_train[:,1],c=Y_train)
#plt.show()
   
