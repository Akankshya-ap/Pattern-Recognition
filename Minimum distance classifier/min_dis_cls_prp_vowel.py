import numpy as np
import matplotlib.pyplot as plt
import pandas
#from sklearn.datasets import make_classifiaction
#from sklearn.dtatsets import make_blobs

ds=pandas.read_csv('C:/Users/Akankshya/Desktop/pattern recog/pyt/vowe.csv')

datay=ds['ans']

datax=ds[['a','b','c']]


###different classes division#####

prop_amt=0.88
print prop_amt

values=np.unique(datay)
c=len(values)
#print c
#print y
r=[]
for value in values:
   # print value
    x= np.count_nonzero(datay==value)
    inds=np.nonzero(datay==value)[0]
    #print inds
    np.random.shuffle(inds)
    p=int(prop_amt*x)
    for i in range(0, p,1):
        r.append(inds[i])
       # print r
#print len(np.array(r))


########Train_test#######
Y_train=datay.loc[ds.index.isin(r)]
X_train=datax.loc[ds.index.isin(r)]
#print X_train
#print Y_train


X_test=datax.loc[~ds.index.isin(X_train.index)]
Y_test=datay.loc[~ds.index.isin(Y_train.index)]
#print X_test
#print Y_test

x=np.array(X_test,dtype=float)
y=np.array(Y_test)

#print y

#print X_train
#print X_test
#print x
#print np.transpose(x)

#########mean#######
mean=X_train.groupby(ds.ans).mean()
#print mean
mean1= np.unique(np.array([y])).reshape(c,1)
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
#print len(x)
for i in range(0,len(x),1):
    #print x[i]
    m=0
    for j in range (0,len(means),1):
        p=discr(x[i],j)
        if(p>m):
            m=p
            n=j
    #print mean1[n]
    l.append(mean1[n])
    
r=np.array(l)

#######printing fitted values####    
#print  r

#####printing actual value###
#print y


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
print 'Accuracy= ' + str(acc)


######plotting#####
#x1,y1=make_classificiation(n_features=2,n_redundant=0,n_informative=2,n_clustres_per_class=1)
#plt.scatter(X_train[:,0],X_train[:,1],c=Y_train)
#plt.show()
   
