import numpy as np
import matplotlib.pyplot as plt
import pandas
#from sklearn.datasets import make_classifiaction
#from sklearn.dtatsets import make_blobs

ds=pandas.read_csv('C:\Users\Akankshya\Desktop\pattern recog\pyt\iris_ds.csv')

datay=ds['flower']

datax=ds[['sw','sl','pw','pl']]


###different classes division#####
values=np.unique(datay)
#print y
r=[]
for value in values:
   # print value
    x= np.count_nonzero(datay==value)
    inds=np.nonzero(datay==value)[0]
    #print inds
    np.random.shuffle(inds)
    p=int(0.6*x)
    for i in range(0, p,1):
        r.append(inds[i])
       # print r
print len(np.array(r))


########Train_test#######
Y_train=datay.loc[ds.index.isin(r)]
X_train=datax.loc[ds.index.isin(r)]
#print X_train
#print Y_train


X_test=datax.loc[~ds.index.isin(X_train.index)]
Y_test=datay.loc[~ds.index.isin(Y_train.index)]
#print X_test


x=np.array(X_test,dtype=float)
y=np.array(Y_test)

'''
def get_train_test_inds(y,train_proportion=0.7):
    y=np.array(datay)
    train_inds = []
    x=[]
    test_inds = np.zeros(len(y),dtype=float)
    values = np.unique(y)
   # v=y.count()
    #print v
    #print values
    for value in values:
        value_inds = np.nonzero(y==value)[0]
        print value_inds
        np.random.shuffle(value_inds)
        n = int(train_proportion*len(value_inds))
        for i in range(0,n,1):
            train_inds.append(value_inds)

        
        #train_inds=np.array(datax.loc[ds.index.isin(value_inds[:n])])
        #print train_inds
        x.append(train_inds)
        Y_train=datay.loc[ds.index.isin(value_inds[:n])]
        #test_inds=datax.loc[~ds.index.isin(value_inds[:n])]
       # test_inds[value_inds[n:]]=True
    print train_inds
    return x #,test_inds

#y = np.array([1,1,2,2,3,3])
train_inds = get_train_test_inds(datay,train_proportion=0.5)
#test_inds=datax.loc[~ds.index.isin(train_inds.index)]
#print train_inds
#print test_inds

'''
#print X_train
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
print 'Accuracy=' + str(acc)


######plotting#####
#x1,y1=make_classificiation(n_features=2,n_redundant=0,n_informative=2,n_clustres_per_class=1)
#plt.scatter(X_train[:,0],X_train[:,1],c=Y_train)
#plt.show()
   
