import numpy as np
import pandas

ds=pandas.read_csv('C:\Users\Akankshya\Desktop\pattern recog\pyt\iris_ds.csv')

###########category_to_numeric#####
#ds['flower']=ds['flower'].astype('category')
#ds['flower']=ds['flower'].cat.codes
ds=pandas.get_dummies(ds,columns=['flower'])
#print ds.head()

datax=ds[['sw','sl','pw','pl']]
datay=ds.columns.isin(datax)
#print datay


values=np.unique(datay)
#print values

c=len(values)

#######random sampling classwise########
r=[]
prop_amt=0.67
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

########array######
X=np.array(X_train)
#print X

Y=np.array(np.array([Y_train]).T)
#print Y

def sigmoid(x):
	return 1/(1+np.exp(-x))

def der_sig(x):
	return x*(1-x)

epoch=5000 ###no_of_times
lr=0.1     ###learning_rate
ip_=X.shape[1]  ####Input neuron size
print ip_      

hl=3     #####no of neurons in hidden layer perceptron
op=1     #####no of neurons in output layer 

w1=np.random.uniform(size=(ip_,hl))   ####weight of input layer
b1=np.random.uniform(size=(1,hl))    ###bias of ip layer
w2=np.random.uniform(size=(hl,op))    ####weight of hidden layer
b2=np.random.uniform(size=(1,op))     ###bias of hidden layr

for i in xrange(5000):
    #forwardpropagation
    hlip1=np.dot(X,w1)
    hlip=hlip1+b1
    hla=sigmoid(hlip)
    opip1=np.dot(hla,w2)
    opip=opip1+b2
    opt=sigmoid(opip)

    ##backpropagation####
    e=Y-opt
    slope_op=der_sig(opt)
    slope_hl=der_sig(hla)

    d_op=e*slope_op
    #print d_op.shape
    
    e_hl=d_op.dot(w2.T)

    d_hl=e_hl*slope_hl

    w2+=hla.T.dot(d_op)*lr
    w1+=X.T.dot(d_hl)*lr

    b2+=np.sum(d_op,axis=0,keepdims=True)*lr
    b1+=np.sum(d_hl,axis=0,keepdims=True)*lr


print opt

#print opt.shape    
#print e.shape
#print d_op.shape


