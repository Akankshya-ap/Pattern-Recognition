import numpy as np

X=np.array([[1,0,1,0],
	[1,0,1,1],
	[0,1,0,1],
            [1,0,1,0],
	[1,0,1,1],
	[0,1,0,1],
            [1,0,1,0],
	[1,0,1,1],
	[0,1,0,1],
            [1,0,1,0],
	[1,0,1,1],
	[0,1,0,1]])

Y=np.array([[1],[1],[0],[1],[1],[0],[1],[1],[0],[1],[1],[0]])

def sigmoid(x):
	return 1/(1+np.exp(-x))

def der_sig(x):
	return x*(1-x)

epoch=5000
lr=0.1
ip_=X.shape[1]
hl=3
op=1

w1=np.random.uniform(size=(ip_,hl))
b1=np.random.uniform(size=(1,hl))
w2=np.random.uniform(size=(hl,op))
b2=np.random.uniform(size=(1,op))

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
    e_hl=d_op.dot(w2.T)

    d_hl=e_hl*slope_hl

    w2+=hla.T.dot(d_op)*lr
    w1+=X.T.dot(d_hl)*lr

    b2+=np.sum(d_op,axis=0,keepdims=True)*lr
    b1+=np.sum(d_hl,axis=0,keepdims=True)*lr


print opt

