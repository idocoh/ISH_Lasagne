'''
Created on Jun 9, 2015

@author: Abigail
'''
from __future__ import print_function
import numpy as np
import scipy
import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelBinarizer
import theano
import theano.tensor as T
import cPickle as pickle

theano.config.exception_verbosity='high'

def run_svm():
    
    HIDDEN_LAYER_OUTPUT_FILE_NAME =  "../results/" + "noLearn_50_3_hiddenLayerOutput_0" 
    with open(HIDDEN_LAYER_OUTPUT_FILE_NAME+".pickle",'r') as f:
        ob,cl = pickle.load(f)
        f.close()

    P=ob[cl.astype(bool)[:,0],:]
    N=ob[~cl.astype(bool)[:,0],:]
        
    dtype = theano.config.floatX
    num_features = 100 #20   # dimensions
    num_classes = 2
    num_pos = P.shape[0]
    P_sigma = 1
    P_mu = 10
#     P = P_sigma * np.random.randn(num_pos,num_features) + P_mu
    num_neg = N.shape[0]
    N_sigma = 1
    N_mu = 1
#     N = N_sigma * np.random.randn(num_neg,num_features) + N_mu
    X = np.vstack([P, N]).astype(theano.config.floatX)
    pos = np.atleast_2d(np.ones(num_pos, dtype=int)).T
    neg = np.atleast_2d(-1 * np.ones(num_neg, dtype=int)).T
    y = np.vstack((pos,neg)) #.astype(theano.config.floatX)
    num_instances = y.shape[0]
    print(num_instances)
    
    plt.scatter(X[:,0],X[:,1])
#     plt.show()

    #Y = LabelBinarizer(neg_label=-1, pos_label=1).fit_transform(y).astype(theano.config.floatX)
    print(y)
    Y = np.hstack((-y,y))
    Y = Y.astype(dtype)
    print(Y)
    
    W0 = np.zeros((num_features, num_classes), dtype=dtype)
    b0 = np.zeros((1, num_classes), dtype=dtype)
    print(X.shape)
    print(Y.shape)
    print(W0.shape)
    print(b0.shape)
    
    W_s = theano.shared(W0, name="W_s")
    b_s = theano.shared(b0, name="b_s", broadcastable=[True,False])
    X_s = T.matrix("X_s", dtype=theano.config.floatX)
    Y_s = T.matrix("Y_s", dtype=theano.config.floatX)
    margin_s = Y_s * (theano.dot(X_s, W_s) + b_s)
    hinge_s = T.ones_like(margin_s) - margin_s
    nneg_s = hinge_s > 0
    nneg_s.astype(theano.config.floatX)
    hinge_s = hinge_s * nneg_s
    svm_cost_s = T.sum(T.mean(hinge_s, axis=0))
    gW_s, gb_s = T.grad(svm_cost_s, [W_s, b_s])
    pred = theano.dot(X_s, W_s) + b_s
    lrate = 0.1
    updates = [(W_s, W_s - lrate * gW_s), (b_s, b_s - lrate * gb_s)]
    
    train = theano.function(inputs=[X_s,Y_s], outputs=[W_s, b_s], updates=updates)
   
    x = np.linspace(-10,10,200)
#     plt.scatter(X[:,0],X[:,1])

    for i in range(10000):
        WW, bb = train(X, Y)
        
        print(WW)
        print('')
        print('')
        print('')
        print(bb)
        if (i%1000 == 0):
            f_x = (-1.0 * WW[0][0] / WW[1][0]) * x - ( bb[0][0] / WW[1][0]) 
            plt.plot(x,f_x)
#             plt.show()
    
    fig1 = plt.figure()    
#     x = np.linspace(-10,10,200)
    f_x = (-1.0 * WW[0][0] / WW[1][0]) * x - ( bb[0][0] / WW[1][0]) 
    plt.plot(x,f_x)
    plt.scatter(X[:,0],X[:,1])
#     plt.show()
    
    plotHyper(X,WW,bb)


def plotHyper(X,W,b):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import numpy as np
    import matplotlib.pyplot as plt
    
    point = np.array([1, 2, 3])
    normal = np.array([1, 1, 2])
    
    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    d = -point.dot(normal)
    
    # create x,y
    xx, yy = np.meshgrid(range(10), range(10))
    
    # calculate corresponding z
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    
    # plot the surface
    plt3d = plt.figure().gca(projection='3d')
    plt.scatter(X[:,0],X[:,1])

    
    Gx, Gy = np.gradient(xx * yy)  # gradients with respect to x and y
    G = (Gx ** 2 + Gy ** 2) ** .5  # gradient magnitude
    N = G / G.max()  # normalize 0..1
    
    plt3d.plot_surface(xx, yy, z, rstride=1, cstride=1,
                       facecolors=cm.jet(N),
                       linewidth=0, antialiased=False, shade=False
    )
    plt.show()
    
    
if __name__ == '__main__':
    run_svm()
#     plotHyper()

