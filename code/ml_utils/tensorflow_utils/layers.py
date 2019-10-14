import tensorflow as tf
import numpy as np
import sonnet as snt
import sys

### Distance Functions ###
def hiddenParams(inputDim,outputDim):
    scale = np.sqrt(6.0/(inputDim + outputDim))
    initial = tf.random_uniform([inputDim,outputDim], minval = -scale, maxval = scale)
    #initial = tf.truncated_normal([inputDim,outputDim], stddev=0.01)

    W = tf.Variable(initial)
    b = tf.Variable(tf.zeros([outputDim]))

    return W,b

def conv2dParams(inputShape,filterDim1,filterDim2,nChannelsOut):
    nChannelsIn = inputShape[2]
    W = tf.Variable(tf.random_uniform([filterDim1,filterDim2,nChannelsIn,nChannelsOut],
                -1.0 / np.sqrt(nChannelsIn),1.0 / np.sqrt(nChannelsIn)))
    b = tf.Variable(tf.zeros([nChannelsOut]))

    return W,b

def hidden(x,outputDim):
    inputDim = int(x.get_shape()[1])

    scale = np.sqrt(6.0/(inputDim + outputDim))
    initial = tf.random_uniform([inputDim,outputDim], minval = -scale, maxval = scale)
    #initial = tf.truncated_normal([inputDim,outputDim], stddev=0.01)

    W = tf.Variable(initial)
    b = tf.Variable(tf.zeros([outputDim]))
    
    return tf.matmul(x,W) + b

def rbfGrid(x,startRange,endRange,rbfDim):
    ### Initialization
    inputDim = int(x.get_shape()[1])
    numRbf = rbfDim ** inputDim 

    means = np.zeros([numRbf,inputDim])

    rbfRange = float(endRange-startRange)
    stepSize = rbfRange/(rbfDim-1.0)

    funcPos = np.arange(startRange,endRange + stepSize,stepSize)

    for d in range(0,inputDim):
        for f in range(0,numRbf):
            p = int(np.floor(f/(rbfDim ** d)) % rbfDim)
            means[f,d] = funcPos[p]
    
    sdInit = np.sqrt( inputDim * (rbfRange/(rbfDim-1.0))**2.0)


    rbfMeans = tf.convert_to_tensor(means,dtype=tf.float32)
    rbfSds = tf.constant(sdInit,shape=[numRbf],dtype=tf.float32)

    # reshape
    rbfMeans = tf.reshape(tf.transpose(rbfMeans),[1,inputDim,numRbf])
    rbfSds = tf.reshape(rbfSds,[1,1,numRbf])

    ### Forward prop
    #pdb.set_trace()
    x = tf.expand_dims(x,2)
    expInput = -tf.reduce_sum(tf.pow(x - rbfMeans,2.0)/tf.pow(rbfSds,2.0),1) # sum across input dim
    expMean = tf.div(tf.reduce_mean(tf.exp(expInput),0),rbfDim ** inputDim) # mean across examples
    output = tf.expand_dims(expMean,0)

    #output = tf.zeros([0])
    #for f in range(0,numRbf):
    #    rbfMeansSlice = tf.slice(rbfMeans,[f,0],[1,inputDim])
    #    expInput = -tf.reduce_sum(tf.pow(x - rbfMeansSlice,2.0)/tf.pow(rbfSds[f],2.0),1)
    #    expMean = tf.div(tf.reduce_mean(tf.exp(expInput)),rbfDim ** inputDim)
    #    expMean = tf.expand_dims(expMean,0)
    #    output = tf.concat([output,expMean],0)

    return output

### Activation functions ###
def linear(x):
    return(x)

def relu(x):
    return tf.nn.relu(x)

def sigmoid(x):
    return tf.nn.sigmoid(x)

def tanh(x):
    return tf.nn.tanh(x)

def elu(x):
    return tf.nn.elu(x)

def absApl(x):
    absVar = tf.Variable(tf.zeros([1]))
    output = tf.nn.relu(x) + tf.abs(absVar)*tf.maximum(-x,0.0)
    return(output)

def apl(x):
    outputs = []
    #paramsShape = [1,int(x.shape[1])]
    paramsShape = [1]
    for hingeIdx,hingePos in enumerate([0.0,1.0,2.0]):
        slopeNeg = tf.get_variable("slopeNeg_%i" % (hingeIdx),initializer=tf.zeros(paramsShape))
        if hingePos == 0.0:
            slopePos = tf.get_variable("slopePos_%i" % (hingeIdx),initializer=tf.zeros(paramsShape)+1)
            #slopePos = tf.Variable(tf.zeros(paramsShape) + 1.0)
        else:
            slopePos = tf.get_variable("slopePos_%i" % (hingeIdx),initializer=tf.zeros(paramsShape))
            #slopePos = tf.Variable(tf.zeros(paramsShape))

        outputs.append(slopePos*tf.maximum(x  - hingePos,0.0) + slopeNeg*tf.maximum(-x - hingePos,0.0))

    output = tf.add_n(outputs)
    return(output)

def andOut(x,onThresh=1.0,blockSize=10):
    assert((int(x.shape[-1]) % blockSize) == 0)

    x_reshape = tf.reshape(x,[-1,int(x.shape[1]/blockSize),blockSize])
    
    whichOff = x_reshape < onThresh

    mask = tf.reduce_min(tf.cast(whichOff,tf.float32),axis=2)
    
    output = mask*tf.reduce_mean(x_reshape,axis=2)

    return(output)

# Some code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/layers/maxout.py
def maxout(x,block_size):
    axis = -1
    shape = x.get_shape().as_list()
    num_channels = shape[axis]
    if num_channels % block_size != 0:
        raise ValueError("number of channels %i is not a multiple of block size %i" % (num_channels,block_size))

    shape[axis] = num_channels/block_size
    shape = shape + [block_size]

    shape = [s if s != None else -1 for s in shape]

    output = tf.reduce_max(tf.reshape(x,shape),axis)

    return(output)

def toSinCos(x):
    eps = 1e-5
    denom = tf.sqrt(tf.pow(x[:,0],2) + tf.pow(x[:,1],2)) + eps
    sin = tf.expand_dims(x[:,0]/denom,1)
    cos = tf.expand_dims(x[:,1]/denom,1)

    return(tf.concat([sin,cos],1))

def getActFunc(actType):
    actType = actType.upper()
    if actType == "RELU":
        actFn = tf.nn.relu
    elif actType == "SIGMOID":
        actFn = tf.nn.sigmoid
    elif actType == "TANH":
        actFn = tf.nn.tanh
    elif actType == "ELU":
        actFn = tf.nn.elu
    elif actType == "ABS":
        actFn = tf.abs
    elif actType == "APL":
        actFn = apl
    elif actType == "ABSAPL":
        actFn = absApl
    elif actType == "LINEAR":
        actFn = linear
    elif actType == "AND":
        actFn = andOut
    else:
        print "ERROR: Unknown activation function %s" % (actType)
        sys.exit()

    return(actFn)

### Layers ###
def concat(x,y):
    output = tf.concat([x,y],1)
    return(output)

def batchNorm(inp,isTraining,center=True,scale=True,axis=1):
    output = tf.layers.batch_normalization(inp,epsilon=1e-5, axis=axis,
                                      fused=True, center=center, scale=scale,
                                      training=isTraining)
    return(output)


def angle_norm(x,outputDim):
    inputDim = int(x.get_shape()[1])

    scale = np.sqrt(6.0/(inputDim + outputDim))
    initial = tf.random_uniform([inputDim,outputDim], minval = -scale, maxval = scale)
    #initial = tf.truncated_normal([inputDim,outputDim], stddev=0.01)

    eps = 0.001

    W = tf.get_variable('weight',initializer=initial)
    b = tf.get_variable('bias',initializer=eps*tf.ones([outputDim]))
    #W = tf.Variable(initial) # TODO change to get_variable
    #b = tf.Variable(tf.zeros([outputDim]))

    Wnorm = tf.expand_dims(tf.sqrt(tf.reduce_sum(W**2, axis=0) + b**2),0)
    xNorm = tf.expand_dims(tf.sqrt(tf.reduce_sum(x**2, axis=1) + eps**2),1)
    output = tf.div(tf.div(tf.matmul(x,W) + eps*b, Wnorm),xNorm)

    #Wnorm = tf.expand_dims(tf.norm(W,axis=0),0)
    #xNorm = tf.expand_dims(tf.norm(x,axis=1),1)

    #output = tf.div(tf.div(tf.matmul(x,W),Wnorm),xNorm) + b

    return(output)


def linear_weight_norm(x,outDim):
    inputDim = int(x.get_shape()[1])

    scale = np.sqrt(6.0/(inputDim + outDim))
    initial = tf.random_uniform([inputDim,outDim], minval = -scale, maxval = scale)
    #initial = tf.truncated_normal([inputDim,outDim], stddev=0.01)

    W = tf.Variable(initial)
    b = tf.Variable(tf.zeros([outDim]))
    coeff = tf.Variable(tf.ones([1,outDim]))

    Wnorm = tf.expand_dims(tf.norm(W,axis=0),0)
    
    output = coeff*tf.div(tf.matmul(x,W),Wnorm) + b

    return(output)

def dense(inp,outDim,actType,isTraining=None,doBatchNorm=False,l2=0.0,weightNorm=False,layerNorm=False,angleNorm=False):
    #if l2 > 0:
    #    regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=l2)}
    
    assert(not (doBatchNorm and layerNorm))
    assert(not (weightNorm and angleNorm))


    if weightNorm == True:
        innerProd = snt.Sequential([lambda x: linear_weight_norm(x,outDim)])(inp)
    elif angleNorm == True:
        innerProd = snt.Sequential([lambda x: angle_norm(x,outDim)])(inp)
    else:
        innerProd = snt.Sequential([snt.Linear(output_size=outDim)])(inp)   

    if doBatchNorm == True:
        center_bn = True
        if weightNorm == True:
            scale_bn = False
        else:
            scale_bn = True

        preAct = batchNorm(innerProd,isTraining,center_bn,scale_bn,1)
    elif layerNorm:
        preAct = tf.contrib.layers.layer_norm(innerProd)
    else:
        preAct = innerProd
        
    output = snt.Sequential([getActFunc(actType)])(preAct)
    #output = getActFunc(actType)(preAct)

    return(output)

def conv2d(inp,numFilts,kernelSize,strides,padding,actType,isTraining=None,doBatchNorm=False,l2=0.0,weightNorm=False):
    convOut = tf.layers.conv2d(inp,numFilts,kernelSize,strides,padding,data_format='channels_last')

    if doBatchNorm == True:
        center_bn = True
        if weightNorm == True:
            scale_bn = False
        else:
            scale_bn = True

        preAct = batchNorm(convOut,isTraining,center_bn,scale_bn,3)
    else:
        preAct = convOut

    output = snt.Sequential([getActFunc(actType)])(preAct)

    return(output)

def resBlock(inp,hDim,actType,numL,isTraining=None,doBatchNorm=False,l2=0.0,weightNorm=False,layerNorm=False,angleNorm=False):
    layerInput = inp
    for layerIdx in range(numL):
        if layerIdx == (numL-1):
            actType_L = "linear"
        else:
            actType_L = actType

        layerInput = dense(layerInput,hDim,actType_L,isTraining,doBatchNorm,l2,weightNorm,layerNorm,angleNorm)

    blockOutPlusInp = tf.add(layerInput, inp)
    
    output = snt.Sequential([getActFunc(actType)])(blockOutPlusInp)
    #output = getActFunc(actType)(blockOutPlusInp)

    return(output)

def resBlockDense(inp,hDim,actType,numBlocks,numL,isTraining=None,doBatchNorm=False,l2=0.0,weightNorm=False,layerNorm=False,angleNorm=False):
    layerInputs = [inp]
    for block in range(numBlocks):
        layerInput = layerInputs[-1]
        for layerIdx in range(numL):
            if layerIdx == (numL-1):
                actType_L = "linear"
            else:
                actType_L = actType

            layerInput = dense(layerInput,hDim,actType_L,isTraining,doBatchNorm,l2,weightNorm,layerNorm,angleNorm)

        Wmult = tf.get_variable('weight_%i_%i' % (block,block),initializer=tf.ones(shape=(1,hDim)))
        blockOutPlusInp = tf.add(layerInput, Wmult*layerInputs[-1])

        for prevBlock in range(0,block):
            Wmult = tf.get_variable('weight_%i_%i' % (prevBlock,block),initializer=tf.zeros(shape=(1,hDim)))
            blockOutPlusInp = tf.add(blockOutPlusInp, Wmult*layerInput[prevBlock])

        
        output = snt.Sequential([getActFunc(actType)])(blockOutPlusInp)

        layerInputs.append(output)   

    return(output)

def resBlockPreAct(inp,hDim,actType,numL,isTraining=None,doBatchNorm=False,l2=0.0,weightNorm=False,layerNorm=False,angleNorm=False):
    layerInput = inp
    if doBatchNorm == True:
        center_bn = True
        if weightNorm == True:
            scale_bn = False
        else:
            scale_bn = True

        layerInput = batchNorm(layerInput,isTraining,center_bn,scale_bn,1)

    layerInput = snt.Sequential([getActFunc(actType)])(layerInput)
    for layerIdx in range(numL):
        if layerIdx == (numL-1):
            actType_L = "linear"
            doBatchNorm_L = False
        else:
            actType_L = actType
            doBatchNorm_L = doBatchNorm

        layerInput = dense(layerInput,hDim,actType_L,isTraining,doBatchNorm_L,l2,weightNorm,layerNorm,angleNorm)

    output = tf.add(layerInput, inp)
    
    return(output)

def resBlockConv2d(inp,numFilts,kernelSize,strides,padding,actType,numL,isTraining=None,doBatchNorm=False,l2=0.0,weightNorm=False):
    layerInput = inp
    for layerIdx in range(numL):
        if layerIdx == (numL-1):
            actType_L = "linear"
        else:
            actType_L = actType

        layerInput = conv2d(layerInput,numFilts,kernelSize,strides,padding,actType_L,isTraining,doBatchNorm,l2,weightNorm)

    blockOutPlusInp = tf.add(layerInput, inp)
    
    output = snt.Sequential([getActFunc(actType)])(blockOutPlusInp)

    return(output)

def skipBlock(inp,hDim,actType,isTraining=None,doBatchNorm=False,l2=0.0,weightNorm=False):
    
    layerOutput = dense(inp,hDim,actType,isTraining=isTraining,doBatchNorm=doBatchNorm,l2=l2,weightNorm=weightNorm)

    output = tf.concat([inp,layerOutput],1)

    print(output.shape)
    return(output)

def getSonnetNet(layerTypes,hDims,activFuncs):
    layers = []
    for layerType,activFunc,hDim in zip(layerTypes,activFuncs,hDims):
        if layerType == "fc":
            layers.append(snt.Linear(output_size=hDim))
        elif layerType == "sin_cos":
            layers.append(toSinCos)

        if activFunc == "relu":
            layers.append(tf.nn.relu)
        elif activFunc == "tanh":
            layers.append(tf.nn.tanh)
        elif activFunc == "sigmoid":
            layers.append(tf.nn.sigmoid)
        elif activFunc != "linear":
            print "ERROR: Unknown activation function %s" % (activFunc)
            sys.exit()
    net = snt.Sequential(layers)

    return(net)

