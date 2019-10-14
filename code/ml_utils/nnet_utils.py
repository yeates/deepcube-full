import tensorflow as tf
from tensorflow.contrib import predictor
import sonnet as snt
import numpy as np
import cPickle as pickle

import os
import sys
sys.path.append('./')

from tensorflow_utils import layers

from multiprocessing import Queue

def nnetPredict(x,inputQueue,predictor,Environment,batchSize=10000,realWorld=True):
    stateVals = np.zeros((0,1))

    numExamples = x.shape[0]
    startIdx = 0
    while startIdx < numExamples:
        endIdx = min(startIdx + batchSize,numExamples)

        x_itr = x[startIdx:endIdx]
        if realWorld == True:
            states_nnet = Environment.state_to_nnet_input(x_itr)
        else:
            states_nnet = x_itr

        states_nnet = np.expand_dims(states_nnet,1)

        inputQueue.put(states_nnet)

        numStates = states_nnet.shape[0]
        stateVals_batch = np.array([predictor.next()['values'].max() for _ in range(numStates)])
        stateVals_batch = np.expand_dims(stateVals_batch,1)

        stateVals = np.concatenate((stateVals,stateVals_batch),axis=0)

        startIdx = endIdx

    assert(stateVals.shape[0] == numExamples)

    return(stateVals)

def getEstimatorPredFn(network,inputDim,Environment,batchSize=10000):
    inputQueue = Queue(1)

    tf_dtype = Environment.tf_dtype()

    def inputGen():
        while True:
            yield inputQueue.get()

    def input_fn_test():
        ds = tf.data.Dataset.from_generator(inputGen,(tf_dtype),(tf.TensorShape([None,1]+inputDim)))
        return(ds)

    predictor = network.predict(input_fn_test)

    def predFn(x,realWorld=True):
        return(nnetPredict(x,inputQueue,predictor,Environment,batchSize,realWorld))

    return(predFn)

def nnetPredict_exported(predict_fn,x,Environment,batchSize=10000,realWorld=True):
    stateVals = np.zeros((0,1))

    numExamples = x.shape[0]
    startIdx = 0
    while startIdx < numExamples:
        endIdx = min(startIdx + batchSize,numExamples)

        x_itr = x[startIdx:endIdx]
        if realWorld == True:
            states_nnet = Environment.state_to_nnet_input(x_itr)
        else:
            states_nnet = x_itr

        states_nnet = np.expand_dims(states_nnet,1)

        stateVals_batch = predict_fn({"x": states_nnet})['output']

        stateVals = np.concatenate((stateVals,stateVals_batch),axis=0)

        startIdx = endIdx

    assert(stateVals.shape[0] == numExamples)

    return(stateVals)

def loadNnet(modelLoc,modelName,useGPU,Environment,batchSize=10000,gpuNum=None):
    assert(modelLoc != "")
    
    argsFile = "%s/args.pkl" % (modelLoc)
    exportDir = "%s/exported_model/" % (modelLoc)
    if os.path.isfile(argsFile) or True:
        CONFIG = tf.ConfigProto()
        CONFIG.gpu_options.allow_growth = True # Prevents tf from grabbing all gpu memory.

        if useGPU and len(os.environ['CUDA_VISIBLE_DEVICES']) > 0:
            print('\nRunning from GPU %s' % str(os.environ['CUDA_VISIBLE_DEVICES']))
        else:
            print('\nRunning from CPU')

        config = tf.estimator.RunConfig(session_config=CONFIG)
        tf.InteractiveSession(config=CONFIG)

        if os.path.isdir(exportDir) or True:
            predict_fn = predictor.from_saved_model(export_dir=exportDir)
            def nnetFn(x,realWorld=True):
                return(nnetPredict_exported(predict_fn,x,Environment,batchSize,realWorld))
        else:
            args = pickle.load(open(argsFile,"rb"))
            nnet_model_fn = lambda features,labels,mode : model_fn(features,labels,mode,args)
            network = tf.estimator.Estimator(model_fn=nnet_model_fn, config=config, model_dir=modelLoc)

            inputDim = list(Environment.state_to_nnet_input(Environment.solvedState).shape[1:])

            nnetFn = getEstimatorPredFn(network,inputDim,Environment,batchSize) # TODO parallel calls to estimator will prob result in errors

    return(nnetFn)

def getModelName(args):
    if args.nnet_name == "":
        labeledDataName = args.labeled_data.split("/")
        labeledDataName.remove("")
        if len(labeledDataName) == 0:
            labeledDataName = ['False']

        nnetName = "l%i_resb_%i_h%i_act%s_bs%i_lri%s_lrd%s_momi%s_momf%s_opt%s_l2%s_dop%s_env%s_sr_%i_%i_lab%s" % (args.num_l,args.num_res,args.num_h,args.act_type.upper(),args.batch_size,args.lr_i,args.lr_d,args.mom_i,args.mom_f,args.opt.upper(),args.l2,args.drop_p,args.env.upper(),args.scramb_min,args.scramb_max,labeledDataName[-1])

        if args.batch_norm:
            nnetName = nnetName + "_bn"
        if args.layer_norm:
            nnetName = nnetName + "_ln"
        if args.weight_norm:
            nnetName = nnetName + "_wn"
        if args.angle_norm:
            nnetName = nnetName + "_an"
    else:
        nnetName = args.nnet_name
        
    if args.debug != 0:
        nnetName = "%s_debug" % (nnetName)

    return(nnetName)

def addNnetArgs(parser):
    parser.add_argument('--env', type=str, default='cube3', help="Environment: cube3, cube4, puzzle15, puzzle24")
    parser.add_argument('--solve_itrs', type=int, default=5000, help="How often to test")

    # Architecture
    parser.add_argument('--num_l', type=int, default=1, help="Number of hidden layers")
    parser.add_argument('--num_res', type=int, default=1, help="Number of residual blocks")
    parser.add_argument('--num_h', type=int, default=100, help="Number of hidden neurons")
    parser.add_argument('--act_type', type=str, default="relu", help="Type of activation function")
    parser.add_argument('--skip_depth', type=int, default=0, help="How far back to concatenate previous layers")
    parser.add_argument('--maxout_bs', type=int, default=2, help="Maxout block size")

    parser.add_argument('--debug', action='store_true', default=False, help="Deletes labeled file after opening")

    parser.add_argument('--batch_norm', action='store_true', default=False, help="Add if doing batch normalization")
    parser.add_argument('--layer_norm', action='store_true', default=False, help="Add if doing layer normalization")
    parser.add_argument('--weight_norm', action='store_true', default=False, help="Add if doing weight normalization")
    parser.add_argument('--angle_norm', action='store_true', default=False, help="Add if doing angle normalization")

    # Gradient descent
    parser.add_argument('--max_itrs', type=int, default=5000000, help="Maxmimum number of iterations")

    parser.add_argument('--lr_i', type=float, default=0.001, help="Initial learning rate")
    parser.add_argument('--lr_d', type=float, default=0.9999993, help="Learning rate decay")
    parser.add_argument('--mom_i', type=float, default=0.0, help="Initial momentum")
    parser.add_argument('--mom_f', type=float, default=0.0, help="Final momentum")
    parser.add_argument('--mom_c_s', type=int, default=500000, help="Momentum change steps")
    parser.add_argument('--batch_size', type=int, default=100, help="Batch size")
    parser.add_argument('--opt', type=str, default="adam", help="Optimization method: sgd or adam")

    parser.add_argument('--bm', type=float, default=0.9, help="b1 for adam")
    parser.add_argument('--bv', type=float, default=0.999, help="b2 for adam")

    # Regularization
    parser.add_argument('--drop_p', type=float, default=0.0, help="Probabiliy of a neuron being dropped out")
    parser.add_argument('--l2', type=float, default=0.0, help="L2 for weight regularization")

    # Input/Output Format
    parser.add_argument('--in_type', type=str, default="fc", help="Type of input")
    parser.add_argument('--out_type', type=str, default="linear", help="Type of output")

    # Problem difficulty
    parser.add_argument('--scramb_min', type=int, default=1, help="Minimum number of scrambles to train on")
    parser.add_argument('--scramb_max', type=int, default=20, help="Maximum number of scrambles to train on")

    parser.add_argument('--max_turns', type=int, default=None, help="Maximum number of turns when solving")

    # Labeled data set
    parser.add_argument('--labeled_data', type=str, required=True, help="File for labeled data")

    parser.add_argument('--nnet_name', type=str, default="", help="Replace nnet name with this name, if exists")
    parser.add_argument('--save_dir', type=str, default="savedModels", help="Director to which to save model")

    parser.add_argument('--delete_labeled', action='store_true', default=False, help="Deletes labeled file after opening")

    parser.add_argument('--model_num', type=int, default=0, help="Model number for progressive learning")

    parser.add_argument('--eps', type=float, default=None, help="Training stops if test set falls below specified error for supervised training. Default is none, meaning no early stopping due to this argument.")

    return(parser)

def statesToStatesList(states,env):
    if env == 'cube3':
        #oneHot_idx = tf.one_hot(states,24,on_value=1,off_value=0)
        oneHot_idx = tf.one_hot(states,6,on_value=1,off_value=0)
        outRep = tf.reshape(oneHot_idx,[-1,int(oneHot_idx.shape[1])*int(oneHot_idx.shape[2])])
    elif env == 'puzzle15':
        oneHot_idx = tf.one_hot(states,4*4,on_value=1,off_value=0)
        outRep = tf.reshape(oneHot_idx,[-1,int(oneHot_idx.shape[1])*int(oneHot_idx.shape[2])])
    elif env == 'puzzle24':
        oneHot_idx = tf.one_hot(states,5*5,on_value=1,off_value=0)
        outRep = tf.reshape(oneHot_idx,[-1,int(oneHot_idx.shape[1])*int(oneHot_idx.shape[2])])
    elif env == 'puzzle35':
        oneHot_idx = tf.one_hot(states,6*6,on_value=1,off_value=0)
        outRep = tf.reshape(oneHot_idx,[-1,int(oneHot_idx.shape[1])*int(oneHot_idx.shape[2])])
    elif env == 'puzzle48':
        oneHot_idx = tf.one_hot(states,7*7,on_value=1,off_value=0)
        outRep = tf.reshape(oneHot_idx,[-1,int(oneHot_idx.shape[1])*int(oneHot_idx.shape[2])])
    elif 'lightsout' in env:
        outRep = states
    elif 'hanoi' in env:
        m = re.search('hanoi([\d]+)d([\d]+)p',env)
        numPegs = int(m.group(2))
        oneHot_idx = tf.one_hot(states,numPegs,on_value=1,off_value=0)
        outRep = tf.reshape(oneHot_idx,[-1,int(oneHot_idx.shape[1])*int(oneHot_idx.shape[2])])
    elif 'sokoban' in env:
        outRep = states

    return(tf.cast(outRep,tf.float32))

def model_fn(features,labels,mode,args):
    if type(features) == type(dict()):
        states = features["x"][:,0,:]
    else:
        states = features[:,0,:]

    if mode == tf.estimator.ModeKeys.TRAIN:
        isTraining = True
    else:
        isTraining = False

    ### Process states
    statesProcessed = statesToStatesList(states,args.env)
    print("Processed shape: {}".format(statesProcessed.shape[1]))

    dropoutSonnet = lambda x: tf.nn.dropout(x,keep_prob=1-args.drop_p)

    nnet_layers = []
    doBatchNorm = args.batch_norm
    layerNorm = args.layer_norm
    weightNorm = args.weight_norm
    angleNorm = args.angle_norm

    nnet_layers.append(lambda x: layers.dense(x,5000,args.act_type,isTraining,doBatchNorm,args.l2,weightNorm,layerNorm,angleNorm))
    for layerIdx in range(0,args.num_l):
        nnet_layers.append(lambda x: layers.dense(x,args.num_h,args.act_type,isTraining,doBatchNorm,args.l2,weightNorm,layerNorm,angleNorm))
    for layerIdx in range(0,args.num_res):
        nnet_layers.append(lambda x: layers.resBlock(x,args.num_h,args.act_type,2,isTraining,doBatchNorm,args.l2,weightNorm,layerNorm,angleNorm))

    nnet_layers.append(dropoutSonnet)

    nnet_layers.append(lambda x: layers.dense(x,1,"linear",isTraining,False,args.l2,False,False,False))

    nnet = snt.Sequential(nnet_layers)

    ### Get curr state value
    stateVals_nnet = nnet(statesProcessed)

    global_step = tf.train.get_global_step()

    predictions = {"values": stateVals_nnet}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return(tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs={"y":tf.estimator.export.PredictOutput(stateVals_nnet)}))

    ### Get state value target
    stateVals_dp = tf.cast(labels,tf.float32)

    ### Cost
    stateVals_dp = tf.stop_gradient(stateVals_dp)
    errs = stateVals_dp - stateVals_nnet

    cost = tf.reduce_mean(tf.pow(errs,2),name="cost")

    if mode == tf.estimator.ModeKeys.EVAL:
        return(tf.estimator.EstimatorSpec(mode, loss=cost))

    ### Tests
    scrambleRange = [args.scramb_min,args.scramb_max]
    scrambleTests = range(scrambleRange[0],scrambleRange[1]+1)
    if args.scramb_max - args.scramb_min > 30:
        scrambleTests = np.linspace(args.scramb_min,args.scramb_max,30,dtype=np.int)

    for scramb in scrambleTests:
        err_val = tf.gather(errs,tf.where(tf.equal(tf.floor(stateVals_dp[:,0]),scramb))[:,0])
        tf.summary.scalar('Cost_%i' % (scramb), tf.reduce_mean(tf.pow(err_val,2)))

    tf.summary.scalar('Cost', cost)
    tf.summary.scalar('batch_size', tf.shape(states)[0])

    ### Optimization
    graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_regularization_loss = tf.reduce_sum(graph_regularizers)

    cost = cost + total_regularization_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        learningRate = tf.train.exponential_decay(args.lr_i, global_step, 1, args.lr_d, staircase=False)
        momentum = args.mom_i + (args.mom_f-args.mom_i)*tf.minimum(tf.to_float(global_step)/float(args.mom_c_s),1.0)
        tf.summary.scalar('lr', learningRate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if args.opt.upper() == "SGD":
                opt = tf.train.MomentumOptimizer(learningRate,momentum).minimize(cost,global_step)
            elif args.opt.upper() == "ADAM":
                opt = tf.train.AdamOptimizer(learningRate,args.bm,args.bv).minimize(cost,global_step)

        return(tf.estimator.EstimatorSpec(mode, predictions=predictions, loss=tf.reduce_mean(tf.pow(errs,2)), train_op=opt))

def getNextStates(cubes,Environment):
    legalMoves = Environment.legalPlays

    nextStates_cube = np.empty([len(cubes),len(legalMoves)] + list(cubes[0].shape),dtype=Environment.dtype)
    nextStateRewards = np.empty([len(cubes),len(legalMoves)])
    nextStateSolved = np.empty([len(cubes),len(legalMoves)],dtype=bool)

    ### Get next state rewards and if solved
    if type(cubes) == type(list()):
        cubes = np.stack(cubes,axis=0)

    for moveIdx,move in enumerate(legalMoves):
        nextStates_cube_move = Environment.next_state(cubes, move)

        isSolved = Environment.checkSolved(nextStates_cube_move)
        nextStateSolved[:,moveIdx] = isSolved
        nextStateRewards[:,moveIdx] = Environment.getReward(nextStates_cube_move,isSolved)
        if type(move[0]) == type(list()):
            nextStateRewards[:,moveIdx] = nextStateRewards[:,moveIdx] - (len(move) - 1) 
        nextStates_cube[:,moveIdx,:] = nextStates_cube_move

    return(nextStates_cube,nextStateRewards,nextStateSolved)

