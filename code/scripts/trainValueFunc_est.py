# Set GPU
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.INFO)
print('\nRunning from GPU %s' % str(os.environ['CUDA_VISIBLE_DEVICES']))

import numpy as np
import cPickle as pickle
import time
import argparse

import shutil

import sys
sys.path.append('./')
from ml_utils import nnet_utils
from ml_utils import search_utils
from environments import env_utils

from multiprocessing import Process, Queue
#from multiprocessing import Pool
#from multiprocessing.pool import ThreadPool

### Data getting functions

def loadFile(fileQueue,resQueue,deleteFile):
    while True:
        labeledFile_path, labeledFile_done = fileQueue.get()
        try:
            if (not deleteFile or os.path.isfile(labeledFile_done)) and os.path.isfile(labeledFile_path):
                with open(labeledFile_path,"rb") as f:
                    data = pickle.load(f)

                states_nnet = data["input"]
                states_nnet = Environment.state_to_nnet_input(data["input"],randTransp=False)
                outputs = data["output"]

                if deleteFile:
                    os.remove(labeledFile_path)
                    os.remove(labeledFile_done)
            else:
                resQueue.put((None,None))
                continue
                
        except EOFError:
            resQueue.put((None,None))
            continue
        except OSError as e:
            print >> sys.stderr,e
            resQueue.put((None,None))
            continue

        if len(outputs.shape) == 1:
            outputs = np.expand_dims(outputs,1)

        states_nnet = np.expand_dims(states_nnet,1)
        
        resQueue.put((states_nnet,outputs))

def loadSupervisedData(dataQueue,batchSize,Environment,labeledData,deleteFile,fileQueue,resQueue):
    states_all = np.array([])
    outputs_all = np.array([])

    while True:
        labeledFiles = [f for f in os.listdir(labeledData) if os.path.isfile(os.path.join(labeledData, f)) and ('.pkl' in f)]
        for labeledFile in labeledFiles:
            labeledFile_path = os.path.join(labeledData,labeledFile)
            labeledFile_done = os.path.join(labeledData,"%s_done" % (labeledFile))
            fileQueue.put((labeledFile_path,labeledFile_done))

        for labeledFile in labeledFiles:
            ### Load file
            states, outputs = resQueue.get()

            if states is None:
                continue

            if states_all.shape[0] == 0:
                states_all = states
                outputs_all = outputs
            else:
                states_all = np.concatenate((states_all,states),axis=0)
                outputs_all = np.concatenate((outputs_all,outputs),axis=0)

            if states_all.shape[0] >= 10*batchSize:
                ### Get batch
                randIdxs = np.random.choice(states_all.shape[0],states_all.shape[0],replace=False)
                startIdx = 0
                while (startIdx + batchSize) < states_all.shape[0]:
                    endIdx = startIdx + batchSize
                    selectIdxs = randIdxs[startIdx:endIdx]

                    states_nnet = states_all[selectIdxs]
                    outputs = outputs_all[selectIdxs]

                    dataQueue.put((states_nnet,outputs))

                    startIdx = endIdx

                states_all = np.array([])
                outputs_all = np.array([])

                if args.debug:
                    print("Data queue size: %i" % (dataQueue.qsize()))

    print >> sys.stderr,"Stopping load supervised data thread"

### Arguments
parser = argparse.ArgumentParser()
parser = nnet_utils.addNnetArgs(parser)

args = parser.parse_args()
if args.max_turns is None:
    args.max_turns = args.scramb_max

nnetName = nnet_utils.getModelName(args)
modelSaveLoc = "%s/%s/%s" % (args.save_dir,nnetName,args.model_num)
if not os.path.exists(modelSaveLoc):
    os.makedirs(modelSaveLoc)

argsSaveLoc = "%s/args.pkl" % (modelSaveLoc)
print("Saving arguments to %s" % (argsSaveLoc))
with open(argsSaveLoc, "wb") as f:
    pickle.dump(args, f, protocol=1)

Environment = env_utils.getEnvironment(args.env)

exampleState = np.expand_dims(Environment.generate_envs(1, [0, 0])[0][0],0)
inputDim = list(Environment.state_to_nnet_input(exampleState).shape[1:])
print("Input shape %s" % (inputDim))

numGPUs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))

scrambleRange = [args.scramb_min,args.scramb_max]
scrambleTests = range(scrambleRange[0],scrambleRange[1]+1)
if args.scramb_max - args.scramb_min > 30:
    scrambleTests = np.linspace(args.scramb_min,args.scramb_max,30,dtype=np.int)

print("Scramble test distances are %s" % (scrambleTests))

assert(args.mom_f >= args.mom_i)
assert(args.mom_c_s >= 1)

assert(args.drop_p < 1.0)


solveItrs = args.solve_itrs
if args.debug:
    displayItrs = 100
else:
    displayItrs = 100

saveItrs = 10000

legalMoves = Environment.legalPlays
print("There are %i legal moves: %s" % (len(legalMoves),legalMoves))

### Initialize data getting
print("Using labeled data in directory %s" % (args.labeled_data))

dataQueue = Queue(100)

numPools = 1
print("Starting %i data queue runners" % (numPools))

fileQueue = Queue()
resQueue = Queue(100)

for i in range(numPools):
    fileProc = Process(target=loadFile, args=(fileQueue,resQueue,args.delete_labeled,))
    fileProc.daemon = True
    fileProc.start()

#loadSupervisedData(dataQueue,args.batch_size,Environment,args.labeled_data,args.delete_labeled)
loadProc = Process(target=loadSupervisedData, args=(dataQueue,args.batch_size,Environment,args.labeled_data,args.delete_labeled,fileQueue,resQueue,))
loadProc.daemon = True
loadProc.start()

### Initialize input dataset
def gen():
    while True:
        yield dataQueue.get()

tf_dtype = Environment.tf_dtype()

def input_fn():
    ds = tf.data.Dataset.from_generator(gen,(tf_dtype,tf.float32),(tf.TensorShape([None,1]+inputDim),tf.TensorShape([None,1])))
    ds = ds.prefetch(buffer_size=5)
    return(ds)

def serving_input_receiver_fn():
    inputs = {"x": tf.placeholder(shape=[None, 1] + inputDim, dtype=tf_dtype)}
    return tf.estimator.export.build_raw_serving_input_receiver_fn(inputs)

serving_input_receiver_fn_ret = serving_input_receiver_fn()

### Initialize
CONFIG = tf.ConfigProto()
CONFIG.gpu_options.allow_growth = True # Prevents tf from grabbing all gpu memory.
runConfig = tf.estimator.RunConfig(save_checkpoints_steps=saveItrs,keep_checkpoint_max=2,save_summary_steps=10000,session_config=CONFIG)

model_fn = lambda features,labels,mode : nnet_utils.model_fn(features,labels,mode,args)
network = tf.estimator.Estimator(model_fn=model_fn, config=runConfig, model_dir=modelSaveLoc)

trainItr = 0
cost = np.inf
aeMax = np.inf

solveItrs = min(solveItrs,args.max_itrs)
while (trainItr < args.max_itrs) and ((args.eps is None) or (cost > args.eps)):
    ### Train network
    network = network.train(input_fn=input_fn,steps=solveItrs)
    trainItr = trainItr + solveItrs

    ### Test network
    heuristicFn = nnet_utils.getEstimatorPredFn(network,inputDim,Environment)
    
    # Check convergence
    if args.labeled_data != "":
        valData = [[],[]]
        for _ in range(0,100):
            data = dataQueue.get()
            valData[0].append(data[0][:,0,:])
            valData[1].append(data[1])

        valData[0] = np.concatenate(valData[0],axis=0)
        valData[1] = np.concatenate(valData[1],axis=0)

        testOutput = heuristicFn(valData[0],realWorld=False)
        errs = valData[1] - testOutput

        errsData = dict()
        errsData["valData"] = valData
        errsData["output"] = testOutput
        errsData["errs"] = errs
        pickle.dump(errsData, open("%s/errs.pkl" % (modelSaveLoc), "wb"),protocol=1)

        costs = np.power(errs,2)
        cost = np.mean(costs)
        costMax = np.max(costs)

        print >> sys.stderr, "Cost" % (cost)

    # Greedy BFS
    for scrambleNum in scrambleTests:
        solve_start_time = time.time()

        ### Generate environments
        testStates_cube, _ = Environment.generate_envs(100,[scrambleNum,scrambleNum])

        ### Get state values
        stateVals = heuristicFn(np.stack(testStates_cube,axis=0))

        ### Solve with GBFS
        isSolved_test, solveSteps = search_utils.solve(testStates_cube,heuristicFn,Environment,maxTurns=args.max_turns)

        ### Get stats
        percentSolved = 100*float(sum(isSolved_test))/float(len(isSolved_test))
        avgSolveSteps = 0.0
        if percentSolved > 0.0:
            avgSolveSteps = np.mean(solveSteps[isSolved_test])

        solve_elapsed_time = time.time() - solve_start_time

        ### Print results
        print >> sys.stderr,"Scramb #: %i, %%Solved: %.2f, avgSolveSteps: %.2f, StateVals Mean(Std/Min/Max): %.2f(%.2f/%.2f/%.2f), time: %.2f" % (scrambleNum, percentSolved, avgSolveSteps,np.mean(stateVals),np.std(stateVals),np.min(stateVals),np.max(stateVals),solve_elapsed_time)

    ### Export model
    exportDir = "%s/%s/" % (modelSaveLoc,"exported_model")
    exportDir_tmp = network.export_savedmodel(export_dir_base=modelSaveLoc,serving_input_receiver_fn=serving_input_receiver_fn_ret)

    if os.path.isdir(exportDir):
        shutil.rmtree(exportDir)
    shutil.move(exportDir_tmp,exportDir)

    
print("DONE")
