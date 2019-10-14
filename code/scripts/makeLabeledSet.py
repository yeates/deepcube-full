import argparse
import sys
import time
import numpy as np
import cPickle as pickle
from multiprocessing import Process, Queue

sys.path.append('./')
from ml_utils import nnet_utils
from ml_utils import search_utils
from environments import env_utils
import os

def heurProc(dataQueue,resQueue,modelLoc,useGPU,Environment,gpuNum=None):
    if modelLoc == "":
        nnet = lambda x,realWorld: np.zeros([x.shape[0]],dtype=np.int)
    else:
        nnet = nnet_utils.loadNnet(modelLoc,"",useGPU,Environment,gpuNum=gpuNum)
    while True:
        data, realWorld = dataQueue.get()
        if data is None:
            resQueue.put(None)
            break
        nnetResult = nnet(data,realWorld=realWorld)
        resQueue.put(nnetResult)

def generateData(dataQueue,numStates,scrambMax,Environment,depth):
    while True:
        ### Generate environments
        startStates, numScrambs = Environment.generate_envs(numStates,[0,scrambMax])
        startStates = np.stack(startStates,axis=0)

        dataQueue.put(startStates)

        
def saveData(saveQueue,Environment,scrambMax):
    while True:
        getData = saveQueue.get()
        if getData is None:
            break
        else:
            testInput, testOutput, testOutput_prev, numScrambs, outFileLoc = getData

        assert(testInput.shape[0] == testOutput.shape[0])

        #print >> sys.stderr, "Saving %i total examples to %s" % (testInput.shape[0],outFileLoc)
        data = dict()
        data["input"] = testInput
        data["output"] = testOutput
        data["output_prev"] = testOutput_prev
        data["numScrambs"] = numScrambs
        data["scrambMax"] = scrambMax
        pickle.dump(data, open(outFileLoc, "wb"),protocol=1)
     
        doneFile = "%s_done" % (outFileLoc)
        try:
            with open(doneFile, 'a'):
                os.utime(doneFile, None)
        except OSError as e:
            print >> sys.stderr,e
            print >> sys.stderr,"ERROR touching %s, continuing anyway" % (doneFile)



def sendToNnet(states,dataQueue_VI,statesAtDepth):
    #statesAtDepth, _, _ = search_utils.Tree.generateToDepth(states,depth,Environment)

    dataQueue_VI.put((statesAtDepth[-1].copy(),True))           

def generateToDepthProc(genToDepthQueue_input,genToDepthQueue,depth,Environment,batchSize_states):
    while True:
        states = genToDepthQueue_input.get()
        startIdx_state = 0
        while startIdx_state < states.shape[0]:
            endIdx_state = min(startIdx_state + batchSize_states,states.shape[0])

            statesAtDepth, rewardsAtDepth, isSolvedAtDepth = search_utils.Tree.generateToDepth(states[startIdx_state:endIdx_state],depth,Environment)

            genToDepthQueue.put((statesAtDepth,rewardsAtDepth,isSolvedAtDepth))

            startIdx_state = endIdx_state

def makeDatasets(outFileLoc,Environment,currStates,dataQueue_VI,resQueue_VI,genToDepthQueue_input,genToDepthQueue,depth,numSteps,batchSize_states):
    startTime = time.time()

    ### Take actions in environment
    legalMoves = Environment.legalPlays
    numLegalMoves = len(legalMoves)

    values = []
    values_prev = []
    states = []

    if numSteps > 1:
        trajs = [set([x.tostring()]) for x in currStates]

    time_nnet = 0.0
    time_backup = 0.0
    time_nextStep = 0.0
    for step in range(numSteps):
        if currStates.shape[0] == 0:
            break

        isSolved = np.zeros(currStates.shape[0],dtype=np.bool)
        genToDepthQueue_input.put(currStates)

        ### Evaluate children
        startIdx_state = 0
        endIdx_state = min(startIdx_state + batchSize_states,currStates.shape[0])

        statesAtDepth_next, rewardsAtDepth_next, isSolvedAtDepth_next = genToDepthQueue.get()
        sendToNnet(currStates[startIdx_state:endIdx_state],dataQueue_VI,statesAtDepth_next)
        
        while startIdx_state < currStates.shape[0]:
            ### Do value iteration backup and take another step if necessary
            idxs_state = np.arange(startIdx_state,endIdx_state,1,dtype=np.int)
            currStates_itr = currStates[idxs_state].copy()
            
            #statesAtDepth, rewardsAtDepth, isSolvedAtDepth = search_utils.Tree.generateToDepth(currStates_itr,depth,Environment)
            statesAtDepth = statesAtDepth_next
            rewardsAtDepth = rewardsAtDepth_next 
            isSolvedAtDepth = isSolvedAtDepth_next

            ### Get values
            #dataQueue_VI.put((statesAtDepth[0],True))           
            #values_prev_VI = resQueue_VI.get()
            #values_prev_VI = np.maximum(values_prev_VI,0.0)
            
            startTime_nnet = time.time()
            nnetValsLastDepth_VI = resQueue_VI.get()
            nnetValsLastDepth = nnetValsLastDepth_VI.copy()
            
            ### Evaluate children of next batch
            startIdx_state = endIdx_state

            if startIdx_state < currStates.shape[0]:
                endIdx_state = min(startIdx_state + batchSize_states,currStates.shape[0])
                statesAtDepth_next, rewardsAtDepth_next, isSolvedAtDepth_next = genToDepthQueue.get()
                sendToNnet(currStates[startIdx_state:endIdx_state],dataQueue_VI,statesAtDepth_next)

            time_nnet = time_nnet + (time.time() - startTime_nnet)

            ### Backup values to root
            startTime_backup = time.time()
            values_VI, _ = search_utils.Tree.backupValues(nnetValsLastDepth_VI,statesAtDepth,rewardsAtDepth,isSolvedAtDepth,Environment)
            values.append(values_VI.copy())

            time_backup = time_backup + (time.time() - startTime_backup)
                
            #values_prev.append(values_prev_VI.copy())
            values_prev.append(np.zeros([statesAtDepth[0].shape[0]],dtype=np.int))
            states.append(currStates_itr.copy())

            ### Take step according to policy
            startTime_nextStep = time.time()
            if step < (numSteps-1):
                ### Check if solved
                isSolved_itr = Environment.checkSolved(currStates_itr)
                isSolved[idxs_state] = isSolved_itr

                idxs_state = idxs_state[np.logical_not(isSolved_itr)]
                currStates_itr = currStates_itr[np.logical_not(isSolved_itr)]

                nnetValsLastDepth = nnetValsLastDepth.reshape(nnetValsLastDepth.shape[0]/numLegalMoves,numLegalMoves)
                nnetValsLastDepth = nnetValsLastDepth[np.logical_not(isSolved_itr),:]

                ### Get next state
                nextMovesSort = np.argsort(nnetValsLastDepth,axis=1)
                #nextMovesSort = np.argsort(-nnetValsLastDepth,axis=1)

                nextMoveIdx = 0
                hasSeen = np.ones(currStates_itr.shape[0],dtype=np.bool)

                prevStates = currStates_itr.copy()
                while (currStates_itr.shape[0] > 0) and (max(hasSeen) == True) and (nextMoveIdx < nextMovesSort.shape[1]):
                    idxs_remain = np.where(hasSeen == True)[0]
                    ### Iterate through moves
                    for moveIdx in range(len(legalMoves)):
                        matchMoveIdxs = np.where(nextMovesSort[idxs_remain,nextMoveIdx] == moveIdx)[0]
                        if matchMoveIdxs.shape[0] > 0:
                            ### Make move
                            idxs_remain_moveMatch = idxs_remain[matchMoveIdxs]
                            currStates_itr[idxs_remain_moveMatch,:] = Environment.next_state(prevStates[idxs_remain_moveMatch,:], legalMoves[moveIdx])

                            ### Check if state has been seen before
                            for idx in idxs_remain_moveMatch:
                                stateIdx_all = idxs_state[idx]
                                hashableVal = currStates_itr[idx,:].tostring()

                                hasSeen[idx] = hashableVal in trajs[stateIdx_all]
                                trajs[stateIdx_all].add(hashableVal)

                    nextMoveIdx = nextMoveIdx + 1
                
                currStates[idxs_state] = currStates_itr

            time_nextStep = time_nextStep + (time.time()-startTime_nextStep)

        ### Remove solved states
        if numSteps > 1:
            currStates = currStates[np.logical_not(isSolved)]
            trajs = [traj for traj,y in zip(trajs,isSolved) if y == False]

    testInput = np.concatenate(states,axis=0)
    testOutput = np.expand_dims(np.concatenate(values,axis=0),1)
    testOutput_prev = np.concatenate(values_prev,axis=0)

    #print >> sys.stderr, testInput.shape

    #print >> sys.stderr, "Total time %s, Nnet time: %s, Backup time: %s, Next step time: %s" % (time.time()-startTime,time_nnet,time_backup,time_nextStep)


    return((testInput,testOutput,testOutput_prev,None,outFileLoc))



parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='cube3', help="Environment: cube3, cube4")

parser.add_argument('--model_loc', type=str, required=True, help="Location of model")

parser.add_argument('--num_states', type=int, default=500, help="Number of states to solve per scramble depth")
parser.add_argument('--scramb_max', type=int, default=20, help="Maximum number of scrambles to produce dataset")

parser.add_argument('--num_steps', type=int, default=1, help="Number of steps to take")

parser.add_argument('--search_depth', type=int, default=1, help="Depth of nnet search")
parser.add_argument('--num_rollouts', type=int, default=1, help="Number of rollouts to do in MCTS")

parser.add_argument('--dir', type=str, default="", help="Base output file name")
parser.add_argument('--base_idx', type=int, default=0, help="Integer to add to file created")
parser.add_argument('--num_itrs', type=int, default=100, help="Number of iterations to get labels for num_states states")

parser.add_argument('--use_gpu', type=int, default=1, help="True if using GPU")

parser.add_argument('--num_parallel', type=int, default=1, help="How many to run at a time")
parser.add_argument('--verbose', type=bool, default=False, help="Print status to screen if true")
args = parser.parse_args()

Environment = env_utils.getEnvironment(args.env)

scrambMax = args.scramb_max
saveDir = args.dir
baseIdx = args.base_idx
numItrs = args.num_itrs
numRollouts = args.num_rollouts
useGPU = bool(args.use_gpu)
verbose = args.verbose

assert(args.num_parallel >= 1)

envName = args.env
envName = envName.lower()

### Start data getters
numDataJobs = 1
dataQueue = Queue(numDataJobs)
for i in range(numDataJobs):
    dataProc = Process(target=generateData, args=(dataQueue,args.num_states,scrambMax,Environment,args.search_depth,))
    dataProc.daemon = True
    dataProc.start()

### Start data saver
numSaveJobs = 1
saveQueue = Queue(numSaveJobs)
for i in range(numSaveJobs):
    saveProc = Process(target=saveData, args=(saveQueue,Environment,scrambMax,))
    saveProc.daemon = True
    saveProc.start()

### Start value iteration network
#print >> sys.stderr, "Loading value iteration model %s" % (args.model_loc)
dataQueue_VI = Queue(1)
resQueue_VI = Queue(1)

heurFn_VI_proc = Process(target=heurProc, args=(dataQueue_VI,resQueue_VI,args.model_loc,useGPU,Environment,))
heurFn_VI_proc.daemon = True
heurFn_VI_proc.start()
#print >> sys.stderr, "Loaded value iteration model"

batchSize_nnet = 5000
legalMoves = Environment.legalPlays

batchSize_states = max(batchSize_nnet/len(legalMoves),1)

### Start generate data to depth proc
#print >> sys.stderr, "Starting data generation model"

genToDepthQueue_input = Queue(1)
genToDepthQueue = Queue(1)

genDepthProc = Process(target=generateToDepthProc, args=(genToDepthQueue_input,genToDepthQueue,args.search_depth,Environment,batchSize_states,))
genDepthProc.daemon = True
genDepthProc.start()

#print >> sys.stderr, "Started data generation model"


### Do value iteration
for itr in range(numItrs):
    itr_start_time = time.time()
    outFileLoc = "%s/data_%i.pkl" % (saveDir,itr + baseIdx)
    
    ### Get starting states
    startTime = time.time()
    currStates = dataQueue.get()
    #print >> sys.stderr, "Get data time %s" % (time.time()-startTime)

    ### Follow exploration policy and update with latest heuristic function
    dataOut = makeDatasets(outFileLoc,Environment,currStates,dataQueue_VI,resQueue_VI,genToDepthQueue_input,genToDepthQueue,args.search_depth,args.num_steps,batchSize_states)

    startTime = time.time()
    saveQueue.put(dataOut)
    #print >> sys.stderr, "Save time: %s" % (time.time()-startTime)

    #print >> sys.stderr, "Itr %i of %i, time: %.2f\n" % (itr+1, numItrs, time.time()-itr_start_time)


### Terminate procs
#print >> sys.stderr,"Stopping save queue thread"
saveQueue.put(None)
saveProc.join()
saveProc.terminate()
#print >> sys.stderr,"Stopped"

#print >> sys.stderr,"Stopping nnet VI thread"
dataQueue_VI.put((None,None))
resQueue_VI.get()
heurFn_VI_proc.join()
heurFn_VI_proc.terminate()

#print >> sys.stderr,"Stopped"


#print >> sys.stderr,"Done"

