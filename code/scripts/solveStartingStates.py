import os

import sys
import numpy as np
import cPickle as pickle
import argparse
import time
from subprocess import Popen, PIPE

from multiprocessing import Process, Queue

sys.path.append('./')
sys.path.append('../')

from environments import env_utils

import socket

import gc

def deleteIfExists(filename):
    if os.path.exists(filename):
        os.remove(filename)


def validSoln(state,soln,Environment):
    solnState = state
    for move in soln:
        solnState = Environment.next_state(solnState,move)

    return(Environment.checkSolved(solnState))

def fileListener(sock,heuristicFn,Environment):
    sock.listen(1)
    exampleState = np.expand_dims(Environment.generate_envs(1, [0, 0])[0][0],0)
    stateDim = exampleState.shape[1]

    maxBytes = 4096
    connection, client_address = sock.accept()
    while True:
        dataRec = connection.recv(8)
        while not dataRec:
            connection, client_address = sock.accept()
            dataRec = connection.recv(8)

        numBytesRecv = np.frombuffer(dataRec,dtype=np.int64)[0]

        #startTime = time.time()
        numBytesSeen = 0
        dataRec = ""
        while numBytesSeen < numBytesRecv:
            conRec = connection.recv(maxBytes)
            dataRec = dataRec + conRec
            numBytesSeen = numBytesSeen + len(conRec)


        states = np.frombuffer(dataRec,dtype=Environment.dtype)
        states = states.reshape(len(states)/stateDim,stateDim)
        #print("Rec Time: %s" % (time.time()-startTime))

        ### Run nnet
        #startTime = time.time()
        results = heuristicFn(states)
        #print("Heur Time: %s" % (time.time()-startTime))

        ### Write file
        #startTime = time.time()
        connection.sendall(results.astype(np.float32))
        #print("Write Time: %s" % (time.time()-startTime))

def dataListener(dataQueue,resQueue,gpuNum=None):
    nnet = nnet_utils.loadNnet(args.model_loc,args.model_name,useGPU,Environment,gpuNum=gpuNum)
    while True:
        data = dataQueue.get()
        nnetResult = nnet(data)
        resQueue.put(nnetResult)

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help="Name of input file")

parser.add_argument('--env', type=str, default='cube3', help="Environment: cube3, cube4")

parser.add_argument('--methods', type=str, default="kociemba,nnet", help="Which methods to use. Comma separated list")

parser.add_argument('--combine_outputs', action='store_true')

parser.add_argument('--model_loc', type=str, default="", help="Location of model")
parser.add_argument('--model_name', type=str, default="model.meta", help="Which model to load")

parser.add_argument('--nnet_parallel', type=int, default=200, help="How many to look at, at one time for nnet")
parser.add_argument('--depth_penalty', type=float, default=0.1, help="Coefficient for depth")
parser.add_argument('--bfs', type=int, default=0, help="Depth of breadth-first search to improve heuristicFn")

parser.add_argument('--startIdx', type=int, default=0, help="")
parser.add_argument('--endIdx', type=int, default=-1, help="")

parser.add_argument('--use_gpu', type=int, default=1, help="1 if using GPU")

parser.add_argument('--verbose', action='store_true', default=False, help="Print status to screen if switch is on")

parser.add_argument('--name', type=str, default="", help="Special name to append to file name")

args = parser.parse_args()

Environment = env_utils.getEnvironment(args.env)

if args.env.upper() == 'CUBE3':
    sys.path.append('./solvers/cube3/')
    from solver_algs import Kociemba
    from solver_algs import Optimal
elif args.env.upper() == 'CUBE4':
    sys.path.append('./solvers/cube4/')
    from solver_algs import Optimal
elif args.env.upper() == 'PUZZLE15':
    sys.path.append('./solvers/puzzle15/')
    from solver_algs import Optimal
elif args.env.upper() == 'PUZZLE24':
    sys.path.append('./solvers/puzzle24/')
    from solver_algs import Optimal

useGPU = bool(args.use_gpu)

methods = [x.lower() for x in args.methods.split(",")]
print("Methods are: %s" % (",".join(methods)))

### Load starting states
filename = args.input
outFileLoc_pre = "../testData/%s/%s_nnetPar%s_depthP%s_bfs%s_%s_solved" % (args.env.lower(),filename.split("/")[-1].split(".")[0],args.nnet_parallel,args.depth_penalty,args.bfs,"_".join(methods))

if args.endIdx != -1:
    outFileLoc_pre = "%s_%i_%i" % (outFileLoc_pre,args.startIdx,args.endIdx)


if args.name != "":
    outFileLoc_pre = "%s_%s" % (outFileLoc_pre,args.name)

outFileLoc = "%s.pkl" % (outFileLoc_pre)
print >> sys.stderr, "Input File: %s, Output File: %s" % (filename,outFileLoc)

inputData = pickle.load(open(filename,"rb"))

states = inputData['states']

if args.endIdx == -1:
    args.endIdx = len(states)

states = states[args.startIdx:args.endIdx]

### Load nnet if needed
if "nnet" in methods:
    from ml_utils import nnet_utils
    from ml_utils import search_utils

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
        gpuNums = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(",")]
    else:
        gpuNums = [None]
    numParallel = len(gpuNums)

    ### Initialize files
    dataQueues = []
    resQueues = []
    for num in range(numParallel):
        dataQueues.append(Queue(1))
        resQueues.append(Queue(1))

        dataListenerProc = Process(target=dataListener, args=(dataQueues[num],resQueues[num],gpuNums[num],))
        dataListenerProc.daemon = True
        dataListenerProc.start()


    def heuristicFn_nnet(x):
        ### Write data
        parallelNums = range(min(numParallel,x.shape[0]))
        splitIdxs = np.array_split(np.arange(x.shape[0]),len(parallelNums))
        for num in parallelNums:
            dataQueues[num].put(x[splitIdxs[num]])

        ### Check until all data is obtaied
        results = [None]*len(parallelNums)
        for num in parallelNums:
            results[num] = resQueues[num].get()

        results = np.concatenate(results)

        return(results)
    
    socketName = "%s_socket" % (outFileLoc_pre)
    try:
        os.unlink(socketName)
    except OSError:
        if os.path.exists(socketName):
            raise

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(socketName)

    fileListenerProc = Process(target=fileListener, args=(sock,heuristicFn_nnet,Environment,))
    fileListenerProc.daemon = True
    fileListenerProc.start()

### Get solutions
data = dict()
data["states"] = states
data["solutions"] = dict()
data["times"] = dict()
data["nodesGenerated_num"] = dict()
for method in methods:
    data["solutions"][method] = [None]*states.shape[0]
    data["times"][method] = [None]*states.shape[0]
    data["nodesGenerated_num"][method] = [None]*states.shape[0]

def runMethods(idx_state):
    idx, state = idx_state
    stateStr = " ".join([str(x) for x in state])
    #print(stateStr)
    for method in methods:
        start_time = time.time()
        if method == "kociemba":
            soln = Kociemba.solve(state)
            nodesGenerated_num = 0

            elapsedTime = time.time() - start_time
        elif method == "nnet":
            runType = "python"
            #if (args.env.upper() in ['CUBE3','PUZZLE15','PUZZLE24','PUZZLE35']) or ('LIGHTSOUT' in args.env.upper()):
            #    runType = "cpp"
            #else:
            #    runType = "python"
            if runType == "cpp":
                popen = Popen(['./ml_utils/parallel_weighted_astar',stateStr,str(args.depth_penalty),str(args.nnet_parallel),socketName,args.env], stdout=PIPE, stderr=PIPE, bufsize=1, universal_newlines=True)
                lines = []
                for stdout_line in iter(popen.stdout.readline, ""):
                    stdout_line = stdout_line.strip('\n')
                    lines.append(stdout_line)
                    if args.verbose:
                        sys.stdout.write("%s\n" % (stdout_line))
                        sys.stdout.flush()

                moves = [int(x) for x in lines[-3].split(" ")[:-1]]
                soln = [Environment.legalPlays[x] for x in moves][::-1]
                nodesGenerated_num = int(lines[-1])
            else:
                BestFS_solve = search_utils.BestFS_solve([state],heuristicFn_nnet,Environment,bfs=args.bfs)
                isSolved, solveSteps, nodesGenerated_num = BestFS_solve.run(numParallel=args.nnet_parallel,depthPenalty=args.depth_penalty,verbose=args.verbose)
                BestFS_solve = []
                del BestFS_solve
                gc.collect()

                soln = solveSteps[0]
                nodesGenerated_num = nodesGenerated_num[0]

            elapsedTime = time.time() - start_time
        elif method == "optimal":
            soln, nodesGenerated_num, elapsedTime = Optimal.solve(state)
        else:
            continue

        data["times"][method][idx] = elapsedTime
        data["nodesGenerated_num"][method][idx] = nodesGenerated_num
    
        assert(validSoln(state,soln,Environment))

        data["solutions"][method][idx] = soln


print("%i total states" % (states.shape[0]))
if 'optimal' in methods:
    solutions, nodesGenerated_num, times = Optimal.solve(states,args.startIdx,args.endIdx,args.combine_outputs)
    data["solutions"]['optimal'] = solutions
    data["times"]['optimal'] = times
    data["nodesGenerated_num"]['optimal'] = nodesGenerated_num
    for state,soln in zip(states,data["solutions"]['optimal']):
        if soln is not None:
            assert(validSoln(state,soln,Environment))
else:
    for idx,state in enumerate(states):
        runMethods((idx,state))

        solveStr = ", ".join(["len/time/#nodes - %s: %i/%.2f/%i" % (method,len(data["solutions"][method][idx]),data["times"][method][idx],data["nodesGenerated_num"][method][idx]) for method in methods])
        print >> sys.stderr, "State: %i, %s" % (idx,solveStr)
    
### Save data
pickle.dump(data, open(outFileLoc, "wb"), protocol=1)

### Print stats
for method in methods:
    solnLens = np.array([len(soln) for soln in data["solutions"][method]])
    times = np.array([solveTime for solveTime in data["times"][method]])
    nodesGenerated_num = np.array([solveTime for solveTime in data["nodesGenerated_num"][method]])
    print("%s: Soln len - %f(%f), Time - %f(%f), # Nodes Gen - %f(%f)" % (method,np.mean(solnLens),np.std(solnLens),np.mean(times),np.std(times),np.mean(nodesGenerated_num),np.std(nodesGenerated_num)))

