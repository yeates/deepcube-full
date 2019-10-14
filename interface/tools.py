import os
import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../code')
sys.path.append('../data')
sys.path.append('./solvers/cube3/')
from werkzeug.datastructures import ImmutableMultiDict
import ast
import numpy as np
import argparse
import time
from subprocess import Popen, PIPE
from multiprocessing import Process, Queue
# from solver_algs import Kociemba
# from solver_algs import Optimal
from environments import env_utils
import socket
import gc
from ml_utils import nnet_utils
from ml_utils import search_utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# --------solve
def getResults(state):

    parser = argparse.ArgumentParser()

    parser.add_argument('--methods', type=str, default="nnet", help="Which methods to use. Comma separated list")

    parser.add_argument('--combine_outputs', action='store_true')

    parser.add_argument('--model_loc', type=str, default="../code/savedModels/cube3/1/", help="Location of model")
    parser.add_argument('--model_name', type=str, default="model.meta", help="Which model to load")

    parser.add_argument('--nnet_parallel', type=int, default=1, help="How many to look at, at one time for nnet")
    parser.add_argument('--depth_penalty', type=float, default=0.2, help="Coefficient for depth")
    parser.add_argument('--bfs', type=int, default=0, help="Depth of breadth-first search to improve heuristicFn")

    parser.add_argument('--startIdx', type=int, default=0, help="")
    parser.add_argument('--endIdx', type=int, default=1, help="")

    parser.add_argument('--use_gpu', type=int, default=1, help="1 if using GPU")

    parser.add_argument('--verbose', action='store_true', default=False, help="Print status to screen if switch is on")

    parser.add_argument('--name', type=str, default="", help="Special name to append to file name")

    args = parser.parse_args()

    Environment = env_utils.getEnvironment('cube3')

    useGPU = bool(args.use_gpu)

    methods = [x.lower() for x in args.methods.split(",")]
    print("Methods are: %s" % (",".join(methods)))

    # convert
    FEToState = [6, 3, 0, 7, 4, 1, 8, 5, 2, 15, 12, \
                       9, 16, 13, 10, 17, 14, 11, 24, 21, 18, \
                       25, 22, 19, 26, 23, 20, 33, 30, 27, 34, \
                       31, 28, 35, 32, 29, 38, 41, 44, 37, 40, \
                       43, 36, 39, 42, 51, 48, 45, 52, 49, 46, \
                       53, 50, 47]
    converted_state = []
    for i in range(len(FEToState)):
        converted_state.append(state[FEToState[i]])

    ### Load starting states
    state = np.array(converted_state, np.int64)

    ### Load nnet if needed
    if "nnet" in methods:

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

            dataListenerProc = Process(target=dataListener, args=(dataQueues[num], resQueues[num], args, useGPU, Environment,gpuNums[num],))
            dataListenerProc.daemon = True
            dataListenerProc.start()

        def heuristicFn_nnet(x):
            ### Write data
            parallelNums = range(min(numParallel, x.shape[0]))
            splitIdxs = np.array_split(np.arange(x.shape[0]), len(parallelNums))
            for num in parallelNums:
                dataQueues[num].put(x[splitIdxs[num]])

            ### Check until all data is obtaied
            results = [None] * len(parallelNums)
            for num in parallelNums:
                results[num] = resQueues[num].get()

            results = np.concatenate(results)

            return (results)

    ### Get solutions
    data = dict()
    data["states"] = state
    data["solutions"] = dict()
    data["times"] = dict()
    data["nodesGenerated_num"] = dict()
    for method in methods:
        data["solutions"][method] = [None] * 1
        data["times"][method] = [None] * 1
        data["nodesGenerated_num"][method] = [None] * 1

    print("%i total states" % (len(state)))

    idx = 0
    runMethods((idx, state), methods, Environment, heuristicFn_nnet, args, data)
    solveStr = ", ".join(["len/time/#nodes - %s: %i/%.2f/%i" % (
        method, len(data["solutions"][method][idx]), data["times"][method][idx],
        data["nodesGenerated_num"][method][idx]) for method in methods])
    print >> sys.stderr, "State: %i, %s" % (idx, solveStr)

    ### Save data
    arr = data['solutions']['nnet'][0]
    moves = []
    moves_rev = []
    solve_text = []
    for i in arr:
        moves.append(str(i[0]) + '_' + str(i[1]))
        moves_rev.append(str(i[0]) + '_' + str(-i[1]))
        if i[1] == -1:
            solve_text.append(str(i[0]) + "'")
        else:
            solve_text.append(str(i[0]))

    results = {"moves": moves, "moves_rev": moves_rev, "solve_text": solve_text}

    ### Print stats
    for method in methods:
        solnLens = np.array([len(soln) for soln in data["solutions"][method]])
        times = np.array([solveTime for solveTime in data["times"][method]])
        nodesGenerated_num = np.array([solveTime for solveTime in data["nodesGenerated_num"][method]])
        print("%s: Soln len - %f(%f), Time - %f(%f), # Nodes Gen - %f(%f)" % (
            method, np.mean(solnLens), np.std(solnLens), np.mean(times), np.std(times), np.mean(nodesGenerated_num),
            np.std(nodesGenerated_num)))

    return results




def runMethods(idx_state, methods, Environment, heuristicFn_nnet, args, data):
    idx, state = idx_state
    stateStr = " ".join([str(x) for x in state])
    # print(stateStr)
    for method in methods:
        start_time = time.time()
        if method == "kociemba":
            soln = Kociemba.solve(state)
            nodesGenerated_num = 0

            elapsedTime = time.time() - start_time
        elif method == "nnet":
            BestFS_solve = search_utils.BestFS_solve([state], heuristicFn_nnet, Environment, bfs=args.bfs)
            isSolved, solveSteps, nodesGenerated_num = BestFS_solve.run(numParallel=args.nnet_parallel,
                                                                        depthPenalty=args.depth_penalty,
                                                                        verbose=args.verbose)
            BestFS_solve = []
            del BestFS_solve
            gc.collect()

            soln = solveSteps[0]
            nodesGenerated_num = nodesGenerated_num[0]

            elapsedTime = time.time() - start_time
        else:
            continue

        data["times"][method][idx] = elapsedTime
        data["nodesGenerated_num"][method][idx] = nodesGenerated_num

        assert (validSoln(state, soln, Environment))

        data["solutions"][method][idx] = soln


def deleteIfExists(filename):
    if os.path.exists(filename):
        os.remove(filename)


def validSoln(state, soln, Environment):
    solnState = state
    for move in soln:
        solnState = Environment.next_state(solnState, move)

    return (Environment.checkSolved(solnState))


def dataListener(dataQueue, resQueue, args, useGPU, Environment, gpuNum=None):
    nnet = nnet_utils.loadNnet(args.model_loc, args.model_name, useGPU, Environment, gpuNum=gpuNum)
    while True:
        data = dataQueue.get()
        nnetResult = nnet(data)
        resQueue.put(nnetResult)

if __name__ == "__main__":
    # ---------processing state
    data = ImmutableMultiDict([('state',
                                '[8, 10, 36,  3,  4, 12, 27, 34, 53,  9,  5,  0, 46, 13, 19, 11, 30,\
            2, 18, 52, 15, 41, 22, 21, 47,  1, 35, 44,  7, 17, 16, 31, 37, 24,\
           48, 29, 45, 14,  6, 25, 40, 39, 42, 28, 33, 26, 23, 38, 43, 49, 50,\
           20, 32, 51]')])
    data = ImmutableMultiDict([('state',
                                '51, 32, 26, 30, 4, 3, 2, 19, 36, 9, 39, 29, 28, 13, 14, 38, 5, 45, 27, 16, 44, 21, 22, 46, 8, 52, 42, 15, 50, 47, 23, 31, 34, 6, 48, 11, 35, 41, 24, 10, 40, 37, 17, 7, 0, 20, 43, 33, 25, 49, 1, 18, 12, 53')])


    data = data.to_dict()
    arr = []
    data['state'] = ast.literal_eval(data['state'])
    print(data['state'])
    for i in data['state']:
        arr.append(int(i))

    # arr = [ 8, 10, 36,  3,  4, 12, 27, 34, 53,  9,  5,  0, 46, 13, 19, 11, 30,
    #         2, 18, 52, 15, 41, 22, 21, 47,  1, 35, 44,  7, 17, 16, 31, 37, 24,
    #        48, 29, 45, 14,  6, 25, 40, 39, 42, 28, 33, 26, 23, 38, 43, 49, 50,
    #        20, 32, 51]
    print(arr)
    print(getResults(arr))