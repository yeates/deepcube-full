#!/usr/bin/env python
# -*- coding: utf-8 -*-


#import pdb
import sys
#import operator

import numpy as np

from random import choice
sys.path.append('./')
import re
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import argparse

class PuzzleN():
    legalPlays = ['U','D','L','R']
    legalPlays_rev = ['D','U','R','L']
    
    def tf_dtype(self):
        import tensorflow as tf
        if self.N <= 15:
            return(tf.uint8)
        else:
            return(tf.int32)

    def __init__(self,N):
        self.N = N
        if self.N <= 15:
            self.dtype = np.uint8
        else:
            self.dtype = np.int

        ### Solved state
        self.solvedState = np.concatenate((np.arange(1,self.N*self.N),[0])).astype(self.dtype)

        ### Heuristic
        self.solvedPos = dict()
        solvedState_mat = np.reshape(self.solvedState,[self.N,self.N])
        for val in range(1,self.N*self.N):
            self.solvedPos[val] = np.where(solvedState_mat == val)

        self.manhattanDistMat = np.zeros((self.N*self.N,self.N,self.N),dtype=np.int)
        for val in range(0,self.N*self.N):
            for i in range(self.N):
                for j in range(self.N):
                    if val == 0:
                        self.manhattanDistMat[val,i,j] = 0
                    else:
                        i_solved,j_solved = self.solvedPos[val]
                        self.manhattanDistMat[val,i,j] = abs(i_solved-i) + abs(j_solved-j)

        self.manhattanDistMat = self.manhattanDistMat.reshape(self.N*self.N,self.N*self.N)

        ### Incremental heuristic
        self.manhattanDistMat_incr = np.zeros((self.N**2,self.N**2,self.N**2),dtype=np.int)
        for val in range(self.N**2):
            for newBlank in range(self.N**2):
                for oldBlank in range(self.N**2):
                    heur_old = self.manhattanDistMat[val,newBlank]
                    heur_new = self.manhattanDistMat[val,oldBlank]

                    self.manhattanDistMat_incr[val,newBlank,oldBlank] = heur_new - heur_old


        ### Next state ops
        self.swapZeroIdxs_dict = dict()
        self.swapZeroIdxs = np.zeros((self.N**2,len(PuzzleN.legalPlays)),dtype=np.int)
        for moveIdx,move in enumerate(PuzzleN.legalPlays):
            for i in range(self.N):
                for j in range(self.N):
                    zIdx = np.ravel_multi_index((i,j),(self.N,self.N))
                    if zIdx not in self.swapZeroIdxs_dict:
                        self.swapZeroIdxs_dict[zIdx] = []

                    state = np.ones((self.N,self.N),dtype=np.int)
                    state[i,j] = 0
                    
                    if move == 'U':
                        isEligible = i < (self.N-1)
                    elif move == 'D':
                        isEligible = i > 0
                    elif move == 'L':
                        isEligible = j < (self.N-1)
                    elif move == 'R':
                        isEligible = j > 0

                    if isEligible:
                        if move == 'U':
                            swap_i = i+1
                            swap_j = j
                        elif move == 'D':
                            swap_i = i-1
                            swap_j = j
                        elif move == 'L':
                            swap_i = i
                            swap_j = j+1
                        elif move == 'R':
                            swap_i = i
                            swap_j = j-1

                        self.swapZeroIdxs[zIdx,moveIdx] = np.ravel_multi_index((swap_i,swap_j),(self.N,self.N))
                        self.swapZeroIdxs_dict[zIdx].append((moveIdx,np.ravel_multi_index((swap_i,swap_j),(self.N,self.N))))
                    else:
                        self.swapZeroIdxs[zIdx,moveIdx] = zIdx

    def next_state(self,states_input,move):
        move = move.upper()
        states = states_input.copy()
        if len(states.shape) == 1:
            states = np.array([states])

        n_all,zIdxs = np.where(states == 0)

        moveIdx = np.where(PuzzleN.legalPlays == np.array(move))[0][0]

        stateIdxs = np.arange(0,states.shape[0])
        swapZIdxs = self.swapZeroIdxs[zIdxs,moveIdx]

        states[stateIdxs,zIdxs] = states[stateIdxs,swapZIdxs]
        states[stateIdxs,swapZIdxs] = 0

        return(states)

    def checkSolved(self,states):
        if len(states.shape) == 1:
            states = np.array([states])

        solvedState_tile = np.tile(np.expand_dims(self.solvedState,0),(states.shape[0],1))
        return(np.all(states == solvedState_tile,axis=1))

    def getReward(self,states,isSolved=None):
        #if type(isSolved) == type(None):
        #    isSolved = self.checkSolved(states)
        #reward = 1.0*isSolved + (-1.0)*(np.logical_not(isSolved))

        reward = np.ones(shape=(states.shape[0]))
        return(reward)

    def state_to_nnet_input(self,states,randTransp=False):
        if len(states.shape) == 1:
            states = np.array([states])
        representation = states

        return(representation)

    def generate_envs(self,numStates,scrambleRange,probs=None):
        assert(scrambleRange[0] >= 0)
        scrambs = range(scrambleRange[0],scrambleRange[1]+1)
        legalMoves = PuzzleN.legalPlays

        states = np.tile(np.expand_dims(self.solvedState,0),(numStates,1))

        scrambleNums = np.random.choice(scrambs,numStates,p=probs)
        numMoves = np.zeros(numStates)
        while (np.max(numMoves < scrambleNums) == True):
            poses = np.where((numMoves < scrambleNums))[0]

            subsetSize = max(len(poses)/len(legalMoves),1)
            poses = np.random.choice(poses,subsetSize)

            move = choice(legalMoves)
            states[poses] = self.next_state(states[poses], move)
            numMoves[poses] = numMoves[poses] + 1

        states = list(states)
        return(states,scrambleNums)

    def manhattanDistance(self,states):
        states = np.reshape(states,[states.shape[0],self.N,self.N])
        
        dists = [None]*states.shape[0]
        
        for stateIdx,state in enumerate(states):
            dist = sum(self.manhattanDistMat[states[stateIdx].flatten(),self.i_flat,self.j_flat])
            dists[stateIdx] = dist

        return(dists)

class InteractiveEnv(plt.Axes):
    def __init__(self,state,env,heuristicFn=None):
        self.state = state
        self.env = env
        self.heuristicFn = heuristicFn
        self.N = np.sqrt(state.shape[1])

        super(InteractiveEnv, self).__init__(plt.gcf(),[0,0,1,1])

        callbacks = fig.canvas.callbacks.callbacks
        del callbacks['key_press_event']

        self.figure.canvas.mpl_connect('key_press_event',self._keyPress)

        self._updatePlot()

    def _keyPress(self, event):
        if event.key.upper() == 'W':
            self.state = Environment.next_state(self.state,'U')
        elif event.key.upper() == 'A':
            self.state = Environment.next_state(self.state,'L')
        elif event.key.upper() == 'D':
            self.state = Environment.next_state(self.state,'R')
        elif event.key.upper() == 'S':
            self.state = Environment.next_state(self.state,'D')
        elif event.key.upper() == 'N':
            self.stepNnet()
        elif event.key.upper() == 'P':
            self.figure.savefig('snapshot.eps')

        self._updatePlot()

        if self.env.checkSolved(self.state)[0]:
            print("SOLVED!")

    def _updatePlot(self):
        self.clear()
        for squareIdx,square in enumerate(self.state[0]):
            color = 'white'
            xPos = int(np.floor(squareIdx / self.N))
            yPos = squareIdx % self.N

            left = yPos/float(self.N)
            right = left + 1.0/float(self.N)
            top = (self.N-xPos-1)/float(self.N)
            bottom = top + 1.0/float(self.N)

            self.add_patch(patches.Rectangle((left,top),1.0/self.N,1.0/self.N,0,linewidth=1,edgecolor='k',facecolor=color))

            if square != 0:
                self.text(0.5*(left+right), 0.5*(bottom+top), str(square),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=30, color='black',
                        transform=self.transAxes)

        self.figure.canvas.draw()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=4, help="")
    parser.add_argument('--init', type=str, default=None, help="")
    args = parser.parse_args()

    Environment = PuzzleN(args.n)
    state = np.array([Environment.solvedState])
    #state = np.array([Environment.generate_envs(1,[1000,1000])[0][0]])
    if args.init is not None:
        state = np.array([[int(x) for x in re.sub("[^0-9,]","",args.init).split(",")]])

    """
    startTime = time.time()
    for i in range(50):
        temp = np.array(Environment.generate_envs(100,[1,500])[0][0])
    print("Generate time %s" % (time.time()-startTime))
    """
    heuristicFn = None
    #if args.n == 4:
    #    heuristicFn = nnet_utils.loadNnet("savedModels/nnet_1_500_0_PUZZLE15/","model.meta",1,Environment)
    #elif args.n == 5:
    #    heuristicFn = nnet_utils.loadNnet("savedModels/nnet_1_500_0_PUZZLE24/","model.meta",1,Environment)

    fig = plt.figure(figsize=(5, 5))
    interactiveEnv = InteractiveEnv(state,Environment,heuristicFn)
    fig.add_axes(interactiveEnv)

    plt.show()
