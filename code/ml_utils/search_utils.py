import numpy as np
import sys
sys.path.append('./')
import nnet_utils
import time

from heapq import heappush, heappop
import gc

#from multiprocessing import Process, Queue

class Node():
    def __init__(self, state, isSolved, reward, parent, depth=None):
            self.visits = 0
            self.value = None
            self.state = state
            self.parent = parent
            self.children = []
            self.depth = depth
    
            self.isSolved = isSolved
            self.reward = reward

            self.strVal = self.state.tostring()
            self.hashVal = hash(self.strVal)

            if self.isSolved:
                self.value = 0.0

    def getParent(self):
        return(self.parent)

    def addChild(self,child):
        self.children.append(child)

    def getChild(self,childIdx):
        return(self.children[childIdx])

    def getChildren(self):
        return(self.children)

    def getDescendents(self):
        descendents = np.array(self.getChildren())
        for child in self.getChildren():
            descendents = np.concatenate([descendents,child.getDescendents()])

        return(descendents)
    
    def getIsSolved(self):
        return(self.isSolved)

    def setValue(self,value):
        self.value = value

    def getValue(self):
        if self.getIsSolved():
            return(0.0)
        else:
            return(self.value)

    def getReward(self):
        return(self.reward)

    def backupValues(self):
        if self.getIsSolved():
            self.value = 0.0
        elif len(self.children) > 0:
            self.value = -np.inf
            for node in self.children:
                node.backupValues()

                childRewardValue = node.getReward() + node.getValue()
                self.value = max(self.value,childRewardValue)

    def visit(self):
        self.visits = self.visits +  1

    def getVisits(self):
        return(self.visits)
 
    def getState(self):
        return(self.state.copy())
    
    def __str__(self):
        return(self.strVal)

    def __hash__(self):
        return(self.hashVal)

    def __repr__(self):
        s = "Solved: %s, Reward: %s, Value: %s, Num Children: %i, Visits: %i" % (self.getIsSolved(),self.getReward(),
                                                                                            self.getValue(),len(self.children),
                                                                                            self.visits)
        return s

    def __eq__(self, other):
        return np.min(self.state == other.state)

    def __ne__(self, other):
        return not self.__eq__(other)

class Tree(object):
    def __init__(self, states, heuristicFn, Environment, values=None, bfs=0):
        self.roots = []

        self.heuristicFn = heuristicFn
        self.Environment = Environment
        self.legalMoves = self.Environment.legalPlays

        self.batchSize = 5000

        self.values = values

        self.bfs = bfs
        
        states = np.stack(states,axis=0)
        isSolved = self.Environment.checkSolved(states)
        rewards = self.Environment.getReward(states,isSolved)

        for idx,state in enumerate(states):
            # 0:state, 1:value, 2:isSolved, 3:reward, 4:parent_move, 5:depth
            state = state.astype(self.Environment.dtype)

            node = [state,None,isSolved[idx],rewards[idx],-1,0]
            self.roots.append(node)
    
    def addNewNode(self,stateHashRep,parentMove,depth,parentHashRep):
        self.seenNodes[stateHashRep] = [parentMove,depth,parentHashRep]

    # input nodes should all be unexpanded, errors will occur, otherwise
    def expand_static(self,states,verbose=False):
        # 0:state, 1:value, 2:isSolved, 3:reward, 4:parent_move, 5:depth
        seenNodes = self.seenNodes

        ### Get next states
        startTime = time.time()
        cStates, cRewards, cIsSolveds = nnet_utils.getNextStates(states,self.Environment) # next states
        cStates = cStates.astype(self.Environment.dtype)
        
        numStates = states.shape[0]
        childrenPerState = cStates.shape[1]
        numChildren = numStates*childrenPerState

        cStates = cStates.reshape((numStates*childrenPerState,cStates.shape[2])) # reshape to numStates*childrenPerState
        cRewards = cRewards.reshape((numStates*childrenPerState))
        cIsSolveds = cIsSolveds.reshape((numStates*childrenPerState))

        cParentMoves = np.array(range(childrenPerState)*numStates)

        self.numGenerated = self.numGenerated + cStates.shape[0]

        nextStateTime = time.time() - startTime

        """
        ### Send data to be evaluated
        resQueue = Queue(1)
        heuristicProc = Process(target=lambda x: resQueue.put(self.computeNodeValues(x)[:,0]), args=(cStates,))
        heuristicProc.daemon = True
        heuristicProc.start()
        """


        ### Get all child information
        startTime = time.time()
        
        cDepths = np.expand_dims([seenNodes[state.tostring()][1] for state in states],axis=1)
        cDepths = np.repeat(cDepths,childrenPerState,axis=1).reshape((numStates*childrenPerState))
        cParentHashReps = []
        for state in states:
            stateHashRep = state.tostring()
            for cIdx in range(childrenPerState):
                cParentHashReps.append(stateHashRep)

        #cDepths = cDepths + np.array([len(self.legalMoves[x]) if type(self.legalMoves[x][0]) == type(list()) else 1 for x in cParentMoves])

        cDepths = cDepths + 1

        cHashReps = [x.tostring() for x in cStates]

        childrenInfoTime = time.time() - startTime

        ### Add states that haven't been seen
        startTime = time.time()
        addToQueue_idxs = []
        for cIdx in range(numChildren):
            cParentMove = cParentMoves[cIdx]
            cDepth = cDepths[cIdx]
            cHashRep = cHashReps[cIdx]
            cParentHashRep = cParentHashReps[cIdx]

            getNode = seenNodes.get(cHashRep)
            if (getNode is None) or (cDepth < getNode[1]):
                addToQueue_idxs.append(cIdx)
                self.addNewNode(cHashRep,cParentMove,cDepth,cParentHashRep)
 
        cStates_add = cStates[addToQueue_idxs]
        cDepths_add = cDepths[addToQueue_idxs]
        cIsSolveds_add = cIsSolveds[addToQueue_idxs]

        checkSeenTime = time.time() - startTime

        ### Compute values
        startTime = time.time()
        if cStates_add.shape[0] > 0:
            cVals_add = self.computeNodeValues(cStates_add)[:,0]
            #cVals_add = resQueue.get()[addToQueue_idxs]
            #heuristicProc.join()
            #heuristicProc.terminate()

            computeValueTime = time.time() - startTime

            ### Push to priority queue
            startTime = time.time()
            heapVals = cVals_add*(np.logical_not(cIsSolveds_add)) + cDepths_add*self.depthPenalty

            for heapVal,cState in zip(heapVals,cStates_add):
                heappush(self.unexpanded,(heapVal,self.nodeCount,cState))
                self.nodeCount = self.nodeCount + 1

            heapPushTime = time.time() - startTime
        else:
            cVals_add = []
            computeValueTime = time.time() - startTime
            heapPushTime = time.time() - startTime


        if verbose:
            print("TIMES - Next state: %.3f, children data proc: %.3f, check seen: %.3f, val comp: %.3f, heappush: %.3f" % (nextStateTime,childrenInfoTime,checkSeenTime,computeValueTime,heapPushTime))
            print("%i Children, %i Added" % (numChildren,len(addToQueue_idxs)))
            #print([int(x) for x in cStates[np.argmin(cVals_add)]])
        return(cVals_add,cDepths_add)
   
    def computeNodeValues(self,states):
        if self.bfs > 0:
            stateVals,_ = self.breadthFirstSearch(states,self.bfs)
        else:
            stateVals = self.heuristicFn(states)
        
        return(stateVals)
     
    def breadthFirstSearch(self,states_root,searchDepth=2,verbose=False):
        statesAtDepth = []
        rewardsAtDepth = []
        isSolvedAtDepth = []

        isSolved_root = self.Environment.checkSolved(states_root)

        statesAtDepth.append(states_root)
        rewardsAtDepth.append(self.Environment.getReward(states_root,isSolved_root))
        isSolvedAtDepth.append(isSolved_root)

        for depth in range(1,searchDepth+1):
            nextStates, nextStateRewards, nextStateSolved = nnet_utils.getNextStates(statesAtDepth[-1],self.Environment)
            nextStates = nextStates.reshape([nextStates.shape[0]*nextStates.shape[1],nextStates.shape[2]])

            statesAtDepth.append(nextStates)
            rewardsAtDepth.append(nextStateRewards)
            isSolvedAtDepth.append(nextStateSolved)

        isSolved = isSolvedAtDepth[-1]
        valsBackup = self.heuristicFn(statesAtDepth[-1])

        valsBackup = valsBackup.reshape(valsBackup.shape[0]/len(self.legalMoves),len(self.legalMoves))
        valsBackup = valsBackup*(np.logical_not(isSolved)) + 0.0*(isSolved)

        for depth in range(len(statesAtDepth)-2,-1,-1):
            valsBackup_children = valsBackup
            rewards_children = rewardsAtDepth[depth+1]

            valsBackup = np.min(rewards_children + valsBackup_children,1)
            
                
            isSolved = isSolvedAtDepth[depth]
            if depth > 0:
                valsBackup = valsBackup.reshape(valsBackup.shape[0]/len(self.legalMoves),len(self.legalMoves))
            valsBackup = valsBackup*(np.logical_not(isSolved)) + 0.0*(isSolved)

    
        rootValsBackup = np.expand_dims(valsBackup,1)
        nextStatesValueReward = valsBackup_children + rewards_children

        return(rootValsBackup, nextStatesValueReward)
     
        
    def combineNodes(self,nodes):
        states= []
        for nodeIdx,node in enumerate(nodes):
            states.append(node[0])

        states = np.stack(states,axis=0)

        return(states)

    def getTrajectory(self,state,seenNodes):
        # 0:state, 1:value, 2:isSolved, 3:reward, 4:parent_move, 5:depth
        moves = []

        state_key = state.tostring()
        while seenNodes[state_key][2] is not None: # While there is a parent
            node = seenNodes[state_key]

            moveIdx = node[0]
            moves.append(self.Environment.legalPlays[moveIdx])

            state_key = node[2]
            

        moves = moves[::-1]
        
        ### Flatten in case of hierarchical moves
        moves_flat = []
        for move in moves:
            if type(move[0]) == type(list()):
                moves_flat.extend(move)
            else:
                moves_flat.append(move)

        moves = moves_flat

        return(moves)

    @staticmethod
    def generateToDepth(states_root,depth,Environment):
        statesAtDepth = []
        rewardsAtDepth = []
        isSolvedAtDepth = []

        isSolved_root = Environment.checkSolved(states_root)

        statesAtDepth.append(states_root)
        rewardsAtDepth.append(Environment.getReward(states_root,isSolved_root))
        isSolvedAtDepth.append(isSolved_root)

        for depth in range(1,depth+1):
            nextStates, nextStateRewards, nextStateSolved = nnet_utils.getNextStates(statesAtDepth[-1],Environment)
            nextStates = nextStates.reshape([nextStates.shape[0]*nextStates.shape[1],nextStates.shape[2]])

            statesAtDepth.append(nextStates)
            rewardsAtDepth.append(nextStateRewards)
            isSolvedAtDepth.append(nextStateSolved)

        return(statesAtDepth,rewardsAtDepth,isSolvedAtDepth)

    @staticmethod
    def backupValues(valsBackup,statesAtDepth,rewardsAtDepth,isSolvedAtDepth,Environment):
        numLegalMoves = len(Environment.legalPlays)

        isSolved = isSolvedAtDepth[-1]

        valsBackup = valsBackup.reshape(valsBackup.shape[0]/numLegalMoves,numLegalMoves)
        valsBackup = valsBackup*(np.logical_not(isSolved)) + 0.0*(isSolved)

        #valsBackup = np.maximum(valsBackup,0.0)

        for depth in range(len(statesAtDepth)-2,-1,-1):
            valsBackup_children = valsBackup
            rewards_children = rewardsAtDepth[depth+1]

            valsBackup = np.min(rewards_children + valsBackup_children,1)
            #valsBackup = np.max(rewards_children + valsBackup_children,1)
                
            isSolved = isSolvedAtDepth[depth]
            if depth > 0:
                valsBackup = valsBackup.reshape(valsBackup.shape[0]/numLegalMoves,numLegalMoves)
            valsBackup = valsBackup*(np.logical_not(isSolved)) + 0.0*(isSolved)

    
        rootValsBackup = valsBackup
        nextStatesValueReward = valsBackup_children + rewards_children

        return(rootValsBackup, nextStatesValueReward)

class BFS(Tree):
    def __init__(self, states, heuristicFn, Environment, values=None, bfs=0):
        Tree.__init__(self, states, heuristicFn, Environment, values, bfs)
        
    def run(self,searchDepth=2,verbose=False):
        statesAtDepth = []
        rewardsAtDepth = []
        isSolvedAtDepth = []

        states_root = self.combineNodes(self.roots)
        isSolved_root = self.Environment.checkSolved(states_root)

        statesAtDepth.append(states_root)
        rewardsAtDepth.append(self.Environment.getReward(states_root,isSolved_root))
        isSolvedAtDepth.append(isSolved_root)

        for depth in range(1,searchDepth+1):
            nextStates, nextStateRewards, nextStateSolved = nnet_utils.getNextStates(statesAtDepth[-1],self.Environment)
            nextStates = nextStates.reshape([nextStates.shape[0]*nextStates.shape[1],nextStates.shape[2]])

            statesAtDepth.append(nextStates)
            rewardsAtDepth.append(nextStateRewards)
            isSolvedAtDepth.append(nextStateSolved)

        isSolved = isSolvedAtDepth[-1]
        valsBackup = self.heuristicFn(statesAtDepth[-1])

        valsBackup = valsBackup.reshape(valsBackup.shape[0]/len(self.legalMoves),len(self.legalMoves))
        valsBackup = valsBackup*(np.logical_not(isSolved)) + 0.0*(isSolved)

        for depth in range(len(statesAtDepth)-2,-1,-1):
            valsBackup_children = valsBackup
            rewards_children = rewardsAtDepth[depth+1]

            valsBackup = np.min(rewards_children + valsBackup_children,1)
            
                
            isSolved = isSolvedAtDepth[depth]
            if depth > 0:
                valsBackup = valsBackup.reshape(valsBackup.shape[0]/len(self.legalMoves),len(self.legalMoves))
            valsBackup = valsBackup*(np.logical_not(isSolved)) + 0.0*(isSolved)

    
        rootValsBackup = valsBackup
        nextStatesValueReward = valsBackup_children + rewards_children

        return(rootValsBackup, nextStatesValueReward)

class BestFS_solve(Tree):
    def __init__(self, states, heuristicFn, Environment, values=None, bfs=0):
        Tree.__init__(self, states, heuristicFn, Environment, values, bfs)
        self.unexpanded = []
        self.seenNodes = []
        self.numExpanded = []
        self.numGenerated = []

        self.nodeCount = 0

        self.roots = np.array(self.roots)

    def run(self,numParallel=100,depthPenalty=0.1,verbose=False):
        # 0:state, 1:value, 2:isSolved, 3:reward, 4:parent_move, 5:depth
        isSolved = False
        solveSteps = np.inf
        solvedNode = None
        self.depthPenalty = depthPenalty

        rootIdx = 0 # TODO make parallelizable, assuming only one for now
        node = self.roots[rootIdx]

        self.seenNodes = dict()
        self.unexpanded = []
        self.numExpanded = 0
        self.numGenerated = 1
        
        rootVal = self.computeNodeValues(np.array([node[0]]))
        
        self.addNewNode(node[0].tostring(),node[4],node[5],None)

        heappush(self.unexpanded,(rootVal*(not node[2]) + node[5]*depthPenalty,self.nodeCount,node[0]))
        self.nodeCount = self.nodeCount + 1

        rolloutNum = 0

        while isSolved == False:
            # 0:state, 1:value, 2:isSolved, 3:reward, 4:parent_move, 5:depth
            rolloutNum = rolloutNum + 1
            if verbose:
                print("Iteration: %i" % (rolloutNum))
            rollout_start_time = time.time()

            ### Get nodes to expand
            startTime = time.time()
            statesToExpand = [heappop(self.unexpanded)[2] for i in range(min(numParallel,len(self.unexpanded)))]
            statesToExpand = np.stack(statesToExpand)

            statePopTime = time.time() - startTime
            ### Check if nodes are solved
            isSolved_where = np.where(self.Environment.checkSolved(statesToExpand))[0]
            if len(isSolved_where) > 0:
                isSolved = True
                solvedNode = statesToExpand[isSolved_where[0]]

            ### Expand nodes
            vals, depths = self.expand_static(statesToExpand,verbose)

            self.numExpanded = self.numExpanded + statesToExpand.shape[0]

            if verbose:
                if len(vals) > 0:
                    # 0:state, 1:value, 2:isSolved, 3:reward, 4:parent_move, 5:depth
                    print("Min/Max - Depth: %i/%i, Value(depth): %.2f(%i)/%.2f(%i), numSeen: %i, numFronteir: %i, PopTime: %s" % (min(depths),max(depths),min(vals),depths[np.argmin(vals)],max(vals),depths[np.argmax(vals)],len(self.seenNodes),len(self.unexpanded),statePopTime))
                else:
                    print("All nodes have values already")
            
            rollout_elapsed_time = time.time() - rollout_start_time
            if verbose:
                print("Time: %0.2f\n" % (rollout_elapsed_time))

        trajChanged = True
        while trajChanged:
            moves = self.getTrajectory(solvedNode,self.seenNodes)
            trajChanged = False                      

        solveSteps = moves

        self.seenNodes = []
        self.unexpanded = []
        self.heuristicFn = []
        self.roots = []
        del self.seenNodes
        del self.unexpanded
        gc.collect()

        isSolved = [isSolved] #TODO change after making search parallelizable
        solveSteps = [solveSteps]
        self.numExpanded = [self.numExpanded]
        self.numGenerated = [self.numGenerated]

        return(isSolved,solveSteps,self.numGenerated)

def solve(cubes,heuristicFn,Environment,maxTurns=50,searchDepth=1,numRollouts=10,searchMethod="BFS",verbose=False):
    legalMoves = Environment.legalPlays
    searchMethod = searchMethod.upper()

    isSolved = np.zeros(len(cubes),dtype=bool)
    solveSteps = np.zeros((len(cubes)),dtype=int)

    cubes = np.stack(cubes)

    for tryIdx in range(maxTurns):
        if verbose:
            print("-------- MOVE: %i --------" % (tryIdx + 1))
        ### Check which ones are done
        isSolved = Environment.checkSolved(cubes)
        if np.min(isSolved) == True:
            break

        solveSteps[np.logical_not(isSolved)] = solveSteps[np.logical_not(isSolved)] + 1

        cubes_unsolved = cubes[np.logical_not(isSolved),:]

        ### Do a search to get the next move
        statesAtDepth, rewardsAtDepth, isSolvedAtDepth = Tree.generateToDepth(cubes_unsolved,searchDepth,Environment)
        _, nextStatesValueReward = Tree.backupValues(heuristicFn(statesAtDepth[-1]),statesAtDepth,rewardsAtDepth,isSolvedAtDepth,Environment)

        nextMoves = np.argmin(nextStatesValueReward,axis=1)
        #nextMoves = np.argmax(nextStatesValueReward,axis=1)

        ### Move cube
        for cubeIdx, cube in enumerate(cubes_unsolved):
            cubes_unsolved[cubeIdx,:] = Environment.next_state(cube, legalMoves[nextMoves[cubeIdx]])

        cubes[np.logical_not(isSolved),:] = cubes_unsolved

    return isSolved, solveSteps

