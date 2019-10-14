#----------------------------------------------------------------------
# Matplotlib Rubik's cube simulator
# Written by Jake Vanderplas
# Adapted from cube code written by David Hogg
#   https://github.com/davidwhogg/MagicCube

import numpy as np
#matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib import widgets
from projection import Quaternion, project_points
from random import choice


import random

import sys
import time
import os
#print(os.getcwd())
sys.path.append('../code/solvers/cube3/')
sys.path.append('./solvers/cube3/')
sys.path.append('../solvers/cube3/')
from solver_algs import Kociemba

import argparse

#from solver_algs import Korf
#from solver_algs import Optimal


"""
Sticker representation
----------------------
Each face is represented by a length [5, 3] array:

  [v1, v2, v3, v4, v1]

Each sticker is represented by a length [9, 3] array:

  [v1a, v1b, v2a, v2b, v3a, v3b, v4a, v4b, v1a]

In both cases, the first point is repeated to close the polygon.

Each face also has a centroid, with the face number appended
at the end in order to sort correctly using lexsort.
The centroid is equal to sum_i[vi].

Colors are accounted for using color indices and a look-up table.

With all faces in an NxNxN cube, then, we have three arrays:

  centroids.shape = (6 * N * N, 4)
  faces.shape = (6 * N * N, 5, 3)
  stickers.shape = (6 * N * N, 9, 3)
  colors.shape = (6 * N * N,)

The canonical order is found by doing

  ind = np.lexsort(centroids.T)

After any rotation, this can be used to quickly restore the cube to
canonical position.
"""

class Cube:

    def tf_dtype(self):
        import tensorflow as tf
        return(tf.uint8)

    def __init__(self,N=3,moveType=None):
        self.dtype = np.uint8
        
        self.N = N

        self.legalPlays_qtm = [[f,n] for f in ['U','D','L','R','B','F'] for n in [-1,1]]
        self.legalPlays_qtm_rev = [[f,n] for f in ['U','D','L','R','B','F'] for n in [1,-1]]
        self.legalPlays = list(self.legalPlays_qtm)
        self.legalPlays_rev = list(self.legalPlays_qtm_rev)

        if moveType == "htm" or moveType == "htmaba":
            [self.legalPlays.append(2*[[x,1]]) for x in ['U','D','L','R','B','F']]
            [self.legalPlays_rev.append(2*[[x,1]]) for x in ['U','D','L','R','B','F']]
        if moveType == "htmaba":
            for x in ['U','D','L','R','B','F']:
                for n in [1,-1]:
                    for y in ['U','D','L','R','B','F']:
                        if x == y:
                            continue
                        for n2 in [1,-1]:
                            move = [[x,n],[y,n2],[x,-n]]
                            move_rev = [[x,n],[y,-n2],[x,-n]]

                            self.legalPlays.append(move)
                            self.legalPlays_rev.append(move_rev)


        ### Solved cube
        self.solvedState = np.array([],dtype=int)
        for i in range(6):
            self.solvedState = np.concatenate((self.solvedState,np.arange(i*(N**2),(i+1)*(N**2))))

        ### Colors to nnet representation
        # TODO only works for 3x3x3
        # WHITE:0, YELLOW:1, BLUE:2, GREEN:3, ORANGE: 4, RED: 5
        self.colorsToGet = []
        self.colorsToNnetRep = np.zeros((6*(N**2),1),dtype=np.uint8)
        idx = 0
        edgePos = 0
        cornerPos = 0
        for face in range(6):
            for i in range(N):
                for j in range(N):
                    ### Encoding of colors to get based on index
                    if i == 1 and j == 1: # center piece
                        self.colorsToNnetRep[idx,0] = 0
                    elif i == 1 or j == 1: # edge piece
                        self.colorsToNnetRep[idx,0] = edgePos
                        edgePos = edgePos + 1
                    else: # corner piece
                        self.colorsToNnetRep[idx,0] = cornerPos
                        cornerPos = cornerPos + 1

                    ### Colors to get
                    if face == 0 or face == 1:
                        if i != 1 or j != 1:
                            self.colorsToGet.append(idx)
                    elif face == 4 or face == 5:
                        if i != 1 and j == 1:
                            self.colorsToGet.append(idx)

                    
                    idx = idx + 1

        self.colorsToGet = np.sort(self.colorsToGet)

        ### Pre-compute rotation idxs
        self.rotateIdxs_old = dict()
        self.rotateIdxs_new = dict()
        for f,n in self.legalPlays_qtm:
            move = "_".join([f,str(n)])

            self.rotateIdxs_new[move] = np.array([],dtype=int)
            self.rotateIdxs_old[move] = np.array([],dtype=int)

            colors = np.zeros((6,N,N),dtype=np.int64)
            colors_new = np.copy(colors)

            # WHITE:0, YELLOW:1, BLUE:2, GREEN:3, ORANGE: 4, RED: 5
            adjFaces = {0:np.array([2,5,3,4]),
                        1:np.array([2,4,3,5]),
                        2:np.array([0,4,1,5]),
                        3:np.array([0,5,1,4]),
                        4:np.array([0,3,1,2]),
                        5:np.array([0,2,1,3])
                        }
            adjIdxs = {0:{2:[range(0,N),N-1],3:[range(0,N),N-1],4:[range(0,N),N-1],5:[range(0,N),N-1]},
                       1:{2:[range(0,N),0],3:[range(0,N),0],4:[range(0,N),0],5:[range(0,N),0]},
                       2:{0:[0,range(0,N)],1:[0,range(0,N)],4:[N-1,range(N-1,-1,-1)],5:[0,range(0,N)]},
                       3:{0:[N-1,range(0,N)],1:[N-1,range(0,N)],4:[0,range(N-1,-1,-1)],5:[N-1,range(0,N)]},
                       4:{0:[range(0,N),N-1],1:[range(N-1,-1,-1),0],2:[0,range(0,N)],3:[N-1,range(N-1,-1,-1)]},
                       5:{0:[range(0,N),0],1:[range(N-1,-1,-1),N-1],2:[N-1,range(0,N)],3:[0,range(N-1,-1,-1)]}
                       }
            faceDict = {'U':0,'D':1,'L':2,'R':3,'B':4,'F':5}
            face = faceDict[f]

            sign = 1
            if n < 0:
                sign = -1

            facesTo = adjFaces[face]
            if sign == 1:
                facesFrom = facesTo[(np.arange(0,len(facesTo))+1) % len(facesTo)]
            elif sign == -1:
                facesFrom = facesTo[(np.arange(len(facesTo)-1,len(facesTo)-1+len(facesTo))) % len(facesTo)]

            ### Rotate face TODO only works for 3x3x3
            cubesIdxs = [[0,range(0,N)],[range(0,N),N-1],[N-1,range(N-1,-1,-1)],[range(N-1,-1,-1),0]]
            cubesTo = np.array([0,1,2,3])
            if sign == 1:
                cubesFrom = cubesTo[(np.arange(len(cubesTo)-1,len(cubesTo)-1+len(cubesTo))) % len(cubesTo)]
            elif sign == -1:
                cubesFrom = cubesTo[(np.arange(0,len(cubesTo))+1) % len(cubesTo)]

            for i in range(4):
                idxsNew = [[idx1,idx2] for idx1 in np.array([cubesIdxs[cubesTo[i]][0]]).flatten() for idx2 in np.array([cubesIdxs[cubesTo[i]][1]]).flatten()]
                idxsOld = [[idx1,idx2] for idx1 in np.array([cubesIdxs[cubesFrom[i]][0]]).flatten() for idx2 in np.array([cubesIdxs[cubesFrom[i]][1]]).flatten()]
                for idxNew,idxOld in zip(idxsNew,idxsOld):
                    flatIdx_new = np.ravel_multi_index((face,idxNew[0],idxNew[1]),colors_new.shape)
                    flatIdx_old = np.ravel_multi_index((face,idxOld[0],idxOld[1]),colors.shape)
                    self.rotateIdxs_new[move] = np.concatenate((self.rotateIdxs_new[move],[flatIdx_new]))
                    self.rotateIdxs_old[move] = np.concatenate((self.rotateIdxs_old[move],[flatIdx_old]))
                #colors_new[face][cubesIdxs[cubesTo[i]]] = colors[face][cubesIdxs[cubesFrom[i]]]

            ### Rotate adjacent faces
            faceIdxs = adjIdxs[face]
            for i in range(0,len(facesTo)):
                faceTo = facesTo[i]
                faceFrom = facesFrom[i]
                idxsNew = [[idx1,idx2] for idx1 in np.array([faceIdxs[faceTo][0]]).flatten() for idx2 in np.array([faceIdxs[faceTo][1]]).flatten()]
                idxsOld = [[idx1,idx2] for idx1 in np.array([faceIdxs[faceFrom][0]]).flatten() for idx2 in np.array([faceIdxs[faceFrom][1]]).flatten()]
                for idxNew,idxOld in zip(idxsNew,idxsOld):
                    flatIdx_new = np.ravel_multi_index((faceTo,idxNew[0],idxNew[1]),colors_new.shape)
                    flatIdx_old = np.ravel_multi_index((faceFrom,idxOld[0],idxOld[1]),colors.shape)
                    self.rotateIdxs_new[move] = np.concatenate((self.rotateIdxs_new[move],[flatIdx_new]))
                    self.rotateIdxs_old[move] = np.concatenate((self.rotateIdxs_old[move],[flatIdx_old]))

                #colors_new[faceTo][faceIdxs[faceTo]] = colors[faceFrom][faceIdxs[faceFrom]]

        ### Precompute transpose
        def rotateFace(colors_cube,face,sign,N):
            colors_cube_new = np.copy(colors_cube)

            cubesIdxs = [[0,range(0,N)],[range(0,N),N-1],[N-1,range(N-1,-1,-1)],[range(N-1,-1,-1),0]]
            cubesTo = np.array([0,1,2,3])
            if sign == 1:
                cubesFrom = cubesTo[(np.arange(len(cubesTo)-1,len(cubesTo)-1+len(cubesTo))) % len(cubesTo)]
            elif sign == -1:
                cubesFrom = cubesTo[(np.arange(0,len(cubesTo))+1) % len(cubesTo)]

            for i in range(4):
                colors_cube_new[face][cubesIdxs[cubesTo[i]]] = colors_cube[face][cubesIdxs[cubesFrom[i]]]

            return(colors_cube_new)

        # WHITE:0, YELLOW:1, BLUE:2, GREEN:3, ORANGE: 4, RED: 5
        self.colorOrdIdxs = dict()
        self.faceSwapIdxs = dict()
        for faceTranspose in [0,2,4,-1]:
            idxsNew = []
            if faceTranspose == 0:
                newFaceOrder = [0,1,4,5,3,2]
                rotateFaces = [0,1]
                rotateDirs = [-1,1]
            elif faceTranspose == 2:
                newFaceOrder = [5,4,2,3,0,1]
                rotateFaces = [2,3,4,4,1,1]
                rotateDirs = [-1,1,1,1,1,1]
            elif faceTranspose == 4:
                newFaceOrder = [2,3,1,0,4,5]
                rotateFaces = [4,5,0,1,2,3]
                rotateDirs = [-1,1,1,1,1,1]
            elif faceTranspose == -1:
                newFaceOrder = [0,1,3,2,4,5]
                rotateFaces = []
                rotateDirs = []

            ### Swap colors
            for face in newFaceOrder:
                for i in range(N):
                    for j in range(N):
                        idx = np.ravel_multi_index((face,i,j),(6,3,3))
                        idxsNew.append(idx)
            idxsNew = np.array(idxsNew)

            idxsNew = idxsNew.reshape([6,N,N])
            for rotateF,rotateD in zip(rotateFaces,rotateDirs):
                idxsNew = rotateFace(idxsNew,rotateF,rotateD,N)

            if faceTranspose == -1:
                idxsNew_tmp = idxsNew.copy()
                for face in range(6):
                    idxsNew[face,0,:] = idxsNew_tmp[face,2,:]
                    idxsNew[face,2,:] = idxsNew_tmp[face,0,:]
                    
            self.colorOrdIdxs[faceTranspose] = idxsNew.flatten()
            
            ### Swap faces
            swappedColors_cube = self.solvedState.reshape([6,N,N])

            swappedColorsFaces_cube = np.zeros([6,N,N],dtype=int)
            for idx,face in enumerate(newFaceOrder):
                swappedColorsFaces_cube[idx] = swappedColors_cube[face]

            for rotateF,rotateD in zip(rotateFaces,rotateDirs):
                swappedColorsFaces_cube = rotateFace(swappedColorsFaces_cube,rotateF,rotateD,N)

            if faceTranspose == -1:
                swappedColorsFaces_cube_tmp = swappedColorsFaces_cube.copy()
                for face in range(6):
                    swappedColorsFaces_cube[face,0,:] = swappedColorsFaces_cube_tmp[face,2,:]
                    swappedColorsFaces_cube[face,2,:] = swappedColorsFaces_cube_tmp[face,0,:]

            self.faceSwapIdxs[faceTranspose] = swappedColorsFaces_cube.flatten()

    def next_state(self,colors, move, layer=0):
        """Rotate Face"""
        colorsNew = colors.copy()

        if type(move[0]) == type(list()):
            for move_sub in move:
                colorsNew = self.next_state(colorsNew,move_sub)
        else:
            moveStr = "_".join([move[0],str(move[1])])

            if len(colors.shape) == 1:
                colorsNew[self.rotateIdxs_new[moveStr]] = colors[self.rotateIdxs_old[moveStr]].copy()
            else:
                colorsNew[:,self.rotateIdxs_new[moveStr]] = colors[:,self.rotateIdxs_old[moveStr]].copy()

        return(colorsNew)

    def state_to_nnet_input(self,colors,randTransp=False):
        #colorsSort = self.get_transposes_color_sort(colors)
        #representation = self.get_nnet_representation(colorsSort)

        colors = colors.astype(self.dtype)
        if len(colors.shape) == 1:
            colors = np.expand_dims(colors,0)
        """
        if randTransp:
            colorsSort = self.get_transposes_color_sort(colors,selectRand=True)
            transpIdx = np.random.randint(colorsSort.shape[1])
            representation = self.get_nnet_representation(colorsSort[:,transpIdx,:])
        else:
            colors = np.argsort(colors,axis=1)
            colors = colors[:,self.colorsToGet]
            
            representation = self.get_nnet_representation(colors)
        """
        
        representation = self.get_nnet_representation(colors)
        return(representation)

    def get_nnet_representation(self,colors):
        colors = colors.astype(self.dtype)
        
        representation = colors/(self.N**2)
        representation = representation.astype(self.dtype)

        #representation = self.colorsToNnetRep[colors]
        #newShape = list(representation.shape[0:-2]) + [representation.shape[-2]*representation.shape[-1]]
        #representation = np.reshape(representation,newShape)

        return(representation)


    def get_transposes_color_sort(self,colors,selectRand=False,colorSort=True):
        colors = colors.astype(np.int8)
        if len(colors.shape) == 1:
            colors = np.expand_dims(colors,0)
        colorsTop0 = np.argsort(colors,axis=1).astype(np.int8) # convert to cube index
        colorsTop5 = self.transpose(colorsTop0,2)
        colorsTop1 = self.transpose(colorsTop5,2)
        colorsTop4 = self.transpose(colorsTop1,2)
        colorsTop2 = self.transpose(colorsTop0,4)
        colorsTop3 = self.transpose(self.transpose(colorsTop2,4),4)

        colorsTopAll = [colorsTop0,colorsTop1,colorsTop2,colorsTop3,colorsTop4,colorsTop5]
        if selectRand == True:
            colorsTopAll = [random.choice(colorsTopAll)]

        colorsList = []
        for colors in colorsTopAll:
            colors_posIndex = np.argsort(colors,axis=1).astype(np.int8)

            colors2 = self.transpose(colors_posIndex,0,indexType="position")
            colors2_posIndex = np.argsort(colors2,axis=1).astype(np.int8)
            
            colors3 = self.transpose(colors2_posIndex,0,indexType="position")
            colors3_posIndex = np.argsort(colors3,axis=1).astype(np.int8)
            
            colors4 = self.transpose(colors3_posIndex,0,indexType="position")
        
            colors_refl = self.transpose(colors_posIndex,-1,indexType="position")
            colors2_refl = self.transpose(colors2_posIndex,-1,indexType="position")
            colors3_refl = self.transpose(colors3_posIndex,-1,indexType="position")
            colors4_refl = self.transpose(colors4,-1)
            
            colorsList.extend([colors,colors2,colors3,colors4,colors_refl,colors2_refl,colors3_refl,colors4_refl])

        transposeList = []
        if colorSort:
            for colors in colorsList:
                colorsSort = colors
                colorsSort = colorsSort[:,self.colorsToGet]
                transposeList.append(colorsSort)
        else:
            for colors in colorsList:
                colors = np.argsort(colors,axis=1).astype(np.int8)
                transposeList.append(colors)


        allTransps = np.stack(transposeList,axis=1)
        cubesRet = allTransps

        #matchIdxs = np.argmax(allTransps[:,:,1] == 0,axis=1)
        #cubesRet = [allTransps[i,matchIdxs[i],:] for i in range(allTransps.shape[0])]
        #cubesRet = np.expand_dims(np.stack(cubesRet,0),1)
        return(cubesRet)

    def checkSolved(self,colors):
        colors = colors.astype(int)
        if len(colors.shape) == 1:
            return(np.min(colors == self.solvedState))
            #return(np.min( colors/(self.N**2) == self.solvedState/(self.N**2) ))
        else:
            solvedState_tile = np.tile(np.expand_dims(self.solvedState,0),(colors.shape[0],1)) 
            return(np.min(colors == solvedState_tile,axis=1))
            #return(np.min(colors/(self.N**2) == solvedState_tile/(self.N**2),axis=1))

    def getReward(self,colors,isSolved=None):
        reward = np.ones(shape=(colors.shape[0]))
        return(reward)

    def transpose(self,colors,faceTranspose,indexType="cube"):
        colors = colors.astype(np.int8)
        if len(colors.shape) == 1:
            colors = np.expand_dims(colors,0)

        ### Swap faces
        if indexType == "cube":
            # Convert to position index
            colors = np.argsort(colors,axis=1).astype(np.int8)
        swappedFaces = colors[:,self.faceSwapIdxs[faceTranspose]]

        ### Swap colors
        colorsArgSort = np.argsort(swappedFaces,axis=1).astype(np.int8) # convert to cube index
        colorsArgSortSelect = colorsArgSort[:,self.colorOrdIdxs[faceTranspose]]
        colorsNew = colorsArgSortSelect

        return(colorsNew)

    def generate_envs(self,numCubes,scrambleRange,probs=None,returnMoves=False):
        assert(scrambleRange[0] >= 0)
        scrambs = range(scrambleRange[0],scrambleRange[1]+1)
        legal = self.legalPlays_qtm
        cubes = []

        scrambleNums = np.zeros([numCubes],dtype=int)
        moves_all = []
        for cubeNum in range(numCubes):
            scrambled = self.solvedState

            # Get scramble Num
            scrambleNum = np.random.choice(scrambs,p=probs)
            scrambleNums[cubeNum] = scrambleNum
            # Scramble cube
            moves = []
            for i in range(scrambleNum):
                move = choice(legal)
                scrambled = self.next_state(scrambled, move)
                moves.append(move)

            cubes.append(scrambled)
            moves_all.append(moves)

        if returnMoves:
            return(cubes,scrambleNums,moves_all)
        else:
            return(cubes,scrambleNums)


class InteractiveCube(plt.Axes):
    # Define some attributes
    base_face = np.array([[1, 1, 1],
                          [1, -1, 1],
                          [-1, -1, 1],
                          [-1, 1, 1],
                          [1, 1, 1]], dtype=float)
    stickerwidth = 0.9
    stickermargin = 0.5 * (1. - stickerwidth)
    stickerthickness = 0.001
    (d1, d2, d3) = (1 - stickermargin,
                    1 - 2 * stickermargin,
                    1 + stickerthickness)
    base_sticker = np.array([[d1, d2, d3], [d2, d1, d3],
                             [-d2, d1, d3], [-d1, d2, d3],
                             [-d1, -d2, d3], [-d2, -d1, d3],
                             [d2, -d1, d3], [d1, -d2, d3],
                             [d1, d2, d3]], dtype=float)

    base_face_centroid = np.array([[0, 0, 1]])
    base_sticker_centroid = np.array([[0, 0, 1 + stickerthickness]])

    # Define rotation angles and axes for the six sides of the cube
    x, y, z = np.eye(3)
    rots = [Quaternion.from_v_theta(x, theta)
            for theta in (np.pi / 2, -np.pi / 2)]
    rots += [Quaternion.from_v_theta(y, theta)
             for theta in (np.pi / 2, -np.pi / 2, np.pi, 2 * np.pi)]

    def __init__(self,N,sess=None,inputTensPH=None,outputTens=None,state=None,interactive=True,view=(0, 0, 10),
                 fig=None, rect=[0, 0.16, 1, 0.84],**kwargs):
        self._move_list = []

        self.N = N
        self.sess = sess
        self.inputTensPH = inputTensPH
        self.outputTens = outputTens
        self._prevStates = []

        self.Environment = Cube(N=3,moveType="qtm")

        self._view = view
        self._start_rot = Quaternion.from_v_theta((1, -1, 0),
                                                  -np.pi / 6)

        if fig is None:
            fig = plt.gcf()

        # disable default key press events
        callbacks = fig.canvas.callbacks.callbacks
        del callbacks['key_press_event']

        # add some defaults, and draw axes
        kwargs.update(dict(aspect=kwargs.get('aspect', 'equal'),
                           xlim=kwargs.get('xlim', (-2.0, 2.0)),
                           ylim=kwargs.get('ylim', (-2.0, 2.0)),
                           frameon=kwargs.get('frameon', False),
                           xticks=kwargs.get('xticks', []),
                           yticks=kwargs.get('yticks', [])))
        super(InteractiveCube, self).__init__(fig, rect, **kwargs)
        self.xaxis.set_major_formatter(plt.NullFormatter())
        self.yaxis.set_major_formatter(plt.NullFormatter())

        self._start_xlim = kwargs['xlim']
        self._start_ylim = kwargs['ylim']

        # Define movement for up/down arrows or up/down mouse movement
        self._ax_UD = (1, 0, 0)
        self._step_UD = 0.01

        # Define movement for left/right arrows or left/right mouse movement
        self._ax_LR = (0, -1, 0)
        self._step_LR = 0.01

        self._ax_LR_alt = (0, 0, 1)

        # Internal state variable
        self._active = False  # true when mouse is over axes
        self._button1 = False  # true when button 1 is pressed
        self._button2 = False  # true when button 2 is pressed
        self._event_xy = None  # store xy position of mouse event
        self._shift = False  # shift key pressed
        self._digit_flags = np.zeros(10, dtype=bool)  # digits 0-9 pressed

        self._current_rot = self._start_rot  #current rotation state
        self._face_polys = None
        self._sticker_polys = None
        
        self.plastic_color = 'black'
        
        # WHITE:0 - U, YELLOW:1 - D, BLUE:2 - L, GREEN:3 - R, ORANGE: 4 - B, RED: 5 - F
        self.face_colors = ["w", "#ffcf00",
                           "#ff6f00", "#cf0000",
                           "#00008f", "#009f0f",
                           "gray", "none"]

        self._initialize_arrays()

        self._draw_cube()

        # connect some GUI events
        self.figure.canvas.mpl_connect('button_press_event',
                                       self._mouse_press)
        self.figure.canvas.mpl_connect('button_release_event',
                                       self._mouse_release)
        self.figure.canvas.mpl_connect('motion_notify_event',
                                       self._mouse_motion)
        self.figure.canvas.mpl_connect('key_press_event',
                                       self._key_press)
        self.figure.canvas.mpl_connect('key_release_event',
                                       self._key_release)

        #self._initialize_widgets()

        # write some instructions
        """
        self.figure.text(0.05, 0.05,
                         "Mouse/arrow keys adjust view\n"
                         "U/D/L/R/B/F keys turn faces\n"
                         "(hold shift for counter-clockwise)",
                         size=10)
        """

        if state is not None:
            self._colors = np.array([int(x) for x in state.split(",")])
            self._draw_cube()


    def _initialize_arrays(self):
        # initialize centroids, faces, and stickers.  We start with a
        # base for each one, and then translate & rotate them into position.

        # Define N^2 translations for each face of the cube
        cubie_width = 2. / self.N
        translations = np.array([[[-1 + (i + 0.5) * cubie_width,
                                   -1 + (j + 0.5) * cubie_width, 0]]
                                 for i in range(self.N)
                                 for j in range(self.N)])

        # Create arrays for centroids, faces, stickers
        face_centroids = []
        faces = []
        sticker_centroids = []
        stickers = []
        colors = []

        factor = np.array([1. / self.N, 1. / self.N, 1])

        for i in range(6):
            M = self.rots[i].as_rotation_matrix()
            faces_t = np.dot(factor * self.base_face
                             + translations, M.T)
            stickers_t = np.dot(factor * self.base_sticker
                                + translations, M.T)
            face_centroids_t = np.dot(self.base_face_centroid
                                      + translations, M.T)
            sticker_centroids_t = np.dot(self.base_sticker_centroid
                                         + translations, M.T)
            #colors_i = i + np.zeros(face_centroids_t.shape[0], dtype=int)
            colors_i = np.arange(i*face_centroids_t.shape[0],(i+1)*face_centroids_t.shape[0])

            # append face ID to the face centroids for lex-sorting
            face_centroids_t = np.hstack([face_centroids_t.reshape(-1, 3),
                                          colors_i[:, None]])
            sticker_centroids_t = sticker_centroids_t.reshape((-1, 3))

            faces.append(faces_t)
            face_centroids.append(face_centroids_t)
            stickers.append(stickers_t)
            sticker_centroids.append(sticker_centroids_t)

            colors.append(colors_i)

        self._face_centroids = np.vstack(face_centroids)
        self._faces = np.vstack(faces)
        self._sticker_centroids = np.vstack(sticker_centroids)
        self._stickers = np.vstack(stickers)
        self._colors = np.concatenate(colors)

    def reset(self):
        self._colors = self.Environment.solvedState

    def getState(self):
        return(self._colors)

    def _initialize_widgets(self):
        self._ax_reset = self.figure.add_axes([0.75, 0.05, 0.2, 0.075])
        self._btn_reset = widgets.Button(self._ax_reset, 'Reset View')
        self._btn_reset.on_clicked(self._reset_view)

        self._ax_solve = self.figure.add_axes([0.55, 0.05, 0.2, 0.075])
        self._btn_solve = widgets.Button(self._ax_solve, 'Solve Cube')
        self._btn_solve.on_clicked(self._solve_cube)

    def _project(self, pts):
        return project_points(pts, self._current_rot, self._view, [0, 1, 0])

    def _draw_cube(self):
        stickers = self._project(self._stickers)[:, :, :2]
        faces = self._project(self._faces)[:, :, :2]
        face_centroids = self._project(self._face_centroids[:, :3])
        sticker_centroids = self._project(self._sticker_centroids[:, :3])

        plastic_color = self.plastic_color
        #self._colors[np.ravel_multi_index((0,1,2),(6,N,N))] = 10
        #self._colors[self.Environment.colorsToGet] = 54
        colors = np.asarray(self.face_colors)[self._colors/(self.N**2)]
        face_zorders = -face_centroids[:, 2]
        sticker_zorders = -sticker_centroids[:, 2]

        if self._face_polys is None:
            # initial call: create polygon objects and add to axes
            self._face_polys = []
            self._sticker_polys = []

            for i in range(len(colors)):
                fp = plt.Polygon(faces[i], facecolor=plastic_color,
                                 zorder=face_zorders[i])
                sp = plt.Polygon(stickers[i], facecolor=colors[i],
                                 zorder=sticker_zorders[i])

                self._face_polys.append(fp)
                self._sticker_polys.append(sp)
                self.add_patch(fp)
                self.add_patch(sp)
        else:
            # subsequent call: update the polygon objects
            for i in range(len(colors)):
                self._face_polys[i].set_xy(faces[i])
                self._face_polys[i].set_zorder(face_zorders[i])
                self._face_polys[i].set_facecolor(plastic_color)

                self._sticker_polys[i].set_xy(stickers[i])
                self._sticker_polys[i].set_zorder(sticker_zorders[i])
                self._sticker_polys[i].set_facecolor(colors[i])

        self.figure.canvas.draw()

    def rotate(self, rot):
        self._current_rot = self._current_rot * rot

    def rotate_face(self, f, n=1, layer=0):
        self._move_list.append((f, n, layer))

        if not np.allclose(n, 0):
            self._colors = self.Environment.next_state(self._colors, [f, n], layer=layer)
            self._draw_cube()

    def move(self,move,draw=True):
        self._colors = self.Environment.next_state(self._colors, move)
        if draw:
            self._draw_cube()

    def _reset_view(self, *args):
        self.set_xlim(self._start_xlim)
        self.set_ylim(self._start_ylim)
        self._current_rot = self._start_rot
        self._draw_cube()

    def _solve_cube(self, *args):
        move_list = self._move_list[:]
        for (face, n, layer) in move_list[::-1]:
            self.rotate_face(face, -n, layer)
        self._move_list = []
    
    def _solve_cube_kociemba(self):
        # WHITE:0 - U, YELLOW:1 - D, BLUE:2 - L, GREEN:3 - R, ORANGE: 4 - B, RED: 5 - F
        moves = Kociemba.solve(self._colors)

        print("Solution length: %i" % (len(moves)))
        for face, n in moves: 
            self.rotate_face(face, n, 0)

    def _solve_cube_nnet(self):
        startTime = time.time()
        BestFS_solve = search_utils.BestFS_solve([self._colors],self.sess,self.inputTensPH,self.outputTens,self.Environment)
        isSolved, solveSteps = BestFS_solve.run(numParallel=200,verbose=True)

        ### Make move
        moves = solveSteps[0]
        print("Cube scrambled %i times. Neural network found solution of length %i (%s)" % (len(self._move_list),len(moves),time.time()-startTime))
        for face, n in moves:
            self.rotate_face(face, n, 0)
            time.sleep(0.1)
                
    def _key_press(self, event):
        """Handler for key press events"""
        if event.key == 'shift':
            self._shift = True
        elif event.key.isdigit():
            self._digit_flags[int(event.key)] = 1
        elif event.key == 'right':
            if self._shift:
                ax_LR = self._ax_LR_alt
            else:
                ax_LR = self._ax_LR
            self.rotate(Quaternion.from_v_theta(ax_LR,
                                                5 * self._step_LR))
        elif event.key == 'left':
            if self._shift:
                ax_LR = self._ax_LR_alt
            else:
                ax_LR = self._ax_LR
            self.rotate(Quaternion.from_v_theta(ax_LR,
                                                -5 * self._step_LR))
        elif event.key == 'up':
            self.rotate(Quaternion.from_v_theta(self._ax_UD,
                                                5 * self._step_UD))
        elif event.key == 'down':
            self.rotate(Quaternion.from_v_theta(self._ax_UD,
                                                -5 * self._step_UD))
        elif event.key.upper() in 'LRUDBF':
            if self._shift:
                direction = -1
            else:
                direction = 1

            if np.any(self._digit_flags[:N]):
                for d in np.arange(N)[self._digit_flags[:N]]:
                    self.rotate_face(event.key.upper(), direction, layer=d)
            else:
                self.rotate_face(event.key.upper(), direction)

        elif event.key.upper() in 'QWEA':
            if event.key.upper() == 'Q':
                self._colors = self.Environment.transpose(self._colors,0,indexType="position").flatten()
                self._colors = np.argsort(self._colors)
            elif event.key.upper() == 'W':
                self._colors = self.Environment.transpose(self._colors,2,indexType="position").flatten()
                self._colors = np.argsort(self._colors)
            elif event.key.upper() == 'E':
                self._colors = self.Environment.transpose(self._colors,4,indexType="position").flatten()
                self._colors = np.argsort(self._colors)
            elif event.key.upper() == 'A':
                self._colors = self.Environment.transpose(self._colors,-1,indexType="position").flatten()
                self._colors = np.argsort(self._colors)

        elif event.key.upper() == 'K':
            self._solve_cube_kociemba()

        elif event.key.upper() == 'O':
            self._solve_cube_korf()

        elif event.key.upper() == 'N':
            self._solve_cube_nnet()

        elif event.key.upper() == 'P':
            self.figure.savefig('cubeSnapshot.eps')

        self._draw_cube()
        isSolved = self.Environment.checkSolved(self._colors)
        if isSolved:
            print("SOLVED!")
            self._move_list = []
            self._prevStates = []

    def _key_release(self, event):
        """Handler for key release event"""
        if event.key == 'shift':
            self._shift = False
        elif event.key.isdigit():
            self._digit_flags[int(event.key)] = 0

    def _mouse_press(self, event, event_x=None, event_y=None):
        """Handler for mouse button press"""
        if event_x != None and event_y != None:
            self._event_xy = (event_x, event_y)
            self._button1 = True
        else:
            self._event_xy = (event.x, event.y)
            if event.button == 1:
                self._button1 = True
            elif event.button == 3:
                self._button2 = True

    def _mouse_release(self, event):
        """Handler for mouse button release"""
        self._event_xy = None
        if event.button == 1:
            self._button1 = False
        elif event.button == 3:
            self._button2 = False

    def _mouse_motion(self, event, event_x=None, event_y=None):
        """Handler for mouse motion"""
        if self._button1 or self._button2:
            if event_x != None and event_y != None:
                dx = event_x - self._event_xy[0]
                dy = event_y - self._event_xy[1]
                self._event_xy = (event_x, event_y)
            else:
                dx = event.x - self._event_xy[0]
                dy = event.y - self._event_xy[1]
                self._event_xy = (event.x, event.y)

            if self._button1:
                if self._shift:
                    ax_LR = self._ax_LR_alt
                else:
                    ax_LR = self._ax_LR
                rot1 = Quaternion.from_v_theta(self._ax_UD,
                                               self._step_UD * dy)
                rot2 = Quaternion.from_v_theta(ax_LR,
                                               self._step_LR * dx)
                self.rotate(rot1 * rot2)

                self._draw_cube()

            if self._button2:
                factor = 1 - 0.003 * (dx + dy)
                xlim = self.get_xlim()
                ylim = self.get_ylim()
                self.set_xlim(factor * xlim[0], factor * xlim[1])
                self.set_ylim(factor * ylim[0], factor * ylim[1])

                self.figure.canvas.draw()

if __name__ == '__main__':
    import sys
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=str, default=None, help="")
    parser.add_argument('--moves', type=str, default=None, help="")
    args = parser.parse_args()

    try:
        N = int(sys.argv[1])
    except:
        N = 3

    if False:
        sys.path.append('./')
        from ml_utils import search_utils
        import tensorflow as tf
        ### Set tf environment
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
        CONFIG = tf.ConfigProto(device_count = {'GPU': 0}, log_device_placement=False, allow_soft_placement=True, inter_op_parallelism_threads=1,intra_op_parallelism_threads=1)
        sess = tf.InteractiveSession(config=CONFIG)
        
        ### Load nnet
        modelLoc = "savedModels/nnet_1_20_2/"
        saver = tf.train.import_meta_graph("%s/model.meta" % (modelLoc))
        saver.restore(sess, tf.train.latest_checkpoint(modelLoc))

        ### Get network variables
        graph = tf.get_default_graph()

        inputTensPH = graph.get_tensor_by_name("Placeholder:0")

        names = [n.name for n in graph.as_graph_def().node if 'add' in n.name and 'sequential' in n.name and 'gradients' not in n.name]
        outTensName = names[-1]
        outTensName = outTensName + ":0"

        print("Output tensor name is %s" % (outTensName))
        outputTens = graph.get_tensor_by_name(outTensName)

        assert(int(outputTens.shape[1]) == 1)
    else:
        sess = None
        inputTensPH = None
        outputTens = None


    fig = plt.figure(figsize=(5, 5))
    interactiveCube = InteractiveCube(N,sess,inputTensPH,outputTens,args.state)
    fig.add_axes(interactiveCube)


    if args.moves is not None:
        fig.savefig('fig_0.eps')
        moves = args.moves.split(",")
        moves = [[x[0],int(x[1:])] for x in moves]
        for idx,move in enumerate(moves):
            interactiveCube.rotate_face(move[0],move[1])
            fig.savefig('fig_%i_%s.eps' % (idx+1,"%s%i" % (move[0],move[1])))

    plt.show()
