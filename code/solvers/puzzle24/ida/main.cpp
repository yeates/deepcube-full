#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <deque>
#include <set>
#include <vector>
#include <map>
#include <fstream>
#include <ctime>
#include <assert.h>
#include <csignal>
#include <cstdlib>
#include <stdint.h>

static const int dim = 5;
static const int numTiles = dim*dim;

void copyArr(int fromArr[], int toArr[], int numElems) {
	for (int i=0; i<numElems; i++) {
		toArr[i] = fromArr[i];
	}
}

void printArr(int arr[], int numElems) {
	for (int i=0; i<numElems; i++) {
		printf("%i ",arr[i]);
	}
	printf("\n");
}

bool fileExists(const char *fileName)
{
	std::ifstream infile(fileName);
	return infile.good();
}


class Tiles {
private:
	int swapZeroIdxs[numTiles][4];
public:
	int zIdx;
	int state[numTiles];
	int state_pos[numTiles];

	Tiles(int state_init[numTiles]) {
		/* Initialize Moves */
		for (int move=0; move<4; move++) {
			for (int i=0; i<dim; i++) {
				for (int j=0; j<dim; j++) {
					int zIdx = i*dim + j;
					bool isEligible;

					int swap_i;
					int swap_j;
					if (move == 0) { // U
							isEligible = i < (dim-1);
							swap_i = i+1;
							swap_j = j;
					} else if (move == 1) { // D
							isEligible = i > 0;
							swap_i = i-1;
							swap_j = j;
					} else if (move == 2) { // L
							isEligible = j < (dim-1);
							swap_i = i;
							swap_j = j+1;
					} else if (move == 3) { // R
							isEligible = j > 0;
							swap_i = i;
							swap_j = j-1;
					}

					if (isEligible) {
						swapZeroIdxs[zIdx][move] = swap_i*dim + swap_j;
					} else {
						swapZeroIdxs[zIdx][move] = zIdx;
					}
				}
			}
		}

		/* Initialize state */
		copyArr(state_init,state,numTiles);
		for (int t=0; t<numTiles; t++) { //initialize state_pos
			state_pos[state[t]] = t;
			if (state[t] == 0) {
				zIdx = t;
			}
		}
	}

	bool noChange(int move) {
		return(zIdx == swapZeroIdxs[zIdx][move]);
	}

	int peekVal(int move) {
		return state[swapZeroIdxs[zIdx][move]];
	}

	int nextZIdx(int move) {
		return swapZeroIdxs[zIdx][move];
	}

	int nextState(int move) {
		int swapZeroIdx = swapZeroIdxs[zIdx][move];

		int val = state[swapZeroIdx];
		state[zIdx] = val;
		state[swapZeroIdx] = 0;
		
		state_pos[val] = zIdx;
		state_pos[0] = swapZeroIdx;

		zIdx = swapZeroIdx;

		return(val);
	}

	void prevState(int zIdx, int swapZeroIdx, int val) {
		this->zIdx = zIdx;

		state[zIdx] = 0;
		state[swapZeroIdx] = val;

		state_pos[val] = swapZeroIdx;
		state_pos[0] = zIdx;
	}
};

class PDB {
private:
	uint8_t *pdb;
	uint64_t *posMult;
	uint64_t posMultAll[numTiles];
	int *tileNames;
	int numTiles_pdb;

	uint64_t numEntries_found;
	int minBound;

	uint64_t currIdx;

	std::deque<uint64_t> unexpanded;

	void visitZPos(Tiles *tiles,int zIdx) {
		if (tiles->state[zIdx] != 0) {
			return;
		} else {
			tiles->state[zIdx] = -1;
			for (int move=0; move<4; move++) {
				if (tiles->noChange(move)) {
					continue;
				}
				int nextZIdx = tiles->nextZIdx(move);
				tiles->zIdx = nextZIdx;

				visitZPos(tiles,nextZIdx);

				tiles->zIdx = zIdx;
			}
		}
	}
	
	int getReachableZPos(Tiles *tiles,int zIdx) {
		Tiles *tilesCopy = new Tiles(tiles->state);
		tilesCopy->zIdx = zIdx;

		visitZPos(tilesCopy,zIdx);


		uint32_t r = 0;
		for (int t=0; t<numTiles; t++) {
			if (tilesCopy->state[t] == -1) {
				r |= (1<<t);
			}
		}

		delete tilesCopy;

		return r;
	}

public:
	PDB() {
	}
	void init(int tileNames[], int numTiles_pdb, const char *filename) {
		this->numTiles_pdb = numTiles_pdb;
		this->tileNames = new int[numTiles_pdb];
		copyArr(tileNames,this->tileNames,numTiles_pdb);
    
		/* Initialize pdb size*/
		for (int i=0; i<numTiles; i++) {
			posMultAll[i] = 0;
		}
		posMult = new uint64_t[numTiles_pdb];
		uint64_t entriesPerPos = 1;
		uint64_t numEntries = 1;
		for (int i=numTiles_pdb-1; i>=0; i--) {
			fprintf(stderr,"I: %i (%i), Entries per pos: %li\n",i,tileNames[i],numEntries);
			posMult[i] = entriesPerPos;
			posMultAll[tileNames[i]] = posMult[i];
			entriesPerPos *= (numTiles);
			numEntries *= (numTiles-i);
		}
		uint64_t pdbSize = entriesPerPos;
		fprintf(stderr,"Pattern database size is %li, number of entires is %li\n",pdbSize,numEntries);

		pdb = new uint8_t[pdbSize];

		if (fileExists(filename)) {
			fprintf(stderr,"Loading file %s\n",filename);
			FILE *pFile = fopen(filename, "rb");
			fread(pdb , sizeof(uint8_t), pdbSize, pFile);
			fclose(pFile);
		} else {
			/* Do BFS to get all pos*/
			fprintf(stderr,"Creating pdb and saving to file %s\n",filename);
			uint32_t *zeroLocs;
			zeroLocs = new uint32_t[pdbSize];

			for (uint64_t i=0; i<pdbSize; i++) {
				pdb[i] = -1;
				zeroLocs[i] = 0;
			}

			int solvedTiles[numTiles] = {0};
			for (int i=0; i<numTiles_pdb; i++) {
				solvedTiles[tileNames[i]-1] = tileNames[i]; // change when zero position changes
			}
			printArr(solvedTiles,numTiles);
			
			Tiles *tiles = new Tiles(solvedTiles);
			numEntries_found = 0;
			int bound = 0;

			uint64_t pdb_idx = lookup_idx(tiles);
			unexpanded.push_back(pdb_idx);
			pdb[pdb_idx] = (uint8_t) bound;
			zeroLocs[pdb_idx] |= getReachableZPos(tiles,numTiles-1); // change when zero position changes
			numEntries_found++;

			bound++;

			while(numEntries_found != numEntries) {
				timespec ts1,ts2;
				clock_gettime(CLOCK_REALTIME, &ts1);

				uint64_t numUnexpanded = unexpanded.size();
				for (uint64_t i=0; i<numUnexpanded; i++) {
					uint64_t pdb_idx_unex = unexpanded.front();
					int *state = idxToState(pdb_idx_unex);
					unexpanded.pop_front();
					Tiles *tiles_node = new Tiles(state);
					
					int zeroLocRep = zeroLocs[pdb_idx_unex];

          if (pdb_idx_unex != lookup_idx(tiles_node)) {
            fprintf(stderr,"ERROR:idxToState not consistent\n");
            std::abort();
          }

					delete [] state;

					for (int zIdx = 0; zIdx < numTiles; zIdx++) {
						if ((zeroLocRep & 1) == 0) {
							zeroLocRep>>=1;
							continue;
						}
						tiles_node->zIdx = zIdx;

						if (tiles_node->state[zIdx] != 0) {
							fprintf(stderr,"ERROR: zIdx is not 0\n");
							std::abort();
						}

						for (int move=0; move<4; move++) {
							if (tiles_node->noChange(move) || tiles_node->peekVal(move) == 0) { //check to skip
								continue;
							}

							int val = tiles_node->nextState(move);
							int swapZeroIdx = tiles_node->zIdx;
							uint64_t pdb_idx = lookup_idx(tiles_node);

							bool pushIdx = false;
							if (pdb[pdb_idx] == (uint8_t) -1) {
								pdb[pdb_idx] = (uint8_t) bound;
								numEntries_found++;
								pushIdx = true;
							}

							if ((zeroLocs[pdb_idx] & (1<<swapZeroIdx)) == 0) {
								zeroLocs[pdb_idx] |= getReachableZPos(tiles_node,swapZeroIdx);
								pushIdx = true;
							}

							if (pushIdx == true) {
								unexpanded.push_back(pdb_idx);
							}

							tiles_node->prevState(zIdx,swapZeroIdx,val);
						}
						zeroLocRep>>=1;
					}
					delete tiles_node;
				}

				clock_gettime(CLOCK_REALTIME, &ts2);
				int timeDiff = int( ts2.tv_sec - ts1.tv_sec );
				fprintf(stderr,"Bound: %i, unexpandedSize (start/now): %li/%li, Entires: %li/%li (%f%%), Time: %i\n",bound,numUnexpanded,unexpanded.size(),numEntries_found,numEntries,100.0*((float) numEntries_found)/((float) numEntries), timeDiff);

				bound++;
			}

			unexpanded.clear();
			delete [] zeroLocs;

			fprintf(stderr,"Saving file to %s\n",filename);
			FILE *pFile = fopen(filename, "wb");
			fwrite(pdb , sizeof(uint8_t), pdbSize, pFile);
			fclose(pFile);

		}
	}
	
	/*
		Find the index in the pattern database
	*/
	uint64_t lookup_idx(Tiles *tiles) {
		int *state_pos = tiles->state_pos;

		uint64_t idx = 0;
		for (int i=0; i<numTiles_pdb; i++) {
			/*
			// Determine position
			const int origPos = state_pos[tileNames[i]];
			int effective_pos = origPos;
			for (int j=0; j<i; j++) {
				if (state_pos[tileNames[j]] < origPos) {
					effective_pos -= 1;
				}
			}
			*/

			//Add position multiplied by precomputed value
			idx += state_pos[tileNames[i]]*posMult[i];
		}

		return idx;
	}

	int* idxToState(uint64_t lookup_idx) {
		int *state = new int[numTiles];
		for (int i=0; i<numTiles; i++) {
			state[i] = 0;
		}

		for (int i=0; i<numTiles_pdb; i++) {
			uint64_t origPos = lookup_idx/posMult[i];
			/*
			int effective_pos = lookup_idx/posMult[i];
	
			int posCount = -1;
			int origPos = -1;
			while (posCount < effective_pos) {
				origPos++;
				if (state[origPos] == 0) {
					posCount++;
				}
			}
			*/
			state[origPos] = tileNames[i];

			lookup_idx = lookup_idx - origPos*posMult[i];
		}
    if (lookup_idx != 0) {
      fprintf(stderr,"ERROR:idxToState failed!\n");
      std::abort();
    }

		return state;
	}

	/*
		Return: Value of state in pdb
	*/
	int lookup(Tiles *tiles) {
		return (int) pdb[lookup_idx(tiles)];
	}

	void init_idx(Tiles *tiles) {
		currIdx = lookup_idx(tiles);
	}

	void update_idx(int val, int posDiff) {
		currIdx += posMultAll[val]*(posDiff);
	}

	int lookup_curr_idx() {
		return (int) pdb[currIdx];
	}

};

class IDAStar {
	int manhattanDist[numTiles][numTiles];
	int manhattanDist_incr[numTiles][numTiles][numTiles];

	const static int numPDBs = 8;
	PDB pdbs[numPDBs];

	long int nodesSeen;
	int minBound;
	int movesToRev[4];
	std::deque<int> moveStack;

	int getHeuristic() {
		int heuristic1 = pdbs[0].lookup_curr_idx() + pdbs[1].lookup_curr_idx() + pdbs[2].lookup_curr_idx() + pdbs[3].lookup_curr_idx();
		int heuristic2 = pdbs[4].lookup_curr_idx() + pdbs[5].lookup_curr_idx() + pdbs[6].lookup_curr_idx() + pdbs[7].lookup_curr_idx();
		int heuristic = std::max(heuristic1,heuristic2);
		return heuristic;
	}
public:
	IDAStar() {
    int tileNames_1[6] = {1,2,6,7,11,12};
    pdbs[0].init(tileNames_1, 6, "6_6_6_1.bin");

    int tileNames_2[6] = {3,4,5,8,9,10};
    pdbs[1].init(tileNames_2, 6, "6_6_6_2.bin");

    int tileNames_3[6] = {13,14,15,19,20,24};
    pdbs[2].init(tileNames_3, 6, "6_6_6_3.bin");

    int tileNames_4[6] = {16,17,18,21,22,23};
    pdbs[3].init(tileNames_4, 6, "6_6_6_4.bin");

    int tileNames_5[6] = {1,2,3,6,7,8};
    pdbs[4].init(tileNames_5, 6, "6_6_6_5.bin");

    int tileNames_6[6] = {4,5,9,10,14,15};
    pdbs[5].init(tileNames_6, 6, "6_6_6_6.bin");

    int tileNames_7[6] = {11,12,16,17,21,22};
    pdbs[6].init(tileNames_7, 6, "6_6_6_7.bin");

    int tileNames_8[6] = {13,18,19,20,23,24};
    pdbs[7].init(tileNames_8, 6, "6_6_6_8.bin");

		int movesToRev_vals[4] = {1,0,3,2};
		copyArr(movesToRev_vals,movesToRev,4);

		/* Initialize Heuristics */
		//Manattan Distance
		for (int val=0; val<numTiles; val++) {
			for (int i = 0; i < dim; i++) {
				for (int j = 0; j < dim; j++) {
					if (val == 0) {
						manhattanDist[val][i*dim + j] = 0;
					} else {
						manhattanDist[val][i*dim + j] = abs(((val-1) / dim) - i) + abs(((val-1) % dim) - j);
					}
				}
			}
		}

		//Incremental heuristic
		for (int val=0; val<numTiles; val++) {
			for (int newBlank = 0; newBlank < numTiles; newBlank++) {
				for (int oldBlank = 0; oldBlank < numTiles; oldBlank++) {
					int heur_old = manhattanDist[val][newBlank];
					int heur_new = manhattanDist[val][oldBlank];

					manhattanDist_incr[val][newBlank][oldBlank] = heur_new - heur_old;
				}
			}
		}
	}

	bool ida(Tiles *tiles, int heuristic, int bound, int g, int moveToRev) {
		nodesSeen++;

		int f = g + heuristic;

		bool isSolved = heuristic == 0;

		if (f > bound) { //Check bound
			if (minBound == -1 || f < minBound) {
				minBound = f;
			}
			return false;
		} else if (isSolved) { //Check solved
			return true;
		}

		for (int move=0; move<4; move++) {
			if (move == moveToRev || tiles->noChange(move)) { //Check to skip
				continue;
			}
			
			// Get next state
			const int zIdx = tiles->zIdx;
			const int val = tiles->nextState(move);
			const int swapZeroIdx = tiles->zIdx;
			const int posDiff = zIdx-swapZeroIdx;

			for (int i=0; i<numPDBs; i++) {
				pdbs[i].update_idx(val,posDiff);
			}

			//int childHeuristic = heuristic + manhattanDist_incr[val][swapZeroIdx][zIdx];
			int childHeuristic = getHeuristic();

			//Search
			bool solnFound = ida(tiles, childHeuristic, bound, g+1, movesToRev[move]);
			if (solnFound) {
				moveStack.push_front(move);
				return solnFound;
			}

			//Reverse state
			tiles->prevState(zIdx,swapZeroIdx,val);

			for (int i=0; i<numPDBs; i++) {
				pdbs[i].update_idx(val,-posDiff);
			}
		}

		return false;
	}

	void search(int state_init[numTiles], int timeLimit) {
		Tiles *tiles = new Tiles(state_init);

		int bound_start = 0;
		for (int t=0; t<numTiles; t++) { //Get starting bound
			bound_start += manhattanDist[tiles->state[t]][t];
		}
		
		for (int i=0; i<numPDBs; i++) {
			pdbs[i].init_idx(tiles);
		}

		bound_start = getHeuristic();

		printf("Starting Bound: %i, Zero Index: %i, PDB_IDX: %li, PDB_VALS: %i,%i\n",bound_start,tiles->zIdx,pdbs[0].lookup_idx(tiles),pdbs[0].lookup(tiles),pdbs[1].lookup(tiles));
				
		int bound = bound_start; //Do IDA
		bool solnFound = false;
		long int totalNodesSeen = 0;
		clock_t begin = clock();
		moveStack.clear();

		timespec searchStart;
		clock_gettime(CLOCK_REALTIME, &searchStart);
		while (solnFound == false) {
			timespec ts1,ts2;
			clock_gettime(CLOCK_REALTIME, &ts1);

			nodesSeen = 0;
			minBound = -1;

			solnFound = ida(tiles,bound_start,bound,0,-1);

			clock_gettime(CLOCK_REALTIME, &ts2);
			int timeDiff = int( ts2.tv_sec - ts1.tv_sec );
			fprintf(stderr,"Bound: %i, New Bound: %i, Nodes Seen: %li, Move stack size: %lu, Time: %i\n",bound,minBound,nodesSeen,moveStack.size(),timeDiff);
			bound = minBound;

			totalNodesSeen += nodesSeen;

			if (timeLimit > 0 && (int( ts2.tv_sec - searchStart.tv_sec ) > timeLimit)) {
				fprintf(stdout,"Time limit reached!\n");
				break;
			}
		}
		clock_t end = clock();
		double timeDiff_search = double(end - begin) / CLOCKS_PER_SEC;
		fprintf(stdout,"Total time: %f\n",timeDiff_search);

		// Print moves
		fprintf(stdout,"Total nodes generated: %li\n",totalNodesSeen);
		char movesToSym[4] = {'U','D','L','R'};
		fprintf(stdout,"Moves are: ");
		for (int i=0; i<(int) moveStack.size(); i++) {
			fprintf(stdout,"%c",movesToSym[moveStack[i]]);
			if (i < (int) moveStack.size() - 1) {
				fprintf(stdout," ");
			}
		}
		fprintf(stdout,"\n");

		delete tiles;
	}
};

int main(int argc, const char *argv[]) {
	printf("The argument supplied is %s\n", argv[1]);
	
	/* Get input from file*/
	int timeLimit = -1;
	std::ifstream in(argv[1]);
  if(!in) {
    std::cout << "Cannot open input file.\n";
    return 1;
  }

	if (argc > 2) {
		timeLimit = atoi(argv[2]);
	}

	printf("Time limit is: %i\n",timeLimit);
	
	/* initialize Pattern Databases*/
	printf("Initializing Pattern Databases\n");
	IDAStar search;

	std::string str;
	int stateNum = 0;
	while(std::getline(in, str)) {
		/* Parse Input */
		int init[numTiles];
		int i = 0;

		std::stringstream ssin(str);
		while (ssin.good()){
			ssin >> init[i];
			++i;
		}

		/* Search */
		printf("State: %i\n",stateNum);

		printArr(init, numTiles);

		search.search(init,timeLimit);

		printf("\n");

		stateNum++;
	}

	/* Clean up*/
	in.close();

	return 0;
}
