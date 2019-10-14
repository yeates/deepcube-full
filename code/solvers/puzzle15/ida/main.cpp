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
#include <sys/time.h>

static const int dim = 4;
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
	PDB(int tileNames[], int numTiles_pdb, const char *filename) {
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

	void update_idx(int val, int prevPos, int currPos) {
		uint64_t multVal = posMultAll[val];
		currIdx -= prevPos*multVal;
		currIdx += currPos*multVal;
	}

	int lookup_curr_idx() {
		return (int) pdb[currIdx];
	}

};

class IDAStar {
	int manhattanDist[numTiles][numTiles];
	int manhattanDist_incr[numTiles][numTiles][numTiles];

	PDB *pdb_1;
	PDB *pdb_2;
	PDB *pdb_3;
	PDB *pdb_4;

	int numPDBs;
	PDB **pdbs;

	long int nodesSeen;
	int minBound;
	int movesToRev[4];
	std::deque<int> moveStack;

	int getHeuristic() {
		int heuristic1 = pdb_1->lookup_curr_idx() + pdb_2->lookup_curr_idx();
		int heuristic2 = pdb_3->lookup_curr_idx() + pdb_4->lookup_curr_idx();
		int heuristic = std::max(heuristic1,heuristic2);
		return heuristic;
	}
public:
	IDAStar() {
    int tileNames_1[8] = {1,2,5,6,9,10,13,14};
    pdb_1 = new PDB(tileNames_1, 8, "7_8_2.bin");

    int tileNames_2[7] = {3,4,7,8,11,12,15};
    pdb_2 = new PDB(tileNames_2, 7, "7_8_1.bin");

    int tileNames_3[8] = {1,2,3,4,5,6,7,8};
    pdb_3 = new PDB(tileNames_3, 8, "7_8_4.bin");

    int tileNames_4[7] = {9,10,11,12,13,14,15};
    pdb_4 = new PDB(tileNames_4, 7, "7_8_3.bin");

		numPDBs = 4;
		pdbs = new PDB*[numPDBs];
		pdbs[0] = pdb_1; pdbs[1] = pdb_2;
		pdbs[2] = pdb_3; pdbs[3] = pdb_4;

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
			int zIdx = tiles->zIdx;
			int val = tiles->nextState(move);
			int swapZeroIdx = tiles->zIdx;

			for (int i=0; i<numPDBs; i++) {
				pdbs[i]->update_idx(val,swapZeroIdx,zIdx);
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
				pdbs[i]->update_idx(val,zIdx,swapZeroIdx);
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
			pdbs[i]->init_idx(tiles);
		}

		bound_start = getHeuristic();

		printf("Starting Bound: %i, Zero Index: %i, PDB_IDX: %li, PDB_VALS: %i,%i\n",bound_start,tiles->zIdx,pdb_1->lookup_idx(tiles),pdb_1->lookup(tiles),pdb_2->lookup(tiles));
				
		int bound = bound_start; //Do IDA
		bool solnFound = false;
		long int totalNodesSeen = 0;
		moveStack.clear();

		timespec searchStart;
		struct timeval tval_before, tval_after, tval_result;

		gettimeofday(&tval_before, NULL);

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

		gettimeofday(&tval_after, NULL);
		timersub(&tval_after, &tval_before, &tval_result);

		printf("Total time: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

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
