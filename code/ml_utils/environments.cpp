#include "environments.h"
#include <map>

uint8_t **getSwapZeroIdxs(int dim) {
  uint8_t **swapZeroIdxs = new uint8_t*[dim*dim];

  for (int i=0; i<(dim*dim); i++) {
    swapZeroIdxs[i] = new uint8_t[4];
  }

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
          swapZeroIdxs[zIdx][move] = (uint8_t) (swap_i*dim + swap_j);
        } else {
          swapZeroIdxs[zIdx][move] = (uint8_t) zIdx;
        }
      }
    }
  }

  return(swapZeroIdxs);
}

Environment::~Environment() {
}

/*** Puzzle15 ***/
uint8_t **Puzzle15::swapZeroIdxs = getSwapZeroIdxs(4);

Puzzle15::Puzzle15(std::vector<uint8_t> state, uint8_t zIdx) {
	this->state = state;
	this->zIdx = zIdx;
}

Puzzle15::Puzzle15(std::vector<uint8_t> state) {
	for (uint8_t i=0; i<state.size(); i++) {
		if (state[i] == 0) {
			this->zIdx = i;
			break;
		}
	}
	this->state = state;
}

Puzzle15::~Puzzle15() {}


Puzzle15 *Puzzle15::getNextState(const int action) const {
	std::vector<uint8_t> newState(this->state);
	
	uint8_t swapZeroIdx = this->swapZeroIdxs[this->zIdx][action];

	uint8_t val = newState[swapZeroIdx];
	newState[this->zIdx] = val;
	newState[swapZeroIdx] = 0;

	Puzzle15 *nextState = new Puzzle15(newState,swapZeroIdx);

	return(nextState);
}

std::vector<Environment*> Puzzle15::getNextStates() const {
	std::vector<Environment*> nextStates;
	for (int i=0; i<numActions; i++) {
		nextStates.push_back(this->getNextState(i));
	}

	return(nextStates);
}

std::vector<uint8_t> Puzzle15::getState() const {
	return(this->state);
}

bool Puzzle15::isSolved() const {
	bool isSolved = true;
	const int numTiles = 16;
	for (int i=0; i<numTiles; i++) {
		isSolved = isSolved & (this->state[i] == ((i+1) % numTiles));
	}

	return(isSolved);
}

float Puzzle15::getReward() const {
	float reward = this->isSolved() ? 1.0 : -1.0;
	return(reward);
}

int Puzzle15::getNumActions() const {
	return(this->numActions);
}

/// Puzzle24
uint8_t **Puzzle24::swapZeroIdxs = getSwapZeroIdxs(5);

Puzzle24::Puzzle24(std::vector<uint8_t> state, uint8_t zIdx) {
	this->state = state;
	this->zIdx = zIdx;
}

Puzzle24::Puzzle24(std::vector<uint8_t> state) {
	for (uint8_t i=0; i<state.size(); i++) {
		if (state[i] == 0) {
			this->zIdx = i;
			break;
		}
	}
	this->state = state;
}

Puzzle24::~Puzzle24() {}


Puzzle24 *Puzzle24::getNextState(const int action) const {
	std::vector<uint8_t> newState(this->state);
	
	uint8_t swapZeroIdx = this->swapZeroIdxs[this->zIdx][action];

	uint8_t val = newState[swapZeroIdx];
	newState[this->zIdx] = val;
	newState[swapZeroIdx] = 0;

	Puzzle24 *nextState = new Puzzle24(newState,swapZeroIdx);

	return(nextState);
}

std::vector<Environment*> Puzzle24::getNextStates() const {
	std::vector<Environment*> nextStates;
	for (int i=0; i<numActions; i++) {
		nextStates.push_back(this->getNextState(i));
	}

	return(nextStates);
}

std::vector<uint8_t> Puzzle24::getState() const {
	return(this->state);
}

bool Puzzle24::isSolved() const {
	bool isSolved = true;
	const int numTiles = 25;
	for (int i=0; i<numTiles; i++) {
		isSolved = isSolved & (this->state[i] == ((i+1) % numTiles));
	}

	return(isSolved);
}

float Puzzle24::getReward() const {
	float reward = this->isSolved() ? 1.0 : -1.0;
	return(reward);
}

int Puzzle24::getNumActions() const {
	return(this->numActions);
}

/// Puzzle35
uint8_t **Puzzle35::swapZeroIdxs = getSwapZeroIdxs(6);

Puzzle35::Puzzle35(std::vector<uint8_t> state, uint8_t zIdx) {
	this->state = state;
	this->zIdx = zIdx;
}

Puzzle35::Puzzle35(std::vector<uint8_t> state) {
	for (uint8_t i=0; i<state.size(); i++) {
		if (state[i] == 0) {
			this->zIdx = i;
			break;
		}
	}
	this->state = state;
}

Puzzle35::~Puzzle35() {}


Puzzle35 *Puzzle35::getNextState(const int action) const {
	std::vector<uint8_t> newState(this->state);
	
	uint8_t swapZeroIdx = this->swapZeroIdxs[this->zIdx][action];

	uint8_t val = newState[swapZeroIdx];
	newState[this->zIdx] = val;
	newState[swapZeroIdx] = 0;

	Puzzle35 *nextState = new Puzzle35(newState,swapZeroIdx);

	return(nextState);
}

std::vector<Environment*> Puzzle35::getNextStates() const {
	std::vector<Environment*> nextStates;
	for (int i=0; i<numActions; i++) {
		nextStates.push_back(this->getNextState(i));
	}

	return(nextStates);
}

std::vector<uint8_t> Puzzle35::getState() const {
	return(this->state);
}

bool Puzzle35::isSolved() const {
	bool isSolved = true;
	const int numTiles = 36;
	for (int i=0; i<numTiles; i++) {
		isSolved = isSolved & (this->state[i] == ((i+1) % numTiles));
	}

	return(isSolved);
}

float Puzzle35::getReward() const {
	float reward = this->isSolved() ? 1.0 : -1.0;
	return(reward);
}

int Puzzle35::getNumActions() const {
	return(this->numActions);
}

///LightsOut
int **getMoveMat(int dim) {
  int **moveMat = new int*[dim*dim];
  for (int move=0; move<(dim*dim); move++) {
    moveMat[move] = new int[5];

		int xPos = move/dim;
		int yPos = move % dim;

		int right = xPos < (dim-1) ? move + dim : move;
		int left = xPos > 0 ? move - dim : move;
		int up = yPos < (dim-1) ? move + 1 : move;
		int down = yPos > 0 ? move - 1 : move;

		moveMat[move][0] = move;
		moveMat[move][1] = right;
		moveMat[move][2] = left;
		moveMat[move][3] = up;
		moveMat[move][4] = down;
  }

	return(moveMat);
}
int **moveMat7 = getMoveMat(7);

LightsOut::LightsOut(std::vector<uint8_t> state, uint8_t dim) {
	this->state = state;
	this->dim = dim;

	this->numActions = (this->dim)*(this->dim);

	if (dim == 7) {
		moveMat = moveMat7;
	}
}

LightsOut::~LightsOut() {}

LightsOut *LightsOut::getNextState(const int action) const {
	std::vector<uint8_t> newState(this->state);
	
	
	for (int i=0; i<5; i++) {
		newState[moveMat[action][i]] = (uint8_t) ((int) (this->state[moveMat[action][i]] + 1)) % 2;
	}

	LightsOut *nextState = new LightsOut(newState,this->dim);

	return(nextState);
}

std::vector<Environment*> LightsOut::getNextStates() const {
	std::vector<Environment*> nextStates;
	for (int i=0; i<numActions; i++) {
		nextStates.push_back(this->getNextState(i));
	}

	return(nextStates);
}

std::vector<uint8_t> LightsOut::getState() const {
	return(this->state);
}

bool LightsOut::isSolved() const {
	bool isSolved = true;
	const int numTiles = (this->dim)*(this->dim);
	for (int i=0; i<numTiles; i++) {
		isSolved = isSolved & (this->state[i] == 0);
	}

	return(isSolved);
}

float LightsOut::getReward() const {
	float reward = this->isSolved() ? 1.0 : -1.0;
	return(reward);
}

int LightsOut::getNumActions() const {
	return(this->numActions);
}


/// Cube3
constexpr int Cube3::rotateIdxs_old[12][24];
constexpr int Cube3::rotateIdxs_new[12][24];

Cube3::Cube3(std::vector<uint8_t> state) {
	this->state = state;
}

Cube3::~Cube3() {}


Cube3 *Cube3::getNextState(const int action) const {
	std::vector<uint8_t> newState(this->state);
	
	for (int i=0; i<24; i++) {
		const int oldIdx = this->rotateIdxs_old[action][i];
		const int newIdx = this->rotateIdxs_new[action][i];
		newState[newIdx] = this->state[oldIdx];
	}

	Cube3 *nextState = new Cube3(newState);

	return(nextState);
}

std::vector<Environment*> Cube3::getNextStates() const {
	std::vector<Environment*> nextStates;
	for (int i=0; i<numActions; i++) {
		nextStates.push_back(this->getNextState(i));
	}

	return(nextStates);
}

std::vector<uint8_t> Cube3::getState() const {
	return(this->state);
}

bool Cube3::isSolved() const {
	bool isSolved = true;
	for (int i=0; i<54; i++) {
		isSolved = isSolved & (this->state[i] == i);
	}

	return(isSolved);
}

float Cube3::getReward() const {
	float reward = this->isSolved() ? 1.0 : -1.0;
	return(reward);
}

int Cube3::getNumActions() const {
	return(this->numActions);
}