This code is for training and running the DeepCubeA algorithm

### REQUIREMENTS ###
- python: 2.7.5
- tensorflow-gpu: version 1.8.0

### Environments ###
- Rubik's Cube 3x3x3 (cube3)
- 15-Puzzle (puzzle15)
- 24-Puzzle (puzzle25)
- 35-Puzzle (puzzle35)
- 48-Puzzle (puzzle48)
- LightsOut 7x7 (lightsout7)
- sokoban (sokoban)


### Arguments ###
- env: put in the corresponding name in parenthesis in the Environments section
- scramb_max or max_s (same thing) argument: 30 for cube3, and 500 for puzzle15 and puzzle24
- eps: the error threshold required before the training algorithm updates the network doing value iteration. In our experiments we set this to 0.05

### TRAINING A MODEL ###
While we used multiple GPUs to train our model, training a model from scratch on a single GPU can take weeks. If you would like to modify our code to parallelize DeepCubeA, the GEN_DATA_CMD should be run with a system, such as SGE, that allows one to submit jobs to different machines.

Additionally, one could decrease the batch size. To obtain our results, we used batch size of 10,000 but we this file is using a batch size of 1000 for a Code Ocean machine. However, this could result in a decrease in performance.

To train a model on a single GPU, use the following commands:
mkdir savedModels/<nnet_name>
sh scripts/trainDeepCubeA_local.sh <nnet arch> <batch_size> <data_save_location> <scramb_max> <num_steps> <env> <nnet_name> <eps> 2>&1 | tee -a savedModels/<nnet_name>/output

To train a model for the 3x3x3 Rubik's Cube from scratch:
mkdir savedModels/cube3_new
export CUDA_VISIBLE_DEVICES="0"
sh scripts/trainDeepCubeA_local.sh "python scripts/trainValueFunc_est.py --num_l 1 --num_res 4 --num_h 1000 --lr_i 0.001 --lr_d 0.9999993 --opt adam --act_type relu --batch_norm --solve_itrs 5000" 1000 labeledData/cube3/ 30 1 cube3 cube3_new 0.05 2>&1 | tee -a savedModels/cube3_new/output


To train a model for the 15-puzzle from scratch:
mkdir savedModels/puzzle15_new
export CUDA_VISIBLE_DEVICES="0"
sh scripts/trainDeepCubeA_local.sh "python scripts/trainValueFunc_est.py --num_l 1 --num_res 4 --num_h 1000 --lr_i 0.001 --lr_d 0.9999993 --opt adam --act_type relu --batch_norm --solve_itrs 5000" 1000 labeledData/puzzle15/ 500 1 puzzle15 puzzle15_new 0.05 2>&1 | tee -a savedModels/cube3_new/output

To train a model for the 24-puzzle from scratch:
mkdir savedModels/puzzle24_new
export CUDA_VISIBLE_DEVICES="0"
sh scripts/trainDeepCubeA_local.sh "python scripts/trainValueFunc_est.py --num_l 1 --num_res 6 --num_h 1000 --lr_i 0.001 --lr_d 0.9999993 --opt adam --act_type relu --batch_norm --solve_itrs 5000" 1000 labeledData/puzzle24/ 500 1 puzzle24 puzzle24_new 0.05 2>&1 | tee -a savedModels/puzzle24_new/output

To train a model for the 35-puzzle from scratch:
mkdir savedModels/puzzle35_new
export CUDA_VISIBLE_DEVICES="0"
sh scripts/trainDeepCubeA_local.sh "python scripts/trainValueFunc_est.py --num_l 1 --num_res 4 --num_h 1000 --lr_i 0.001 --lr_d 0.9999993 --opt adam --act_type relu --batch_norm --solve_itrs 5000" 1000 labeledData/puzzle35/ 1000 100 puzzle35 puzzle35_new 1.0 2>&1 | tee -a savedModels/puzzle35_new/output

To train a model for the 48-puzzle from scratch:
mkdir savedModels/puzzle48_new
export CUDA_VISIBLE_DEVICES="0"
sh scripts/trainDeepCubeA_local.sh "python scripts/trainValueFunc_est.py --num_l 1 --num_res 4 --num_h 1000 --lr_i 0.001 --lr_d 0.9999993 --opt adam --act_type relu --batch_norm --solve_itrs 5000" 1000 labeledData/puzzle48/ 1000 100 puzzle48 puzzle48_new 1.0 2>&1 | tee -a savedModels/puzzle48_new/output

To train a model for the Lights Out from scratch:
mkdir savedModels/lightsout7
export CUDA_VISIBLE_DEVICES="0"
sh scripts/trainDeepCubeA_local.sh "python scripts/trainValueFunc_est.py --num_l 1 --num_res 4 --num_h 1000 --lr_i 0.001 --lr_d 0.9999993 --opt adam --act_type relu --batch_norm --solve_itrs 5000" 1000 labeledData/lightsout7/ 500 50 lightsout7 lightsout7_new 1.0 2>&1 | tee -a savedModels/lightsout7_new/output

To train a model for the Sokoban from scratch:
mkdir savedModels/sokoban
export CUDA_VISIBLE_DEVICES="0"
sh scripts/trainDeepCubeA_local.sh "python scripts/trainValueFunc_est.py --num_l 1 --num_res 4 --num_h 1000 --lr_i 0.001 --lr_d 0.9999993 --opt adam --act_type relu --batch_norm --solve_itrs 5000" 1000 labeledData/sokoban/ 1000 50 sokoban sokoban_new 1.0 2>&1 | tee -a savedModels/sokoban_new/output


### Testing a Model with Weighted A* search ###
python scripts/solveStartingStates.py --input ../data/<env>/states.pkl --env <env> --methods nnet --model_loc <location of model> --nnet_parallel <parallelize search, recommend starting with 100> --depth_penalty <value from 0.0 to 1.0> --verbose

- Runs weighted A* search with the model given by --model_loc on states given by --input.
- Outputs results to same directory as input
- --depth_penalty: 0.0 is GBFS and 1.0 is A* search. Higher values typically lead to better results but are also more likely to run out of memory
- --nnet_parallel: Higher values typically lead to better results, search will slow down for higher values, but takes advantage of GPU parallelism

### Example Testing for Rubik's Cube ###
python scripts/solveStartingStates.py --input ../data/cube3/states.pkl --env cube3 --methods nnet --model_loc savedModels/cube3/1/ --nnet_parallel 100 --depth_penalty 0.2
