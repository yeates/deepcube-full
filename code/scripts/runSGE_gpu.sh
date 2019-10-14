MACHINE=$1
NUM_GPUS=$2
CMD=$3

# Parse command
FILE_NAME=$(echo $CMD | sed 's/\s\+/_/g')

# Create file name
FILE_NAME=$(echo $FILE_NAME | sed 's/^[^\/]*\///')
FILE_NAME=$(echo $FILE_NAME | sed 's/\//_/g')
FILE_NAME=$(echo $FILE_NAME | sed 's/--//g')
FILE_NAME=$(echo $FILE_NAME | sed 's/"//g')
FILE_NAME="SGEScripts/${FILE_NAME}.sh"

# Print to screen
echo "CMD: $CMD"
echo "Filename: $FILE_NAME"

# Write to file
echo "cd /home/baldig-projects/forest/RubiksCube_cost/" > $FILE_NAME
echo $CMD >> $FILE_NAME

# qsub
qsub $MACHINE -P arcus_gpu.p -l gpu=$NUM_GPUS -o /home/baldig-projects/forest/RubiksCube_cost/log/ -e /home/baldig-projects/forest/RubiksCube_cost/log/ $FILE_NAME
