# PARAMS #
export PYTHONPATH="/segmentation_task/:$PYTHONPATH"

# ARGS #
CFG=etc/config.yml

python src/train.py --cfg $CFG