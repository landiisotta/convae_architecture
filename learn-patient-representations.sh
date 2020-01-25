#! /bin/zsh

clear
virtualenvdir=../stratification_ILRM/myvenv/bin/python
indir=./data_example

gpu=0

test_set=$1

if [ ! -z "$test_set" ]
then
    test_set="--test_set $test_set"
fi

CUDA_VISIBLE_DEVICES=$gpu $virtualenvdir -u ./patient_representations.py $indir $test_set
