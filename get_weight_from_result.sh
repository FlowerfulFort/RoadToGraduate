#!/bin/bash

if [ $# -ne 1 ]
then
    echo "Usage: $0 [index_of_model]"
    exit 1
fi

file_number=$1
cp /results/weights/2m_4fold_512_30e_d0.2_g0.2/fold${file_number}_best.pth $(dirname $(realpath $0))/.
