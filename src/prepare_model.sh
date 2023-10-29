#!/bin/bash

path_here=$(dirname $(realpath $0))

# file check
if [ $(ls -l ${path_here}/weights | grep ^- | wc -l) -eq 0 ]
then
    echo "No weights here.. Please check your pth file."
    exit 1
fi

mkdir -p /results/weights/2m_4fold_512_30e_d0.2_g0.2/
cp ${path_here}/weights/* /results/weights/2m_4fold_512_30e_d0.2_g0.2/
chmod -R 777 /results
