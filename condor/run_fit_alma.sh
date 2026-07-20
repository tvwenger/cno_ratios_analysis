#!/bin/bash

mkdir data
tar -xzf alma.tar.gz -C data/

# temporary pytensor compiledir
tmpdir=`mktemp -d`
echo "starting to analyze $1"
PYTENSOR_FLAGS="base_compiledir=$tmpdir" python fit_alma.py $1 $2
rm -rf $tmpdir

rm -rf data/
