#!/bin/bash

tar -xzvf alma.tar.gz

# temporary pytensor compiledir
tmpdir=`mktemp -d`
echo "starting to analyze $1"
PYTENSOR_FLAGS="base_compiledir=$tmpdir" python fit_alma.py $1 $2
rm -rf $tmpdir
