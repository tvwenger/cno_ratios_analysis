#!/bin/bash

tar -xzvf 107-23.tar.gz
tar -xzvf 042-25.tar.gz

# temporary pytensor compiledir
tmpdir=`mktemp -d`
echo "starting to analyze $1"
PYTENSOR_FLAGS="base_compiledir=$tmpdir" python fit_iram.py $1
rm -rf $tmpdir
