#!/bin/bash

# temporary pytensor compiledir
tmpdir=`mktemp -d`
echo "starting to analyze $1"
PYTENSOR_FLAGS="base_compiledir=$tmpdir" python run_bayes_cn_hfs.py $1
rm -rf $tmpdir
