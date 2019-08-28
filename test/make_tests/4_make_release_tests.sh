#!/bin/bash

matlabdir='aquacrop_matlab_test_data'
pythondir='aquacrop_python_test_data'
aostestdir='../release'

for dir in $pythondir/*/
do
    basedir=`basename $dir`
    newdir=test_"${basedir}"
    echo $dir
    echo $newdir
    if [ -d $aostestdir/$newdir ]
    then
        rm -r $aostestdir/$newdir
    fi    

    cp -r $pythondir/$basedir $aostestdir/$newdir
    for yrdir in $aostestdir/$newdir/*/; do
        yr=`basename $yrdir`
        outdir=$aostestdir/$newdir/$yr/Output
        if [ "$(ls -A $outdir)" ]
        then
            rm -r $outdir/*
        fi        
    done

    for yrdir in $matlabdir/$basedir/*/
    do
        yr=`basename $yrdir`
        cp -r $matlabdir/$basedir/$yr/Output $aostestdir/$newdir/$yr/Expected_Output
    done    
done

# clean up
rm -r $pythondir
