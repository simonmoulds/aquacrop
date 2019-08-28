#!/bin/bash

aosdir='../../aquacrop'
testdir='aquacrop_python_test_data'

function run_python_tests {
    arr="$1"
    years="$2"
    for elem in "${arr[@]}"
    do
        for year in "${years[@]}"
        do
            cwd=`pwd`
            wd="${testdir}"/"${elem}"/"${year}"
            cd $wd
            config_fn=`echo aos_"${elem}"_"${year}"_config.ini | awk '{print tolower($0)}'`
            config_fn=Config/$config_fn
            runner_fn="${aosdir}"/AquaCrop_Py/run.py
            $runner_fn $config_fn
            cd $cwd
        done
    done
}

# Chapter 7 tests
# ###############

declare -a arr=("Ch7_Ex1a_Tunis_Wheat")
years=($(seq 1979 1 1979))
run_python_tests "$arr" "$years"

# declare -a arr=("Ch7_Ex1a_Tunis_Wheat"
#                 "Ch7_Ex1b_Tunis_Wheat")
# years=($(seq 1979 1 2001))
# run_python_tests "$arr" "$years"

# declare -a arr=("Ch7_Ex2a_Tunis_Wheat"
#                 "Ch7_Ex2b_Tunis_Wheat")
# years=($(seq 1979 1 2001))
# run_python_tests "$arr" "$years"

# declare -a arr=("Ch7_Ex3a_Tunis_Wheat"
#                 "Ch7_Ex3b_Tunis_Wheat"
#                 "Ch7_Ex3c_Tunis_Wheat"
#                 "Ch7_Ex3d_Tunis_Wheat")
# years=($(seq 1988 1 1988))
# run_python_tests "$arr" "$years"

# # TODO: Ex4

# declare -a arr=("Ch7_Ex6_Tunis_Wheat")
# years=($(seq 1979 1 2001))
# run_python_tests "$arr" "$years"

# declare -a arr=("Ch7_Ex7a_Tunis_Wheat"
#                 "Ch7_Ex7b_Tunis_Wheat"
#                 "Ch7_Ex7c_Tunis_Wheat"
#                 "Ch7_Ex7d_Tunis_Wheat"
#                 "Ch7_Ex7e_Tunis_Wheat")
# years=($(seq 1979 1 2001))
# run_python_tests "$arr" "$years"

# # Chapter 8 tests
# # ###############

# declare -a arr=("Ch8_Ex2a_Hyderabad_Cereal"
#                 "Ch8_Ex2b_Hyderabad_Cereal")
# years=($(seq 2000 1 2010))
# run_python_tests "$arr" "$years"

# declare -a arr=("Ch8_Ex3a_Hyderabad_Cereal"
#                 "Ch8_Ex3b_Hyderabad_Cereal"
#                 "Ch8_Ex3c_Hyderabad_Cereal"
#                 "Ch8_Ex3d_Hyderabad_Cereal")
# years=($(seq 2002 1 2002))
# run_python_tests "$arr" "$years"

# declare -a arr=("Ch8_Ex6_Hyderabad_Cereal")
# years=($(seq 2000 1 2009))
# run_python_tests "$arr" "$years"

# # Chapter 9 tests
# # ###############

# declare -a arr=("Ch9_Ex1_Brussels_Potato")
# years=($(seq 1985 1 2005))
# run_python_tests "$arr" "$years"

# declare -a arr=("Ch9_Ex4_Brussels_Potato")
# years=($(seq 1985 1 2005))
# run_python_tests "$arr" "$years"

# declare -a arr=("Ch9_Ex5a_Brussels_Potato"
#                 "Ch9_Ex5b_Brussels_Potato"
#                 "Ch9_Ex5c_Brussels_Potato")
# years=($(seq 1985 1 2005))
# run_python_tests "$arr" "$years"

# declare -a arr=("Ch9_Ex6a_Brussels_Potato")
# years=($(seq 1976 1 2005))
# run_python_tests "$arr" "$years"

# declare -a arr=("Ch9_Ex6b_Brussels_Potato")
# years=($(seq 2041 1 2070))
# run_python_tests "$arr" "$years"
