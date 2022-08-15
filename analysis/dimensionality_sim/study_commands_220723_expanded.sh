
###################################

# run the 4 models

run_tests_4_models () {
    runscript="$1"
    configs="${@:2}"
    export SILENT_MODE=1
    echo '' > $runscript
    for seed in `seq 100`; do
    echo "python batch_220726a_across_noise.py $configs --increase_sharing 0 --clumpy_mfs 0 --model local_random_expanded2 --seed $seed "  >> $runscript
    echo "python batch_220726a_across_noise.py $configs --increase_sharing 0 --clumpy_mfs 1 --model local_random_expanded2 --seed $seed "  >> $runscript
    echo "python batch_220726a_across_noise.py $configs --increase_sharing 1 --clumpy_mfs 0 --model local_random_expanded2 --seed $seed "  >> $runscript
    echo "python batch_220726a_across_noise.py $configs --increase_sharing 1 --clumpy_mfs 1 --model local_random_expanded2 --seed $seed "  >> $runscript
    done
    parallel -j 4 < $runscript
}

# configs=""
# run_tests_4_models "runscript_dimex_112825_balin" $configs
# configs="--pattern_type uniform"
# run_tests_4_models "runscript_dimex_112826_balin" $configs


run_tests_dim () {
    runscript="$1"
    configs="${@:2}"
    export SILENT_MODE=1
    echo '' > $runscript
    for seed in `seq 100`; do
    echo "python batch_220726a_across_noise.py $configs --n_grcs 1459 --n_mfs 630 --seed $seed "  >> $runscript
    done
    parallel -j 4 < $runscript
}

# configs="--model observed"
# run_tests_dim "runscript_dimex_112825_dwalin" $configs
# configs="--model local_random"
# run_tests_dim "runscript_dimex_112825_dwalin" $configs
# configs="--model local_random2"
# run_tests_dim "runscript_dimex_112825_dwalin" $configs

# configs="--pattern_type uniform --model observed"
# run_tests_dim "runscript_dimex_112825_dwalin0" $configs
# configs="--pattern_type uniform --model local_random"
# run_tests_dim "runscript_dimex_112825_dwalin0" $configs
# configs="--pattern_type uniform --model local_random2"
# run_tests_dim "runscript_dimex_112825_dwalin0" $configs






