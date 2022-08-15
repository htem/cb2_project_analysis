
###################################
# 220725
# now we also loop over synapse pct

run_tests_2b () {
    runscript="$1"
    configs="${@:2}"
    export SILENT_MODE=1
    echo '' > $runscript
    # for seed in `seq 200 500`; do
    for seed in `seq 250`; do
    for model in local_random_expanded2 ; do
    # echo "python batch_220725_snr_synapse.py $configs --increase_sharing 0 --clumpy_mfs 0 --model local_random_expanded2 --seed $seed --skip_finished 1"  >> $runscript
    echo "python batch_220725_snr_synapse.py $configs --increase_sharing 0 --clumpy_mfs 1 --model local_random_expanded2 --seed $seed --skip_finished 1"  >> $runscript
    # echo "python batch_220725_snr_synapse.py $configs --increase_sharing 1 --clumpy_mfs 0 --model local_random_expanded2 --seed $seed --skip_finished 1"  >> $runscript
    echo "python batch_220725_snr_synapse.py $configs --increase_sharing 1 --clumpy_mfs 1 --model local_random_expanded2 --seed $seed --skip_finished 1"  >> $runscript
    done
    done
    parallel < $runscript
}

# configs="--pattern_size_pct 30"
# run_tests_2b "runscript_220725_2_dwalin" $configs
# configs="--pattern_size_pct 100"
# run_tests_2b "runscript_220725_6_dwalin" $configs

# configs="--num_patterns 12 --pattern_size_pct 30"
# run_tests_2b "runscript_220725_2_balin" $configs
    configs="--num_patterns 16 --pattern_size_pct 30"
    run_tests_2b "runscript_220725_3_balin" $configs
    # configs="--num_patterns 32 --pattern_size_pct 30"
    # run_tests_2b "runscript_220725_4_balin" $configs
