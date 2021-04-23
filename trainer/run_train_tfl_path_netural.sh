
output_dir=res_channel_gene_path_netural/v1
mkdir -p ${output_dir}

n=0
for m in `cat ../finalize_score/gMVP_model_paths.txt`
do
    python trainer_tfl.py \
        --init_model $m \
        --config config_tfl_path_netural.json  > $output_dir/$n.log
    n=$((n+1))
done
