
python ../trainer/predict.py \
    --model_list gMVP_model_paths.txt \
    --output gMVP_raw_score_Feb24_part2.tsv \
    --config ../trainer/config_predict.json     \
    --transcript_list ./transcript_id_part2.list     \
    --feature_dir /share/BigData/hz2529/MVPContext/combined_feature_2021_v2/ \
    --batch_size 512
