

output=cv/model_width129_v2

mkdir -p ${output}

for i in `seq 0 4`
do
    python trainer.py --config config.json --cv $i --random 0  > ${output}/$i.log
done
