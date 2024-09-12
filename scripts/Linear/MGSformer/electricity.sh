if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=MGSformer

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

random_seed=2021
for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name_$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 321 \
    --e_layers 2 \
    --n_heads 20 \
    --x 9\
    --d_model 128 \
    --d_ff 204 \
    --dropout 0.31173761314157555\
    --fc_dropout 0.14776342038685644\
    --head_dropout 0.0661134740854798\
    --patch_len 16\
    --stride 8\
    --des 'Exp' \
    --train_epochs 100\
    --patience 10\
    --lradj 'TST'\
    --pct_start 0.2\
    --period 36\
    --momentum 0.5157849724080178 \
    --itr 1 --batch_size 23 --learning_rate 0.00019775888211389357 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done