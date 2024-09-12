# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=DLinear

#python -u run_longExp.py \
#  --is_training 1 \
#  --root_path ./dataset/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_$seq_len'_'96 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --seq_len $seq_len \
#  --pred_len 96 \
#  --enc_in 7 \
#  --des 'Exp' \
#  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'96.log
#
#python -u run_longExp.py \
#  --is_training 1 \
#  --root_path ./dataset/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_$seq_len'_'192 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --seq_len $seq_len \
#  --pred_len 192 \
#  --enc_in 7 \
#  --des 'Exp' \
#  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'192.log
#
#python -u run_longExp.py \
#  --is_training 1 \
#  --root_path ./dataset/ \
#  --data_path ETTh1.csv \
#  --model_id ETTh1_$seq_len'_'336 \
#  --model $model_name \
#  --data ETTh1 \
#  --features M \
#  --seq_len $seq_len \
#  --pred_len 336 \
#  --enc_in 7 \
#  --des 'Exp' \
#  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'336.log

#python -u run_longExp.py \
#  --is_training 1 \
#  --root_path ./dataset/ \
#  --data_path shejiangdianwang.csv \
#  --model_id ETTh1_$seq_len'_'720 \
#  --model $model_name \
#  --data custom \
#  --features M \
#  --seq_len $seq_len \
#  --fc_dropout 0.8677107409585632\
#  --dropout 0.5113542437771635\
#  --momentum 0.968195178228064\
#  --pred_len 480 \
#  --enc_in 7 \
#  --des 'Exp' \
#  --itr 1 --batch_size 662 --learning_rate 0.0065922090115510005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'480.log

python -u run_longExp.py \
  --is_training 0 \
  --root_path ./dataset/ \
  --data_path shejiangdianwang.csv \
  --model_id ETTh1_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 1 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'96.log
