model_name=iTransformer

# if [ ! -d "./logs" ]; then
#     mkdir ./logs
# fi

# if [ ! -d "./logs/Dataset_Channel_dependent" ]; then
#     mkdir ./logs/Dataset_Channel_dependent
# fi

model_id_name=Dataset_Channel_dependent
data_name=Dataset_Channel_dependent
seq_len=336
pred_len=96

python -u run.py \
  --train_epochs 100 \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id $model_id_name_$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 3 \
  --n_heads 1 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --des 'Dataset_Channel_dependent' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 512 \
  --learning_rate 0.001 \
  --use_multi_gpu \
  --devices '0,1' \
  --itr 1