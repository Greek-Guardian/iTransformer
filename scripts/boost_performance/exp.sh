# export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
# model_name=Transformer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 100 \
  --patience 20 \
  --use_multi_gpu \
  --devices 0,1 \
  --no_embedding \
  # --load_path /home/liangzida/workspace/iTransformer/checkpoints/ECL_96_96_iTransformer_custom_M_ft96_sl48_ll96_pl512_dm8_nh2_el1_dl2048_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth \
