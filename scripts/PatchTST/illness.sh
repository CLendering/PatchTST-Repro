if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/supervised" ]; then
    mkdir ./logs/supervised
fi

model_name=PatchTST
model_identifier=patchtst_illness
dataset=illness
input_length=104

for prediction_length in 24 36 48 60
do
    python -u src/training/train.py \
      --train_mode \
      --model_identifier $model_identifier'_'$input_length'_'$prediction_length \
      --model $model_name \
      --dataset $dataset \
      --features M \
      --input_length $input_length \
      --prediction_length $prediction_length \
      --encoder_input_size 7 \
      --num_encoder_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_fcn 128 \
      --dropout 0.3 \
      --fc_dropout 0.3 \
      --head_dropout 0 \
      --patch_length 24 \
      --stride 2 \
      --epochs 100 \
      --patience 20 \
      --learning_rate_adjustment constant \
      --bootstrap_iterations 1 --batch_size 16 --learning_rate 0.0025 >logs/supervised/$model_identifier'_'$input_length'_'$prediction_length.log 
done