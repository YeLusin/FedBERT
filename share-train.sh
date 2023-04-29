#!/bin/bash
# sequential


TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=16        # Number of sequences per batch (batch size)
UPDATE_FREQ=16          # Increase the batch size 16x
MAX_EPOCH=500           # Max local epoch of each client

ROOT_DIR=./base_share/datasets_clients
CLIENTS_NUMBER=10 

for client in {6..9}; do
    echo Start training CLIENT $client
    echo Based on $ROOT_DIR/client_$[$client-1]/

    CUDA_VISIBLE_DEVICES=0,1,2,3
    nohup fairseq-train --fp16 $ROOT_DIR/client_$client/data-bin \
    --task masked_lm --criterion masked_lm \
    --arch roberta_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 --tensorboard-logdir $ROOT_DIR/client_$client/logdir \
    --save-interval 50 \
    --max-epoch $[$MAX_EPOCH+$[$MAX_EPOCH*$client]] \
    --save-dir $ROOT_DIR/client_$client/checkpoints \
    --restore-file $ROOT_DIR/client_$[$client-1]/checkpoints/checkpoint_last.pt;\
done