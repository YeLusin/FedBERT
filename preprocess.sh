#!/bin/bash
# Encode and binaize data for each client



# CLIENTS_NUMBER=10


# encode raw data with the GPT-2 BPE
# .raw --> .bpe

#!!! need to change the number of clients in "for"
# and inputs and outputs dir
for client in {0..9}; do\
    echo Start processing CLIENT $client:
    for SPLIT in train valid test; do \
        python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json ./gpt2_bpe/encoder.json \
        --vocab-bpe ./gpt2_bpe/vocab.bpe \
        --inputs ./datasets_clients/client_$client/${SPLIT}.raw \
        --outputs ./datasets_clients/client_$client/${SPLIT}.bpe \
        --keep-empty \
        --workers 60; \
        echo $SPLIT.bpe Generated
    done
    #echo CLIENT $client Encoding Finished.
done



# preprocess/binarize .bpe using the GPT-2 fairseq dictionary
# .bpe --> data-bin/


for client in {0..9}; do\
    echo Start generating data-bin of CLIENT $client:
    fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref ./datasets_clients/client_$client/train.bpe \
    --validpref ./datasets_clients/client_$client/valid.bpe \
    --testpref ./datasets_clients/client_$client/test.bpe \
    --destdir ./datasets_clients/client_$client/data-bin \
    --workers 60
    echo CLIENT $client: data-bin/ Generated
done