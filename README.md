# FedBERT
The repository is based on fairseq
https://github.com/facebookresearch/fairseq \
git clone https://github.com/pytorch/fairseq

### Download data
Wikitext103
```
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip
mkdir -p gpt2_bpe
wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
```
GLUE benchmark
``` python download_glue_data.py --data_dir glue_data --tasks all ```

### Preprocess
```
python sample.py --number_clients 10 --input_dir ./wikitext-103-raw/ --output_dir ./datasets_clients
bash preprocess.sh
```

### Train
```
bash avg-train.sh
bash emdavg-train.sh
bash head_emd-train.sh
bash share-train.sh
```
### Test
```
bash fine-tune.sh
bash inference.sh
```
