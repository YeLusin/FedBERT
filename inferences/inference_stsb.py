from fairseq.models.roberta import RobertaModel
from scipy.stats import pearsonr

roberta = RobertaModel.from_pretrained(
    'base_heavg/fine_tuning_c0/STS-B/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='STS-B-bin'
)

roberta.cuda()
roberta.eval()
gold, pred = [], []
with open('glue_data/STS-B/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[7], tokens[8], float(tokens[9])
        tokens = roberta.encode(sent1, sent2)
        features = roberta.extract_features(tokens)
        predictions = 5.0 * roberta.model.classification_heads['stsb_head'](features)
        # print(predictions.item(), target)
        gold.append(target)
        pred.append(predictions.item())

print('| Pearson: ', pearsonr(gold, pred))