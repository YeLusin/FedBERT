# Evaluate QNLI
# Metrics: Accuracy
# Sentence pair classification
# 2 classes

from fairseq.models.roberta import RobertaModel

roberta = RobertaModel.from_pretrained(
    model_name_or_path='base_heavg/fine_tuning_c0/QNLI',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='QNLI-bin/'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()

with open('glue_data/QNLI/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[1], tokens[2], tokens[3]
        # print(tokens)
        #print(sent1, sent2, target)
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('qnli_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))