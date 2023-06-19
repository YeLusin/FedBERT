# Evaluate CoLA
# Metrics: Matthew's Corr
# Single sentence classification
# 2 classes

from fairseq.models.roberta import RobertaModel
import math

roberta = RobertaModel.from_pretrained(
    'fed/checkpoints/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='CoLA-bin/'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)

tp, tn, fp, fn = 0, 0, 0, 0
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()

with open('glue_data/CoLA/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        target, sent = tokens[1], tokens[3]
        # print(sent, target)
        tokens = roberta.encode(sent)
        prediction = roberta.predict('cola_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        if prediction_label == target and prediction_label == '1':
            tp += 1
        elif prediction_label != target and prediction_label == '1':
            fp += 1
        elif prediction_label == target and prediction_label == '0':
            tn += 1
        elif prediction_label != target and prediction_label == '0':
            fn += 1
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))
print('tp, tn, fp, fn:\n',tp, tn, fp, fn)
print('|Matthew\'s Corr', float((tp*tn - fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))