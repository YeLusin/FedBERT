# Evaluate SST-2
# Metrics: Accuracy
# Single sentence classification
# 2 classes

from fairseq.models.roberta import RobertaModel

roberta = RobertaModel.from_pretrained(
    'base_heavg/fine_tuning_c0/SST-2',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='SST-2-bin/'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
with open('glue_data/SST-2/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent, target = tokens[0], tokens[1]
        # print(tokens)
        # print(sent, target)
        tokens = roberta.encode(sent)
        prediction = roberta.predict('sst2_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))