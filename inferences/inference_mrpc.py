# Evaluate MRPC
# Metrics: F1/Accuracy
# Sentence pair classfication
# 2 classes

from fairseq.models.roberta import RobertaModel

roberta = RobertaModel.from_pretrained(
    model_name_or_path='base_heavg/fine_tuning_c0/MRPC/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='MRPC-bin/'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()

with open('glue_data/MRPC/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        target, sent1, sent2 = tokens[0], tokens[3], tokens[4]
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('mrpc_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        # print(sent1, sent2, target, prediction_label)
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))
# print("F1: ", )