# Evaluate MNLI
# Metrics: Accuracy
# Sentence pair classification
# 3 classes

from fairseq.models.roberta import RobertaModel

roberta = RobertaModel.from_pretrained(
    model_name_or_path='base_single/client_0/MNLI/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='MNLI-bin/'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
Matched_Accuracy, Mismatched_Accuracy = 0.0, 0.0
roberta.cuda()
roberta.eval()

with open('glue_data/MNLI/dev_matched.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[8], tokens[9], tokens[15]
        # print(sent1, sent2, target)
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('mnli_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        ncorrect += int(prediction_label == target)
        nsamples += 1
        Matched_Accuracy = float(ncorrect)/float(nsamples)

ncorrect, nsamples = 0, 0

with open('glue_data/MNLI/dev_mismatched.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[8], tokens[9], tokens[15]
        # print(sent1, sent2, target)
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('mnli_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        ncorrect += int(prediction_label == target)
        nsamples += 1
        Mismatched_Accuracy = float(ncorrect)/float(nsamples)

print('| Matched Accuracy: ', Matched_Accuracy)
print('| Mismatched Accuracy: ', Mismatched_Accuracy)