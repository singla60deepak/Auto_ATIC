# variables
target = 'Line_item_target'
feature = 'line_item_narative'

# model hyperparams
num_labels = 2
learning_rate = 2e-5
epochs = 3 #change this to 1 when data is less
batch_size = 16
loss='binary_crossentropy'
metric='accuracy'
model_name='bert-base-cased'

input_path='/home/jupyter/NLP_Track/Deepak/Git/Files/spam_aug_v11.csv'

file_for_pred_input_path='/home/jupyter/NLP_Track/Deepak/Git/Files/sample_kw_spam.csv'
final_classification_report_output="/home/jupyter/NLP_Track/Deepak/Git/Files/classification_report.txt"