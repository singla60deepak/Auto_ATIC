import pandas as pd
import numpy as np
import torch
import random
from numpy import array
from numpy import argmax

import tensorflow as tf
from tensorflow import keras

from transformers import BertTokenizer, TFBertForSequenceClassification

import util
import config

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    
set_seed(42)    #Set seed for reproducibility




import os

def aug_data_prep(input_path,aug_column_val):
    train_df=pd.read_csv(input_path)
#     train_df=train_df.drop(["line_item_narative"],axis=1)
#     train_df=train_df.rename(columns={"description":"line_item_narative"})
    train_df=train_df[train_df["aug_flag"].isin(["default",aug_column_val])]
    train_df=train_df.drop(["aug_flag"],axis=1)
    return train_df


def y_ohe(Y):
    
    #label encode
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    Y_integer_encoded = label_encoder.fit_transform(list(Y))

    # one hot encode labels
    from sklearn.preprocessing import OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse=False)
    Y_integer_encoded_reshaped = Y_integer_encoded.reshape(len(Y_integer_encoded), 1)
    Y_one_hot_encoded = onehot_encoder.fit_transform(Y_integer_encoded_reshaped)
    
    return Y_one_hot_encoded
    

def train_test_split(X,Y):
    
    # one-hot encode y
    Y_one_hot_encoded=y_ohe(Y)
    
    # train test split
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(X, Y_one_hot_encoded, test_size=0.20, random_state=42, stratify=Y)

#     X_train=X_train[feature]
#     X_val=X_val[feature]
    
    return X_train, X_val, y_train, y_val


def data_prep(X,y,model_name):
    
    from transformers import BertTokenizer, TFBertForSequenceClassification
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    #tokenise
    X_tokens = tokenizer(list(X.astype(str)),
                            truncation=True,
                            padding=True)

    
    # create TF datasets as input for BERT model
    bert_ds = tf.data.Dataset.from_tensor_slices((
        dict(X_tokens),
        y
    ))
    
    return X_tokens, bert_ds


def model_run(num_labels,learning_rate,epochs,batch_size,loss,metric,model_name, bert_train_ds,bert_val_ds):

    
    # create BERT model
    bert_categorical_partial = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    bert_categorical_partial.compile(optimizer=optimizer, loss=loss, metrics= metric)

    with tf.device('/device:GPU:0'):
        history = bert_categorical_partial.fit(bert_train_ds.shuffle(100).batch(batch_size),
              epochs=epochs,
              validation_data=bert_val_ds.shuffle(100).batch(batch_size)
    #                                            , class_weight=class_weights
                                              )
        
    return bert_categorical_partial


def get_tokens(X_tokens):
    
    X_tokens_new = {'input_ids': np.asarray(X_tokens['input_ids']),
                     'token_type_ids': np.asarray(X_tokens['token_type_ids']),
                     'attention_mask': np.asarray(X_tokens['attention_mask']),
                     }
    
    return X_tokens_new

# def class_report(X_tokens,savedModel, y):
#     from sklearn.metrics import classification_report, confusion_matrix

#     X_tokens_new=get_tokens(X_tokens)

#     pred = savedModel.predict(X_tokens_new)
#     pred_proba = tf.math.softmax(pred['logits'], axis=-1).numpy()
#     pred_labels = np.argmax(pred['logits'], axis=1)

#     y_true = y.argmax(axis = 1)
    
#     print(classification_report(y_true,pred_labels))
    
    
def class_report(X_tokens,savedModel, y):
    from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, fbeta_score
    import numpy as np
    import tensorflow as tf

    X_tokens_new=get_tokens(X_tokens)

    pred = savedModel.predict(X_tokens_new)
    pred_proba = tf.math.softmax(pred['logits'], axis=-1).numpy()
    pred_labels = np.argmax(pred['logits'], axis=1)

    y_true = y.argmax(axis = 1)
    
    bal_acc=balanced_accuracy_score(y_true,pred_labels)
    f2_score=fbeta_score(y_true, pred_labels, average=None, beta=2)

    class_report_str = classification_report(y_true,pred_labels)
    bal_acc_str = "Balanced Accuracy:  " + str(bal_acc)
    f2_score_str = "F2 score:  " + str(f2_score)

    # Combine all strings into one and return
    return '\n'.join([class_report_str, bal_acc_str, f2_score_str])


def pipeline_run(input_path, aug_column_val, target, feature, num_labels,learning_rate,epochs,batch_size,loss,metric,model_name,path):
    
    train_df=aug_data_prep(input_path,aug_column_val)
#     print(train_df.columns)
#     train_df=pd.read_csv(input_path)
#     train_df=train_df.sample(frac=0.01).reset_index(drop=True)
    
    X = train_df[feature]
    Y = train_df[target]
    
    X_train, X_val, y_train, y_val = train_test_split(X,Y)
    
    print(X_train.shape)
    print(X_val.shape)
    print(y_train.shape)
    print(y_val.shape)
    
    X_train_tokens, bert_train_ds=data_prep(X_train,y_train,model_name)
    X_val_tokens, bert_val_ds=data_prep(X_val,y_val,model_name)
    
    
    bert_categorical_partial=model_run(num_labels,learning_rate,epochs,batch_size,loss,metric,model_name, bert_train_ds,bert_val_ds)
    
    print(bert_categorical_partial.summary())
    bert_categorical_partial.save(path)
    print("Training of bert model is done")
    
    #Train Classification Report
    print("Train classification report")
    class_report(X_tokens=X_train_tokens,savedModel=bert_categorical_partial, y=y_train)
    
    #Val Classification Report
    print("Val classification report")
    class_report(X_tokens=X_val_tokens,savedModel=bert_categorical_partial, y=y_val)
    
    
def data_prep_inference_pipeline(file_for_pred_input_path,feature,target,model_name):
    df=pd.read_csv(file_for_pred_input_path)
    train_df=df[df["train_test_tag"]=="train"]
    test_df=df[df["train_test_tag"]=="test"]
    
    ## Preperation of Training Data
    X=train_df[feature]
    Y=train_df[target]

    Y_train_one_hot_encoded=y_ohe(Y)

    X_train_tokens,bert_train_ds=data_prep(X,Y_train_one_hot_encoded,model_name)

    X_train_tokens_new=get_tokens(X_train_tokens)
    
    ## Preperation of Test Data
    X_test=test_df[feature]
    Y_test=test_df[target]

    Y_test_one_hot_encoded=y_ohe(Y_test)

    X_test_tokens,bert_test_ds=data_prep(X_test,Y_test_one_hot_encoded,model_name)

    X_test_tokens_new=get_tokens(X_test_tokens)
    
    return X_train_tokens_new, Y_train_one_hot_encoded, X_test_tokens_new, Y_test_one_hot_encoded