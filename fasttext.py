import numpy as np
import pandas as pd 
import jieba
#warnings.simplefilter('ignore')
#import warnings
#from simpletransformers.classification import MultiLabelClassificationModel

data_train=pd.read_csv('.../train.csv')
data_test=pd.read_csv('.../nlp_test.csv')
sub=pd.read_csv('.../sample_submission.csv')

#合并数据
data_train['type']='train'
data_test['type']='test'
data_all=pd.concat([data_train,data_test],axis=0,ignore_index=True,sort=False)

data_all.shape
data_all.columns
data_all['Question Sentence'].apply(lambda x:len(x))


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


data_all['Question']=data_all['Question Sentence'].apply(lambda x: " ".join(jieba.cut(x)))

import fasttext
from sklearn.metrics import f1_score
# A

data_df=data_all[['Question','category_A']].head(5000)
data_df['label_ft'] = '__label__' + data_train['category_A'].head(5000).astype(str)
data_df[['Question','label_ft']].to_csv('train.csv', index=None, header=None, sep='\t')


model = fasttext.train_supervised('train.csv', lr=0.05, wordNgrams=2, 
                                  verbose=2, minCount=1, epoch=500, loss="hs")

val_pred_A = [model.predict(x)[0][0].split('__')[-1] for x in data_all['Question'][5000:]]


sub['category_A']=val_pred_A

print(sub['category_A'].value_counts()/3000)
print(data_train['category_A'].value_counts()/5000)

#B
data_df=data_all[['Question','category_B']].head(5000)
data_df['label_ft'] = '__label__' + data_train['category_B'].head(5000).astype(str)
data_df[['Question','label_ft']].to_csv('train.csv', index=None, header=None, sep='\t')

#loss function {ns, hs, softmax, ova}
model = fasttext.train_supervised('train.csv', lr=0.05, wordNgrams=2, 
                                  verbose=2, minCount=1, epoch=500, loss="hs")


val_pred_B = [model.predict(x)[0][0].split('__')[-1] for x in data_all['Question'][5000:]]


sub['category_B']=val_pred_B

print(sub['category_B'].value_counts()/3000)
print(data_train['category_B'].value_counts()/5000)

#C

data_df=data_all[['Question','category_C']].head(5000)
data_df['label_ft'] = '__label__' + data_train['category_C'].head(5000).astype(str)
data_df[['Question','label_ft']].to_csv('train.csv', index=None, header=None, sep='\t')


model = fasttext.train_supervised('train.csv', lr=0.05, wordNgrams=2, 
                                  verbose=2, minCount=1, epoch=500, loss="hs")

val_pred_C = [model.predict(x)[0][0].split('__')[-1] for x in data_all['Question'][5000:]]


sub['category_C']=val_pred_C

print(sub['category_C'].value_counts()/3000)
print(data_train['category_C'].value_counts()/5000)

#D

data_df=data_all[['Question','category_D']].head(5000)
data_df['label_ft'] = '__label__' + data_train['category_D'].head(5000).astype(str)
data_df[['Question','label_ft']].to_csv('train.csv', index=None, header=None, sep='\t')


model = fasttext.train_supervised('train.csv', lr=0.05, wordNgrams=2, 
                                  verbose=2, minCount=1, epoch=500, loss="hs")

val_pred_D = [model.predict(x)[0][0].split('__')[-1] for x in data_all['Question'][5000:]]

sub['category_D']=val_pred_D

print(sub['category_D'].value_counts()/3000)
print(data_train['category_D'].value_counts()/5000)


#E

data_df=data_all[['Question','category_E']].head(5000)
data_df['label_ft'] = '__label__' + data_train['category_E'].head(5000).astype(str)
data_df[['Question','label_ft']].to_csv('train.csv', index=None, header=None, sep='\t')
model = fasttext.train_supervised('train.csv', lr=0.05, wordNgrams=2, 
                                verbose=2, minCount=1, epoch=500, loss="hs")

val_pred_E = [model.predict(x)[0][0].split('__')[-1] for x in data_all['Question'][5000:]]
sub['category_E']=val_pred_E
print(sub['category_E'].value_counts()/3000)
print(data_train['category_E'].value_counts()/5000)

#F

data_df=data_all[['Question','category_F']].head(5000)
data_df['label_ft'] = '__label__' + data_train['category_F'].head(5000).astype(str)
data_df[['Question','label_ft']].to_csv('train.csv', index=None, header=None, sep='\t')

#loss function {ns, hs, softmax, ova}
model = fasttext.train_supervised('train.csv', lr=0.05, wordNgrams=2, 
                                  verbose=2, minCount=1, epoch=500, loss="hs")
val_pred_F = [model.predict(x)[0][0].split('__')[-1] for x in data_all['Question'][5000:]]


sub['category_F']=val_pred_F

print(sub['category_F'].value_counts()/3000)
print(data_train['category_F'].value_counts()/5000)

sub.to_csv('HOSIP_FASTTEST_predictions.csv', index=False)
