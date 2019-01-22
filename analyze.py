
# coding: utf-8

# 取り扱い説明
# このファイルは、学習用データと予測用データを決められたフォーマットで指定し読ませることで、
# 「いい感じ」にデータ予測させるプログラムである。
# 第一引数：学習用ファイルパス
# 第二引数：学習用ファイルの種類(tsv or csv)
# 第三引数：予測用ファイルパス
# 第四引数：予測用ファイルの種類(tsv or csv)
# 実行例：python Analyze.py data/train.tsv tsv data/test.tsv tsv

# 諸注意
# trainファイルとtestファイルのカラム差分は必ず1つだけにすること。
# trainファイルとtestファイルのIDカラム名は「id」で統一すること。
# ファイルの種類はtsvとcsvのみ対応。(今のところ)
# アルゴリズムはランダムフォレストを使用。
# 欠損値補完は全て0埋め。


# In[1]:

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from sklearn.model_selection import train_test_split
pd.set_option('display.float_format', lambda x:'%.5f' % x)
import numpy as np

# arg設定
import sys
if len(sys.argv) != 5:
    print("arg error")
    sys.exit(1)
    
def delimiter_decision(delimiter):
	if delimiter in ['tsv', 't']:
		return '\t'
	if delimiter in ['csv', 'c']:
		return ','

train_file = sys.argv[1]
train_delimiter = delimiter_decision(sys.argv[2])
test_file = sys.argv[3]
test_delimiter = delimiter_decision(sys.argv[4])


# In[2]:

# csvファイルからPandas DataFrameへ読み込み
train = pd.read_csv(train_file, delimiter=train_delimiter, low_memory=True)
test = pd.read_csv(test_file, delimiter=test_delimiter, low_memory=True)


# In[3]:

# trainとtestのカラム差を出す
train_set = set(train.columns)
test_set = set(test.columns)
dis = train_set - test_set
if len(dis) != 1:
    #error
    print('column error')
    sys.exit(1)

list_dis = list(dis)
eval_column = list_dis[0]
print(eval_column)


# In[4]:

# 両方のセットへ「is_train」のカラムを追加
# 1 = trainのデータ、0 = testデータ
train['is_train'] = 1
test['is_train'] = 0
 
# trainの評価したいカラム以外のデータをtestと連結
train_test_combine = pd.concat([train.drop([eval_column], axis=1),test],axis=0)
 
# 念のためデータの中身を表示させましょう
train_test_combine.head()


# In[5]:

train_test_combine = train_test_combine.fillna(0)


# In[6]:

for key in train_test_combine.select_dtypes(include=['object']):
    train_test_combine[key] = train_test_combine[key].astype("category")
    train_test_combine[key] = train_test_combine[key].cat.codes


# In[7]:

# 「is_train」のフラグでcombineからtestとtrainへ切り分ける
df_test = train_test_combine.loc[train_test_combine['is_train'] == 0]
df_train = train_test_combine.loc[train_test_combine['is_train'] == 1]
 
# 「is_train」をtrainとtestのデータフレームから落とす
df_test = df_test.drop(['is_train'], axis=1)
df_train = df_train.drop(['is_train'], axis=1)


# In[9]:

# df_trainへ評価したいカラムを戻す
df_train[eval_column] = train[eval_column]
 
# 評価したいカラムをlog関数で処理
df_train[eval_column] = df_train[eval_column].apply(lambda x: np.log(x) if x>0 else x)
 
# df_trainを表示して確認
df_train.head()


# In[11]:

# x ＝ price以外の全ての値、y = price（ターゲット）で切り分ける
x_train, y_train = df_train.drop([eval_column], axis=1), df_train[eval_column]
 
# モデルの作成
m = RandomForestRegressor(n_jobs=-1, min_samples_leaf=5, n_estimators=200)
m.fit(x_train, y_train)
 
# スコアを表示
m.score(x_train, y_train)


# In[13]:

# 作成したランダムフォレストのモデル「m」に「df_test」を入れて予測する
preds = m.predict(df_test)
 
# 予測値 predsをnp.exp()で処理
np.exp(preds)
 
# Numpy配列からpandasシリーズへ変換
preds = pd.Series(np.exp(preds))
 
# テストデータのIDと予測値を連結
submit = pd.concat([df_test['id'], preds], axis=1)
 
# カラム名を提出指定の名前をつける
submit.columns = ['id', eval_column]
 
# 提出ファイルとしてCSVへ書き出し
submit.to_csv('submit_rf_base.csv', index=False)


# In[ ]:



