#%% md
# # Metrics, validation strategies and baselines
# 
# В данном jupyter notebook рассматриваются примеры того, какие схемы валидации и метрики используются в рекомендательных системах.
# Также построим простые модели (бейзлайны) на данных МТС Библиотеки. 
# 
# * [Preprocessing](#preprocessing)
# * [General remarks](#general-remarks)
# * [Metrics](#metrics)
#     * [Regression](#regression)
#     * [Classification](#classification)
#     * [Ranking](#ranking)
# * [Validation strategies](#validation)
# * [Baselines](#baselines)
#%%
#!pip install pandas
#%%
import os
import statistics
import numpy as np 
import pandas as pd 
from itertools import islice, cycle
from more_itertools import pairwise
from collections import defaultdict, Counter

print('Dataset:')
for dirname, _, filenames in os.walk('/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#%% md
# <a id="preprocessing"></a>
# # Preprocessing
# 
# Загрузим наши данные, теперь уже с фичами, и применим знания из [pandas-scipy-for-recsys](https://www.kaggle.com/sharthz23/pandas-scipy-for-recsys)
#%%
df = pd.read_csv('../mts-library/interactions.csv')
df_users = pd.read_csv('../mts-library/users.csv')
df_items = pd.read_csv('../mts-library/items.csv')
#%% md
# ## Interactions
#%%
df.info()
#%%
df['start_date'] = pd.to_datetime(df['start_date'])
#%%
duplicates = df.duplicated(subset=['user_id', 'item_id'], keep=False)
df_duplicates = df[duplicates].sort_values(by=['user_id', 'start_date'])
df = df[~duplicates]
#%%
df_duplicates = df_duplicates.groupby(['user_id', 'item_id']).agg({
    'progress': 'max',
    'rating': 'max',
    'start_date': 'min'
})
df = pd.concat([df,df_duplicates], ignore_index=True)
#%%
df_duplicates.shape
#%%
df['progress'] = df['progress'].astype(np.int8)
df['rating'] = df['rating'].astype(pd.SparseDtype(np.float32, np.nan))
#%%
df.info()
#%%
df.to_pickle('interactions_preprocessed.pickle')
#%%
!ls -lah
#%% md
# ## Users
#%%
df_users.head()
#%%
df_users.info()
#%%
df_users.nunique()
#%%
df_users['age'] = df_users['age'].astype('category')
df_users['sex'] = df_users['sex'].astype(pd.SparseDtype(np.float32, np.nan))
#%%
df_users.info()
#%%
interaction_users = df['user_id'].unique()

common_users = len(np.intersect1d(interaction_users, df_users['user_id']))
users_only_in_interaction = len(np.setdiff1d(interaction_users, df_users['user_id']))
users_only_features = len(np.setdiff1d(df_users['user_id'], interaction_users))
total_users = common_users + users_only_in_interaction + users_only_features
print(f'Кол-во пользователей - {total_users}')
print(f'Кол-во пользователей c взаимодействиями и фичами - {common_users} ({common_users / total_users * 100:.2f}%)')
print(f'Кол-во пользователей только c взаимодействиями - {users_only_in_interaction} ({users_only_in_interaction / total_users * 100:.2f}%)')
print(f'Кол-во пользователей только c фичами - {users_only_features} ({users_only_features / total_users * 100:.2f}%)')
#%%
df_users.to_pickle('users_preprocessed.pickle')
#%%
!ls -lah
#%% md
# ## Items
#%%
df_items.head()
#%%
df_items.info(memory_usage='full')
#%%
def num_bytes_format(num_bytes, float_prec=4):
    units = ['bytes', 'Kb', 'Mb', 'Gb', 'Tb', 'Pb', 'Eb']
    for unit in units[:-1]:
        #print(unit)
        if abs(num_bytes) < 1000:
            return f'{num_bytes:.{float_prec}f} {unit}'
        num_bytes /= 1000
    return f'{num_bytes:.4f} {units[-1]}'
#%%
#df_items.memory_usage(deep=True).sum()
#%%
num_bytes = df_items.memory_usage(deep=True).sum()
num_bytes_format(num_bytes)
#%%
df_items.nunique()
#%% md
# Почему колонка `year` типа `object`, а не `int`?
#%%
df_items['year'].value_counts().tail(25)
#%%
df_items[df_items['year'] == '1898, 1897, 1901']
#%%
for col in ['genres', 'authors', 'year']:
    df_items[col] = df_items[col].astype('category')
#%%
df_items.info(memory_usage='full')
#%%
df_items.groupby(level='genres')['rank'].transform(np.size)
#%%
num_bytes = df_items.memory_usage(deep=True).sum()
num_bytes_format(num_bytes)
#%%
interaction_items = df['item_id'].unique()

common_items = len(np.intersect1d(interaction_items, df_items['id']))
items_only_in_interaction = len(np.setdiff1d(interaction_items, df_items['id']))
items_only_features = len(np.setdiff1d(df_items['id'], interaction_items))
total_items = common_items + items_only_in_interaction + items_only_features
print(f'Кол-во книг - {total_items}')
print(f'Кол-во книг c взаимодействиями и фичами - {common_items} ({common_items / total_items * 100:.2f}%)')
print(f'Кол-во книг только c взаимодействиями - {items_only_in_interaction} ({items_only_in_interaction / total_items * 100:.2f}%)')
print(f'Кол-во книг только c фичами - {items_only_features} ({items_only_features / total_items * 100:.2f}%)')
#%%
df_items.to_pickle('items_preprocessed.pickle')
#%%
!ls -lah
#%% md
# <a id="general-marks"></a>
# # General marks
# 
# Основная цель валидации - оценить качество модели перед внедрением в "продакшен" или сабмитом в соревновании. Отсюда вытекает одно из самых важных требование к процессу валидации - *схема должна максимально точно воспроизводить условия, в которых модель будет использоваться.* Это касается как продакшена, так и соревнований, причем не только в рекомендательных системах, а в любой задаче машинного обучения. **Правильная валидация - ключ к успеху** :)
# 
# Ключевые вопросы, на которые надо ответить:
# * Что хотим от модели?
#     * Предсказанное значение для пары пользователь-объект (рейтинг или вероятность)
#     * Ранжирование объектов для пользователя (топ фильмов)
# * Как будет использоваться модель? 
#     * Рекомендации будут считаться батчами раз в период (1 раз в день)
#     * Онлайн-рекомендации
#     * Связь бизнес-метрик с оффлайн-метриками
# * Какие технические ограничения?
#     * По времени построения как системы в целом, так и отдельных рекомендаций (чем меньше "железа", тем проще должна быть модель)
#     * Какие данные будут доступны в онлайн-режиме
# * Как есть особенности у задачи?
#     * Рекомендуем ли для пользователя объекты, с которым он уже взаимодействовал (для фильмов или книг - скорее всего нет, для продуктового ритейла - может быть и да)
#     * Cold-start - как много появится пользователей или объектов, для которых не известная история по взаимодействиям
#     * Как много взаимодействий по пользователям есть в данных
#%% md
# <a id="metrics"></a>
# # Metrics
# 
# Как это водится в машинном обучении, для подсчета метрик нам нужно два массива: правильный ответы и предсказанные ответы.
# И если в классическом машинном обучении это действительно два массива, то в рекомендательных системах нам нужна структура, которая в качевстве ключа или индекса будет содержать пары пользователь-объект.
# 
# В качестве такой структуры вполне подойдет уже известный pandas.DataFrame. 
# Правильные метки (true) мы можем представить в виде dataframe со следующими столбцами:
# * user_id - ID пользователя
# * item_id - ID объекта
# * value - оценка взаимодействия (опционально, может и не быть, если мы, например, за взаимодействие берем просто клик или факт покупки)
# 
# Соответственно, рекомендации/предсказания (recs) в следующем:
# * user_id - ID пользователя
# * item_id - ID объекта
# * value - предсказанная оценка взаимодействия 
#     * Численная оценка, например рейтинг
#     * Позиция, если мы ранжируем контент
#     
# Для подсчета метрик мы можем объединить эти два датафрема по `(user_id, item_id)`. В качестве метода объединения можно использовать методы join/merge из pandas.DataFrame.
#%% md
# <a id="regression"></a>
# ## Regression
# 
# Эти метрики оценивают качество предсказанных численных оценок. Примеры метрик:
# * Mean Squared Error - среднеквадратичная ошибка
# * Root Mean Squared Error - корень из среднеквадратичной ошибки
# * Mean Absolute Error - средний ошибка по модулю
# ![image.png](attachment:image.png)
# Эти метрики показываются, как хорошо модель "восстанавливает" какие-то численные оценки, но не то, насколько хороши рекомендации
#%%
df_true = pd.DataFrame({
    'user_id': ['Аня',                'Боря',               'Вася',         'Вася'],
    'item_id': ['Мастер и Маргарита', '451° по Фаренгейту', 'Зеленая миля', 'Рита Хейуорт и спасение из Шоушенка'],
    'value':   [4,                    5,                    3,            5]
})
df_true
#%%
df_recs = pd.DataFrame({
    'user_id': ['Аня',                'Боря',               'Вася',         'Вася'],
    'item_id': ['Мастер и Маргарита', '451° по Фаренгейту', 'Зеленая миля', 'Рита Хейуорт и спасение из Шоушенка'],
    'value':   [3.28,                 3.5,                  4.06,           4.73]
})
df_recs
#%%
df_true = df_true.set_index(['user_id', 'item_id'])
df_recs = df_recs.set_index(['user_id', 'item_id'])

df_merged = df_true.join(df_recs, how='left', lsuffix='_true', rsuffix='_recs')
df_merged
#%%
df_merged['MAE'] = (df_merged['value_true'] - df_merged['value_recs']).abs()
df_merged['MSE'] = (df_merged['value_true'] - df_merged['value_recs']) ** 2
df_merged
#%%
print(f"MAE  - {df_merged['MAE'].mean():.4f}")
print(f"MSE  - {df_merged['MSE'].mean():.4f}")
print(f"RMSE - {np.sqrt(df_merged['MSE'].mean()):.4f}")
#%% md
# <a id="classification"></a>
# ## Classification
# 
# Эти метрики оценивают качество топ-N рекомендаций. В рекомендательные системы напрямую перекочевали из методов оценки качества бинарной классификации.
# Все считается на основе 4 базовых случаев:
# * True positive  (TP) - модель рекомендовала объект, с которым пользователь провзаимодействовал
# * False positive (FP) - модель рекомендовала объект, с которым пользователь не провзаимодействовал
# * True negative  (TN) - модель не рекомендовала объект, с которым пользователь не провзаимодействовал
# * False negative (FN) - модель не рекомендовала объект, с которым пользователь провзаимодействовал
# 
# Что из этого всего важней? В первую очередь это True positive. Мы хотим строить наиболее релевантные рекомендации для пользователя. 
# Во вторую очередь это False negative, опять же потому, что мы не хотим, чтобы модель "теряла" релевантные рекомендации.
# 
# А что с FP и TN? На самом деле, эти величины не показательны. Они обычно очень больше, так как пользователи взаимодействуют с очень малым количество объектов относительно общего числа объектов.
# И практика показывает, что этими значениями можно пренебречь.
# 
# Для измерения доли TP и FN применяются следующие метрики:
# * **Precision@K** - доля релевантных рекомендаций среди всех рекомендаций
#     * Формула - `TP / (TP + FP)`
#     * Можно заметить, что под positives мы понимаем рекомендованные объекты, то есть наш топ-К, значит `TP + FP = K`
#     * Итоговая формула - `TP / K`
#     * Считаем по каждому пользователю и для некторых К
#     * Усредняем по всем пользователя
# * **Recall@K** - доля релевантных рекомендаций среди всех релевантных объектов
#     * Формула - `TP / (TP + FN)`
#     * `TP + FN` это количество известных релевантых объектов для пользователя
#     * Считаем по каждому пользователю и для некторых К
#     * Усредняем по всем пользователя
# 
# Это основные метрики, но на основе TP, FP, TN, FN можно посчитать все что угодно :)
# ![image.png](attachment:image.png)
# Источник - [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
#%%
df_true = pd.DataFrame({
    'user_id': ['Аня',                'Боря',               'Вася',         'Вася'],
    'item_id': ['Мастер и Маргарита', '451° по Фаренгейту', 'Зеленая миля', 'Рита Хейуорт и спасение из Шоушенка'],
})
df_true
#%%
df_recs = pd.DataFrame({
    'user_id': [
        'Аня', 'Аня', 'Аня', 
        'Боря', 'Боря', 'Боря', 
        'Вася', 'Вася', 'Вася',
    ],
    'item_id': [
        'Отверженные', 'Двенадцать стульев', 'Герои нашего времени', 
        '451° по Фаренгейту', '1984', 'О дивный новый мир',
        'Десять негритят', 'Искра жизни', 'Зеленая миля', 
    ],
    'rank': [
        1, 2, 3,
        1, 2, 3,
        1, 2, 3,
    ]
})
df_recs
#%%
df_merged = df_true.set_index(['user_id', 'item_id']).join(df_recs.set_index(['user_id', 'item_id']), how='left')
df_merged
#%% md
# Вначале посчитаем метрик для топ-2 (т.е. К = 2). Алгоритм следующий:
# * Релевантные объекты, которые не были рекомендованы игнорируем (NaN)
# * Определяем, какие релеватные рекомендации попали в топ-2 (hit)
#     * True positive для каждого пользователя
# * Делим TP на K  
# * Считаем Precision@K для каждого пользователя как сумму его TP/K
# * Все Precision@K усредняем
#%%
df_merged['hit@2'] = df_merged['rank'] <= 2
df_merged
#%%
df_merged['hit@2/2'] = df_merged['hit@2'] / 2
df_merged
#%%
df_prec2 = df_merged.groupby(level=0)['hit@2/2'].sum()
df_prec2
#%%
print(f'Precision@2 - {df_prec2.mean()}')
#%% md
# Но вообще шаг с группировкой по пользователям для Precision@K не нужен
#%%
df_merged['hit@2/2'].sum() / df_merged.index.get_level_values('user_id').nunique()
#%%
users_count = df_merged.index.get_level_values('user_id').nunique()
for k in [1, 2, 3]:
    hit_k = f'hit@{k}'
    df_merged[hit_k] = df_merged['rank'] <= k
    print(f'Precision@{k} = {(df_merged[hit_k] / k).sum() / users_count:.4f}')
#%% md
# C Recall@K похожая история, нам также нужно получить hit@K, но делить уже будем на количество релевантных объектов у пользователя
#%%
df_merged['users_item_count'] = df_merged.groupby(level='user_id')['rank'].transform(np.size)
df_merged
#%%
for k in [1, 2, 3]:
    hit_k = f'hit@{k}'
    # Уже посчитано
    # df_merged[hit_k] = df_merged['rank'] <= k  
    print(f"Recall@{k} = {(df_merged[hit_k] / df_merged['users_item_count']).sum() / users_count:.4f}")
#%% md
# Precision@K и Recall@K неплохие метрики, чтобы оценить качество рекомендаций, но они учитывают только "попадания" (hits, true positives). 
# Но на самом деле нам важно насколько высоко по позициям находятся эти самые попадания. 
# 
# Простой пример, пусть две модели рекомендаций для одного пользователя получили такие hit@3 на тесте:
# * model1 - 1, 0, 0, 1
# * model2 - 1, 0, 1, 0
# 
# Precision@4 для них будет одинаковый - 0.5, хотя model2 немного лучше, так как 2-ое попадание находится выше, чем у model1
#%% md
# <a id="ranking"></a>
# ## Ranking
# 
# Эти метрики оценивают качество топ-N рекомендаций c учетом рангов/позиций. Основная идея - оценить "попадания" с весом, зависящим от позиции (обычно это обратная пропорциальная зависимость, то есть чем больше позиция, тем меньше вес).
# Основные метрики следующие:
#%% md
# **Mean Reciprocal Rank**
# ![image.png](attachment:image.png)
# Где Q - это query или наш пользователь, а rank_i - позиция первой релевантной рекомендации
#%% md
# ![image.png](attachment:image.png)
#%% md
# **Mean Average Precision**
# ![image.png](attachment:image.png)
# То есть MAP - это усреднение AveragePrecision по всем пользователям. 
# А AveragePrecision в свою очередь, это средний Precision@K по релевантным объектам одного пользователя
#%%
df_true = pd.DataFrame({
    'user_id': ['Аня',                'Боря',               'Вася',         'Вася'],
    'item_id': ['Мастер и Маргарита', '451° по Фаренгейту', 'Зеленая миля', 'Рита Хейуорт и спасение из Шоушенка'],
})
df_true 
#%%
df_recs = pd.DataFrame({
    'user_id': [
        'Аня', 'Аня', 'Аня', 
        'Боря', 'Боря', 'Боря', 
        'Вася', 'Вася', 'Вася',
    ],
    'item_id': [
        'Отверженные', 'Двенадцать стульев', 'Герои нашего времени', 
        '451° по Фаренгейту', '1984', 'О дивный новый мир',
        'Десять негритят', 'Рита Хейуорт и спасение из Шоушенка', 'Зеленая миля', 
    ],
    'rank': [
        1, 2, 3,
        1, 2, 3,
        1, 2, 3,
    ]
})
df_recs
#%%
df_merged = df_true.set_index(['user_id', 'item_id']).join(df_recs.set_index(['user_id', 'item_id']), how='left')
df_merged = df_merged.sort_values(by=['user_id', 'rank'])
df_merged
#%%
df_merged['reciprocal_rank'] = 1 / df_merged['rank']
df_merged
#%%
mrr = df_merged.groupby(level='user_id')['reciprocal_rank'].max()
mrr
#%%
print(f"MRR = {mrr.fillna(0).mean()}")
#%%
# df_merged['cumulative_rank1'] = df_merged.groupby(level='user_id').cumcount() + 1
# df_merged['cumulative_rank'] = df_merged['cumulative_rank1'] / df_merged['rank']
# df_merged
#%%
df_merged['cumulative_rank'] = df_merged.groupby(level='user_id').cumcount() + 1
df_merged['cumulative_rank'] = df_merged['cumulative_rank'] / df_merged['rank']
df_merged['users_item_count'] = df_merged.groupby(level='user_id')['rank'].transform(np.size)
df_merged
#%%
users_count = df_merged.index.get_level_values('user_id').nunique()
map3 = (df_merged["cumulative_rank"] / df_merged["users_item_count"]).sum() / users_count
print(f"MAP@3 = {map3}")
#%% md
# <a id="validation"></a>
# # Validation
# 
# Для получения train/test (или train/validation/test) глобально есть два подхода:
# * Случайное разбиение
#     * По всем взаимодействиям
#     * По пользователю или объекту
# * Разбиение по времени
# 
# ## Случайное разбиение
# 
# Обычно применяется схема Leave-one-out или Leave-P-out. Идея проста, давайте для одного (Leave-one-out) или нескольких (Leave-P-out) пользователей, для которых есть больше 2 взаимодействий, оставим одно взаимодействие в качестве теста.
# На практике такой метод редко применяется по двум причинам:
# * Дорогостоящая схема проверки, на боевых данных такое считаться будет долго
# * Часто в данных присутствует временная зависимость
# 
# Реализовать можно через pandas.DataFrame.sample
#%% md
# ## Разбиение по времени
# 
# Чаще всего встречается на практике. Обычно выбирается размер test по времени и период дат для разделения на train/test.
# 
# Например: test - 1 день, период для тестирования 7 дней.
# 
# Для наших данных по МТС Библиотеке выбрем 7 последних дней и будем тестировать на них последовательно.
#%%
test_dates = df['start_date'].unique()[-7:]
test_dates
#%% md
# Соберем из этих дат последовательные пары. Первая дата будет использоваться для ограничения на train, и обе даты будут использоваться для получения test
#%%
test_dates = list(pairwise(test_dates))
test_dates
#%%
split_dates = test_dates[0]
train = df[df['start_date'] < split_dates[0]]
test = df[(df['start_date'] >= split_dates[0]) & (df['start_date'] < split_dates[1])]
test = test[(test['rating'] >= 4) | (test['rating'].isnull())]
split_dates, train.shape, test.shape
#%% md
# <a id="baselines"></a>
# # Baselines
# 
# Самым популярным бейзлайном является просто построение популярного :)
# Гиперпараметром такой модели может быть например окно, за которое мы считаем популярное.
# 
# Модель можно расширять засчет учета фичей, чтобы считать популярное в рамках каких-то групп.
# 
# Но на самом деле бейзлайны в первую очередь зависят от типа данных. В некоторых случаях это могут быть простые модели (или даже бизнес-правила), которые просто учитывают контекст задачи.
#%%
class PopularRecommender():
    def __init__(self, max_K=100, days=30, item_column='item_id', dt_column='date'):
        self.max_K = max_K
        self.days = days
        self.item_column = item_column
        self.dt_column = dt_column
        self.recommendations = []
        
    def fit(self, df, ):
        min_date = df[self.dt_column].max().normalize() - pd.DateOffset(days=self.days)
        self.recommendations = df.loc[df[self.dt_column] > min_date, self.item_column].value_counts().head(self.max_K).index.values
        #print(self.recommendations)
    
    def recommend(self, users=None, N=10):
        recs = self.recommendations[:N]
        if users is None:
            return recs
        else:
            #print(list(islice(cycle([recs]), len(users))))
            return list(islice(cycle([recs]), len(users)))
#%%
pop_model = PopularRecommender(days=7, dt_column='start_date')
pop_model.fit(train)
#%%
top10_recs = pop_model.recommend()
top10_recs
#%%
item_titles = pd.Series(df_items['title'].values, index=df_items['id']).to_dict()
#item_titles
#%%
list(map(item_titles.get, top10_recs))
#%%
recs = pd.DataFrame({'user_id': test['user_id'].unique()})
top_N = 10
recs['item_id'] = pop_model.recommend(recs['user_id'], N=top_N)
recs.head()
#%%
#pop_model.recommend(recs['user_id'], N=top_N)
#%%
recs = recs.explode('item_id')
recs.head(top_N + 2)
#%%
recs['rank'] = recs.groupby('user_id').cumcount() + 1
recs.head(top_N + 2)
#%%
test_recs = test.set_index(['user_id', 'item_id']).join(recs.set_index(['user_id', 'item_id']))
test_recs = test_recs.sort_values(by=['user_id', 'rank'])
test_recs.tail()
#%%
test_recs['users_item_count'] = test_recs.groupby(level='user_id', sort=False)['rank'].transform(np.size)
test_recs['reciprocal_rank'] = 1 / test_recs['rank']
test_recs['reciprocal_rank'] = test_recs['reciprocal_rank'].fillna(0)
test_recs['cumulative_rank'] = test_recs.groupby(level='user_id').cumcount() + 1
test_recs['cumulative_rank'] = test_recs['cumulative_rank'] / test_recs['rank']
test_recs.tail()
#%%
test_recs[test_recs['rank'].notnull()].head()
#%%
print(f'Метрик по test ({str(split_dates[0])[:10]}, {str(split_dates[1])[:10]})')
users_count = test_recs.index.get_level_values('user_id').nunique()
for k in range(1, top_N + 1):
    hit_k = f'hit@{k}'
    test_recs[hit_k] = test_recs['rank'] <= k
    print(f'Precision@{k} = {(test_recs[hit_k] / k).sum() / users_count:.4f}')
    print(f"Recall@{k} = {(test_recs[hit_k] / test_recs['users_item_count']).sum() / users_count:.4f}")

mapN = (test_recs["cumulative_rank"] / test_recs["users_item_count"]).sum() / users_count
print(f"MAP@{top_N} = {mapN}")

mrr = test_recs.groupby(level='user_id')['reciprocal_rank'].max().mean()
print(f"MRR = {mrr}")
#%% md
# ### В качестве домашнего задания:
# * Попробуйте посчитать теперь метрик по всем фолдам в test_dates и оценить mean, std во времени. Стабильна ли модель во времени?
# * Постройте популярное по группам: возрастам пользователей или жанрам книг. Метрики стали лучше?
#%% md
# ### Calculate metrics by all folds
#%%
class PopularRecommender():
    def __init__(self, max_K=100, days=30, item_column='item_id', dt_column='date'):
        self.max_K = max_K
        self.days = days
        self.item_column = item_column
        self.dt_column = dt_column
        self.recommendations = []
        
    def fit(self, df, ):
        min_date = df[self.dt_column].max().normalize() - pd.DateOffset(days=self.days)
        self.recommendations = df.loc[df[self.dt_column] > min_date, self.item_column].value_counts().head(self.max_K).index.values
        #print(self.recommendations)
    
    def recommend(self, users=None, N=10):
        recs = self.recommendations[:N]
        if users is None:
            return recs
        else:
            #print(list(islice(cycle([recs]), len(users))))
            return list(islice(cycle([recs]), len(users)))
#%%
test_dates
#%%
def collect_metrics(i):
    split_dates = test_dates[i]
    train = df[df['start_date'] < split_dates[0]]
    test = df[(df['start_date'] >= split_dates[0]) & (df['start_date'] < split_dates[1])]
    test = test[(test['rating'] >= 4) | (test['rating'].isnull())]
    pop_model = PopularRecommender(days=7, dt_column='start_date')
    pop_model.fit(train)
    
    recs = pd.DataFrame({'user_id': test['user_id'].unique()})
    top_N = 10
    recs['item_id'] = pop_model.recommend(recs['user_id'], N=top_N)
    recs = recs.explode('item_id')
    recs['rank'] = recs.groupby('user_id').cumcount() + 1
    test_recs = test.set_index(['user_id', 'item_id']).join(recs.set_index(['user_id', 'item_id']))
    test_recs = test_recs.sort_values(by=['user_id', 'rank'])
    test_recs['users_item_count'] = test_recs.groupby(level='user_id', sort=False)['rank'].transform(np.size)
    test_recs['reciprocal_rank'] = 1 / test_recs['rank']
    test_recs['reciprocal_rank'] = test_recs['reciprocal_rank'].fillna(0)
    test_recs['cumulative_rank'] = test_recs.groupby(level='user_id').cumcount() + 1
    test_recs['cumulative_rank'] = test_recs['cumulative_rank'] / test_recs['rank']
    print(f'Метрик по test ({str(split_dates[0])[:10]}, {str(split_dates[1])[:10]})')
    users_count = test_recs.index.get_level_values('user_id').nunique()
    for k in range(1, top_N + 1):
        hit_k = f'hit@{k}'
        test_recs[hit_k] = test_recs['rank'] <= k
        precision = (test_recs[hit_k] / k).sum() / users_count
        recall = (test_recs[hit_k] / test_recs['users_item_count']).sum() / users_count
        precision_at_k[k].append(precision)
        recall_at_k[k].append(recall)
        # print(f'Precision@{k} = {precision:.4f}')
        # print(f"Recall@{k} = {recall:.4f}")
    
    mapN = (test_recs["cumulative_rank"] / test_recs["users_item_count"]).sum() / users_count
    #print(f"MAP@{top_N} = {mapN}")
    map_at_10.append(mapN)
    mrr = test_recs.groupby(level='user_id')['reciprocal_rank'].max().mean()
    #print(f"MRR = {mrr}")
    mrr_at_10.append(mrr)
#%%
precision_at_k = defaultdict(list)
recall_at_k = defaultdict(list)
map_at_10 = []
mrr_at_10 = []
for i in range(len(test_dates)): 
    collect_metrics(i)
    
for key, value in precision_at_k.items():
    print(f'precision at {key}, mean={statistics.mean(value):.6f}, stdev={statistics.stdev(value):.6f}' )
for key, value in recall_at_k.items():
    print(f'precision at {key}, mean={statistics.mean(value):.6f}, stdev={statistics.stdev(value):.6f}' )
print(f'map, mean={statistics.mean(map_at_10):.6f}, stdev={statistics.stdev(map_at_10):.6f}' )
print(f'mrr, mean={statistics.mean(mrr_at_10):.6f}, stdev={statistics.stdev(mrr_at_10):.6f}' )
#%%
#df_users
#%% md
# 
#%% md
# ### Calculate metrics by all folds by age
# 
#%%
df_age = pd.merge(df, df_users, on='user_id', how='left')

#%%
df.shape
#%%
df_age.shape
#%%
#df_age['age'].unique()
#%%
def collect_metrics_age(i, age, df):
    
    split_dates = test_dates[i]
    df = df[df['age'] == age]
    train = df[df['start_date'] < split_dates[0]]
    test = df[(df['start_date'] >= split_dates[0]) & (df['start_date'] < split_dates[1])]
    test = test[(test['rating'] >= 4) | (test['rating'].isnull())]
    pop_model = PopularRecommender(days=7, dt_column='start_date')
    pop_model.fit(train)
    
    recs = pd.DataFrame({'user_id': test['user_id'].unique()})
    top_N = 10
    recs['item_id'] = pop_model.recommend(recs['user_id'], N=top_N)
    recs = recs.explode('item_id')
    recs['rank'] = recs.groupby('user_id').cumcount() + 1
    test_recs = test.set_index(['user_id', 'item_id']).join(recs.set_index(['user_id', 'item_id']))
    test_recs = test_recs.sort_values(by=['user_id', 'rank'])
    test_recs['users_item_count'] = test_recs.groupby(level='user_id', sort=False)['rank'].transform(np.size)
    test_recs['reciprocal_rank'] = 1 / test_recs['rank']
    test_recs['reciprocal_rank'] = test_recs['reciprocal_rank'].fillna(0)
    test_recs['cumulative_rank'] = test_recs.groupby(level='user_id').cumcount() + 1
    test_recs['cumulative_rank'] = test_recs['cumulative_rank'] / test_recs['rank']
    #print(f'Метрик по test ({str(split_dates[0])[:10]}, {str(split_dates[1])[:10]})')
    users_count = test_recs.index.get_level_values('user_id').nunique()
    for k in range(1, top_N + 1):
        hit_k = f'hit@{k}'
        test_recs[hit_k] = test_recs['rank'] <= k
        precision = (test_recs[hit_k] / k).sum() / users_count
        recall = (test_recs[hit_k] / test_recs['users_item_count']).sum() / users_count
        precision_at_k[k].append(precision)
        recall_at_k[k].append(recall)
        # print(f'Precision@{k} = {precision:.4f}')
        # print(f"Recall@{k} = {recall:.4f}")
    
    mapN = (test_recs["cumulative_rank"] / test_recs["users_item_count"]).sum() / users_count
    #print(f"MAP@{top_N} = {mapN}")
    map_at_10.append(mapN)
    mrr = test_recs.groupby(level='user_id')['reciprocal_rank'].max().mean()
    #print(f"MRR = {mrr}")
    mrr_at_10.append(mrr)
#%%
precision_at_k = defaultdict(list)
recall_at_k = defaultdict(list)
map_at_10 = []
mrr_at_10 = []

for age in df_age[df_age['age'].notna()]['age'].unique():
    precision_at_k = defaultdict(list)
    recall_at_k = defaultdict(list)
    map_at_10 = []
    mrr_at_10 = []
    print(age)
    for i in range(len(test_dates)):

        #print(i, age)
        collect_metrics_age(i, age, df_age)
    
    # for key, value in precision_at_k.items():
    #     print(f'precision at {key}, mean={statistics.mean(value):.6f}, stdev={statistics.stdev(value):.6f}' )
    # for key, value in recall_at_k.items():
    #     print(f'precision at {key}, mean={statistics.mean(value):.6f}, stdev={statistics.stdev(value):.6f}' )
    print(f'map, mean={statistics.mean(map_at_10):.6f}, stdev={statistics.stdev(map_at_10):.6f}' )
    print(f'mrr, mean={statistics.mean(mrr_at_10):.6f}, stdev={statistics.stdev(mrr_at_10):.6f}' )
#%%
df_age[df_age['age'].notna()]['age'].unique()
#%%
df_age['age'].unique()
#%% md
# ### Calculate metrics by all folds by items_genres
# 
#%%
df_items['item_id'] = df_items['id']
#%%
df_items
#%%
df_genres = pd.merge(df, df_items, on='item_id', how='left')

#%%
for key, _ in Counter(df_genres['genres']).most_common(10):
    print(key)
#%%
def collect_metrics_age(i, genres, df):
    
    split_dates = test_dates[i]
    df = df[df['genres'] == genres]
    train = df[df['start_date'] < split_dates[0]]
    test = df[(df['start_date'] >= split_dates[0]) & (df['start_date'] < split_dates[1])]
    test = test[(test['rating'] >= 4) | (test['rating'].isnull())]
    pop_model = PopularRecommender(days=7, dt_column='start_date')
    pop_model.fit(train)
    
    recs = pd.DataFrame({'user_id': test['user_id'].unique()})
    top_N = 10
    recs['item_id'] = pop_model.recommend(recs['user_id'], N=top_N)
    recs = recs.explode('item_id')
    recs['rank'] = recs.groupby('user_id').cumcount() + 1
    test_recs = test.set_index(['user_id', 'item_id']).join(recs.set_index(['user_id', 'item_id']))
    test_recs = test_recs.sort_values(by=['user_id', 'rank'])
    test_recs['users_item_count'] = test_recs.groupby(level='user_id', sort=False)['rank'].transform(np.size)
    test_recs['reciprocal_rank'] = 1 / test_recs['rank']
    test_recs['reciprocal_rank'] = test_recs['reciprocal_rank'].fillna(0)
    test_recs['cumulative_rank'] = test_recs.groupby(level='user_id').cumcount() + 1
    test_recs['cumulative_rank'] = test_recs['cumulative_rank'] / test_recs['rank']
    #print(f'Метрик по test ({str(split_dates[0])[:10]}, {str(split_dates[1])[:10]})')
    users_count = test_recs.index.get_level_values('user_id').nunique()
    for k in range(1, top_N + 1):
        hit_k = f'hit@{k}'
        test_recs[hit_k] = test_recs['rank'] <= k
        precision = (test_recs[hit_k] / k).sum() / users_count
        recall = (test_recs[hit_k] / test_recs['users_item_count']).sum() / users_count
        precision_at_k[k].append(precision)
        recall_at_k[k].append(recall)
        # print(f'Precision@{k} = {precision:.4f}')
        # print(f"Recall@{k} = {recall:.4f}")
    
    mapN = (test_recs["cumulative_rank"] / test_recs["users_item_count"]).sum() / users_count
    #print(f"MAP@{top_N} = {mapN}")
    map_at_10.append(mapN)
    mrr = test_recs.groupby(level='user_id')['reciprocal_rank'].max().mean()
    #print(f"MRR = {mrr}")
    mrr_at_10.append(mrr)
#%%
precision_at_k = defaultdict(list)
recall_at_k = defaultdict(list)
map_at_10 = []
mrr_at_10 = []

for key, _ in Counter(df_genres['genres']).most_common(10):
    precision_at_k = defaultdict(list)
    recall_at_k = defaultdict(list)
    map_at_10 = []
    mrr_at_10 = []
    print(key)
    for i in range(len(test_dates)):

        #print(i, age)
        collect_metrics_age(i, key, df_genres)
    
    # for key, value in precision_at_k.items():
    #     print(f'precision at {key}, mean={statistics.mean(value):.6f}, stdev={statistics.stdev(value):.6f}' )
    # for key, value in recall_at_k.items():
    #     print(f'precision at {key}, mean={statistics.mean(value):.6f}, stdev={statistics.stdev(value):.6f}' )
    print(f'map, mean={statistics.mean(map_at_10):.6f}, stdev={statistics.stdev(map_at_10):.6f}' )
    print(f'mrr, mean={statistics.mean(mrr_at_10):.6f}, stdev={statistics.stdev(mrr_at_10):.6f}' )