#%% md
# # Pandas & SciPy for RecSys datasets
# 
# В данном jupyter notebook рассматриваются примеры использования библиотек pandas и scipy при работе с данным для построения рекомендательных систем.
# 
# * [Preprocessing](#preprocessing)
# * [Pandas](#pandas)
#     - [СategoryDType](#categorydtype)
#     - [IntegerDType](#integerdtype)
#     - [Sparse Type](#sparse-type)
# * [SciPy.Sparse](#scipy)
#     - [Matrix types](#matrix-types)
#     - [Pandas to matrix](#pandas-to-matrix)
# * [Links](#links)
#%%
import numpy as np 
import pandas as pd 
import scipy.sparse as sp

import os
for dirname, _, filenames in os.walk(''):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#%%
#!pip install scipy

#%% md
# <a id="preprocessing"></a>
# # Preprocessing
# 
# Загрузим взаимодействия
#%%
df = pd.read_csv('../mts-library/interactions.csv')
df.head()
#%%
df.shape
#%%
df['start_date'] = pd.to_datetime(df['start_date'])
#%%
df.info()
#%% md
# Проверим данные на дубликаты
#%%
duplicates = df.duplicated(subset=['user_id', 'item_id'], keep=False)
duplicates.sum()
#%%
df_duplicates = df[duplicates].sort_values(by=['user_id', 'start_date'])
df = df[~duplicates]
#%%
df_duplicates.shape
#%%
df.shape
#%%
df_duplicates = df_duplicates.groupby(['user_id', 'item_id']).agg({
    'progress': 'max',
    'rating': 'max',
    'start_date': 'min'
})
df_duplicates.info()
#%%
#df = df.append(ddf_duplicates, ignore_index=True)
df.info()
#%%
df['item_id'].nunique()
#%%
df.nunique()
#%% md
# Как видно, у нас 1.5 миллиона строк, но уникальных значений гораздо меньше. 
# Это свойство называется **низкой кардинальностью** и встречается во многих датасетах с "взаимодействиями". 
#%% md
# <a id="pandas"></a>
# # Pandas
# 
# 
# <a id="categorydtype"></a>
# ## CategoryDType
# 
# [CategoryDType](https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html) - тип в pandas, который позволяет оптимизировать потребление памяти для строковых столбцов и задать логический порядок для значений в них.
# 
#%%
df_cat = pd.DataFrame({'city': ['Moscow', 'London', 'Tokyo', 'Moscow']})
df_cat
#%%
df_cat['city'] = df_cat['city'].astype('category')
df_cat
#%%
df_cat['city_codes'] = df_cat['city'].cat.codes
df_cat
#%%
mapping = dict(enumerate(df_cat['city'].cat.categories))
mapping
#%% md
# Рассмотрим то, как такая конвертация типов может экономить память.
#%%
df_user_item = df[['user_id', 'item_id']].copy()
#%%
def num_bytes_format(num_bytes, float_prec=4):
    units = ['bytes', 'Kb', 'Mb', 'Gb', 'Tb', 'Pb', 'Eb']
    for unit in units[:-1]:
        if abs(num_bytes) < 1000:
            return f'{num_bytes:.{float_prec}f} {unit}'
        num_bytes /= 1000
    return f'{num_bytes:.4f} {units[-1]}'
#%%
num_bytes_ints = df_user_item.memory_usage(deep=True).sum()
num_bytes_format(num_bytes_ints)
#%%
df_user_item = df_user_item.astype('string')
num_bytes_str = df_user_item.memory_usage(deep=True).sum()
num_bytes_format(num_bytes_str)
#%%
df_user_item = df_user_item.astype('category')
num_bytes_cat = df_user_item.memory_usage(deep=True).sum()
num_bytes_format(num_bytes_cat)
#%%
print(f'Экономия category относительно string: {(1 - num_bytes_cat / num_bytes_str) * 100:.2f}%')
print(f'Экономия ints относительно category: {(1 - num_bytes_ints / num_bytes_cat) * 100:.2f}%')
#%%
df_user_item = df_user_item.astype(np.int64).astype('category')
num_bytes_int_cat = df_user_item.memory_usage(deep=True).sum()
num_bytes_format(num_bytes_int_cat)
#%%
print(f'Экономия category on int64 относительно category on string: {(1 - num_bytes_int_cat / num_bytes_cat) * 100:.2f}%')
#%%
df_user_item['user_id'].cat.codes.dtype
#%% md
# <a id="integerdtype"></a>
# ## IntegerDType
# 
# [IntegerDType](https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html) - группа типов в pandas, который позволяет целочисленным столбцам содержать пропущенные значения. 
# 
# Для классического целочисленного типа есть свой "двойник". Отличаются только первыми заглавными буквами, например:
# * numpy.int32 - pd.Int32DType
# * numpy.uint32 - pd.UInt64DType
# 
# 
# Этот тип стоит использовать только для экономии памяти при хранении датафрейма или при простых операциях на ним. Большинство библиотек этот тип в данный момент не поддерживают.
#%%
ratings = df['rating'].astype(np.float32).copy()
#%%
num_bytes_float = ratings.memory_usage(deep=True)
num_bytes_format(num_bytes_float)
#%%
ratings = ratings.astype(pd.Int32Dtype())
num_bytes_Int32 = ratings.memory_usage(deep=True)
num_bytes_format(num_bytes_Int32)
#%%
ratings = ratings.astype(pd.Int8Dtype())
num_bytes_Int8 = ratings.memory_usage(deep=True)
num_bytes_format(num_bytes_Int8)
#%%
ratings
#%%
print(f'Экономия Int8DType относительно float64: {(1 - num_bytes_Int8 / num_bytes_float) * 100:.2f}%')
#%% md
# <a id="sparse-type"></a>
# ## Sparse Type
# 
# [Sparse Type](https://pandas.pydata.org/pandas-docs/stable/user_guide/sparse.html) - тип данных в pandas для работы с разреженными данными.
# 
# Идея проста - храним только "известные" значения, остальное не храним и имитируем константой.
# 
# Сам тип создается на основе двух значений:
# * dtype - тип сохраняемых значений
# * fill_value - константа для пропущенных значений
#%%
sparse_type = pd.SparseDtype(np.float32, np.nan)
ratings = ratings.astype(np.float32).astype(sparse_type)
#%%
ratings
#%%
num_bytes_sparse = ratings.memory_usage(deep=True)
num_bytes_format(num_bytes_sparse)
#%%
print(f'Экономия sparse относительно Int8DType: {(1 - num_bytes_sparse / num_bytes_Int8) * 100:.2f}%')
print(f'Экономия sparse относительно float32: {(1 - num_bytes_sparse / num_bytes_float) * 100:.2f}%')
#%%
ratings.sparse.density
#%% md
# <a id="scipy"></a>
# # SciPy.Sparse
# 
# 
# <a id="matrix-types"></a>
# ## Matrix types
# 
# [Sparse matrix](https://docs.scipy.org/doc/scipy/reference/sparse.html) - разреженная матрица, которая хранит только известные значения.
# 
# Виды разреженных матриц в scipy.sparse:
# * **coo_matrix** - A sparse matrix in COOrdinate format
# * **csc_matrix** - Compressed Sparse Column matrix
# * **csr_matrix** - Compressed Sparse Row matrix
# * **bsr_matrix** - Block Sparse Row matrix
# * **dia_matrix** - Sparse matrix with DIAgonal storage
# * **dok_matrix** - Dictionary Of Keys based sparse matrix
# * **lil_matrix** - Row-based list of lists sparse matrix
# 
# Их можно разделить на две группы:
# * Классы для создания матриц
#   * coo_matrix - тройки (строка, столбец, значение)
#   * dok_matrix - словарь, где ключ - кортеж из строки и столбца, а значение - это значение :)
#   * lil_matrix - список из списков, где внутренние списки - это строки
# * Классы оптимизированные под оптимальное хранение и операции над матрицами
#   * csr_matrix - сжатое построчное представление
#   * csc_matrix - сжатое представление по столбцам
#   * bsr_matrix - как csr_matrix, только хранятся "плотные блоки"
#   * dia_matrix - диагональное представление
#   
#  
# На практике чаще всего встречаются coo_matrix, csr_matrix и csc_matrix 
#%% md
# ### coo_matrix
# 
# 
#%%
rows =   [1,  1, 0,  4,   2, 2]
cols =   [0,  1, 0,  5,   3, 3]
values = [-2, 7, 19, 1.0, 6, 8]

coo = sp.coo_matrix((values, (rows, cols)))
coo
#%%
coo.todense()
#%%
coo.row, coo.col, coo.data
#%% md
# ### csr_matrix/csc_matrix
#%%
csr = coo.tocsr()
csr
#%%
csr.todense()
#%%
csr.indptr, csr.indices, csr.data
#%% md
# * indptr - указатели, которые рассматриваются парами. Имеют два значения:
#     * Индекс пары - номер строки 
#     * Значения пары - начало и конец строки в data и indices
# * indices - номер столбца
# * data - значение
# 
# Как это работает:
# * indptr -> (1, 3) -> 1-ая строка.
# * 3 - 1 = 2 -> кол-во заполненных значений
# * indices[1:3] = (0, 1) -> столбцы 2 значений
# * data[1:3] = (-2, 7) -> сами значения
# * -2 -> (1, 0)
# * 7  -> (1, 1)
# 
# 
#%%
csc = coo.tocsc()
csc
#%%
csc.todense()
#%%
csc.indptr, csc.indices, csc.data
#%% md
# <a id="pandas-to-matrix"></a>
# ## Pandas to matrix
# 
# Для создания разреженной матрицы из dataframe с взаимодействиям нужно вначале определить соответствия между user/item ID и номерами строк/столбцов.
# 
# По сути, мы просто должны пронумеровать (начиная с 0) все уникальные ID.
#%%
df.head()
#%%
df.info()
#%%
df.nunique()
#%% md
# * users_mapping - конвертация ID в номер строки
# * users_inv_mapping - номер строки в ID
#%%
users_inv_mapping = dict(enumerate(df['user_id'].unique()))
users_mapping = {v: k for k, v in users_inv_mapping.items()}
len(users_mapping)
#%%
users_mapping[126706], users_inv_mapping[0]
#%%
items_inv_mapping = dict(enumerate(df['item_id'].unique()))
items_mapping = {v: k for k, v in items_inv_mapping.items()}
len(items_mapping)
#%%
items_mapping[14433], items_inv_mapping[0]
#%% md
# Имея данные отображения в виде словарей, мы теперь можем легко конвертировать наши столбцы user_id и item_id в массивы строк и столбцов и закинуть их в coo_matrix
#%%
rows = df['user_id'].map(users_mapping.get)
cols = df['item_id'].map(items_mapping.get)

rows.isna().sum(), cols.isna().sum()
#%% md
# Заполняем единицей (аля implicit feedback)
#%%
coo = sp.coo_matrix((
    np.ones(df.shape[0], dtype=np.int8),
    (rows, cols)
))
coo
#%%
num_bytes_format(coo.data.nbytes + coo.row.nbytes + coo.col.nbytes)
#%% md
# Или значением из dataframe. Придумаем страшную формулу для взвешивания каждого взаимодействия на основе оценки и проценту прочитанного.
#%%
df['weight'] = ((df['progress'] + 1) / 101) * (2 ** df['rating'])
df['weight'] = df['weight'].astype(np.float32)
#%%
ax = df['weight'].plot.hist()
#%%
coo = sp.coo_matrix((
    df['weight'],
    (rows, cols)
))
coo
#%%
num_bytes_format(coo.data.nbytes + coo.row.nbytes + coo.col.nbytes)
#%% md
# <a id="links"></a>
# # Links
# * https://medium.com/@aakashgoel12/pandas-optimize-memory-and-speed-operation-17d8a66c8be4 - отличный гайд по эффективному использованию Pandas
# * https://matteding.github.io/2019/04/25/sparse-matrices/ - шикарные визуализации разреженных матриц
# * https://rushter.com/blog/scipy-sparse-matrices/ - еще один хороший разбор разреженных матриц