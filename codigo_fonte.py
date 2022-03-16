import numpy as np
import pandas as pd
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import joblib

caminho_bases = pathlib.Path('dataset')
base_airbnb = pd.DataFrame()
meses = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6,
         'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}

for arquivo in caminho_bases.iterdir():
    nome_mes = arquivo.name[:3]
    mes = meses[nome_mes]

    ano = arquivo.name[-8:]
    ano = int(ano.replace('.csv', ''))

    df = pd.read_csv(caminho_bases / arquivo.name)
    df['ano'] = ano
    df['mes'] = mes
    base_airbnb = base_airbnb.append(df)

# Fazendo uma análise para excluir algumas colunas desnecessárias

# gerando um arquivo csv para visualizar colunas no excel
base_airbnb.head(1000).to_csv('primeiros_registros.csv', sep=';')

# trazendo os nomes das colunas restantes após o tratamento feito no arquivo filtrando_colunas.py
colunas = ['host_response_time', 'host_response_rate', 'host_is_superhost', 'host_listings_count', 'latitude',
           'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type',
           'amenities', 'price', 'security_deposit', 'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights',
           'maximum_nights', 'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
           'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'instant_bookable',
           'is_business_travel_ready', 'cancellation_policy', 'ano', 'mes']

# limpando colunas com muitos NaN
base_airbnb = base_airbnb.loc[:, colunas]
for coluna in base_airbnb:
    if base_airbnb[coluna].isnull().sum() > 300000:
        base_airbnb = base_airbnb.drop(coluna, axis=1)

# limpando linhas com NaN
base_airbnb = base_airbnb.dropna()

# verificando tipos
# price e extra_people estão como object e precisam ser valores numéricos
base_airbnb['price'] = base_airbnb['price'].str.replace('$', '')    # removendo sifrão
base_airbnb['price'] = base_airbnb['price'].str.replace(',', '')    # removendo vírgula do separador de milhar
base_airbnb['price'] = base_airbnb['price'].astype(np.float32, copy=False)  # transformando em float

base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace('$', '')    # removendo sifrão
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace(',', '')    # removendo vírgula do separador de milhar
base_airbnb['extra_people'] = base_airbnb['extra_people'].astype(np.float32, copy=False)  # transformando em float

'''
Ver a correlação entre as features e decidir se manteremos todas as features que temos.
Excluir outliers (usaremos como regra, valores abaixo de Q1 - 1.5xAmplitude e valores acima de Q3 + 1.5x Amplitude). Amplitude = Q3 - Q1
Confirmar se todas as features que temos fazem realmente sentido para o nosso modelo ou se alguma delas não vai nos ajudar e se devemos excluir
Vamos começar pelas colunas de preço (resultado final que queremos) e de extra_people (também valor monetário). Esses são os valores numéricos contínuos.

Depois vamos analisar as colunas de valores numéricos discretos (accomodates, bedrooms, guests_included, etc.)

Por fim, vamos avaliar as colunas de texto e definir quais categorias fazem sentido mantermos ou não.
'''

plt.figure(figsize=(15, 10))
sns.heatmap(base_airbnb.corr(), annot=True)


def limites(colun):
    q1 = colun.quantile(0.25)
    q3 = colun.quantile(0.75)
    amplitude = q3 - q1
    return q1 - 1.5 * amplitude, q3 + 1.5 * amplitude


def excluir_outliers(df, nome_coluna):
    qtde_linhas = df.shape[0]
    lim_inf, lim_sup = limites(df[nome_coluna])
    df = df.loc[(df[nome_coluna] >= lim_inf) & (df[nome_coluna] <= lim_sup), :]
    linhas_removidas = qtde_linhas - df.shape[0]
    return df, linhas_removidas


def diagrama_caixa(colun):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)
    sns.boxplot(x=colun, ax=ax1)
    ax2.set_xlim(limites(colun))
    sns.boxplot(x=colun, ax=ax2)
    plt.show()


def histograma(colun):
    plt.figure(figsize=(15, 5))
    sns.distplot(colun, hist=True)
    plt.show()


def grafico_barra(colun):
    plt.figure(figsize=(15, 5))
    ax = sns.barplot(x=colun.value_counts().index, y=colun.value_counts())
    ax.set_xlim(limites(colun))


# analisando a coluna price
diagrama_caixa(base_airbnb['price'])
histograma(base_airbnb['price'])

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'price')  # excluindo linhas abaixo e acima do limite

# analisando a coluna extra_people
diagrama_caixa(base_airbnb['extra_people'])
histograma(base_airbnb['extra_people'])

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'extra_people')  # excluindo linhas abaixo e acima do limite

# analisando host_listings_count
diagrama_caixa(base_airbnb['host_listings_count'])
grafico_barra(base_airbnb['host_listings_count'])

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'host_listings_count')  # excluindo linhas abaixo e acima do limite

# analisando accommodates
diagrama_caixa(base_airbnb['accommodates'])
grafico_barra(base_airbnb['accommodates'])

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'accommodates')  # excluindo linhas abaixo e acima do limite

# analisando bathrooms
diagrama_caixa(base_airbnb['bathrooms'])
plt.figure(figsize=(15, 5))
sns.barplot(x=base_airbnb['bathrooms'].value_counts().index, y=base_airbnb['bathrooms'].value_counts())

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bathrooms')  # excluindo linhas abaixo e acima do limite

# analisando bedrooms
diagrama_caixa(base_airbnb['bedrooms'])
grafico_barra(base_airbnb['bedrooms'])

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bedrooms')  # excluindo linhas abaixo e acima do limite

# analisando beds
diagrama_caixa(base_airbnb['beds'])
grafico_barra(base_airbnb['beds'])

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'beds')  # excluindo linhas abaixo e acima do limite

# analisando guests_included
# diagrama_caixa(base_airbnb['guests_included'])
# grafico_barra(base_airbnb['guests_included'])
# removendo, pois aparentemente os dados estão mal preenchidos

base_airbnb = base_airbnb.drop('guests_included', axis=1)

# analisando minimum_nights
diagrama_caixa(base_airbnb['minimum_nights'])
grafico_barra(base_airbnb['minimum_nights'])

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'minimum_nights')  # excluindo linhas abaixo e acima do limite

# analisando maximum_nights
# diagrama_caixa(base_airbnb['maximum_nights'])
# grafico_barra(base_airbnb['maximum_nights'])

base_airbnb = base_airbnb.drop('maximum_nights', axis=1)    # removendo por problemas de preenchimento

# analisando number_of_reviews
diagrama_caixa(base_airbnb['number_of_reviews'])
grafico_barra(base_airbnb['number_of_reviews'])

base_airbnb = base_airbnb.drop('number_of_reviews', axis=1)    # removendo pensando em novos users

# tratamento de colunas com valores textuais

# tratando property_type
plt.figure(figsize=(15, 5))
grafico = sns.countplot('property_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

tabela_tipos_casa = base_airbnb['property_type'].value_counts()
colunas_agrupar = []

for tipo in tabela_tipos_casa.index:
    if tabela_tipos_casa[tipo] < 2000:
        colunas_agrupar.append(tipo)

for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['property_type'] == tipo, 'property_type'] = 'Outros'

plt.figure(figsize=(15, 5))
grafico = sns.countplot('property_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

# tratando room_type
plt.figure(figsize=(15, 5))
grafico = sns.countplot('room_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

# tratando bed_type
plt.figure(figsize=(15, 5))
grafico = sns.countplot('bed_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

# agrupando categorias de bed_type
tabela_bed = base_airbnb['bed_type'].value_counts()
colunas_agrupar = []

for tipo in tabela_bed.index:
    if tabela_bed[tipo] < 10000:
        colunas_agrupar.append(tipo)

for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['bed_type']==tipo, 'bed_type'] = 'Outros'

plt.figure(figsize=(15, 5))
grafico = sns.countplot('bed_type', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

# tratando cancellation_policy
plt.figure(figsize=(15, 5))
grafico = sns.countplot('cancellation_policy', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

# agrupando categorias de cancellation_pollicy
tabela_cancellation = base_airbnb['cancellation_policy'].value_counts()
colunas_agrupar = []

for tipo in tabela_cancellation.index:
    if tabela_cancellation[tipo] < 10000:
        colunas_agrupar.append(tipo)

for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['cancellation_policy']==tipo, 'cancellation_policy'] = 'strict'

plt.figure(figsize=(15, 5))
grafico = sns.countplot('cancellation_policy', data=base_airbnb)
grafico.tick_params(axis='x', rotation=90)

# tratando amenities
print(base_airbnb['amenities'].iloc[1].split(','))
print(len(base_airbnb['amenities'].iloc[1].split(',')))

base_airbnb['n_amenities'] = base_airbnb['amenities'].str.split(',').apply(len)
base_airbnb = base_airbnb.drop('amenities', axis=1)

diagrama_caixa(base_airbnb['n_amenities'])
grafico_barra(base_airbnb['n_amenities'])

base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'n_amenities')

# visualização de mapa das propriedades
amostra = base_airbnb.sample(n=50000)
centro_mapa = {'lat': amostra.latitude.mean(), 'lon': amostra.longitude.mean()}
mapa = px.density_mapbox(amostra, lat='latitude', lon='longitude', z='price', radius=2.5, center=centro_mapa, zoom=10,
                         mapbox_style='stamen-terrain')
mapa.show()

# encoding
colunas_tf = ['host_is_superhost', 'instant_bookable', 'is_business_travel_ready']
base_airbnb_cod = base_airbnb.copy()
for coluna in colunas_tf:
    base_airbnb_cod.loc[base_airbnb_cod[coluna] == 't', coluna] = 1
    base_airbnb_cod.loc[base_airbnb_cod[coluna] == 'f', coluna] = 0

colunas_categorias = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
base_airbnb_cod = pd.get_dummies(data=base_airbnb_cod, columns=colunas_categorias)

# modelo de previsão


def avaliar_modelo(nome_modelo, y_teste, previsao):
    r2 = r2_score(y_teste, previsao)
    RSME = np.sqrt(mean_squared_error(y_teste, previsao))
    return f'Modelo {nome_modelo}:\nR²:{r2}\nRSME:{RSME}'


modelo_rf = RandomForestRegressor()
modelo_lr = LinearRegression()
modelo_et = ExtraTreesRegressor()

# testando modelos
'''modelos = {'RandomForest': modelo_rf, 'LinearRegression': modelo_lr, 'ExtraTrees': modelo_et}

y = base_airbnb_cod['price']
x = base_airbnb_cod.drop('price', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)

for nome_modelo, modelo in modelos.items():
    # treinar
    modelo.fit(x_train, y_train)
    # testar
    previsao = modelo.predict(x_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))

for nome_modelo, modelo in modelos.items():
    # testar
    previsao = modelo.predict(x_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))'''

# escolhendo o modelo extra_trees por apresentar maior valor R² e menor RSME
# extra_tree -- R²=0.9750 RSME=41.8755
base_airbnb_cod = base_airbnb_cod.drop('is_business_travel_ready', axis=1)

y = base_airbnb_cod['price']
X = base_airbnb_cod.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

modelo_et.fit(X_train, y_train)
previsao = modelo_et.predict(X_test)

base_teste = base_airbnb_cod.copy()
for coluna in base_teste:
    if 'bed_type' in coluna:
        base_teste = base_teste.drop(coluna, axis=1)
print(base_teste.columns)
y = base_teste['price']
X = base_teste.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

modelo_et.fit(X_train, y_train)
previsao = modelo_et.predict(X_test)
print(avaliar_modelo('ExtraTrees', y_test, previsao))
print(previsao)

# deploy
X['price'] = y
X.to_csv('dados.csv')

# exportar modelo
joblib.dump(modelo_et, 'modelo.joblib')

