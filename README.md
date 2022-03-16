# aplicacao_datascience_mercado
Ferramenta de Previsão de Preço de Imóvel para pessoas comuns

As bases de dados foram retiradas do site kaggle: https://www.kaggle.com/allanbruno/airbnb-rio-de-janeiro

Utilizando as bases de dados e várias bibliotecas do python, implementei essa ferramenta para fins acadêmicos.

O código principal encontra-se no arquivo "codigo_fonte.py", mas é preciso fazer o download dos arquivos no link acima para que ele funcione.

O arquivo "app.py" contém um aplicação local que usa como base o arquivo "modelo.joblib" que é gerado ao final da execução do arquivo "codigo_fonte"
A aplicação é bem simples e possui apenas os campos para definir as métricas que serão usadas para calcular o preço da diária de um imóvel dentro do Rio de Janeiro.
OBS. o arquivo app.py deve ser executado pelo terminal através da biblioteca streamlit

ROTEIRO:

Importar Bibliotecas e Bases de Dados
Consolidar Base de Dados
Se tivermos muitas colunas, já vamos identificar quais colunas podemos excluir
Tratar Valores Faltando
Verificar Tipos de Dados em cada coluna
Análise Exploratória e Tratar Outliers
Encoding
Modelo de Previsão
Análise do Melhor Modelo
Ajustes e Melhorias no Melhor Modelo
