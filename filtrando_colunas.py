import pandas as pd

tabela = pd.read_csv('primeiros_registros.csv', sep=';')

# limpando colunas vazias
tabela2 = tabela.dropna(axis=1, how='all')
# salvando um novo arquivo para que o mesmo não seja afetado após executar o codigo_fonte.py novamente
tabela2.to_csv('primeiros_registros_filtrados.csv', sep=';', index=False)

# criando uma lista com os nomes das colunas
lista_colunas = []
for colunas in tabela2:
    lista_colunas.append(colunas)

# usaremos os nomes nesta lista para filtrar a nossa base airbnb no arquivo codigo_fonte.py
print(lista_colunas)
# esse foi o resultado após analisarmos manualmente e apagar as colunas que não seriam uteis para o nosso desafio
'''
['host_response_time', 'host_response_rate', 'host_is_superhost', 'host_listings_count', 'latitude', 'longitude',
 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 'price',
  'security_deposit', 'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights',
   'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
    'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 
    'instant_bookable', 'is_business_travel_ready', 'cancellation_policy', 'ano', 'mes']
'''
