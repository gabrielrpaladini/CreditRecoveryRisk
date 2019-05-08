import pandas as pd
import numpy as np
import datetime

class Dataset:

    def __init__(self, file_train, file_prod):
        self.file_train = file_train
        self.file_prod = file_prod
    
    #Tratar as colunas do dataset que vai ser utilizado para treino 
    def normalize_dataset_train(self):
        df_treino = pd.read_excel(self.file_train)
        df_treino.drop('Cpf', axis=1, inplace=True)
        
        x = df_treino.drop('Pagamento', axis=1)
        y = df_treino['Pagamento']
        return x, y
    
    #Normalizar o dataset que vai ser utilizado na produção
    def normalize_dataset_production(self):
        df_production = pd.read_excel(self.file_prod)
        real_cpf = df_production['Cpf']
        normalized_df_production = df_production
        del normalized_df_production['Cpf']
        return real_cpf, normalized_df_production

        


