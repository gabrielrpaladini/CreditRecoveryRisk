import datetime
import pandas as pd

class Log:

    def __init__(self, real_value, predict_value):
        self.real_value = real_value
        self.predict_value = predict_value
        self.date = datetime.datetime.now()
    
    #Metódo para inserir os logs de treino, previsões dos valores reais e dos valores preditos
    def log_train(self):

        file_train = open('Log/log_train_{}.txt'.format(self.date.strftime("%a_%d")), 'w')

        for i in range(len(self.real_value)):
            file_train.write('O valor real foi: {} e o valor predito foi: {} \n'.format(self.real_value.iloc[i],self.predict_value[i]))

        file_train.close()

        return print('Log de treino inserido!!')

    #resultado principal do modelo, com dados de produção
    def results(self):

        results = open('Results/ResultadoIAGAV_{}.txt'.format(self.date.strftime("%a_%d")), 'w')

        for i in range(len(self.real_value)):
            results.write('{}   {}\n'.format(self.real_value.iloc[i], self.predict_value[i]))
        results.close()
        
        return print('Resultado inserido com sucesso!! verificar na pasta de Results ')
