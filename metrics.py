import datetime
from sklearn.metrics import mean_squared_error

class Metrics:
    
    def __init__(self, real_value, predict_value):
        self.real_value = real_value
        self.predict_value = predict_value
        self.date = datetime.datetime.now()

    #Escreve as métricas de erro do modelo    
    def write_metrics(self):
        metrics_file = open('Metrics/Metrics file.txt', 'a')
        metrics_file.write('\n A media quadratica calculada de erro para o arquivo do dia {} é: {}'.format(self.date.strftime("%a_%d"), mean_squared_error(self.real_value, self.predict_value)))
        metrics_file.close()
        return print('Metrica de erro inserida!!')