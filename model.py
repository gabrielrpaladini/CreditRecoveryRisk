from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import  mean_squared_error
from log import Log
from dataset import Dataset
from metrics import Metrics

class Model:
    
    def __init__(self, x, y, production):
        self.x = x
        self.y = y
        self.production = production

    #Modelo principal, fita o modelo divide métricas...
    def model_learning(self):
        
        #Dividir dados de treino e de teste
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.30)

        #Criar o objeto do modelo -- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
        gradient = GradientBoostingRegressor()
        gradient.fit(X_train, y_train)
        gb_prediction_train = gradient.predict(X_test)
        
        #Valores reais e valores preditos
        real_value = y_test
        predict_value = gb_prediction_train

        #Métricas de erro
        metrics = Metrics(real_value, predict_value)
        metrics.write_metrics()

        #log de previsão
        log = Log(real_value, predict_value)
        log.log_train()

        #Fazer a previsão em produção
        prediction_production = gradient.predict(self.production)
        
        return prediction_production
