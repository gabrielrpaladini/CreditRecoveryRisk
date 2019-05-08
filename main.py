#classe para tratar os dados do dataset
from dataset import Dataset
#classe para tratar os dados do modelo
from model import Model
#classe para logs e metricas
from log import Log

dataset = Dataset('dados1', 'dados2')

x, y = dataset.normalize_dataset_train()

real_cpf, normalized_df_production = dataset.normalize_dataset_production()

model = Model(x, y, normalized_df_production)

predict_value = model.model_learning()

log = Log(real_cpf, predict_value)

log.results()