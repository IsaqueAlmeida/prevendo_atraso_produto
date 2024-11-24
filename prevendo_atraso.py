import pandas as pd
from sklearn.model_selection import train_test_split


class DataPrep:
    def __init__(self, data):
        self.data = data
  
    def _one_hot_encoder(df: pd.DataFrame, variavel: str) -> pd.DataFrame:
        # Transforma uma variável categórica utilizando o One Hot Encoding
        dummies = pd.get_dummies(df[variavel], prefix=variavel)
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=variavel, inplace=True)
        return df

    def tratar_variaveis_categoricas(self) -> None:
        # Usar One Hot Encoder para tratar as variáveis categóricas
        self.data = DataPrep._one_hot_encoder(self.data, 'Warehouse_block')

        self.data = DataPrep._one_hot_encoder(self.data, 'Mode_of_Shipment')

        self.data = DataPrep._one_hot_encoder(self.data, 'Product_importance')

        self.data = DataPrep._one_hot_encoder(self.data, 'Gender')
    
    def remover_colunas(self) -> None:
        pass
