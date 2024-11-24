import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class DataPrep:
    def __init__(self, data):
        self.data = data
  
    def _one_hot_encoder(df: pd.DataFrame, variavel: str) -> pd.DataFrame:
        # Transforma uma variável categórica utilizando o One Hot Encoding
        dummies = pd.get_dummies(df[variavel], prefix=variavel)
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=variavel, inplace=True)
        return df

    def _tratar_variaveis_categoricas(self) -> None:
        # Usar One Hot Encoder para tratar as variáveis categóricas
        self.data = DataPrep._one_hot_encoder(self.data, 'Warehouse_block')

        self.data = DataPrep._one_hot_encoder(self.data, 'Mode_of_Shipment')

        self.data = DataPrep._one_hot_encoder(self.data, 'Product_importance')

        self.data = DataPrep._one_hot_encoder(self.data, 'Gender')
    
    def _remover_colunas(self) -> None:
        # Remove as variáveis que não serão utilizadas pelo modelo
        self.data.drop(columns='ID', inplace=True)
    
    def _normalizar_dados(self) -> None:
        variaveis = self.data.drop(columns='Reached.on.Time_y.N')
        var_cols = variaveis.columns
        resposta = self.data['Discount_offered']

        scaler = MinMaxScaler()
        variaveis = scaler.fit_transform(variaveis)
        variaveis = pd.DataFrame(variaveis, columns=var_cols)

        self.data = pd.concat([variaveis, resposta], axis=1)
    
    def _separar_treino_teste(self):
        # Divide o dataset em conjunto de treino e teste
        treino, teste = train_test_split(self.data, test_size=0.3, random_state=2024)
        return treino, teste

    def preparar_dados(self):
        # Aplica todas as transformações de dados
        self.tratar_variaveis_categoricas()
        self._remover_colunas()
        self._normalizar_dados()
        treino, teste = self._separar_treino_teste()

        return treino, teste
