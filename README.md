# Prevendo Atraso na Entrega de Produtos

Este projeto utiliza um modelo de Machine Learning para prever se um produto será entregue com atraso, baseado em características do pedido e do cliente. Ele aplica técnicas de pré-processamento de dados e manipulação de variáveis categóricas para preparar o conjunto de dados para modelagem.

---

## Estrutura do Dataset

O dataset contém 10.999 observações e 12 variáveis. Abaixo está a descrição das colunas:

- **ID**: Identificador único de cada cliente.
- **Warehouse_block**: Bloco do armazém onde o pedido foi armazenado, identificado por letras (A, B, C, D, F).
- **Mode_of_Shipment**: Meio de transporte utilizado (Ship = Marítimo, Flight = Aéreo, Road = Terrestre).
- **Customer_care_calls**: Número de chamadas para suporte ao cliente sobre o pedido.
- **Customer_rating**: Avaliação do cliente sobre a experiência (1 a 5).
- **Cost_of_the_product**: Custo do produto (em dólares americanos).
- **Prior_purchases**: Quantidade de compras anteriores realizadas pelo cliente.
- **Product_importance**: Importância do produto (Low, Medium, High).
- **Gender**: Sexo do cliente (Male, Female).
- **Discount_offered**: Desconto aplicado ao pedido.
- **Weight_in_gms**: Peso do produto (em gramas).
- **Reached_on_time**: Variável resposta (0 = Chegou no prazo; 1 = Chegou atrasado).

---

## Estrutura do Programa

### **1. Classe `DataPrep`**

A classe `DataPrep` é responsável por transformar e preparar os dados para análise. 

#### **Métodos**

- **`__init__(self, data: pd.DataFrame)`**  
  Construtor que inicializa o DataFrame que será processado.

- **`_one_hot_encoder(df: pd.DataFrame, variavel: str) -> pd.DataFrame`**  
  Método privado que aplica o *One Hot Encoding* a uma variável categórica, criando novas colunas para cada categoria.

- **`tratar_variaveis_categoricas(self) -> None`**  
  Aplica o *One Hot Encoding* às variáveis categóricas do dataset, transformando-as em variáveis binárias. As variáveis tratadas são:  
  - `Warehouse_block`  
  - `Mode_of_Shipment`  
  - `Product_importance`  
  - `Gender`

- **`remover_colunas(self) -> None`**  
  Método reservado para ser implementado futuramente.

---

## Fluxo do Programa

1. **Importação das Bibliotecas**  
   São utilizadas as seguintes bibliotecas:
   - `pandas` para manipulação de dados.
   - `sklearn.model_selection` para dividir os dados em conjuntos de treino e teste.

2. **Pré-processamento**  
   O dataset é carregado e processado por meio da classe `DataPrep`:
   - *One Hot Encoding* é aplicado às variáveis categóricas para torná-las utilizáveis em modelos de Machine Learning.

3. **Modelagem (a ser implementada futuramente)**  
   O programa atualmente foca na preparação dos dados. Etapas como divisão em treino e teste e treinamento de um modelo serão adicionadas posteriormente.

---

## Exemplo de Uso

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from dataprep import DataPrep

# Carregar o dataset
df = pd.read_csv('train_data_science.csv')

# Instanciar a classe e preparar os dados
dp = DataPrep(df)
dp.tratar_variaveis_categoricas()

# Exemplo: dividir os dados em treino e teste
train, test = train_test_split(dp.data, test_size=0.3, random_state=42)

print(train.head())
```
## Contribuições Futuras
- Adicionar o tratamento de colunas irrelevantes com o método `remover_colunas`.
- Implementar o treinamento de um modelo de Machine Learning para prever atrasos.
- Adicionar análises exploratórias e visualizações para melhor entendimento dos dados.