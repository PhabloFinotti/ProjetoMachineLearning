### Introdução
## Informações do Dataset:

Este dataset contém **9358 respostas registradas** a cada hora de um Dispositivo Multisensorial de Qualidade Química do Ar que contém uma lista com **5 sensores de óxidos de metais**. Esse dispositivo estava localizado em uma área consideravelmente poluída, próxima a estradas de uma cidade italiana. Os dados foram registrados entre Março de 2004 a Fevereiro de 2005 (**um ano**) representando o registro gratuito mais longo de um dispositivo de qualidade química do ar. Registros verificados mostram concentrações de: CO, Hidrocarboneto não metânico, Benzeno, Óxidos totais de Nitrogênio (NOx) e Dióxido de Nitrogênio (NO2) e foram fornecidas por um analisador certificado da região. Evidências de **sensibilidades cruzadas**, bem como **desvios de conceito e sensor** estão presentes, conforme descrito em _De Vito et al., Sens. And Act. B, Vol. 129,2,2008_, que eventualmente podem afetar as estimações e concentrações dos sensores. **Valores faltantes foram atribuídos o valor "-200"**.
Esse dataset só pode ser usado para fins de pesquisa. Excluindo assim, usos comerciais.

### Detalhes dos registros
 - 0 Data (DD/MM/AAAA)
 - 1 Tempo (HH.MM.SS)
 - 2 Concentração média por hora de CO in mg/m^3 (analisador de referência)
 - 3 PT08.S1 (Óxido de Estanho) concentração média por hora (nominalmente direcionada a CO)
 - 4 Concentração média por hora de Hidrocarbonetos não metânicos em microg/m3 (analisador de referência)
 - 5 Concentração média por hora de Benzeno em microg/m3 (analisador de referência)
 - 6 PT08.S2 (titânia) resposta média por hora do sensor (nominalmente direcionada ao NMHC)
 - 7 Concentração média por hora de NOX em ppb (analisador de referência)
 - 8 PT08.S3 (óxido de tungstênio) resposta média do sensor (nominalmente direcionada NO2)
 - 9 Concentração média por hora de NO2 em microg/m3 (analisador de referência)
 - 10 PT08.S4 (óxido de tungstênio) resposta média por hora (nominalmente direcionada NO2)
 - 11 PT08.S5 (óxido de índio) resposta média por hora (nominalmente direcionada O3)
 - 12 Temperatura em ºC
 - 13 Umidade relativa (%)
 - 14 Umidade Absoluta (AH)

Começamos importando a biblioteca **```pandas```** que possui uma extensa quantidade de comandos e funções que auxiliarão bastante na resolução deste caso, então, fazemos a leitura do arquivo **.csv** que contém todos os dados da pesquisa e mostramos as 5 primeiras linhas com o seguinte código:

```python
import pandas as pd
df = pd.read_csv("data05.csv")
df.head()
```

Após executar essas linhas de código nos deparamos com uma tabela contendo 16 colunas com nomes não muito convenientes para o nosso projeto, mas logo falaremos disso. Em seguida, podemos executar mais alguns comandos para obtermos mais informações sobre esses dados, sendo eles:

```python
df.shape() - Apresenta quantas linhas e colunas existem no arquivo. No caso, 9357 linhas e 16 colunas.
df.info() - Apresenta a quantidade de linhas sem valores nulos e quais os tipos de dados de cada coluna.
df.describe(include="all") - Apresenta diversas informações sobre nossos dados, por exemplo: quantidade, valores únicos, média, mínima, máxima, padrão, etc. Colocamos o parâmetro include='all', para que sejam resgatadas todas as informações do DataFrame, sendo dados numéricos ou objetos
```

Depois de termos essa visão geral do projeto, podemos seguir para o tratamento dos dados.

## Tratamento de dados ausentes

##### Verificando valores nulos
Começamos com uma checagem simples de valores Nulos:

```python
df.isnull()
```

O código acima nos retornará um valor booleano, sendo ```True``` onde o valor seja nulo.

Podemos adicionar outra função em frente ao código anterior para obtermos a quantidade de valores nulos por coluna.

```python
df.isnull().sum()
```

##### Derrubando valores nulos
&nbsp;
```python
df.dropna(how='all').inplace=True
```

Utilizando ```dropna(how='all')```, excluiremos as linhas aonde contém somente valores nulos. Porém nesse caso, gostariamos de definir quais colunar iremos considerar para remover valores ausentes com: ```dropna(subset=[])```.
```inplace=True``` "salvará" nossas alterações ao DataFrame.

```python
df.dropna(subset = ['Date','Time','CO(GT)','PT08.S1(CO)','NMHC(GT)','C6H6(GT)','PT08.S2(NMHC)','NOx(GT)','PT08.S3(NOx)','NO2(GT)','PT08.S4(NO2)','PT08.S5(O3)','T','RH','AH','classe'], inplace=True)
```

E então redefinimos a indexação do nosso DataFrame com:
```python
df = df.reset_index(drop=True)
```

#### Substituindo Valores

Para melhorar a comodidade e visualização do DataFrame iremos derrubar algumas colunar irrelevantes para os nossos testes e renomear algumas outras.

```python
df.columns = ['Data','Hora','Cobalto', 'OxidoEstanho','HidrocarbonetosNãoMetanicos', 'Benzeno','Titania','ConcentraçãoNOx','OxidoTungstênioNOx','ConcentraçãoNO2','OxidoTungstênioNO2','OxidoÍndioO3','Temperatura-C°','UmidadeRelativa (%)','UmidadeAbsoluta','Classe']
df_2 = df
df_2 = df.drop(columns=['Data','Hora','Temperatura-C°','UmidadeRelativa (%)','UmidadeAbsoluta'])
```

De acordo com o enunciado, valores faltantes foram atribuídos o valor **-200**. Então iremos trocar esses valores por **0** (zero).

```python
df_2.replace([-200], 0, inplace=True)
```

#### Realizando Testes

Selecionamos quais colunas serão utilizadas para os nossos testes e atribuímos a variável **x** e também atribuímos a coluna **Classes** a variável **y** 

```python
feature_columns = ['Cobalto', 'OxidoEstanho','HidrocarbonetosNãoMetanicos','Benzeno','Titania','ConcentraçãoNOx','OxidoTungstênioNOx','ConcentraçãoNO2','OxidoTungstênioNO2','OxidoÍndioO3','Classe']
x = df_2[feature_columns].values
y = df_2['Classe'].values
```

##### Scikit KNeighborsClassifier

Utilizando a biblioteca ```SciKit Learn``` importamos a função ```train_test_split```, em que os paramêtros são:
- **x**: As colunas selecionadas acima
- **y**: A coluna de classificação
- **test_size**: A fração do DataFrame que será utilizada para testes.
- **random_state**: Quantidade de "embaralhamento" dos dados antes de fazer os testes.

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=2)
```

Também do **_SciKit Learn_** importamos o ```KNeighborsClassifier```, que passaremos a quantidade de dados "vizinhos" que serão utilizados nos testes (```model.fit```).

```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier (n_neighbors=3)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
```

E então com a variável ```y_pred``` podemos testar a acurácia dos testes com a função ```accuracy_score```, onde obtemos o valor de **98.61%** de acurácia!

```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)*100
print ('Acuracia: ', str(round(accuracy,2))+ "%")
```

Entre outras métricas, conseguimos algumas como:
 - **99%** de Precisão;
 - **99%** de Recall;
 - **99%** de F1.

#### Novo Tratamento

Decidimos fazer mais alguns testes, dessa vez, alterando os valores **-200** pela mediana da coluna. Um exemplo disso:

```python
df_3Cobalto=df_3[df_3['Cobalto']>-200.0]
df_3Cobalto['Cobalto'].median()   # Resultou em 1.8
df_3['Cobalto']= df_3['Cobalto'].replace([-200],1.8)
```

Foram feitos os mesmo processos anteriores. Obtivemos as seguintes métricas:
 - **97.91%** de Acurácia;
 - **98%** de Precisão;
 - **98%** de Recall;
 - **98%** de F1.
 
#### Outros Tratamentos

Além desses tratamentos, também decidimos fazer:
 - Tirando a coluna **data** e **hora** e fazer os teste com os valores **-200**:
   - **98.13%** de Acurácia;
   - **98%** de Precisão;
   - **99%** de Recall;
   - **98%** de F1.
   
 - Normalização, substituindo os valores **-200** para **0**:
   - **97.43%** de Acurácia;
   - **97%** de Precisão;
   - **98%** de Recall;
   - **98%** de F1.
  
 - Padronização, substituindo todos os valores e colocando eles entre **0** e **1**:
   - **97.48%** de Acurácia;
   - **97%** de Precisão;
   - **99%** de Recall;
   - **98%** de F1.









