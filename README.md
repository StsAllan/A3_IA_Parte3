
# Previsão de Segundo Ataque Cardíaco

## Objetivo
Este projeto tem como objetivo desenvolver um modelo preditivo para identificar a probabilidade de pacientes que já sofreram um ataque cardíaco terem um segundo ataque. O modelo busca auxiliar na implementação de medidas preventivas mais eficazes.

## Conjunto de Dados
Os dados utilizados contêm informações sobre pacientes que sofreram um ataque cardíaco. Os atributos disponíveis incluem:
- **Idade**: Idade em anos.
- **Estado civil**: Codificado em valores numéricos (0 = nunca casado, 1 = casado, 2 = divorciado, 3 = viúvo).
- **Gênero**: 0 para mulheres, 1 para homens.
- **Categoria de peso**: 0 para peso normal, 1 para sobrepeso, 2 para obesidade.
- **Colesterol**: Nível de colesterol no momento do tratamento.
- **Administração de estresse**: Participação em curso de gestão de estresse (0 = não, 1 = sim).
- **Tratamento de ansiedade**: Pontuação de ansiedade em uma escala de 0 a 100.
- **Segundo ataque cardíaco**: Variável alvo (0 = não, 1 = sim).

## Metodologia
O problema foi tratado como uma tarefa de classificação binária utilizando redes neurais multicamadas. O pipeline do projeto inclui as seguintes etapas:

### 1. **Pré-processamento**
- Conversão da variável alvo (`Sim`/`Não`) para valores binários (1/0).
- Normalização das variáveis independentes usando `StandardScaler`.

### 2. **Divisão dos Dados**
Os dados foram divididos em:
- **Treinamento**: 80% dos dados.
- **Teste**: 20% dos dados.

### 3. **Arquiteturas Testadas**
O modelo de rede neural foi configurado com diferentes arquiteturas:
- Número de neurônios na camada escondida: 5 ou 9.
- Função de ativação: ReLU ou Logística.
- Algoritmo de aprendizado: Adam ou L-BFGS-B.

### 4. **Métricas de Avaliação**
- **Acurácia**: Proporção de previsões corretas.
- **Matriz de Confusão**: Representa os verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos.

## Resultados
Foram simuladas 8 combinações de arquiteturas. Os resultados para cada arquitetura incluíram a acurácia e a matriz de confusão.

### Melhores Resultados:
- Redes com função de ativação ReLU e solver Adam apresentaram melhor desempenho, alcançando uma acurácia de 100%.
- A ativação logística apresentou desempenho inferior em geral.

### Piores Resultados:
- Modelos com solver Adam e ativação logística tiveram as menores acurácias (~89%).

## Ferramentas Utilizadas
- **Python 3.11**
- Bibliotecas:
  - `pandas`: Manipulação de dados.
  - `scikit-learn`: Normalização, criação de redes neurais e métricas.
  - `matplotlib`: Visualização de dados (opcional).

## Como Executar
1. Certifique-se de ter o Python 3.x instalado.
2. Instale as dependências:
   ```bash
   pip install pandas scikit-learn matplotlib
   ```
3. Execute o código no arquivo `.py` fornecido.

## Conclusão
Os resultados mostram que redes neurais multicamadas podem ser eficazes para prever a probabilidade de um segundo ataque cardíaco. A escolha de parâmetros como a função de ativação e o solver afeta significativamente o desempenho do modelo.

Este modelo pode ser utilizado para identificar pacientes em maior risco e implementar medidas preventivas personalizadas.
