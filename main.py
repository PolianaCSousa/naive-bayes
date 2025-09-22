import pandas as pd
from pathlib import Path

from ClassificaNaiveBayes import classifica_naive_bayes
from TreinaNaiveBayes import treina_naive_bayes


def read_file(file_name):
    try:
        dataset = pd.read_csv(file_name, header=None)
        print(dataset)
    except FileNotFoundError:
        print("Erro: O arquivo não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
    return dataset


#converte os dados em dados numéricos
def process_data(dataset):
    # dicionários de mapeamento
    map_values = {"o": 0, "x": 1, "b": 2}
    map_class = {"positive": 1, "negative": 0}

    # aplica nos 9 primeiros atributos
    dataset.iloc[:, :-1] = dataset.iloc[:, :-1].replace(map_values)

    # aplica na classe (última coluna)
    dataset.iloc[:, -1] = dataset.iloc[:, -1].replace(map_class)

    # salva em CSV sem índice e sem cabeçalho
    dataset.to_csv("numeric-tic-tac-toe.csv", index=False, header=True)

    return dataset



#divide o dataset em dados de treinamento (70%) e teste (30%)
def divide_data(dataset):
    # Suponha que df seja seu DataFrame
    df = pd.DataFrame(dataset)
    treino = df.sample(frac=0.7, random_state=42)  # 70% aleatório
    teste = df.drop(treino.index)  # o resto (30%)
    return treino,teste


if __name__  == '__main__':
    # Traz o dataset pra memoria
    dataset = read_file("tic-tac-toe.data")

    # Processa os dados
    # - Se o arquivo processado ja existir, le ele
    if Path("numeric-tic-tac-toe.csv").exists():
        processed_dataset = read_file("numeric-tic-tac-toe.csv")
    else:
        processed_dataset = process_data(dataset)

    # Divide o dataset em treino e teste
    dataset_treino, dataset_teste = divide_data(processed_dataset)

    # Treina o modelo Naive Bayes
    p_condicional, p_prior = treina_naive_bayes(dataset_treino)

    # Roda a classificacao pra cada linha do dataset de teste
    classe_predita = [
        classifica_naive_bayes(p_prior, p_condicional, dataset_teste.iloc[[i]])
        for i in range(len(dataset_teste))
    ]

    # Printa os resultados
    print(f"DATASET TREINO:\n{dataset_treino}")
    print(f"DATASET TESTE:\n{dataset_teste}")
    print("Classe prevista para cada exemplo do teste:")
    for i, classe in enumerate(classe_predita):
        print(f"Exemplo {i}: Classe prevista = {classe}")