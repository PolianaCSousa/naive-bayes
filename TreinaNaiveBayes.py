# 1: p_prior ← ∅
# 2: p_condicional ← ∅
# 3: for cada classe ‘c’ em dados_treino do
# 4: p_prior[c] ← count(c) / len(dados_treino)
# 5: end for
# 6: for cada classe ‘c’ em dados_treino do
# 7: for cada atributo ‘a’ no dados_treino do
# 8: for cada valor ‘v’ do atributo ‘a’ do
# 9: p_condicional[a][v][c] ← count(a == v and c) / count(c)
# 10: end for
# 11: end for
# 12: end for
# 13: return p_prior, p_condicional =0
import pandas as pd
from collections import defaultdict
import math

from numpy.ma.core import equal


def treina_naive_bayes(dataset_treino):
    p_prior = {}
    p_condicional = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    classe = 9
    classes = [0, 1]
    atributos = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    valores = [0, 1, 2]

    # Probabilidade a priori
    for c in classes: # Para cada classe existente conta quantas vezes apareceu determinada classe em treino e divide pelo total de dados em treino
        p_prior[c] = len(dataset_treino[dataset_treino[classe] == c]) / len(dataset_treino)


    # Probabilidade Condicional
    for c in classes:
        # A probabilidade de cada atributo ocorrer dado a classe
        for a in atributos:
            for v in valores:
                # Contando quantos atributos presentes em treino são iguais a cada valor possivel
                # Dividindo pela quantidade de vezes que a classe ocorreu
                p_condicional[a][v][c] = len(dataset_treino[(dataset_treino[a] == v) & (dataset_treino[classe] == c)]) / len(dataset_treino[dataset_treino[classe] == c])

    print("Probabilidades a priori (P(c)):\n")
    for c, p in p_prior.items():
        print(f"Classe {c}: {p:.4f} ({p * 100:.2f}%)")

    return p_condicional, p_prior

