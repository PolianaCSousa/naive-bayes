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


