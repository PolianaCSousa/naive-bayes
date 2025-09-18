# 1: melhor_probabilidade ← 0
# 2: classe_predita ← nulo
# 3: for cada classe ‘c’ em p_prior do
# 4: prob_classe ← p_prior[c]
# 5: for cada atributo ‘a’ no exemplo_novo do
# 6: valor ← valor do atributo ‘a’ no exemplo_novo
# 7: prob_cond ← p_condicional[a][valor][c]
# 8: prob_classe ← prob_classe · prob_cond
# 9: end for
# 10: if (prob_classe > melhor_probabilidade) then
# 11: melhor_probabilidade ← prob_classe
# 12: classe_predita ← c
# 13: end if
# 14: end for
# 15: return classe_predita =0