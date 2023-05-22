[diretorio, _] = get_absolute_file_path()
exec(diretorio + 'inputs.sce', -1)

/* ===================================================================
    PROCESSAMENTO E SAÍDAS
   ===================================================================*/

// Análise de controlabilidade
[c, U] = contr(A, B)
d_c = det(U)

// Análise de observabilidade
[o, V] = contr(A', C')
d_o = det(V)

// Plot dos resultados
disp("Matriz A", A)
disp("Matriz B", B)
disp("matriz C", C)

disp("Matriz de controlabilidade", clean(U))
disp("Determinante da matriz de controlabilidade", d_c)
disp("Posto da matriz de controlabilidade", c)

disp("Matriz de observabilidade", V)
disp("Determinante da matriz de observabilidade", d_o)
disp("Posto da matriz de observabilidade", o)
