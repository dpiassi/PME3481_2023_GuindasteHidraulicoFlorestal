[diretorio, _] = get_absolute_file_path()
exec(diretorio + 'inputs.sce', -1)

/* ===================================================================
    PROCESSAMENTO E SAÍDAS
   ===================================================================*/

// Inicialização do sistema linear em malha aberta:
sl_OpenLoop = syslin('c', A, B, C, D)

// Inicialização da função de transferência em malha aberta:
h = ss2tf(sl_OpenLoop)
disp(h)

// Obtenção dos pólos em malha aberta:
eigenvalues = spec(A)
printf("\nPólos em malha aberta:")
disp(eigenvalues)

// Plotagem dos pólos do sistema linear:
scf()
plzr(sl_OpenLoop)
xtitle("Pólos do sistema em malha aberta", "Eixo real", "Eixo imaginário")
legend(["Pólos"])

// Obtenção do polinômio característico:
characteristic_polynomial = poly(A, 's')
printf("\nPolinômio característico: ")
disp(characteristic_polynomial)

// Coeficientes e pólos do polinômio:
printf("\nPólos:")
Poles = roots(characteristic_polynomial)
disp(Poles)

printf("\nCoeficientes:")
Coeffs = coeff(characteristic_polynomial)
disp(Coeffs)

// Aplicação do critério de Routh Hurwitz:
RouthHurwitz = routh_t(characteristic_polynomial)
first_column = RouthHurwitz(:, 1)
printf("\nTabela de Routh-Hurwitz:")
disp(RouthHurwitz)
