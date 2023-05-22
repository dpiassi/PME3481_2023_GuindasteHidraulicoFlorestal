[diretorio, _] = get_absolute_file_path()
exec(diretorio + 'inputs.sce', -1)

// Pólos desejados:
po1 = -2.40E-04 + 1.72E+00 * %i
po2 = -2.40E-04 + -1.72E+00 * %i
po3 = -2.21E-01 + 8.58E-04 * %i
po4 = -2.21E-01 + -8.58E-04 * %i
po5 = -1.41E+00 + 0.00E+00 * %i
po6 = -1.41E+00 + 0.00E+00 * %i
P = [po1, po2, po3, po4, po5, po6]
printf("\nPólos desejados:")
disp(P')


// Pólos do observador:
P_o = P - ones(1, 6)
printf("\nPólos do observador:")
disp(P_o')


Ko = (ppol(A', C', P))'

/* ===================================================================
    VARIÁVEIS AUXILIARES
   ===================================================================*/
step = 0.05
duration = 5
t = 0:step:duration

// Variável de saída:
z = zeros(2, length(t))


/* ===================================================================
    PROCESSAMENTO E SAÍDAS
   ===================================================================*/
printf("\n=========================================")
printf("\nRESPOSTA SEM CONTROLE")
printf("\n=========================================")
printf("\n")
   
// Definição do sistema linear:
linear_system = syslin("c", A, B(:, 1), C, D(:, 1))

// Solução do sistema linear:
[y, x] = csim('step', t, linear_system, x0)

// Plotagem do sistema linear:
scf()
plot(t, x(1, :), "r")
plot(t, x(3, :), "g")
plot(t, x(5, :), "b")
xtitle("Resposta ao degrau — sem controle", "Tempo", "Vetor de saídas")
hl=legend(['theta_1'; 'theta_2'; 'theta_3'])
xs2png(gcf(), "Resposta ao degrau — sem controle")

printf("\n=========================================")
printf("\nRESPOSTA COM CONTROLE")
printf("\n=========================================")
printf("\n")

// ppol: obter matriz de ganhos de controle para os pólos dados
K = ppol(A, B, P)
printf("\nMatriz de ganhos de controle:")
disp(K)

// F é a matriz de estados para resposta controlada
F = A - B * K
printf("\nF = A - B * K")
disp(F)

// Definição do sistema linear:
controlled_linear_system = syslin("c", F, B(:, 1), C, D(:, 1))

// Solução do sistema linear:
[y, x] = csim('step', t, controlled_linear_system, x0)

// Plotagem do sistema linear com controle:
scf()
plot(t, x(1, :), "r")
plot(t, x(3, :), "g")
plot(t, x(5, :), "b")
xtitle("Resposta ao degrau — com controle", "Tempo", "Vetor de saídas")
hl=legend(['theta_1'; 'theta_2'; 'theta_3'])
xs2png(gcf(), "Resposta ao degrau — com controle")


printf("\n=========================================")
printf("\nDETERMINAÇÃO DO OBSERVADOR")
printf("\n=========================================")
printf("\n")

printf("\nMatriz de ganhos do observador:")
K_o = ppol(A', C', P_o)'
disp(K_o)

printf("\nÂ=A-K_o*C")
Â = A - K_o * C
disp(Â)


printf("\n=========================================")
printf("\nRESPOSTA COM CONTROLE E OBSERVADOR")
printf("\n=========================================")
printf("\n")

Lambda = [A, -B*K; K_o*C, A-B*K-K_o*C]
printf("\nLambda:")
disp(Lambda)

function z_dot=SeparationPrinciple(t, z)
    z_dot = Lambda * z
endfunction

z0 = [zeros(6, 1); x0]
z = ode(z0, 0, t, SeparationPrinciple)

scf(3)
plot(t, z(1, :), "g")
plot(t, z(7, :), "r")
xtitle("Sistema controlado com observador — theta_1", "Tempo", "Vetor de saídas")
hl=legend(['theta_1'; '~theta_1'])
xs2png(gcf(), "Sistema controlado com observador — theta_1")

scf(4)
plot(t, z(3, :), "g")
plot(t, z(9, :), "r")
xtitle("Sistema controlado com observador — theta_2", "Tempo", "Vetor de saídas")
hl=legend(['theta_2'; '~theta_2'])
xs2png(gcf(), "Sistema controlado com observador — theta_2")

scf(5)
plot(t, z(5, :), "g")
plot(t, z(11, :), "r")
xtitle("Sistema controlado com observador — theta_3", "Tempo", "Vetor de saídas")
hl=legend(['theta_3'; '~theta_3'])
xs2png(gcf(), "Sistema controlado com observador — theta_3")
