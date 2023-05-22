// Busca os dados de entrada
diretorio = get_absolute_file_path();
exec(diretorio + 'inputs.sce')

// Definição dos polos desejados
po1=complex(-2,2)
po2=po1'
po3=complex(-1,1)
po4=po3'
po5=complex(-3,3)
po6=po5'
poles = [po1,po2,po3,po4,po5,po6]
F = ppol(A,B,poles)

// Calculo da nova estabilidade a partir dos novos polos
Pc = poly(A-B*F, "s")
r = roots(Pc) 

sys = syslin('c', A-B*F,B,C,D)
scf(1)
plzr(sys)

disp("Pós alocação de polos", clean(r));

// Aplicação do critério de Routh Hurwitz:
RouthHurwitz = routh_t(Pc)
first_column = RouthHurwitz(:, 1)
printf("\nTabela de Routh-Hurwitz:")
disp(RouthHurwitz)

latex1 = prettyprint(RouthHurwitz)
disp(latex1)

last_sign = sign(first_column(1))
sign_changes = 0
for i=2:length(first_column)
    current_sign = sign(first_column(i))
    if current_sign ~= last_sign then
        last_sign = current_sign
        sign_changes = sign_changes + 1
    end
end

printf("\n")
if sign_changes == 0 then
    printf("Como não há mudanças de sinal na primeira coluna, o sistema é ESTÁVEL.\n")
else
    printf("%d mudanças de sinal na primeira coluna.\nPortanto, o sistema é INSTÁVEL com %d polos instáveis\n", sign_changes, sign_changes)
end

