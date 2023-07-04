// Executar script de inputs:
cwd = get_absolute_file_path();
exec(cwd + 'inputs.sce')

// Polinômio característico em malha aberta:
Pc = poly(A, "s")

// Aplicação do critério de Routh Hurwitz:
RouthHurwitz = routh_t(Pc)
first_column = RouthHurwitz(:, 1)
printf("\nTabela de Routh-Hurwitz:")
disp(RouthHurwitz)
disp(prettyprint(RouthHurwitz))

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

