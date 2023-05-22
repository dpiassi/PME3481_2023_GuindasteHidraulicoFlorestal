[diretorio, _] = get_absolute_file_path()
exec(diretorio + 'inputs.sce', -1)

/* ===================================================================
    PROCESSAMENTO E SAÍDAS
   ===================================================================*/
 
// Pólos do controlador por alocação de pólos:
po1 = complex(-2, 2)
po2 = po1'
po3 = complex(-1, 1)
po4 = po3'
po5 = complex(-3, 3)
po6 = po5'
poles = [po1, po2, po3, po4, po5, po6] - ones(1, 6)
Ko = (ppol(A', C', poles))'

mprintf("\n> Alocação de pólos")
mprintf("\nKₒ - Matriz de ganho de observação:")
disp(Ko)
disp(prettyprint(Ko, "latex"))


// Pólos do controlador LQR:
po1 = -2.40E-04 + 1.72E+00 * %i
po2 = -2.40E-04 + -1.72E+00 * %i
po3 = -2.21E-01 + 8.58E-04 * %i
po4 = -2.21E-01 + -8.58E-04 * %i
po5 = -1.41E+00 + 0.00E+00 * %i
po6 = -1.41E+00 + 0.00E+00 * %i  
poles = [po1, po2, po3, po4, po5, po6] - ones(1, 6)
Ko = (ppol(A', C', poles))'

mprintf("\n> LQR")
mprintf("\nKₒ - Matriz de ganho de observação:")
disp(Ko)
disp(prettyprint(Ko, "latex"))
