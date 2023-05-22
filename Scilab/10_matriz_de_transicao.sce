[diretorio, _] = get_absolute_file_path()
exec(diretorio + 'inputs.sce', -1)

/* ===================================================================
    PROCESSAMENTO E SAÍDAS
   ===================================================================*/
   
// Matriz de transição:
Phi = expm(A * dt)

// Matriz dos termos forçantes:
Gamma = inv(A) * (Phi - eye(A))

disp(Phi)
disp(Gamma)
