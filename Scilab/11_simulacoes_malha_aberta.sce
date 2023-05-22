[diretorio, _] = get_absolute_file_path()
exec(diretorio + 'inputs.sce', -1)

/* ===================================================================
    FUNÇÕES AUXILIARES
   ===================================================================*/
function FormatCurrentAxes(legend_titles)
    l = legend(legend_titles)
    l.font_size = 3
    a = gca()
    a.x_label.font_size = 3
    a.y_label.font_size = 3
    a.title.font_size = 4
endfunction

/* ===================================================================
    PROCESSAMENTO E SAÍDAS
   ===================================================================*/

//

for i=1:4
    sl = syslin('c', A, B(:, i), C, D(:, i))
    [y, x] = csim('step', t, sl, x0)
    G=ss2tf(sl)
    Captions=["θ₁"; "ω₁"; "θ₂"; "ω₂"; "θ₃"; "ω₃"];
//    bode(G, Captions);
    
    scf()
    plot(t, y(1, :), 'r:', 'LineWidth', 2)
    plot(t, y(3, :), 'g-', 'LineWidth', 2)
    plot(t, y(5, :), 'b-.', 'LineWidth', 2)
    xtitle("Simulação em malha aberta", "t (s)", "θᵢ (rad)")
    FormatCurrentAxes(["θ₁", "θ₂", "θ₃"])
    
    scf()
    plot(t, y(2, :), 'r:', 'LineWidth', 2)
    plot(t, y(4, :), 'g-', 'LineWidth', 2)
    plot(t, y(6, :), 'b-.', 'LineWidth', 2)
    xtitle("Simulação em malha aberta", "t (s)", "ωᵢ (rad/s)")
end

FormatCurrentAxes(["ω₁", "ω₂", "ω₃"])

// Matriz de transição:
Phi = expm(A * dt)

// Matriz dos termos forçantes:
Gamma = inv(A) * (Phi - eye(A))

// Simulação


x(:, 1) = x0
for i = 2:length(t);
    x(:, i) =  Phi * x(:, i - 1) + Gamma * B(:, 1)
end

//scf()
//plot(t, x(1, :), 'r:', 'LineWidth', 2)
//plot(t, x(3, :), 'g-', 'LineWidth', 2)
//plot(t, x(5, :), 'b-.', 'LineWidth', 2)
//xtitle("Variação de θ pela simulação em malha aberta", "t (s)", "θᵢ (rad)")
//FormatCurrentAxes(["θ₁", "θ₂", "θ₃"])
//
//scf()
//plot(t, x(2, :), 'r:', 'LineWidth', 2)
//plot(t, x(4, :), 'g-', 'LineWidth', 2)
//plot(t, x(6, :), 'b-.', 'LineWidth', 2)
//xtitle("Variação de ω pela simulação em malha aberta", "t (s)", "ωᵢ (rad/s)")
//FormatCurrentAxes(["ω₁", "ω₂", "ω₃"])
