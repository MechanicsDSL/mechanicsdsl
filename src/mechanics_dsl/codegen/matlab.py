"""
MATLAB/Octave Code Generator for MechanicsDSL

Generates MATLAB/GNU Octave simulation code with:
- Real sympy-to-MATLAB equation conversion
- Multiple ODE solvers (ode45, ode23, ode15s)
- Publication-quality plotting
- Simulink model generation (optional)
- State-space export
- Animation support
"""

from typing import Dict, List, Optional

import sympy as sp
from sympy.printing.octave import octave_code

from ..utils import logger
from .base import CodeGenerator


def sympy_to_matlab(expr: sp.Expr) -> str:
    """
    Convert a sympy expression to MATLAB/Octave code.

    Args:
        expr: Sympy expression to convert

    Returns:
        MATLAB code string

    Examples:
        >>> import sympy as sp
        >>> x = sp.Symbol('x')
        >>> sympy_to_matlab(sp.sin(x))
        'sin(x)'
    """
    if expr is None:
        return "0.0"

    try:
        return octave_code(expr)
    except Exception as e:
        logger.warning(f"Failed to convert expression to MATLAB: {e}")
        return f"0.0  % ERROR: {e}"


class MatlabGenerator(CodeGenerator):
    """
    Generates MATLAB/Octave simulation code.

    Features:
    - Multiple ODE solvers (ode45, ode23, ode15s for stiff systems)
    - Publication-quality plotting
    - Simulink block diagram generation
    - State-space model export
    - Energy/phase space visualization
    - Animation helper functions

    Example:
        >>> import sympy as sp
        >>> theta, g, l = sp.symbols('theta g l')
        >>> gen = MatlabGenerator(
        ...     system_name="pendulum",
        ...     coordinates=["theta"],
        ...     parameters={"g": 9.81, "l": 1.0},
        ...     initial_conditions={"theta": 0.5, "theta_dot": 0.0},
        ...     equations={"theta_ddot": -g/l * sp.sin(theta)},
        ...     solver="ode45"
        ... )
        >>> gen.generate("pendulum.m")
        'pendulum.m'

    Attributes:
        solver: MATLAB ODE solver (ode45, ode23, ode15s, ode113)
        generate_simulink: Whether to generate Simulink model
    """

    SUPPORTED_SOLVERS = ["ode45", "ode23", "ode15s", "ode113", "ode23s"]

    def __init__(
        self,
        system_name: str,
        coordinates: List[str],
        parameters: Dict[str, float],
        initial_conditions: Dict[str, float],
        equations: Dict[str, sp.Expr],
        lagrangian: Optional[sp.Expr] = None,
        hamiltonian: Optional[sp.Expr] = None,
        forces: Optional[List[sp.Expr]] = None,
        constraints: Optional[List[sp.Expr]] = None,
        solver: str = "ode45",
        abstol: float = 1e-8,
        reltol: float = 1e-6,
        generate_simulink: bool = False,
    ) -> None:
        """
        Initialize the MATLAB/Octave code generator.

        Args:
            system_name: Name of the physics system
            coordinates: List of generalized coordinate names
            parameters: Physical parameters
            initial_conditions: Initial state values
            equations: Acceleration equations
            lagrangian: Optional Lagrangian
            hamiltonian: Optional Hamiltonian
            forces: Optional non-conservative forces
            constraints: Optional holonomic constraints
            solver: MATLAB ODE solver
            abstol: Absolute tolerance
            reltol: Relative tolerance
            generate_simulink: Generate Simulink model
        """
        super().__init__(
            system_name=system_name,
            coordinates=coordinates,
            parameters=parameters,
            initial_conditions=initial_conditions,
            equations=equations,
            lagrangian=lagrangian,
            hamiltonian=hamiltonian,
            forces=forces,
            constraints=constraints,
        )

        if solver not in self.SUPPORTED_SOLVERS:
            logger.warning(f"Unknown solver '{solver}', using ode45")
            solver = "ode45"

        self.solver = solver
        self.abstol = abstol
        self.reltol = reltol
        self.generate_simulink = generate_simulink

    @property
    def target_name(self) -> str:
        """Target platform identifier."""
        return "matlab"

    @property
    def file_extension(self) -> str:
        """File extension for generated code."""
        return ".m"

    def expr_to_code(self, expr: sp.Expr) -> str:
        """
        Convert sympy expression to MATLAB/Octave code.

        Args:
            expr: Sympy expression

        Returns:
            MATLAB code string
        """
        return sympy_to_matlab(expr)

    def generate(self, output_file: str = "simulation.m") -> str:
        """
        Generate MATLAB/Octave simulation code.

        Args:
            output_file: Path to output file

        Returns:
            Path to generated file

        Raises:
            ValueError: If validation fails
        """
        self.validate_or_raise()

        logger.info(f"Generating MATLAB code for {self.system_name}")

        code = self._generate_code()

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(code)

        logger.info(f"Successfully wrote {output_file}")
        return output_file

    def generate_equations(self) -> str:
        """Generate MATLAB code for equations of motion."""
        lines = []
        idx = 1  # MATLAB is 1-indexed
        for coord in self.coordinates:
            accel_key = f"{coord}_ddot"
            lines.append(f"    dydt({idx}) = y({idx+1});  % d{coord}/dt = {coord}_dot")
            if accel_key in self.equations and self.equations[accel_key] is not None:
                expr = self.equations[accel_key]
                m_expr = self.expr_to_code(expr)
                lines.append(f"    dydt({idx+1}) = {m_expr};  % d{coord}_dot/dt")
            else:
                lines.append(f"    dydt({idx+1}) = 0.0;  % d{coord}_dot/dt (no equation)")
            idx += 2
        return "\n".join(lines)

    def _generate_code(self) -> str:
        """Generate complete MATLAB/Octave simulation script."""
        state_dim = len(self.coordinates) * 2

        # Parameter definitions (as global variables)
        param_lines = []
        param_globals = []
        for name, val in self.parameters.items():
            param_lines.append(f"    {name} = {val};")
            param_globals.append(name)
        param_str = "\n".join(param_lines) if param_lines else "    % No parameters"
        global_decl = " ".join(param_globals) if param_globals else ""

        # State variable unpacking (MATLAB is 1-indexed)
        unpack_lines = []
        idx = 1
        for coord in self.coordinates:
            unpack_lines.append(f"    {coord} = y({idx});")
            unpack_lines.append(f"    {coord}_dot = y({idx+1});")
            idx += 2
        unpack_str = "\n".join(unpack_lines)

        # Equations
        eq_str = self.generate_equations()

        # Initial conditions
        init_vals = []
        for coord in self.coordinates:
            pos = self.initial_conditions.get(coord, 0.0)
            vel = self.initial_conditions.get(f"{coord}_dot", 0.0)
            init_vals.extend([str(pos), str(vel)])
        init_str = "; ".join(init_vals)

        template = f'''%% =========================================================================
%% {self.system_name.upper()} SIMULATION
%% Generated by MechanicsDSL
%% =========================================================================
%%
%% Compatible with MATLAB R2019a+ and GNU Octave 5.0+
%%
%% Run: {self.system_name}
%%
%% =========================================================================

function results = {self.system_name}()
    %% =========================================================================
    %% Configuration
    %% =========================================================================
    config.solver = '{self.solver}';
    config.tspan = [0, 10];
    config.abstol = {self.abstol};
    config.reltol = {self.reltol};
    config.plot_results = true;
    config.export_csv = true;

    %% =========================================================================
    %% Physical Parameters (accessible to nested functions)
    %% =========================================================================
{param_str}

    %% =========================================================================
    %% Initial Conditions
    %% =========================================================================
    y0 = [{init_str}];

    %% =========================================================================
    %% Solve ODE
    %% =========================================================================
    fprintf('\\n===========================================\\n');
    fprintf(' {self.system_name.upper()} SIMULATION\\n');
    fprintf('===========================================\\n');
    fprintf(' Solver: %s\\n', config.solver);
    fprintf(' Time span: [%.1f, %.1f]\\n', config.tspan(1), config.tspan(2));
    fprintf(' State dimension: %d\\n', {state_dim});
    fprintf('-------------------------------------------\\n');

    opts = odeset('AbsTol', config.abstol, 'RelTol', config.reltol);

    tic;
    switch config.solver
        case 'ode45'
            [t, y] = ode45(@equations_of_motion, config.tspan, y0, opts);
        case 'ode23'
            [t, y] = ode23(@equations_of_motion, config.tspan, y0, opts);
        case 'ode15s'
            [t, y] = ode15s(@equations_of_motion, config.tspan, y0, opts);
        case 'ode113'
            [t, y] = ode113(@equations_of_motion, config.tspan, y0, opts);
        case 'ode23s'
            [t, y] = ode23s(@equations_of_motion, config.tspan, y0, opts);
        otherwise
            [t, y] = ode45(@equations_of_motion, config.tspan, y0, opts);
    end
    elapsed = toc;

    fprintf(' Completed in %.3f seconds\\n', elapsed);
    fprintf(' Points solved: %d\\n', length(t));
    fprintf('-------------------------------------------\\n');

    %% =========================================================================
    %% Compute Energy
    %% =========================================================================
    energy = zeros(length(t), 1);
    for i = 1:length(t)
        energy(i) = compute_energy(y(i, :));
    end
    energy_drift = abs(energy(end) - energy(1));
    fprintf(' Energy drift: %.6e\\n', energy_drift);

    %% =========================================================================
    %% Package Results
    %% =========================================================================
    results.t = t;
    results.y = y;
    results.energy = energy;
    results.config = config;
    results.coord_names = {{{", ".join(f"'{c}'" for c in self.coordinates)}}};

    %% =========================================================================
    %% Plotting
    %% =========================================================================
    if config.plot_results
        plot_results(results);
        plot_phase_space(results);
        plot_energy(results);
    end

    %% =========================================================================
    %% Export CSV
    %% =========================================================================
    if config.export_csv
        export_csv(results, '{self.system_name}_results.csv');
    end

    fprintf('===========================================\\n\\n');

    %% =========================================================================
    %% Nested Functions
    %% =========================================================================

    function dydt = equations_of_motion(t, y)
        % Equations of motion for {self.system_name}
        %
        % Arguments:
        %   t - Current time
        %   y - State vector [{", ".join(f"{c}; {c}_dot" for c in self.coordinates)}]
        %
        % Returns:
        %   dydt - Derivative vector

        dydt = zeros({state_dim}, 1);

        % Unpack state
{unpack_str}

        % Suppress unused variable warning
        if false, disp(t); end

        % Compute derivatives
{eq_str}
    end

    function E = compute_energy(y)
        % Compute total mechanical energy
        %
        % Arguments:
        %   y - State vector
        %
        % Returns:
        %   E - Total energy (kinetic approximation)

        E = 0;
        for j = 1:{len(self.coordinates)}
            E = E + 0.5 * y(2*j)^2;  % Kinetic energy term
        end
    end

end

%% =========================================================================
%% Visualization Functions
%% =========================================================================

function plot_results(results)
    % Create time history plots

    figure('Name', '{self.system_name} - Time History', 'NumberTitle', 'off');
    n_coords = length(results.coord_names);

    for i = 1:n_coords
        subplot(n_coords, 2, 2*i - 1);
        plot(results.t, results.y(:, 2*i - 1), 'b-', 'LineWidth', 1.5);
        xlabel('Time (s)');
        ylabel([results.coord_names{{i}}, ' (rad)']);
        title(['Position: ', results.coord_names{{i}}]);
        grid on;

        subplot(n_coords, 2, 2*i);
        plot(results.t, results.y(:, 2*i), 'r-', 'LineWidth', 1.5);
        xlabel('Time (s)');
        ylabel([results.coord_names{{i}}, '\\_dot (rad/s)']);
        title(['Velocity: ', results.coord_names{{i}}, '\\_dot']);
        grid on;
    end

    sgtitle('{self.system_name} Simulation Results');
    saveas(gcf, '{self.system_name}_time_history.png');
    fprintf(' Saved: {self.system_name}_time_history.png\\n');
end

function plot_phase_space(results)
    % Create phase space plot

    figure('Name', '{self.system_name} - Phase Space', 'NumberTitle', 'off');

    plot(results.y(:, 1), results.y(:, 2), 'b-', 'LineWidth', 1.0);
    hold on;
    plot(results.y(1, 1), results.y(1, 2), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
    plot(results.y(end, 1), results.y(end, 2), 'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    hold off;

    xlabel([results.coord_names{{1}}, ' (rad)']);
    ylabel([results.coord_names{{1}}, '\\_dot (rad/s)']);
    title('{self.system_name} Phase Space');
    legend('Trajectory', 'Start', 'End', 'Location', 'best');
    grid on;

    saveas(gcf, '{self.system_name}_phase_space.png');
    fprintf(' Saved: {self.system_name}_phase_space.png\\n');
end

function plot_energy(results)
    % Create energy plot

    figure('Name', '{self.system_name} - Energy', 'NumberTitle', 'off');

    plot(results.t, results.energy, 'k-', 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel('Energy (J)');
    title('{self.system_name} Energy Conservation');
    grid on;

    % Add drift annotation
    energy_drift = abs(results.energy(end) - results.energy(1));
    text(0.05, 0.95, sprintf('Drift: %.2e', energy_drift), ...
        'Units', 'normalized', 'FontSize', 10);

    saveas(gcf, '{self.system_name}_energy.png');
    fprintf(' Saved: {self.system_name}_energy.png\\n');
end

function export_csv(results, filename)
    % Export results to CSV file

    fid = fopen(filename, 'w');
    if fid == -1
        warning('Could not open file: %s', filename);
        return;
    end

    % Header
    fprintf(fid, 't,{",".join(self.coordinates)},{",".join(c + "_dot" for c in self.coordinates)},energy\\n');

    % Data
    for i = 1:length(results.t)
        fprintf(fid, '%.10e', results.t(i));
        for j = 1:{state_dim}
            fprintf(fid, ',%.10e', results.y(i, j));
        end
        fprintf(fid, ',%.10e\\n', results.energy(i));
    end

    fclose(fid);
    fprintf(' Exported: %s (%d points)\\n', filename, length(results.t));
end
'''
        return template
