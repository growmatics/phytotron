"""Physics-only economic MPC on the GreenLight model (benchmark arm 3).

Direct-collocation nonlinear MPC on a **state-scaled** GreenLight ODE (scaling is
what makes the stiff 28-state model solvable in an optimizer). The module depends
only on the ODE from ``gl_gym`` (imported lazily). It is validated in layers: unit
tests in ``tests/test_mpc.py`` (L1), ``examples/validate_mpc.py`` (L2 model fidelity,
L3 solver status), and ``examples/l5_reproduce.py`` (L5, the published numbers of
van Laatum et al.). Results are in ``docs/validation.md``.

Inverse-crime boundary: it holds only a nominal model, never the simulator's
ground-truth parameters (enforced by ``PhysicsController``).

The stable-solver recipe (L-BFGS Hessian, relaxed IPOPT tolerances, L1 soft-bound
penalties, and a 10% input-rate constraint on the state-scaled collocation model)
follows van Laatum et al. [vanlaatum2026rlmpc]. Cold-start hardening (a warm control
history, a safe hold-last fallback, and strictly-positive CO2/vapour/biomass box bounds
that avoid the ODE's photosynthesis/LAI singularities at zero) and a solve-speed pass
(a build-once parametric Opti NLP [andersson2019casadi], control warm-start, capped
iterations with acceptable-point termination [wachter2006ipopt], a salvaged last
iterate, an MX->SX ``expand``, and Radau degree 2, benchmarked against degree 3 for
<1% plant fidelity) bring the per-step solve to about 1 s median or less.

The full recipe, its provenance, and the references behind the bracketed keys are in
``docs/mpc.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from phytotron.config import BenchmarkConfig, EconomicConfig
from phytotron.controllers.base import (
    ControlContext,
    NominalParameters,
    PhysicsController,
)
from phytotron.models.outputs import outputs_from_state

# 28-state scale: temps/soil ~1e1, CO2/vapour ~1e3, biomass ~1e5, tCanSum ~1e3, time 1.
STATE_SCALE: NDArray[np.float64] = np.array(
    [1e3, 1e3] + [1e1] * 13 + [1e3, 1e3] + [1e1] * 5 + [1e5, 1e5, 1e5, 1e5, 1e3, 1e0],
    dtype=float,
)

# Fallback economic prices (mirror benchmark.yaml economic block / GLG2 GreenhouseReward).
# The live benchmark value comes from the economic config (benchmark.yaml); see __init__.
DEFAULT_PRICES = {"fruit": 1.6, "heat": 0.09, "elec": 0.3, "co2": 0.3}

# Warm control history (heating + CO2 on): a cold season start needs heating, and
# the rate limit is applied relative to this so the MPC does not start from u=0.
U_INIT: NDArray[np.float64] = np.array([0.5, 0.3, 0.0, 0.0, 0.0, 0.0])

# State indices used by the economic objective / soft bounds.
_I_CO2, _I_TAIR, _I_CFRUIT, _I_TIME = 0, 2, 25, 27
_TEMP_IDX = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21)

# Tolerance for accepting a salvaged (non-clean) IPOPT iterate's controls as in-bounds.
_U_TOL = 1e-6

# Characteristic scales that normalise the economic penalty's climate outputs to O(1)
# (raw CO2 ppm ~1000, RH ~100 %, air temp ~10 degC) so the soft-bound constraints do
# not wreck the KKT conditioning. Penalty value is preserved (scale folds into weight).
_CO2_SCALE, _TEMP_SCALE, _RH_SCALE = 1000.0, 10.0, 100.0

# Van Laatum et al. [vanlaatum2026rlmpc] penalise RH from 2 %-points below its ceiling
# (their RL-SMPC code: P >= w * (rh - (rh_max - 2))) -- a small margin that keeps the MPC
# off the humidity bound. Applied to the RH upper bound only.
_RH_UPPER_MARGIN = 2.0

# Scaled-state indices the GreenLight ODE DIVIDES by: CO2 air/top (0, 1), vapour
# pressure air/top (15, 16), and live leaf/stem/fruit biomass (23, 24, 25). IPOPT
# evaluates the model at infeasible trial points during line search and restoration
# (it has no "stay in the domain" mode), so a bound alone does not prevent the ODE from
# seeing a non-positive divisor and returning NaN -- which is fatal. We therefore extend
# the vector field to be FINITE and smooth everywhere by flooring these states with
# sqrt(z^2 + eps^2) before the ODE: a differentiable surrogate outside the physical box
# that equals the state to O(eps^2) inside it (eps << the box floors, so the feasible
# region is unchanged). Standard interior-point practice for domain-restricted models.
_SINGULAR_IDX = (0, 1, 15, 16, 23, 24, 25)
_DOMAIN_EPS = 1e-3  # scaled units


def _domain_safe(z: Any, nx: int) -> Any:
    """Smoothly floor the ODE's divisor states so the vector field is finite everywhere."""
    import casadi as ca

    return ca.vertcat(
        *[
            ca.sqrt(z[i] * z[i] + _DOMAIN_EPS * _DOMAIN_EPS) if i in _SINGULAR_IDX else z[i]
            for i in range(nx)
        ]
    )


def _state_box(nx: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Loose conditioning box, EXCEPT temperatures, which get a physical-generous bound.

    ``_domain_safe`` makes the divisor states (CO2/vapour/biomass) finite everywhere, so
    those stay loose -- a *tight* physical bound (e.g. a CO2 upper limit that dosing
    legitimately crosses) makes the OCP infeasible, since IPOPT never violates a variable
    bound. But the ODE's temperature-dependent terms (exp/satVp in the aux states) still
    blow up at *extreme* temperatures, and a uniformly-loose box lets an iterate reach
    -500 degC and NaN the model. So temperatures keep a generous physical bound
    [-40, 80] degC (never binds; real greenhouse temp is -20..60), while every other state
    stays loose. The physical climate range is still enforced softly in the penalty.
    """
    lb = np.full(nx, -50.0)
    ub = np.full(nx, 1e4)
    for i in _TEMP_IDX:  # temperatures: [-40, 80] degC (scale 1e1); avoids the ODE NaN
        lb[i], ub[i] = -4.0, 8.0
    lb[_I_TIME], ub[_I_TIME] = -1.0, 1e6
    return lb, ub


def collocation_integrator(ode: dict[str, Any], dt: float, degree: int) -> Any:
    """CasADi's collocation integrator over ONE finite element of Radau degree ``degree``.

    This is the same discretisation the OCP in :meth:`PhysicsOnlyMPC._build` writes out
    by hand with :func:`radau_coefficients` (one Radau IIA element per control interval),
    so a prediction made with it is a prediction of the model the MPC plans with.
    ``tests/test_mpc.py`` checks the two constructions agree on a known ODE.
    """
    import casadi as ca

    options = {
        "number_of_finite_elements": 1,
        "collocation_scheme": "radau",
        "interpolation_order": degree,
    }
    return ca.integrator("model", "collocation", ode, 0.0, dt, options)


def radau_coefficients(degree: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Radau collocation differentiation matrix ``C`` and end-weights ``D`` (pure)."""
    import casadi as ca

    tau = [0.0, *ca.collocation_points(degree, "radau")]
    coll = np.zeros((degree + 1, degree + 1))
    end = np.zeros(degree + 1)
    for j in range(degree + 1):
        poly = np.poly1d([1.0])
        for r in range(degree + 1):
            if r != j:
                poly = poly * np.poly1d([1.0, -tau[r]]) / (tau[j] - tau[r])
        end[j] = poly(1.0)
        dpoly = poly.deriv()
        for r in range(degree + 1):
            coll[j, r] = dpoly(tau[r])
    return coll, end


OBJECTIVES = ("economic", "energy_band")


@dataclass(frozen=True)
class MpcSettings:
    """Everything the MPC is built with, except its nominal model.

    ``objective`` has no default: every construction says whether it runs the
    benchmark's economic objective (the GreenLight-Gym2 reward in EPI form: fruit
    revenue minus operating costs minus CO2, temperature and RH penalties, the objective
    that reproduces the study) or the ``energy_band`` placeholder (a modest energy cost
    plus a soft air-temperature band, kept for solver diagnosis). The benchmark builds
    its MPC with :meth:`from_config`. The solver and discretisation choices are
    explained in docs/mpc.md.
    """

    objective: str
    horizon: int = 6  # control intervals
    degree: int = 2  # Radau collocation degree per interval
    dt: float = 900.0  # control interval [s]
    prices: Mapping[str, float] | None = (
        None  # energy_band only; economic prices come from economic
    )
    temp_bounds: tuple[float, float] = (15.0, 34.0)  # energy_band: the soft band [C]
    w_energy: float = 1.0  # energy_band weights: the band penalty must dominate the energy cost
    w_temp: float = 50.0
    w_fruit: float = 0.0
    delta_u_max: float = 0.1  # largest control change per interval
    u_init: tuple[float, ...] | None = None  # None: U_INIT
    state_scale: tuple[float, ...] | None = None  # None: STATE_SCALE
    linear_solver: str = "mumps"  # the one shipped with the CasADi wheel; deterministic
    expand: bool = True  # MX to SX for faster evaluation
    jit: bool = False  # compile the NLP functions to C once
    economic: EconomicConfig | None = None
    climate_bounds: Mapping[str, tuple[float, float]] | None = None
    max_iter: int = 150  # IPOPT iteration cap
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {self.objective!r}")
        if self.objective == "economic" and (self.economic is None or self.climate_bounds is None):
            raise ValueError("objective='economic' requires economic= and climate_bounds=")

    @classmethod
    def from_config(cls, cfg: BenchmarkConfig) -> MpcSettings:
        """The benchmark's MPC: the economic objective with the config's horizon, step,
        prices and simulator band."""
        if cfg.mpc is None or cfg.economic is None:
            raise ValueError("configs/benchmark.yaml needs the mpc and economic blocks")
        return cls(
            objective="economic",
            horizon=cfg.mpc.horizon_steps,
            dt=float(cfg.episode.timestep_seconds),
            economic=cfg.economic,
            climate_bounds=cfg.constraints.bounds(),
        )


class PhysicsOnlyMPC(PhysicsController):
    """Direct-collocation economic MPC over the GreenLight ODE, built from
    :class:`MpcSettings` and a nominal model (never the simulator's own parameters)."""

    control_kind = "absolute"  # outputs absolute controls -> harness uses step_raw_control

    def __init__(self, nominal_params: NominalParameters, settings: MpcSettings) -> None:
        super().__init__(nominal_params)
        self.settings = settings
        self.objective = settings.objective
        self.economic = settings.economic
        self.climate_bounds = settings.climate_bounds
        self.max_iter = settings.max_iter  # IPOPT iteration cap (bounds worst-case latency)
        self.verbose = settings.verbose  # IPOPT print_level 5 for solver diagnosis
        self.horizon = settings.horizon
        self.degree = settings.degree
        self.dt = settings.dt
        self.linear_solver = settings.linear_solver
        self.expand = settings.expand
        self.jit = settings.jit
        # Prices are the single source of truth in benchmark.yaml, reaching here via the
        # economic config for the economic objective (the benchmark path). DEFAULT_PRICES is
        # only a fallback for the energy_band objective, never the live benchmark value.
        if settings.prices is not None:
            self.prices = dict(settings.prices)
        elif self.economic is not None:
            self.prices = {
                "fruit": self.economic.fruit_price,
                "heat": self.economic.heating_price,
                "elec": self.economic.elec_price,
                "co2": self.economic.co2_price,
            }
        else:
            self.prices = dict(DEFAULT_PRICES)
        self.temp_bounds = settings.temp_bounds
        self.w_energy = settings.w_energy
        self.w_temp = settings.w_temp
        self.w_fruit = settings.w_fruit
        self.delta_u_max = settings.delta_u_max
        self._u_init = (
            np.asarray(settings.u_init, dtype=float)
            if settings.u_init is not None
            else U_INIT.copy()
        )
        self._u_prev = self._u_init.copy()
        self.scale = np.asarray(
            STATE_SCALE if settings.state_scale is None else settings.state_scale, dtype=float
        )
        self._params_vec = np.asarray(list(nominal_params.values.values()), dtype=float)
        self._warm: dict | None = None
        self._C, self._D = radau_coefficients(settings.degree)
        # The parametric NLP is built once (lazily, on the first solve) and reused:
        # rebuilding the symbolic ODE and collocation graph every step is the dominant
        # cost. State, previous control and the forecast are parameters (docs/mpc.md).
        self._opti: Any = None  # ca.Opti, built lazily; casadi is stubless
        self._built_nd: int | None = None
        self._x_nodes: list = []  # interval-end state variables, for prediction readout
        self._model: Any = None  # the one-interval prediction function, built on first use
        self._last_pred: dict | None = None  # {"x": [...unscaled states...], "u": [...]}
        # Per-solve outcome tally: ``clean`` = IPOPT converged; ``salvage`` = non-optimal
        # but its last iterate was usable; ``hold`` = unusable, the last control repeated.
        # The benchmark reports solve success = (clean + salvage) / total per arm.
        self.solve_stats: dict[str, int] = {"clean": 0, "salvage": 0, "hold": 0}

    def reset(self, seed: int = 0) -> None:
        """Clear the warm-start and control history (start of a season).

        The compiled NLP is kept: its structure is season-independent, only the
        warm-start guess and control history reset.
        """
        self._warm = None
        self._u_prev = self._u_init.copy()
        self.solve_stats = {"clean": 0, "salvage": 0, "hold": 0}

    def act(self, ctx: ControlContext) -> NDArray[np.float64]:  # pragma: no cover - integration
        """Controller-protocol adapter: plan from the full state and weather forecast.

        The physics arm ignores ``ctx.obs`` (lossy) and ``ctx.env``; it needs the full
        state and the disturbance forecast, which is what ``solve`` consumes.
        """
        return self.solve(ctx.state, ctx.forecast)

    def model_step(  # pragma: no cover - integration (needs gl_gym)
        self, x: NDArray[np.float64], u: NDArray[np.float64], d: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """One-interval prediction with the model the MPC plans with.

        The same scaled, domain-safe ODE and the same discretisation as the OCP in
        :meth:`_build`: one Radau element of ``degree`` per ``dt`` interval, with the
        nominal parameters baked in. Used by ``examples/validate_mpc.py`` to measure the
        prediction error against the plant (validation layer 2); a finer integrator
        here would measure a model the controller never uses.
        """
        import casadi as ca

        s = self.scale
        if self._model is None:
            from gl_gym.environments.models.ode import ODE

            nx = s.size
            z = ca.SX.sym("z", nx)
            u_sym = ca.SX.sym("u", 6)
            d_sym = ca.SX.sym("d", np.asarray(d).size)
            ode = {
                "x": z,
                "u": u_sym,
                "p": d_sym,
                "ode": ODE(ca.DM(s) * _domain_safe(z, nx), u_sym, d_sym, ca.DM(self._params_vec))
                / ca.DM(s),
            }
            self._model = collocation_integrator(ode, self.dt, self.degree)
        z_next = self._model(x0=ca.DM(np.asarray(x, dtype=float) / s), u=ca.DM(u), p=ca.DM(d))
        return np.asarray(z_next["xf"], dtype=float).flatten() * s

    def _economic_stage_cost(  # pragma: no cover - integration (needs casadi Opti)
        self, opti: Any, uk: Any, xn: Any, s: NDArray[np.float64]
    ) -> Any:
        """One-interval economic cost: operating costs + CO2/temp/RH soft penalties.

        Matches GLG2 ``GreenhouseReward``: heating/CO2/lamp costs via the GreenLight
        ``p[]`` conversions, and ``lambda . violations`` on the *unscaled* climate
        outputs (CO2 ppm, air temperature, RH) against ``climate_bounds``. Slacks keep
        the penalty differentiable. Returns the stage cost to be minimised.
        """
        import casadi as ca

        ec = self.economic
        bounds = self.climate_bounds
        assert ec is not None and bounds is not None  # guaranteed by __init__
        p = self._params_vec
        pi = ec.param_indices
        h_idx, c_idx, l_idx = ec.cost_input_indices
        dt_kwh = self.dt / 3600 * 1e-3  # W/aFlr -> kWh/m2 over one interval
        # operating costs over one dt interval (per GLG2 GreenhouseReward conversions)
        heat = uk[h_idx] * p[pi["heat_power"]] / p[pi["floor_area"]] * dt_kwh * ec.heating_price
        elec = uk[l_idx] * p[pi["lamp_power"]] * dt_kwh * ec.elec_price
        co2c = (
            uk[c_idx] * p[pi["co2_capacity"]] / p[pi["floor_area"]] * self.dt * 1e-6 * ec.co2_price
        )
        stage = heat + elec + co2c
        # output soft-bound penalties on the unscaled outputs. Each output is
        # normalised by a characteristic scale so its soft-bound constraint is O(1)
        # (raw ppm ~1000, RH ~100, temp ~10 otherwise wreck the KKT conditioning and
        # blow up IPOPT's duals). The penalty VALUE is unchanged: the slack is now in
        # normalised units, so the effective weight absorbs the scale
        # (weight*char*slack == lambda*raw_violation).
        co2_ppm, temp, rh = outputs_from_state(xn * ca.DM(s))
        lam = ec.penalty_weights  # order [CO2, temp, RH]
        rh_lo, rh_hi = bounds["rh_percent"]
        for y, (lo, hi), w, char in (
            (co2_ppm, bounds["co2_ppm"], lam[0], _CO2_SCALE),
            (temp, bounds["air_temp_c"], lam[1], _TEMP_SCALE),
            (rh, (rh_lo, rh_hi - _RH_UPPER_MARGIN), lam[2], _RH_SCALE),  # van Laatum RH margin
        ):
            s_lo = opti.variable()
            s_hi = opti.variable()
            opti.subject_to(s_lo >= 0)
            opti.subject_to(s_hi >= 0)
            opti.subject_to(y / char >= lo / char - s_lo)
            opti.subject_to(y / char <= hi / char + s_hi)
            stage += (w * char) * (s_lo + s_hi)
        return stage

    def _build(self, nd: int) -> None:  # pragma: no cover - integration (needs gl_gym)
        """Construct the parametric collocation NLP once (reused every step).

        Initial state, previous control, and the per-interval forecast are Opti
        *parameters*, so a step changes only their values, not the symbolic graph.
        The nominal parameters are baked in as constants (they never change). The order
        in which variables are created is part of the problem IPOPT sees and must not
        change: control, collocation states and the interval-end state, then the stage
        cost's slacks, interval by interval.
        """
        import casadi as ca

        s = self.scale
        nx, nu = s.size, 6
        f_scaled = self._dynamics(nd)
        opti = ca.Opti()
        p_x0 = opti.parameter(nx)  # initial state (scaled)
        p_uprev = opti.parameter(nu)  # control applied before this horizon
        p_d = [opti.parameter(nd) for _ in range(self.horizon)]  # forecast per interval
        cost = 0.0
        controls: list = []
        seed_states: list = []  # every state variable, seeded to x0 on a cold start
        x_nodes: list = []  # interval-end states, for the prediction readout
        xk = p_x0  # stage-0 state is the parameter itself (no extra variable)
        prev_u = p_uprev
        for k in range(self.horizon):
            uk = opti.variable(nu)
            opti.subject_to(opti.bounded(0, uk, 1))
            opti.subject_to(opti.bounded(-self.delta_u_max, uk - prev_u, self.delta_u_max))
            prev_u = uk
            controls.append(uk)
            xn, states = self._collocation_constraints(opti, f_scaled, xk, uk, p_d[k])
            seed_states.extend(states)
            x_nodes.append(xn)
            cost += self._stage_cost(opti, uk, xn)
            xk = xn
        cost += self._crop_reward(p_x0, xk)
        opti.minimize(cost)
        opti.solver("ipopt", self._solver_options())
        self._opti = opti
        self._p_x0, self._p_uprev, self._p_d = p_x0, p_uprev, p_d
        self._u_first = controls[0]
        self._controls = controls
        self._seed_states = seed_states
        self._x_nodes = x_nodes
        self._built_nd = nd

    def _dynamics(self, nd: int) -> Any:  # pragma: no cover - integration (needs gl_gym)
        """The scaled, domain-safe GreenLight ODE as a CasADi function of (z, u, d)."""
        import casadi as ca
        from gl_gym.environments.models.ode import ODE

        s = self.scale
        nx = s.size
        z_sym = ca.SX.sym("z", nx)
        u_sym = ca.SX.sym("u", 6)
        d_sym = ca.SX.sym("d", nd)
        return ca.Function(
            "f",
            [z_sym, u_sym, d_sym],
            [
                ODE(ca.DM(s) * _domain_safe(z_sym, nx), u_sym, d_sym, ca.DM(self._params_vec))
                / ca.DM(s)
            ],
        )

    def _collocation_constraints(  # pragma: no cover - integration (needs casadi Opti)
        self, opti: Any, f_scaled: Any, xk: Any, uk: Any, dk: Any
    ) -> tuple[Any, list[Any]]:
        """One interval of Radau collocation: the collocation states, their equations and
        the interval-end state. Returns the end state and every state variable created."""
        import casadi as ca

        nx = self.scale.size
        lb, ub = _state_box(nx)
        x_end = self._D[0] * xk
        coll = [opti.variable(nx) for _ in range(self.degree)]
        for j in range(1, self.degree + 1):
            opti.subject_to(opti.bounded(ca.DM(lb), coll[j - 1], ca.DM(ub)))
            xp = self._C[0, j] * xk
            for r in range(1, self.degree + 1):
                xp = xp + self._C[r, j] * coll[r - 1]
            opti.subject_to(self.dt * f_scaled(coll[j - 1], uk, dk) == xp)
            x_end = x_end + self._D[j] * coll[j - 1]
        xn = opti.variable(nx)
        opti.subject_to(xn == x_end)
        return xn, [*coll, xn]

    def _stage_cost(self, opti: Any, uk: Any, xn: Any) -> Any:  # pragma: no cover - integration
        """One interval's cost under the chosen objective."""
        s = self.scale
        if self.objective == "economic":
            return self._economic_stage_cost(opti, uk, xn, s)
        # placeholder: modest energy cost + a smooth soft air-temperature band via slacks
        # (differentiable, unlike fmax(0,.)^2).
        t_air = xn[_I_TAIR] * s[_I_TAIR]
        s_lo = opti.variable()
        s_hi = opti.variable()
        opti.subject_to(s_lo >= 0)
        opti.subject_to(s_hi >= 0)
        opti.subject_to(t_air >= self.temp_bounds[0] - s_lo)
        opti.subject_to(t_air <= self.temp_bounds[1] + s_hi)
        return self.w_energy * (uk[0] + uk[1] + uk[4]) + self.w_temp * (s_lo + s_hi)

    def _crop_reward(self, p_x0: Any, x_last: Any) -> Any:  # pragma: no cover - integration
        """Minus the fruit grown over the horizon (a telescoping sum of fruit growth): in
        the economic objective at the per-kg-dry-weight price (the GreenhouseReward
        conversion, mg dry matter to EUR), in the placeholder with ``w_fruit``."""
        s = self.scale
        cfruit_gain = x_last[_I_CFRUIT] * s[_I_CFRUIT] - p_x0[_I_CFRUIT] * s[_I_CFRUIT]
        if self.objective == "economic":
            assert self.economic is not None  # guaranteed by MpcSettings
            return -1e-6 * self.economic.price_per_kg_dw * cfruit_gain
        return -self.w_fruit * cfruit_gain

    def _solver_options(self) -> dict[str, Any]:
        """IPOPT options: the recipe that makes the stiff model tractable (docs/mpc.md).

        A quasi-Newton (L-BFGS) Hessian avoids the exact second-order ODE sensitivity
        that fails; relaxed tolerances stop at an approximately feasible, approximately
        optimal point; an iteration cap with a short acceptable-point window bounds the
        worst-case latency; a wall-clock cap lets the salvage or hold fallback take over on
        a pathological state. MUMPS is the one linear solver in the CasADi wheel and is
        deterministic.
        """
        options: dict[str, Any] = {"expand": self.expand}  # MX -> SX for faster evaluation
        if self.jit:  # compile the NLP functions to C once; amortised over the season
            options["jit"] = True
            options["compiler"] = "shell"
            options["jit_options"] = {"flags": ["-O2"], "verbose": False}
        return {
            **options,
            "ipopt.print_level": 5 if self.verbose else 0,
            "print_time": 0,
            "ipopt.linear_solver": self.linear_solver,
            "ipopt.max_iter": self.max_iter,
            "ipopt.acceptable_iter": 5,
            "ipopt.max_cpu_time": 30.0,
            "ipopt.hessian_approximation": "limited-memory",
            "ipopt.tol": 0.1,
            "ipopt.acceptable_tol": 0.1,
            "ipopt.constr_viol_tol": 1e-2,
            "ipopt.acceptable_constr_viol_tol": 0.1,
        }

    def solve(  # pragma: no cover - integration (needs gl_gym + solver)
        self,
        x: NDArray[np.float64],
        weather: NDArray[np.float64],
        state_guess: list[NDArray[np.float64]] | None = None,
    ) -> NDArray[np.float64]:
        """Return the first control ``u0`` in [0,1]^6 for state ``x`` and a weather
        forecast ``weather`` of shape ``(>=horizon, nd)`` (disturbances per interval).

        ``state_guess`` (an optional list of ``horizon`` unscaled states, one per interval
        end) warm-starts the collocation states from a physically consistent forward
        trajectory (a learned controller could pass its rollout's states). Defaults to
        seeding every state to ``x0``, the cold-start guess that rarely fails; a shifted
        previous-solve seed backfired on the stiff solves, but a fresh forward rollout
        from the current ``x0`` does not.
        """
        s = self.scale
        nu = 6
        nd = int(weather.shape[1])
        if self._opti is None or self._built_nd != nd:
            self._build(nd)
        opti = self._opti
        z0 = np.asarray(x, dtype=float) / s
        opti.set_value(self._p_x0, z0)
        opti.set_value(self._p_uprev, self._u_prev)
        for k in range(self.horizon):
            opti.set_value(self._p_d[k], weather[k])
        # Seed the states fresh from the current x0 every step (they shift too much
        # between steps to reuse) and warm-start only the controls from the previous
        # solve (they persist well). Reusing the full primal, including states, made
        # the stiff cold-region solves fail; controls-only is both faster and rarely fails.
        if state_guess is not None:
            # Warm-start each interval's states (degree collocation points + the interval
            # node) from the forward-rollout state at that interval's end; fall back to x0
            # for any interval past the guess length.
            per = self.degree + 1
            for k in range(self.horizon):
                zk = np.asarray(state_guess[k], dtype=float) / s if k < len(state_guess) else z0
                for v in self._seed_states[k * per : (k + 1) * per]:
                    opti.set_initial(v, zk)
        else:
            for v in self._seed_states:
                opti.set_initial(v, z0)
        warm_u = self._warm["u"] if self._warm is not None else None
        for k, uk in enumerate(self._controls):
            opti.set_initial(uk, warm_u[k] if warm_u is not None else 0.3)
        try:
            sol = opti.solve()
        except RuntimeError:
            # IPOPT stopped non-optimal (max_iter / a stiff-ODE corner). Its last
            # iterate (via CasADi's Opti.debug, Andersson et al. 2019
            # [andersson2019casadi]) is usually a finite, in-bounds *approximate*
            # optimum under the relaxed tolerances, so prefer it (an honest approximate
            # solve) over the crude hold-last-control; hold only if it is unusable.
            u_try = np.asarray(opti.debug.value(self._u_first), dtype=float).reshape(nu)
            if (
                np.all(np.isfinite(u_try))
                and np.all(u_try >= -_U_TOL)
                and np.all(u_try <= 1 + _U_TOL)
            ):
                u0 = np.clip(u_try, 0.0, 1.0)
                self._u_prev = u0
                self._warm = None  # drop the warm start after a non-clean solve
                self.solve_stats["salvage"] += 1
                return u0
            self._warm = None
            self.solve_stats["hold"] += 1
            return np.asarray(self._u_prev, dtype=float).reshape(nu)
        u0 = np.asarray(sol.value(self._u_first), dtype=float).reshape(nu)
        self._u_prev = u0
        self.solve_stats["clean"] += 1
        controls_pred = [np.asarray(sol.value(c), dtype=float).reshape(nu) for c in self._controls]
        self._warm = {"u": controls_pred}
        # cache the predicted (unscaled) state trajectory and controls for diagnosis
        self._last_pred = {
            "x": [np.asarray(sol.value(xn), dtype=float).flatten() * s for xn in self._x_nodes],
            "u": controls_pred,
        }
        return u0
