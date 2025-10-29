# simplified_solar_search.py

import heapq
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import logging
from engines.advanced_roi_calculator import FinancialParameters
import math

# Import TimeHorizon if available from the correct module
try:
    from engines.advanced_roi_calculator import TimeHorizon
except ImportError:
    # Fallback: define a minimal stub if not available
    class TimeHorizon:
        LONG_TERM = "long_term"

log = logging.getLogger(__name__)

# reuse Decision enum from your file or re-declare small helper:
from enum import Enum
class Decision(Enum):
    INSTALL_NOW = "install_now"
    WAIT_3_MONTHS = "wait_3_months"
    WAIT_6_MONTHS = "wait_6_months"
    WAIT_12_MONTHS = "wait_12_months"

# Minimal state dataclass used by search
@dataclass(frozen=True)
class TimeState:
    months_waited: int
    tariff: float
    tech_cost_per_kw: float
    budget: float

    def __hash__(self):
        # coarse hashing to avoid huge closed sets
        return hash((int(self.months_waited), int(self.tariff * 100), int(self.tech_cost_per_kw)))

@dataclass
class SearchNode:
    state: TimeState
    g_cost: float                # accumulated opportunity cost (lost savings) so far
    heuristic: float             # estimated "cost" to reach preferred goal (lower payback preferred)
    decision_path: List[Decision]
    payback: Optional[float] = None
    coverage_pct: Optional[float] = None

    @property
    def f(self):
        return self.g_cost + self.heuristic

    def __lt__(self, other):
        return self.f < other.f

class TimeAStarSolarSearch:
    """
    A* search that explores 'when to install' (time dimension only).
    It queries AdvancedROICalculator for each candidate install time to get payback/coverage.
    Objective: find earliest install time that minimizes payback, while ensuring coverage >= 100% where possible.
    """

    def __init__(self, roi_calculator, max_months: int = 36):
        """
        roi_calculator: instance of AdvancedROICalculator (or wrapper) with method calculate_comprehensive_roi(...)
        max_months: how far into future to consider (default 3 years)
        """
        self.roi_calculator = roi_calculator
        self.max_months = max_months

    def _months_to_action(self, months: int) -> Decision:
        if months == 0:
            return Decision.INSTALL_NOW
        if months <= 3:
            return Decision.WAIT_3_MONTHS
        if months <= 6:
            return Decision.WAIT_6_MONTHS
        return Decision.WAIT_12_MONTHS

    def search(
        self,
        user_budget: float,
        current_tariff: float,
        required_capacity_kw: float,
        current_tech_cost_per_kw: float,
        monthly_consumption_kwh: float,
        # dynamic rates from pipeline:
        tariff_growth_rate: float,
        tech_decline_rate: float,
        # environment objects required by AdvancedROICalculator:
        system_spec_template,   # SolarSystemSpec with fields to override capacity
        location_data,          # LocationData instance
        financial_params_template,  # FinancialParameters template to override tariff/costs
        risk_params             # RiskParameters used in Monte Carlo
    ) -> Dict:
        """
        Returns dict: recommended decision, months_to_wait, payback, coverage, metadata
        """

        # helper to compute tariff and tech cost after m months (months as integer)
        def project_rates(months: int):
            years = months / 12.0
            proj_tariff = current_tariff * ((1 + tariff_growth_rate) ** years)
            proj_tech_cost = current_tech_cost_per_kw * ((1 - tech_decline_rate) ** years)
            return proj_tariff, proj_tech_cost

        # helper to call ROI calculator and return payback + coverage
        def evaluate_install_at(months:int):
            proj_tariff, proj_tech_cost = project_rates(months)

            # build system_spec for this evaluation
            system_spec = system_spec_template
            system_spec.capacity_kw = required_capacity_kw

            # financial params clone and override
            fin = FinancialParameters(**financial_params_template.__dict__)
            fin.system_cost_per_kw = proj_tech_cost  # dynamic cost per kW
            fin.electricity_tariff = proj_tariff
            # keep net_metering, subsidy etc same

            # call ROI calculator (synchronous). It returns ROIResults object
            try:
                roi_results = self.roi_calculator.calculate_comprehensive_roi(
                    system_spec=system_spec,
                    location_data=location_data,
                    financial_params=fin,
                    risk_params=risk_params,
                    time_horizon=TimeHorizon.LONG_TERM  # use long-term payback analysis
                )
            except Exception as e:
                log.exception("ROI calculator failed during A* evaluation")
                # gracefully degrade: estimate payback using simple formula
                annual_generation = required_capacity_kw * 1400
                annual_savings = min(annual_generation, monthly_consumption_kwh * 12) * proj_tariff
                system_cost_est = required_capacity_kw * proj_tech_cost
                payback_est = system_cost_est / annual_savings if annual_savings > 0 else float('inf')
                coverage = (required_capacity_kw * 1400) / (monthly_consumption_kwh * 12) * 100
                return payback_est, coverage, {'fallback': True}

            # Use payback and coverage from ROIResults
            payback = roi_results.payback_period
            # estimated coverage (month-based generation / consumption)
            annual_generation = required_capacity_kw * (location_data.annual_irradiance * 0.001) * (system_spec.panel_efficiency/20.0) * 0.85
            # But to be robust, use roi_results.energy_production if available
            try:
                year1_gen = roi_results.energy_production[0]
                coverage_pct = (year1_gen) / (monthly_consumption_kwh * 12) * 100
            except Exception:
                coverage_pct = (required_capacity_kw * 1400) / (monthly_consumption_kwh * 12) * 100

            return payback, coverage_pct, {'fallback': False, 'roi_results': roi_results}

        # Start A* over discrete month increments: 0,3,6,9,... up to max_months
        step_choices = [0, 3, 6, 12]  # allowed waiting steps (in months)
        # We'll expand nodes by incrementing these steps (additive)
        start_state = TimeState(months_waited=0, tariff=current_tariff,
                                tech_cost_per_kw=current_tech_cost_per_kw, budget=user_budget)

        start_node = SearchNode(state=start_state, g_cost=0.0,
                                heuristic=self._heuristic_payback_estimate(0, current_tariff, current_tech_cost_per_kw, required_capacity_kw, monthly_consumption_kwh, tariff_growth_rate, tech_decline_rate),
                                decision_path=[])
        open_heap = [start_node]
        closed = set()
        best_solution = None
        iterations = 0
        max_iterations = 200

        while open_heap and iterations < max_iterations:
            node = heapq.heappop(open_heap)
            iterations += 1
            if node.state in closed:
                continue
            closed.add(node.state)

            months = node.state.months_waited

            # If we choose to install at this state (i.e., evaluate install now)
            payback, coverage, meta = evaluate_install_at(months)
            node.payback = payback
            node.coverage_pct = coverage

            # Goal condition: coverage >= 100% (prefer) and payback is finite and small
            # We'll accept a solution if coverage >= 100% AND payback is finite
            if coverage >= 100.0 and math.isfinite(payback):
                # Keep best (prefer smaller payback; tie-breaker earlier install)
                if not best_solution or (payback < best_solution.payback) or (abs(payback - best_solution.payback) < 1e-6 and months < best_solution.state.months_waited):
                    best_solution = node
                    # we can break early if payback is good enough (e.g., <6 years) — but we prefer to keep exploring a bit to ensure not missing better
                    if payback <= 6.0:
                        break

            # If we haven't reached max horizon, expand wait actions
            if months < self.max_months:
                for step in step_choices:
                    if step == 0:
                        continue
                    new_months = months + step
                    if new_months > self.max_months:
                        continue

                    # project tariff and tech cost to new_months
                    proj_tariff, proj_tech_cost = project_rates(new_months)
                    # opportunity cost (lost savings during this extra wait step)
                    # estimate monthly generation and monthly savings at current state's tariff
                    annual_gen_per_kw = 1400  # conservative proxy; pipeline provides a more accurate figure but we keep proxy for op-cost
                    monthly_generation = required_capacity_kw * annual_gen_per_kw / 12
                    monthly_savings_now = monthly_generation * node.state.tariff
                    # lost savings during the step
                    lost_savings = monthly_savings_now * step

                    new_state = TimeState(months_waited=new_months, tariff=proj_tariff, tech_cost_per_kw=proj_tech_cost, budget=node.state.budget)
                    if new_state in closed:
                        continue

                    # heuristic = estimated payback if install at new_state (we convert payback years into comparable "cost" units)
                    heuristic_payback = self._heuristic_payback_estimate(new_months, proj_tariff, proj_tech_cost, required_capacity_kw, monthly_consumption_kwh, tariff_growth_rate, tech_decline_rate)
                    # convert heuristic years -> "cost-like" by multiplying with system cost to get comparable scale
                    estimated_system_cost = required_capacity_kw * proj_tech_cost
                    heuristic_value = heuristic_payback * 0.5 * estimated_system_cost / max(1.0, estimated_system_cost)  # scaled heuristic

                    new_node = SearchNode(
                        state=new_state,
                        g_cost=node.g_cost + lost_savings,
                        heuristic=heuristic_value,
                        decision_path=node.decision_path + [self._months_to_action(step)]
                    )
                    heapq.heappush(open_heap, new_node)

        # If we found best_solution, return it; else fallback to recommend install_now if budget allows
        if best_solution:
            first_action = best_solution.decision_path[0] if best_solution.decision_path else Decision.INSTALL_NOW
            return {
                'optimal_scenario_type': first_action.value,
                'months_to_wait': best_solution.state.months_waited,
                'payback_years': round(best_solution.payback, 2) if best_solution.payback is not None else None,
                'coverage_pct': round(best_solution.coverage_pct, 1) if best_solution.coverage_pct is not None else None,
                'iterations': iterations,
                'search_successful': True,
            }
        else:
            # fallback logic: if budget covers current cost -> install, else wait 6 months
            current_system_cost = required_capacity_kw * current_tech_cost_per_kw
            if user_budget >= current_system_cost * 1.05:
                return {'optimal_scenario_type': Decision.INSTALL_NOW.value, 'months_to_wait': 0, 'payback_years': None, 'coverage_pct': None, 'search_successful': False}
            else:
                return {'optimal_scenario_type': Decision.WAIT_6_MONTHS.value, 'months_to_wait': 6, 'payback_years': None, 'coverage_pct': None, 'search_successful': False}

    def _heuristic_payback_estimate(self, months:int, tariff:float, tech_cost_per_kw:float, required_capacity_kw:float, monthly_consumption_kwh:float, tariff_growth_rate:float, tech_decline_rate:float) -> float:
        """
        Fast heuristic: estimate payback years if install at `months` (lower is better).
        Uses quick formula: system_cost / annual_savings.
        Doesn't call ROI calculator (cheap).
        """
        # project years
        years = months / 12.0
        proj_tech_cost = tech_cost_per_kw * ((1 - tech_decline_rate) ** years)
        proj_tariff = tariff * ((1 + tariff_growth_rate) ** years)

        system_cost = required_capacity_kw * proj_tech_cost
        annual_generation = required_capacity_kw * 1400  # proxy
        annual_savings = min(annual_generation, monthly_consumption_kwh * 12) * proj_tariff

        if annual_savings <= 0:
            return float('inf')
        return system_cost / annual_savings


# Simplified integration function



from typing import Dict, Optional # Make sure Optional is imported

# Simplified integration function
def run_simplified_heuristic_search(
    user_budget: float,
    monthly_consumption: float,
    current_tariff: float,
    monthly_bill: float,
    current_tech_cost_per_kw: float,
    required_capacity_kw: Optional[float] = None,
    # ADD DYNAMIC PARAMETERS
    tariff_growth_rate: float = 0.05,  # Default fallback
    tech_decline_rate: float = 0.03    # Default fallback
) -> Dict:
    """
    FIXED: Integration that now accepts dynamic rates from the pipeline
    and passes them through to the heuristic search engine.
    """
    try:
        # This block now works correctly because required_capacity_kw is defined.
        if required_capacity_kw is None:
            # Annual consumption / annual generation per kW
            annual_consumption = monthly_consumption * 12
            capacity = annual_consumption / 1400
            # Clamp the value to a reasonable range (e.g., 2kW to 10kW)
            required_capacity_kw = max(2, min(10, capacity))
            log.info(f"Simplified search calculated capacity: {required_capacity_kw:.2f} kW")

        # Run simplified A* search
        search_engine = TimeAStarSolarSearch()

        # FIXED: Pass the DYNAMIC rates instead of hardcoded values
        best_decision, payback_years, metadata = search_engine.search_optimal_decision(
            user_budget=user_budget,
            current_tariff=current_tariff,
            required_capacity_kw=required_capacity_kw,
            current_tech_cost=current_tech_cost_per_kw,
            monthly_consumption=monthly_consumption,
            tariff_growth_rate=tariff_growth_rate,  # DYNAMIC from pipeline
            tech_decline_rate=tech_decline_rate     # DYNAMIC from pipeline
        )

        # Log the dynamic values being used
        log.info(f"Using DYNAMIC rates in heuristic search:")
        log.info(f"  Tariff growth: {tariff_growth_rate:.1%} (from pipeline tariff forecast)")
        log.info(f"  Tech decline: {tech_decline_rate:.1%} (from pipeline tech trends)")

        # Calculate metrics for the successful result
        system_cost = required_capacity_kw * current_tech_cost_per_kw
        annual_savings = (required_capacity_kw * 1400) * current_tariff
        roi_percent = (annual_savings / system_cost) * 100 if system_cost > 0 else 0
        coverage_percent = (required_capacity_kw * 1400) / (monthly_consumption * 12) * 100

        log.info(f"Simplified A* result: {best_decision.value}, payback: {payback_years:.1f}y")

        # Single, clean return statement for the success case
        return {
            'optimal_scenario_type': best_decision.value,
            'payback_period': round(payback_years, 1),
            'roi': round(roi_percent, 2),
            'cost': system_cost,
            'capacity_kw': round(required_capacity_kw, 2),
            'coverage': round(coverage_percent, 2),
            'confidence': 0.8 if metadata.get('search_successful', False) else 0.6,
            'risk_score': 4.0,
            'search_metadata': {
                'algorithm': 'Simplified A* Search',
                'iterations': metadata.get('iterations', 0),
                'solution_quality': 'good' if metadata.get('search_successful', False) else 'fallback',
                'dynamic_tariff_growth': f"{tariff_growth_rate:.1%}",
                'dynamic_tech_decline': f"{tech_decline_rate:.1%}",
                'rates_source': 'pipeline_dynamic'
            }
        }

    except Exception as e:
        log.error(f"Simplified search failed catastrophically: {e}", exc_info=True)
        # Robust fallback dictionary.
        return {
            'optimal_scenario_type': 'fallback_error',
            'payback_period': None,
            'roi': None,
            'cost': None,
            'capacity_kw': round(required_capacity_kw, 2) if required_capacity_kw else None,
            'coverage': None,
            'confidence': 0.2,
            'risk_score': 8.0,
            'search_metadata': {
                'algorithm': 'Simplified A* Search',
                'error_message': str(e),
                'solution_quality': 'failed',
                'rates_source': 'fallback'
            }
        }
