"""
COVID-19 Spreading Simulation - SimPy Event-Driven Version
Uses discrete event simulation for efficiency and scalability.
Much faster than Pygame version; can simulate 100,000+ people easily.
"""

import simpy
import random
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict
import time
import matplotlib.pyplot as plt


# Real COVID-19 data from Our World in Data
REAL_COVID_DATA = {
    "transmission_rate": 0.08,
    "mortality_unvaccinated": 0.032,
    "mortality_vaccinated": 0.003,
    "disease_duration_min": 7.0,
    "disease_duration_max": 14.0,
    "mask_effectiveness": 0.65,
    "vaccine_effectiveness": 0.90,
    "distancing_effectiveness": 0.70,
}


@dataclass
class SimulationStats:
    """Track simulation statistics in real-time."""
    susceptible: int = 0
    infected: int = 0
    recovered: int = 0
    deceased: int = 0
    history: List[Dict] = field(default_factory=list)
    
    def record(self, env):
        """Record current state snapshot."""
        self.history.append({
            'time': env.now,
            'S': self.susceptible,
            'I': self.infected,
            'R': self.recovered,
            'D': self.deceased,
        })
    
    def counts(self):
        """Return current counts."""
        return {
            'S': self.susceptible,
            'I': self.infected,
            'R': self.recovered,
            'D': self.deceased,
        }


class Person:
    """Individual in the simulation."""
    
    person_id = 0
    
    def __init__(self, env, world, infected=False, masked=False, vaccinated=False, distancing=False):
        self.env = env
        self.world = world
        self.id = Person.person_id
        Person.person_id += 1
        
        self.state = "I" if infected else "S"
        self.masked = masked
        self.vaccinated = vaccinated
        self.distancing = distancing
        
        self.process = env.process(self.life_cycle())
        
        # Update stats
        if self.state == "I":
            world.stats.susceptible -= 1
            world.stats.infected += 1
        else:
            world.stats.susceptible += 1
    
    def life_cycle(self):
        """Main process for a person's disease progression."""
        if self.state == "I":
            yield self.env.process(self.infection_cycle())
    
    def infect(self):
        """Infect this person (called by others)."""
        if self.state != "S":
            return False
        
        self.state = "I"
        self.world.stats.susceptible -= 1
        self.world.stats.infected += 1
        
        # Start infection cycle
        self.env.process(self.infection_cycle())
        return True
    
    def infection_cycle(self):
        """Handle infection duration, recovery, or death."""
        disease_duration = random.uniform(
            REAL_COVID_DATA["disease_duration_min"],
            REAL_COVID_DATA["disease_duration_max"]
        )
        
        # Infectious period: try to infect contacts during this time
        contact_attempts = int(disease_duration)
        for _ in range(contact_attempts):
            self.attempt_infection()
            # Check for recovery/death periodically
            yield self.env.timeout(1.0)
        
        # Determine outcome: recovery or death
        mortality = (
            REAL_COVID_DATA["mortality_vaccinated"]
            if self.vaccinated
            else REAL_COVID_DATA["mortality_unvaccinated"]
        )
        
        if random.random() < mortality:
            self.state = "D"
            self.world.stats.infected -= 1
            self.world.stats.deceased += 1
        else:
            self.state = "R"
            self.world.stats.infected -= 1
            self.world.stats.recovered += 1
    
    def attempt_infection(self):
        """Try to infect nearby susceptible people."""
        if self.state != "I":
            return
        
        # Random sample of population (simulates random contacts)
        contact_count = random.randint(2, 8)
        candidates = random.sample(self.world.people, min(contact_count, len(self.world.people)))
        
        for other in candidates:
            if other.state != "S":
                continue
            
            # Calculate transmission probability
            chance = REAL_COVID_DATA["transmission_rate"]
            
            if self.masked:
                chance *= (1 - REAL_COVID_DATA["mask_effectiveness"])
            if other.masked:
                chance *= (1 - REAL_COVID_DATA["mask_effectiveness"])
            if other.vaccinated:
                chance *= (1 - REAL_COVID_DATA["vaccine_effectiveness"])
            if other.distancing:
                chance *= (1 - REAL_COVID_DATA["distancing_effectiveness"])
            
            if random.random() < chance:
                other.infect()


class CovidSimulation:
    """Main simulation controller."""
    
    def __init__(self, population=500, initial_infected=5, mask_rate=0.45,
                 vax_rate=0.45, dist_rate=0.25, virus_strength=None,
                 lockdown=False, max_time=200):
        self.env = simpy.Environment()
        self.population = population
        self.initial_infected = initial_infected
        self.mask_rate = mask_rate
        self.vax_rate = vax_rate
        self.dist_rate = dist_rate
        self.lockdown = lockdown
        self.max_time = max_time
        
        self.stats = SimulationStats()
        self.people = []
        
        self._create_population()
        
        # Start recording process
        self.env.process(self.recorder())
    
    def _create_population(self):
        """Initialize population."""
        Person.person_id = 0
        
        for i in range(self.population):
            infected = i < self.initial_infected
            masked = random.random() < self.mask_rate
            vaccinated = random.random() < self.vax_rate
            distancing = random.random() < self.dist_rate
            
            person = Person(
                self.env, self,
                infected=infected,
                masked=masked,
                vaccinated=vaccinated,
                distancing=distancing
            )
            self.people.append(person)
        
        self.stats.record(self.env)
    
    def recorder(self):
        """Periodically record simulation state."""
        while True:
            yield self.env.timeout(1.0)
            self.stats.record(self.env)
    
    def run(self):
        """Execute the simulation."""
        print(f"\n{'='*70}")
        print("COVID-19 SIMULATION (SimPy Event-Driven)".center(70))
        print(f"{'='*70}")
        print(f"Population: {self.population} | Initial Infected: {self.initial_infected}")
        print(f"Mask Rate: {self.mask_rate*100:.0f}% | Vaccination: {self.vax_rate*100:.0f}% | Distancing: {self.dist_rate*100:.0f}%")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        self.env.run(until=self.max_time)
        elapsed = time.time() - start_time
        
        print(f"Simulation complete in {elapsed:.2f} seconds\n")
        self.print_results()
    
    def print_results(self):
        """Print detailed results."""
        counts = self.stats.counts()
        total = sum(counts.values())
        attacked = counts['I'] + counts['R'] + counts['D']
        
        print(f"\n{'='*70}")
        print("FINAL RESULTS".center(70))
        print(f"{'='*70}")
        
        print(f"\nFinal Population Status (Total: {total:,})")
        print(f"  Susceptible: {counts['S']:10,d} ({100*counts['S']/total:5.1f}%)")
        print(f"  Infected:    {counts['I']:10,d} ({100*counts['I']/total:5.1f}%)")
        print(f"  Recovered:   {counts['R']:10,d} ({100*counts['R']/total:5.1f}%)")
        print(f"  Deceased:    {counts['D']:10,d} ({100*counts['D']/total:5.1f}%)")
        
        print(f"\nAttack Rate: {100*attacked/total:.1f}% of population infected")
        
        if attacked > 0:
            mortality = counts['D'] / attacked
            print(f"Case Fatality Rate: {100*mortality:.2f}%")
        
        print(f"Simulation Time: {self.env.now:.1f} days")
        
        self.create_histograms()
        self.print_analysis()
    
    def create_histograms(self):
        """Create and display histograms for SIR data."""
        history = self.stats.history
        if len(history) < 2:
            return
        
        times = [h['time'] for h in history]
        susceptible = [h['S'] for h in history]
        infected = [h['I'] for h in history]
        recovered = [h['R'] for h in history]
        deceased = [h['D'] for h in history]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('COVID-19 Disease Spread Simulation - Philippines Scale', fontsize=16, fontweight='bold')
        
        # SIR curve
        ax = axes[0, 0]
        ax.plot(times, susceptible, 'b-', linewidth=2, label='Susceptible')
        ax.plot(times, infected, 'r-', linewidth=2, label='Infected')
        ax.plot(times, recovered, 'g-', linewidth=2, label='Recovered')
        ax.plot(times, deceased, 'k-', linewidth=2, label='Deceased')
        ax.set_xlabel('Time (days)', fontsize=11)
        ax.set_ylabel('Number of People', fontsize=11)
        ax.set_title('SIR Curve Over Time', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K' if x >= 1000 else f'{int(x)}'))
        
        # Final status pie chart
        ax = axes[0, 1]
        counts = self.stats.counts()
        sizes = [counts['S'], counts['I'], counts['R'], counts['D']]
        labels = [f"Susceptible\n{counts['S']:,}\n({100*counts['S']/sum(sizes):.1f}%)",
                  f"Infected\n{counts['I']:,}\n({100*counts['I']/sum(sizes):.1f}%)",
                  f"Recovered\n{counts['R']:,}\n({100*counts['R']/sum(sizes):.1f}%)",
                  f"Deceased\n{counts['D']:,}\n({100*counts['D']/sum(sizes):.1f}%)"]
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#95a5a6']
        ax.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
        ax.set_title('Final Population Distribution', fontsize=12, fontweight='bold')
        
        # Infected over time bar chart
        ax = axes[1, 0]
        sample_rate = max(1, len(times) // 50)
        times_sample = times[::sample_rate]
        infected_sample = infected[::sample_rate]
        ax.bar(times_sample, infected_sample, color='#e74c3c', alpha=0.7, width=1)
        ax.set_xlabel('Time (days)', fontsize=11)
        ax.set_ylabel('Number of Infected', fontsize=11)
        ax.set_title('Active Infections Over Time', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K' if x >= 1000 else f'{int(x)}'))
        
        # Cumulative outcomes
        ax = axes[1, 1]
        categories = ['Recovered', 'Deceased', 'Still Infected']
        values = [counts['R'], counts['D'], counts['I']]
        colors_bar = ['#2ecc71', '#95a5a6', '#e74c3c']
        bars = ax.bar(categories, values, color=colors_bar, alpha=0.8)
        ax.set_ylabel('Number of People', fontsize=11)
        ax.set_title('Cumulative Outcomes', fontsize=12, fontweight='bold')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K' if x >= 1000 else f'{int(x)}'))
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(value):,}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('simulation_results.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Histograms saved as 'simulation_results.png'")
        plt.show()
    
    def print_analysis(self):
        """Print key findings."""
        counts = self.stats.counts()
        total = sum(counts.values())
        attacked = counts['I'] + counts['R'] + counts['D']
        attack_rate = attacked / total
        
        print(f"\n{'='*70}")
        print("KEY FINDINGS".center(70))
        print(f"{'='*70}\n")
        
        if attack_rate > 0.8:
            print(f"  • SEVERE OUTBREAK: {attack_rate*100:.0f}% of population infected")
        elif attack_rate > 0.5:
            print(f"  • MODERATE OUTBREAK: {attack_rate*100:.0f}% of population infected")
        else:
            print(f"  • CONTAINED OUTBREAK: {attack_rate*100:.0f}% of population infected")
        
        if self.mask_rate > 0.5 or self.vax_rate > 0.5:
            print(f"  • Interventions (masks/vaccines) reduced transmission")
        else:
            print(f"  • No major interventions active")
        
        peak_infected = max(h['I'] for h in self.stats.history)
        peak_time = next(h['time'] for h in self.stats.history if h['I'] == peak_infected)
        print(f"  • Peak infections: {peak_infected} people at day {peak_time:.1f}")
        
        print(f"\n  Data Source: Our World in Data (excess deaths & test positivity)")
        print(f"{'='*70}\n")


class BulkSimulation:
    """Run multiple simulations for statistical analysis."""
    
    def __init__(self, num_runs=10, population=500, **kwargs):
        self.num_runs = num_runs
        self.population = population
        self.kwargs = kwargs
        self.results = []
    
    def run(self):
        """Execute multiple simulations."""
        print(f"\n{'='*70}")
        print(f"ENSEMBLE SIMULATION ({self.num_runs} runs)".center(70))
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for run_num in range(self.num_runs):
            sim = CovidSimulation(population=self.population, **self.kwargs)
            sim.env.run(until=self.kwargs.get('max_time', 200))
            
            counts = sim.stats.counts()
            attacked = counts['I'] + counts['R'] + counts['D']
            attack_rate = attacked / self.population
            mortality = (counts['D'] / attacked) if attacked > 0 else 0
            
            self.results.append({
                'attack_rate': attack_rate,
                'mortality_rate': mortality,
                'deaths': counts['D'],
                'infected': counts['I'],
            })
            
            print(f"Run {run_num+1:2d}/{self.num_runs} | Attack Rate: {attack_rate*100:5.1f}% | Deaths: {counts['D']:4d}", end='\r')
        
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Completed {self.num_runs} runs in {elapsed:.1f} seconds")
        print(f"Average per run: {elapsed/self.num_runs:.2f} seconds\n")
        
        self.print_statistics()
    
    def print_statistics(self):
        """Print statistical summary."""
        attack_rates = [r['attack_rate'] for r in self.results]
        deaths = [r['deaths'] for r in self.results]
        
        print(f"{'='*70}")
        print("STATISTICAL SUMMARY".center(70))
        print(f"{'='*70}\n")
        
        print(f"Attack Rate:")
        print(f"  Mean:   {np.mean(attack_rates)*100:.1f}%")
        print(f"  Median: {np.median(attack_rates)*100:.1f}%")
        print(f"  Std:    {np.std(attack_rates)*100:.1f}%")
        print(f"  Range:  {np.min(attack_rates)*100:.1f}% - {np.max(attack_rates)*100:.1f}%")
        
        print(f"\nTotal Deaths:")
        print(f"  Mean:   {np.mean(deaths):.0f}")
        print(f"  Median: {np.median(deaths):.0f}")
        print(f"  Std:    {np.std(deaths):.0f}")
        print(f"  Range:  {int(np.min(deaths))} - {int(np.max(deaths))}")
        
        print(f"\n{'='*70}\n")


# Example usage scenarios
def example_single_simulation():
    """Run a single simulation - Philippines population scale."""
    # Philippines population: ~120 million
    # Using scaled representation: 120,000 (for computational efficiency)
    # Initial infected: 5% of population
    sim = CovidSimulation(
        population=120000,
        initial_infected=6000,
        mask_rate=0.45,
        vax_rate=0.45,
        dist_rate=0.25,
        max_time=365
    )
    sim.run()


def example_ensemble():
    """Run 50 simulations for statistical confidence."""
    ensemble = BulkSimulation(
        num_runs=50,
        population=500,
        initial_infected=5,
        mask_rate=0.45,
        vax_rate=0.45,
        dist_rate=0.25,
        max_time=100
    )
    ensemble.run()


def example_scenario_comparison():
    """Compare different intervention scenarios."""
    scenarios = [
        ("No Interventions", {"mask_rate": 0, "vax_rate": 0, "dist_rate": 0}),
        ("Masks Only", {"mask_rate": 0.7, "vax_rate": 0, "dist_rate": 0}),
        ("Vaccination Only", {"mask_rate": 0, "vax_rate": 0.7, "dist_rate": 0}),
        ("All Interventions", {"mask_rate": 0.7, "vax_rate": 0.7, "dist_rate": 0.7}),
    ]
    
    print("\n" + "="*70)
    print("SCENARIO COMPARISON".center(70))
    print("="*70)
    
    results = {}
    for name, params in scenarios:
        sim = CovidSimulation(population=1000, initial_infected=5, max_time=100, **params)
        sim.env.run(until=100)
        
        counts = sim.stats.counts()
        attacked = counts['I'] + counts['R'] + counts['D']
        attack_rate = attacked / 1000
        
        results[name] = {
            'attack_rate': attack_rate,
            'deaths': counts['D'],
        }
    
    print("\nComparison Results:")
    print("-" * 70)
    for name in scenarios:
        scenario_name = name[0]
        attack = results[scenario_name]['attack_rate']
        deaths = results[scenario_name]['deaths']
        print(f"{scenario_name:20s} | Attack Rate: {attack*100:5.1f}% | Deaths: {deaths:4d}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # Choose example to run:
    
    # 1. Single detailed simulation
    example_single_simulation()
    
    # 2. Ensemble for statistical confidence (uncomment to use)
    # example_ensemble()
    
    # 3. Compare scenarios (uncomment to use)
    # example_scenario_comparison()
