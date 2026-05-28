# Disease Spreading Simulation

A Python-based epidemiological simulation that models the spread of infectious diseases using real COVID-19 data and Pygame visualization.

## Features

- **Real-world data**: Uses actual COVID-19 transmission and mortality rates from Our World in Data
- **Visual simulation**: Interactive Pygame-based visualization with real-time statistics
- **Agent-based modeling**: Simulates individual agents with health states (susceptible, infected, recovered, vaccinated)
- **Intervention effects**: Models real intervention strategies including:
  - Vaccination (90% effectiveness)
  - Mask usage (65% effectiveness)
  - Social distancing (70% effectiveness)
- **Multiple implementations**: Includes both Pygame and SimPy-based versions

## Getting Started

### Requirements
- Python 3.8+
- pygame
- pygame-ce for python 3.14+
- simpy (for SimPy version)

### Installation

```bash
pip install pygame simpy
```

### Installation for python 3.14+

```bash
pip install pygame-ce simpy
```

### Running the Simulation

```bash
python DiseaseSpreadingSim.py
```

For the SimPy-based version:
```bash
python DiseaseSpreadingSimPy.py
```

## How It Works

The simulation models disease transmission between agents in a spatial grid. Each agent has a health state and can be infected, vaccinated, or recovered. The simulation uses real epidemiological data to calibrate transmission and mortality rates.
