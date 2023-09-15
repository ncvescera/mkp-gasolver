# mkp-gasolver

A Multidimensional Knapsack Problem solver using Genetic Algorithm 🧬

## Install

Install the environment and python requirements:

```bash
make install
```

## Run

Activate environment, enter on `scr/` folder and run `main.py` script.

```bash
source .env/bin/acrivate
cd src/
python main.py data/MKP11.txt -pmut .05 -pcross .97 -ngen 250 -plen 100 --log INFO -tk 61 
```

If in doubt, run:

```bash
python main.py --help

usage: main.py [-h] [-plen POPULATION_LENGHT] [-pcross CROSSOVER_PROBABILITY]
               [-pmut MUTATION_PROBABILITY] [-ngen NUMBER_GENERATION] [-tk TOURNAMENT_K]
               [-log {DEBUG,INFO,WARNINGS}]
               path

Multidimensional Knapsack Problem Solver

positional arguments:
  path                  Instance File Path

options:
  -h, --help            show this help message and exit
  -plen POPULATION_LENGHT, --population_lenght POPULATION_LENGHT
                        Initial Population Lenght
  -pcross CROSSOVER_PROBABILITY, --crossover_probability CROSSOVER_PROBABILITY
                        Crossover Probability (from 0 to 1)
  -pmut MUTATION_PROBABILITY, --mutation_probability MUTATION_PROBABILITY
                        Mutation probability (from 0 to 1)
  -ngen NUMBER_GENERATION, --number_generation NUMBER_GENERATION
                        Number of generations
  -tk TOURNAMENT_K, --tournament_k TOURNAMENT_K
                        Tournament random solution to select
  -log {DEBUG,INFO,WARNINGS}, --log_level {DEBUG,INFO,WARNINGS}
                        Logging Level
```

