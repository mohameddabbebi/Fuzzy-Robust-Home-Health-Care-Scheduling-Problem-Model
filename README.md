# Home Health Care Scheduling Optimization

This project aims to solve the home health care scheduling problem, where the objective is to determine the best route for each nurse in order to maximize the number of tasks completed, while considering various constraints such as time windows, maximum working hours, and workload balancing. The project uses a genetic algorithm to optimize the schedule and improve the efficiency of healthcare service delivery.

## Problem Overview

The goal of the project is to assign tasks to nurses in a way that:
- Maximizes the number of tasks completed.
- Respects the time windows for each client.
- Ensures that the working hours of each nurse do not exceed the predefined limits.
- Balances the workload across the nurses to prevent overworking any individual.

The problem does not focus on matching caregivers' skills and qualifications to specific patient needs but rather focuses on optimizing the route scheduling.

## Algorithm Used

The project employs a **Genetic Algorithm (GA)** for solving the optimization problem. The genetic algorithm evolves a population of possible schedules over several generations to find the most efficient solution.

### Key Steps:
1. **Fitness Function**: The fitness function evaluates each schedule based on how well it maximizes task completion, respects time windows, and balances the workload.
2. **Selection**: The algorithm selects the best-performing schedules to create new offspring.
3. **Crossover**: The selected schedules are combined to create new solutions.
4. **Mutation**: A small random change is introduced to ensure diversity in the population.

## Benchmark Dataset

The project uses the **Solomon Benchmark dataset** for testing and evaluating the performance of the optimization algorithm. This dataset includes various instances of vehicle routing problems, which are adapted to represent nurse scheduling problems.

## Technologies Used

- **Python**: The core language used to implement the algorithm.
- **Genetic Algorithm**: For optimizing nurse schedules.
- **Solomon Benchmark Dataset**: For testing and validating the solution.

## Setup

To run the project on your local machine:

### Prerequisites:
- Python 3.x
- Required Python libraries (can be installed via `pip`).
