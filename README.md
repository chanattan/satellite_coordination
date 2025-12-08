# Satellite Coordination project

This project takes place as a project in the course *"COCOMA"* for coordination of agents.

The project is based on the article with auction-based coordinations from Gauthier Picard. The reference being:

*Auction-based and Distributed Optimization Approaches for Scheduling Observations in
Satellite Constellations with Exclusive Orbit Portions, G. Picard, AAMAS ’22 : Proceedings of
the 21st International Conference on Autonomous Agents and Multiagent Systems, May 2022,
Pages 1056–1064*

# Projet structure

The code is available on this GitHub, the presentation of the project is on the associated Python Notebook, along with results and illustrations.

The project follows this structure:

### Preparation: Instances and planification of tasks\
- We implement a generator of random ESOP instances, given a number of satellites, exlusive users, and tasks, as objects along with their text format.
- We implement a method to plan the execution of a set of tasks for a satellite with temporal constraints on the tasks. This step is crucial as it needs to be optimal computation-wise.

### Part 1: Optimization of distributed constraints

Based on the article, the problem of agents' coordination with exclusive users and a central planner can be modeled by a DCOP. We use the Pydcop library to solve such problems.
- Based on an instance I, we implement a method to generate the DCOP problem associated.
- We solve such instances with Pydcop.
- We experimentally compare the resolution times and the quality of solutions by the different algorithms of Pydcop, on different problems of varying size.

### Part 2: Auctions

We focus on the coordination of agents with negotiation protocols, to coordinate the allocation of tasks. We are interested in parallel auctions, sequential auctions, and regret-based sequential auctions.

The third approach is explained in the article, the first two are seen in class. We implement the three and compare them through the same instance generator we have implemented.

# Authors

Chanattan Sok and Tom Bouscarat.
