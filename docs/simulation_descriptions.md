Simulation Descriptions
=======================

Welfare Descriptions
--------------------

| Name            |  Description                             |
| --------------- | ---------------------------------------- |
| Master          | Multiple layers, but no special emphasis |
| Robustness      | (Number of agents)+1 variants, where each variant but the first has one agent disabled. The same listen-weights are used for all variants, so we get a single topology resilient to single-agent loss |
| Disparate Goals | Each agent's welfare is based on averaging three environment nodes (agent's number + 1,2,3 % number-of-env-nodes) |
| Binary Goals    | Each agent's welfare is based on averaging even or odd nodes (agent number % 2) |
| Selfish         | There is an emphasis on distribution of labor, where listening to the environment is exponentially expensive per agent |

We can also combine the above welfare functions, such as "selfish binary goals", or "robust disparate goals".

Trial Descriptions
------------------

We use five trials to measure the effectiveness of a welfare function in different scenarios:

| Name            |  Description                             | Technical Detail                   |
| --------------- | ---------------------------------------- | ---------------------------------- |
| Trial 1         | Baseline                                 | None                               |
| Trial 2         | Listening to env extra expensive         | envobsnoise 5 instead of 2         |
| Trial 3         | Listening to agents extra expensive      | innoise 10 instead of 2            |
| Trial 4         | Double environment nodes                 | num_environment 10 instead of 5    |
| Trial 5         | Double agent nodes                       | num_agents 20 instead of 10        |
