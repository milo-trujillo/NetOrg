Simulation Descriptions
=======================

Welfare Descriptions
--------------------

| Name            |  Description                             |
| --------------- | ---------------------------------------- |
| Master          | Multiple layers, but no special emphasis |
| Robustness      | Agents+1 layers, where each layer but first has one agent disabled |
| Disparate Goals | Each agent's welfare is based on averaging three env nodes (agent's number + 1,2,3 % number-of-env-nodes) |
| Binary Goals    | Each agent's welfare is based on averaging even or odd nodes (agent number % 2) |
| Selfish         | There is an emphasis on distribution of labor, where listening to the environment is exponentially expensive per agent |

Trial Descriptions
------------------

| Name            |  Description                             | Technical Detail                   |
| --------------- | ---------------------------------------- | ---------------------------------- |
| Trial 1         | Baseline                                 | None                               |
| Trial 2         | Listening to env extra expensive         | envobsnoise 5 instead of 2         |
| Trial 3         | Listening to agents extra expensive      | innoise 10 instead of 2            |
| Trial 4         | Double environment nodes                 | num_environment 10 instead of 5    |
| Trial 5         | Double agent nodes                       | num_agents 20 instead of 10        |
