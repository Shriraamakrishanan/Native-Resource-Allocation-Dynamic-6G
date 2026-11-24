# AI-Native Resource Allocation for Dynamic Network Slicing in 6G
 
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-RL_Environment-0081A5)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-FF6F00)
![PyTorch](https://img.shields.io/badge/PyTorch-Policy_Network-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Simulation-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plots-11557C)
![pandas](https://img.shields.io/badge/pandas-Results-150458?logo=pandas&logoColor=white)
![ns-3](https://img.shields.io/badge/ns--3_+_5G--LENA-Future_Work-lightgrey)
 
## Purpose
 
A 6G network splits one physical radio resource pool into virtual **slices**: eMBB (high throughput), URLLC (very low latency) and mMTC (massive IoT). Static splits waste capacity when one slice is idle and another is starved. This project trains a **PPO reinforcement-learning agent** to reallocate bandwidth across the three slices every step, and benchmarks it against a proportional-fair heuristic on **SLA violations** and **bandwidth utilization**.
 

## Architecture

```mermaid
flowchart LR
    A["Traffic Generator<br/>Bursty demand per slice<br/>eMBB | URLLC | mMTC"]
    B["6G Slicing Simulator<br/>Queues + Latency Model"]
    C["Gymnasium Wrapper<br/>State + Action + Reward"]
    D["PPO Agent<br/>Stable-Baselines3 + PyTorch"]
    E["Heuristic Baseline<br/>Proportional Fair + URLLC Boost"]
    F["Performance Evaluation<br/>SLA Violations + Utilization + Latency"]
    G["Results and Visualization<br/>Plots + Benchmark Tables"]

    A --> B
    B --> C
    C -->|"State"| D
    D -->|"Bandwidth Actions"| B
    E -->|"Baseline Actions"| B
    C --> F
    F --> G
```



 
**Research workflow (pseudo code)**
 
```text
FOR each experiment IN [reward_shaping, ppo_hyperparams, traffic_generalization, action_cap]:
    env      <- NetworkSlicingEnv(config = experiment.config, seed = s)
    agent    <- PPO(policy = MLP, env, lr, n_steps, clip_range)
    FOR step = 1 .. total_timesteps:
        state  <- [demand_i, queue_i, alloc_i  for i in slices]
        action <- agent.policy(state)                 # raw shares
        alloc  <- normalize(action) * total_bandwidth
        served <- min(demand, alloc);  queue <- max(queue + demand - served - drain, 0)
        latency <- f(queue);  violation <- max(latency - SLA, 0)
        reward <- -w1 * sum(violation) + w2 * sum(served) / total_bandwidth
        agent.update(clipped PPO objective)
    FOR policy IN [agent, heuristic]:
        log mean SLA violations, utilization, latency over N unseen episodes
    compare(agent, heuristic); save table + plots to docs/
```
 
## Setup
 
```bash
git clone https://github.com/<your-username>/6g-network-slicing-rl.git
cd 6g-network-slicing-rl
python3 -m venv .venv && source .venv/bin/activate
pip install gymnasium "stable-baselines3[extra]" numpy matplotlib pandas
```
 
## Usage
 
```bash
python train.py       # trains PPO (200k steps), saves ppo_slicing_agent.zip
python evaluate.py    # RL vs heuristic benchmark, prints and saves results
```
 
Files: `slicing_env.py` (environment), `train.py`, `baseline_heuristic.py`, `evaluate.py`.
 
## Results
 
Mean over [20] evaluation episodes of 200 steps, seed [__]. Replace the placeholders with the output of `evaluate.py`.
 
| Method | SLA violations / episode | Utilization | Mean URLLC latency (ms) |
|---|---|---|---|
| PPO agent | [x.xx] | [xx%] | [x.xx] |
| Proportional-fair heuristic | [x.xx] | [xx%] | [x.xx] |
 
![Training reward curve](docs/reward_curve.png)
 
## Reflection
 
The biggest hurdle was getting PPO to learn anything beyond an equal split. Early training showed a flat reward curve. Logging each reward component separately showed the SLA-violation penalty, which grows into the hundreds during bursts, swamped the utilization reward capped at 5, so the agent had no useful gradient. I normalized both terms to comparable magnitudes, added queue length and current allocation to the observation so the agent could see the effect of its last action, and seeded the environment's random generator so runs were repeatable. After this, the agent began shifting bandwidth toward URLLC during bursts and violations fell from [X] to [Y] per episode against the heuristic. The lesson: in reinforcement learning, debugging the reward signal matters more than tuning the algorithm. *(Edit this to match what actually happened in your runs.)*
 
## Release
 
```bash
git add -A && git commit -m "Complete mock-environment PPO pipeline with benchmark"
git tag -a v1.0 -m "v1.0: PPO vs heuristic network slicing benchmark (mock environment)"
git push origin main && git push origin v1.0
```
 
**Roadmap:** ns-3 + 5G-LENA integration via ns3-ai / ns3-gym for realistic PHY/MAC behavior.
 
