from src.agents import RandomAgent, TDAgent, KPlyAgentFused
from src.environment import Environment


def test_k_ply_agent_fused():
    agent_1 = RandomAgent(sign=1)
    agent_2 = TDAgent.from_saved_model(sign=-1, export_dir="/tmp/backgammon_agent_saved_model")
    agent_2 = KPlyAgentFused(sign=-1, k=1, agent=agent_2)
    env = Environment(agents=[agent_1, agent_2])
    res = env.contest(num_episodes=1, verbose=True)
    print(res)
    assert True
