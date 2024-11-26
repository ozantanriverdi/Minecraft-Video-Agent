from tqdm import tqdm

if __name__ == '__main__':
    
    rl_phrases = [
    "reinforcement",
    "reward",
    #"reward function",
    #"reward signal",
    "policy optimization",
    "q-learning",
    "temporal difference",
    "markov decision process",
    "mdp",
    #"exploration-exploitation",
    "value function",
    "actor-critic",
    "policy gradient",
    "dqn",
    "a3c",
    "ppo",
    "td",
    "sarsa",
    "trajectory optimization",
    #"learning rate",
    "off-policy",
    "on-policy",
    #"experience replay",
    #"environment interaction",
    "bellman equation",
    "stochastic policy",
    "advantage function",
    "transition probability",
    "discount factor",
    #"agent training",
    #"control tasks"
    ]
    
    
    with open("article_titles.txt", "r", encoding="utf-8") as file:
        with open("article_titles_pre_filtered.txt", "w", encoding="utf-8") as file_2:
            with open("rl_papers.txt", "w", encoding="utf-8") as file_3:
                for line in tqdm(file):
                    if any(phrase in line.lower() for phrase in rl_phrases):
                        print(line, end="")
                        file_3.write(line)
                    else:
                        file_2.write(line)
                    