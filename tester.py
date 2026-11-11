import pandas as pd
from load_tracking import load_tracking

# players, ball = load_tracking("data/tactical/BAR-ATH_tactical_atd.txt")

# players.to_csv("data/tactical/players_labelled.csv", index=False)
# ball.to_csv("data/tactical/ball_labelled.csv", index=False)
b_df= pd.read_csv('/Users/rishavsingh/Downloads/BarcaTester/data/tactical/ball_labelled.csv')
p_df = pd.read_csv('/Users/rishavsingh/Downloads/BarcaTester/data/tactical/players_labelled.csv')

