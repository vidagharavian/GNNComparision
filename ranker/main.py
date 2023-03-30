# This is a sample Python script.
import pandas as pd

for i in range(1,200):
  data = pd.read_csv(f"generations/{i}.csv")
  data


data2 = data[['se2so0', 'lte2m', 'wte2m', 'lscore']].copy()
