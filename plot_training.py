import pandas as pd
import json
from matplotlib import pyplot as plt

with open("policy_transfer/data/ppo_DartHopperPT-v10__UP/progress.json", 'r') as json_file:
    json_text = '['
    for line in json_file:
        json_text += line + ','
    json_text = json_text[:-1] + ']'
    # print(json_text)
    data = pd.read_json(json_text)


plt.figure()
plt.plot(data['EpRewMean'])
plt.title("Mean Episode Reward")
plt.figure()
plt.plot(data['loss_vf_loss'] + data['loss_pol_surr'])
plt.title("Policy Loss")
plt.show()