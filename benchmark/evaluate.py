import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
# from your_model_module import NCF  # Replace with the actual module where your NCF class is defined

test = pd.read_csv('data.csv')
test['rank_latest'] = test.groupby(['user_id'])['timestamp'] \
                                .rank(method='first', ascending=False)
test_ratings = test[test['rank_latest'] == 1]
test_ratings = test_ratings[['user_id', 'movie_id', 'rating']]

num_users = test['user_id'].max()+1
num_items = test['movie_id'].max()+1

all_movieIds = test['movie_id'].unique()

# Initialize the model
model = NCF(num_users, num_items)  # Make sure to replace with actual dimensions
checkpoint_path = '../models/lightning_logs/version_0/checkpoints/epoch=9-step=13890.ckpt'  # Replace with the path to your checkpoint
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# User-item pairs for testing
test_user_item_set = set(zip(test_ratings['user_id'], test_ratings['movie_id']))

# Dict of all items that are interacted with by each user
user_interacted_items = test.groupby('user_id')['movie_id'].apply(list).to_dict()

# Evaluate the model on the test set
hits = []

for (u, i) in tqdm(test_user_item_set):
    interacted_items = user_interacted_items[u]
    not_interacted_items = list(set(all_movieIds) - set(interacted_items))
    selected_not_interacted = list(np.random.choice(not_interacted_items, 99))
    test_items = selected_not_interacted + [i]

    user_input = torch.tensor([u] * len(test_items), dtype=torch.long).unsqueeze(0)
    test_items_input = torch.tensor(test_items, dtype=torch.long).unsqueeze(0)

    predicted_labels = np.squeeze(model(user_input, test_items_input).detach().numpy())

    top10_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][:10].tolist()]

    if i in top10_items:
        hits.append(1)
    else:
        hits.append(0)

# Calculate and print the Hit Ratio @ 10
hit_ratio_at_10 = np.average(hits)
print("The Hit Ratio @ 10 is {:.2f}".format(hit_ratio_at_10))
