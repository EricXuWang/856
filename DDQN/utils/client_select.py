import numpy as np

def client_selection(num_users, C):
    # Calculate the number of clients that will join federated learning at this communication round
    m = max(int(C * num_users), 1)
    # Select the indexes of clients randomly from all clients
    idxs_users = np.random.choice(range(num_users), m, replace = False)
    return idxs_users