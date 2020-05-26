# Auto-encdoer

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


# Getting the number of users and movies
from torch.optim import RMSprop


def get_num_user_movies(training_set, test_set):
    nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
    nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))
    return nb_users, nb_movies


# converting the data into an array with users in lines and movies in columns
def convert_bm_matrix(data, nb_users, nb_movies):
    """
    return a matrix of user's movie ratings. A row is a user's rating to all the movies. A column is a movie's rating
    from all the users.

    If using the function multiple times, you need to make sure that the user_id and movie_id across all
    of your dataset are consistent.
    :param data: the dataset that is going to be convert to user-movie_rating matrix
    :param nb_users: the number of users in the dataset
    :param nb_movies: the number of movies in the dataset
    :return: a matrix of user's movie ratings
    """
    new_data = []

    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings  # get all the rated movies indexes
        new_data.append(ratings)

    return new_data


def convert_to_binary(data: torch.FloatTensor, threshold=3):
    rtn = data.clone()
    rtn[data == 0.] = -1.
    rtn[[x & y for (x, y) in zip([data > 0.], [data < threshold])]] = 0.
    rtn[data >= threshold] = 1.
    return rtn

if __name__ == '__main__':
    # Script body ##########################################################################################
    all_movie = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
    all_users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
    all_ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

    # prepare the training set and the test set
    training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
    training_set = np.array(training_set, dtype='int')
    # prepare the test set and the test set
    test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
    test_set = np.array(test_set, dtype='int')

    num_users, num_movies = get_num_user_movies(training_set, test_set)
    training_set = convert_bm_matrix(training_set, nb_users=num_users, nb_movies=num_movies)
    test_set = convert_bm_matrix(test_set, nb_users=num_users, nb_movies=num_movies)

    training_set = torch.FloatTensor(training_set)
    test_set = torch.FloatTensor(test_set)

    class SAE(nn.Module):
        def __init__(self, input_len, units1, units2):
            super(SAE, self).__init__()
            self.fc1 = nn.Linear(input_len, units1)  # first fully connected layer
            self.fc2 = nn.Linear(units1, units2)
            self.fc3 = nn.Linear(units2, units1)  # start of the decoding
            self.fc4 = nn.Linear(units1, input_len)  # output of the decoder
            self.activation = nn.Sigmoid()

        def forward(self, x):
            x = self.activation(self.fc1(x))  # encode with the first layer
            x = self.activation(self.fc2(x))  # encode with the second layer
            x = self.activation(self.fc3(x))  # decode with the third layer
            x = self.fc4(x)  # decode with the forth layer, no activation at the output
            return x


    sae = SAE(num_movies, units1=20, units2=10)
    criterion = nn.MSELoss()  # criterion for the loss function
    optimizer = RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

    # Training the SAE
    nb_epoch = 200
    for epoch in range(1, nb_epoch + 1):
        train_loss = 0
        s = 0.
        for id_user in range(num_users):
            input_x = Variable(training_set[id_user]).unsqueeze(0)
            target = input_x.clone()
            if torch.sum(target.data > 0) > 0:
                    output = sae.forward(input_x)
                    target.require_grad = False  # don't compute gradient with respect to target
                    output[target == 0] = 0  # ground the non-ratings
                    loss = criterion(target, output)
                    mean_corrector = num_movies/float(torch.sum(target.data > 0) + 1e-10)
                    loss.backward()
                    train_loss += np.sqrt(loss.data.item() * mean_corrector)
                    s += 1.
                    optimizer.step()

        print('epoch ' + str(epoch) + ': train loss is ' + str(train_loss/s))


    # check the test set loss
    test_loss = 0
    s = 0.

    for id_user in range(num_users):
        input_x = Variable(training_set[id_user]).unsqueeze(0)
        target = Variable(test_set[id_user]).unsqueeze(0)
        if torch.sum(target.data > 0) > 0:
            output = sae.forward(input_x)
            target.require_grad = False  # don't compute gradient with respect to target
            output[target == 0] = 0  # ground the non-ratings
            loss = criterion(target, output)
            mean_corrector = num_movies / float(torch.sum(target.data > 0) + 1e-10)
            test_loss += np.sqrt(loss.data.item() * mean_corrector)
            s += 1.
    print('The test loss is ' + str(test_loss/s))