import re
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

def read_file(file_name):
    with open(file_name, 'r') as f:
        data = f.read()
    return data

def remove_non_alpha_characters(data):
    data = data.lower()
    data = re.sub(r'[^a-zA-Z\s]', '', data)
    data = re.sub(r'\s+', ' ', data)
    return data

def return_unique(data):
    unique = set(data)
    return list(unique)

def remove_stopwords(data):
    stopwords = ['a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'at', 'from', 'by', 'on', 'off', 'for', 'in', 'out', 'over', 'to', 'into', 'with', ""]
    data = [word for word in data if word not in stopwords]
    return data

def return_list_without_a_value(data, value):
    return [x for x in data if x != value]

def one_hot_encode(words):
    length = len(words.keys())
    encoded_words = {}
    for key, value in words.items():
        one_hot = np.zeros(length)
        one_hot[value] = 1
        tensor = torch.from_numpy(one_hot).to(torch.int64)
        encoded_words[key] = tensor

    return encoded_words

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.init_emb()

    def init_emb(self):
        init_mean = 0
        init_std = 0.01
        self.u_embeddings.weight.data.normal_(init_mean, init_std)
        self.v_embeddings.weight.data.normal_(init_mean, init_std)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score  = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = 0
        for i in range(len(neg_v[0])):
            neg_score += F.logsigmoid(-torch.sum(torch.mul(emb_u, emb_neg_v[:,i,:]), dim=1))

        return torch.mean(score + neg_score)


def train(model, data_loader, optimizer, criterion, dictionary_length, negative_sample_length):
    model.train()

    total_loss = 0

    for i, (pos_u, pos_v) in enumerate(data_loader):
        optimizer.zero_grad()
        neg_v = torch.randint(0, dictionary_length, (1, negative_sample_length))
        score = model.forward(pos_u, pos_v, neg_v)
        loss = criterion(model, score)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss

def test(model, data_loader, criterion):
    model.eval()

    total_loss = 0

    for i, (pos_u, pos_v, neg_v) in enumerate(data_loader):
        score = model.forward(pos_u, pos_v, neg_v)
        loss = criterion(model, score)
        total_loss += loss.item()

    return total_loss

raw_data = read_file('shakespeare.txt')
data = remove_non_alpha_characters(raw_data)
data = data.split(" ")
data = remove_stopwords(data)
unique_words = return_unique(data)

unique_dict = {word: i for i, word in enumerate(unique_words)}
dictionary_length = len(unique_dict)
encoded_data = one_hot_encode(unique_dict)

window_size = 5
dataset = []
sample_data = data

for i, val in enumerate(sample_data):
    if i > len(sample_data) - window_size:
        break
    sub = sample_data[i:i+window_size]
    included = return_list_without_a_value(sub, val)
    for target in included:
        dataset.append((unique_dict[val],unique_dict[target]))

batch_size = 100
n_iters = 3000
negative_sampling = 5
num_epochs = 100
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

model = SkipGramModel(len(unique_words), 300)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()


for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, dictionary_length, negative_sampling)
    print("Epoch: %d, Loss: %.4f" % (epoch+1, train_loss))


# Testing

test_data = data[:100]
test_dataset = []

for i, val in enumerate(test_data):
    if i > len(test_data) - window_size:
        break
    sub = test_data[i:i+window_size]
    included = return_list_without_a_value(sub, val)
    for target in included:
        test_dataset.append((unique_dict[val],unique_dict[target]))

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loss = test(model, test_loader, criterion)
print("Test Loss: %.4f" % test_loss)

