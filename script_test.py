# %%
import re
import random
import torch.nn as nn
import torch
import numpy as np

# %%
def read_file(file_name):
    with open(file_name, 'r') as f:
        data = f.read()
    return data

# %%
raw_data = read_file('shakespeare.txt')


# %%
def remove_non_alpha_characters(data):
    data = data.lower()
    # use regex to remove all non-alphanumeric characters
    data = re.sub(r'[^a-zA-Z\s]', '', data)
    # use regex to remove all whitespace characters
    data = re.sub(r'\s+', ' ', data)
    return data

def return_unique(data):
    unique = set(data)
    return list(unique)

def remove_stopwords(data):
    stopwords = ['a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'at', 'from', 'by', 'on', 'off', 'for', 'in', 'out', 'over', 'to', 'into', 'with', ""]
    data = [word for word in data if word not in stopwords]
    return data

# %%
data = remove_non_alpha_characters(raw_data)
data = data.split(" ")
data = remove_stopwords(data)
unique_words = return_unique(data)

# %%
unique_dict = {word: i for i, word in enumerate(unique_words)}

def one_hot_encode(words):
    length = len(words.keys())
    encoded_words = {}
    for key, value in words.items():
        one_hot = np.zeros(length)
        one_hot[value] = 1
        tensor = torch.from_numpy(one_hot).to(torch.int64)
        encoded_words[key] = tensor

    return encoded_words

encoded_data = one_hot_encode(unique_dict)

# %%
def return_list_without_a_value(data, value):
    return [x for x in data if x != value]


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

# %%
batch_size = 100
n_iters = 3000
num_epochs = 100
num_epochs = int(num_epochs)
# create a train_loader that will randomly generate examples forever

train_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

# %%
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
        # reshape the input tensors to have the right dimensions
        emb_u = self.u_embeddings(pos_u).view(-1, 1, self.embedding_dim).squeeze()
        emb_v = self.v_embeddings(pos_v).view(-1, self.embedding_dim).squeeze()
        # get the dot product of the two embeddings using the bmm function
        score = torch.bmm(emb_u.unsqueeze(1), emb_v.unsqueeze(2)).squeeze()
        score = torch.sigmoid(score)
        neg_emb_v = self.v_embeddings(neg_v).view(-1, self.embedding_dim, 5)
        neg_score = torch.bmm(emb_u.unsqueeze(1), neg_emb_v).squeeze()
        neg_score = torch.sigmoid(neg_score)
        return score, neg_score
    
    def forward_without_negatives(self, word1, word2):
        pos_u = torch.tensor([unique_dict[word1]])
        pos_v = torch.tensor([unique_dict[word2]])
        emb_u = self.u_embeddings(pos_u).view(-1, 1, self.embedding_dim).squeeze()
        emb_v = self.v_embeddings(pos_v).view(-1, self.embedding_dim).squeeze()
        score = torch.dot(emb_u, emb_v)
        print(score)
        score = torch.sigmoid(score)
        return score

    def get_dict_embeddings(self):
        return self.u_embeddings.weight.data.cpu().numpy()
    
    def get_embedding_from_word(self, word):
        index = unique_dict[word]
        return self.u_embeddings.weight.data[index]
    
    def get_embedding_from_index(self, index):
        return self.u_embeddings.weight.data[index]

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings
        fout = open(file_name, 'w')
        fout.write('{} {}\n'.format(len(id2word), self.embedding_dim))
        for wid, w in id2word.items():
            e = ' '.join(map(lambda x: str(x), self.get_embedding_from_index(wid)))
            fout.write('{} {}\n'.format(w, e))
        fout.close()
    
    def import_embeddings(self, file_name):
        fin = open(file_name, 'r')
        n, d = map(int, fin.readline().split())
        embedding = np.zeros((n, d))
        word2id = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            word2id[tokens[0]] = len(word2id)
            embedding[word2id[tokens[0]]] = list(map(float, tokens[1:]))
        return embedding, word2id

def non_scalar_loss(score, neg_score, lr, weight_decay):
    pos_loss = -torch.mean(torch.log(score))
    neg_loss = -torch.mean(torch.sum(torch.log(1 - neg_score), dim=1))
    loss = pos_loss + neg_loss
    # add L2 regularization term
    l2_loss = 0
    for param in model.parameters():
        l2_loss += torch.sum(param**2)
        loss += weight_decay * l2_loss
    return loss
  
embedding_dim = 100
window_size = 5

dictionary_length = len(unique_words)

model = SkipGramModel(dictionary_length, embedding_dim)



# %%

criterion = nn.BCELoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
# optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)

loss_sum = 0

negative_sample_length = 5

pos_u_data = torch.ones(batch_size)
neg_v_data = torch.zeros(batch_size*negative_sample_length)
concat_data = torch.cat([pos_u_data, neg_v_data], dim=0)
print(concat_data)
step_interval = 200

epochs = 15
for epoch in range(epochs):
    for i, (x, y) in enumerate(train_loader):
        pos_u = torch.tensor(x)
        pos_v = torch.tensor(y)
        neg_v = torch.randint(0, dictionary_length, (batch_size, negative_sample_length))
        # print(neg_v.shape)
        optimizer.zero_grad()
        pos_score, neg_score = model(pos_u, pos_v, neg_v)
        # score = torch.cat([pos_score, neg_score.flatten()], dim=0)
        # combined_len = len(pos_score) + len(neg_score)
        # # add a column of ones to pos_u_data
        # pos_u_data = torch.ones(len(pos_u), 1)
        # neg_v_data = torch.zeros(len(neg_score.flatten()), 1)
        # print(score.shape, concat_data.shape)
        loss =non_scalar_loss(pos_score, neg_score, learning_rate, 0.0001)
        # loss = criterion(score, concat_data)
        
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        if i % step_interval == 0:
            print(' Step [{}/{}], Loss: {:.4f}' 
                    .format(i+1, len(dataset)//batch_size, loss_sum/step_interval))
            loss_sum = 0
        if i > len(train_loader) - batch_size:
            break
        

# %%


# %%
def subtract_vector(vector1,vector2):
    return get_emb(vector1) - get_emb(vector2)

def add_vector(vector1,vector2):
    return get_emb(vector1) + get_emb(vector2)

def cos_sim(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

def cos_sim_word(word1, word2):
    vector1 = get_emb(word1)
    vector2 = get_emb(word2)
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

def get_emb(word):
    return model.get_embedding_from_word(word)

def invert_dictionary(dictionary):
    return {v: k for k, v in dictionary.items()}

def get_closest_vector(vector):
    max = 0
    target = None
    for key,item in unique_dict.items():
        comparative = get_emb(key)
        comparison = cos_sim(vector, comparative)
        if comparison > max:
            max = comparison
            target = key

        
    return target


# %%
vector = subtract_vector("king", "man")
vector = vector +get_emb("woman") 


# %%
print(cos_sim_word("flower", "rose"),("flower", "rose"))
print(cos_sim_word("flower", "tree"), ("flower", "tree"))
print(cos_sim_word("flower", "dog"), ("flower", "dog"))
print(cos_sim_word("flower", "cat"), ("flower", "cat"))
print(cos_sim_word("flower", "car"), ("flower", "car"))
print(cos_sim_word("cat", "dog"), ("cat", "dog"))
print(cos_sim_word("king", "queen"), ("king", "queen"))
print(cos_sim_word("king", "royalty"), ("king", "royalty"))
print(cos_sim_word("queen", "royalty"), ("queen", "royalty"))
print(cos_sim_word("man", "king"), ("man", "king"))
print(cos_sim_word("woman", "king"), ("woman", "king"))



# %%
reversed_unique_dict = invert_dictionary(unique_dict)
model.forward_without_negatives("king", "man")

# %%
index1 = unique_dict["king"]
index2 = unique_dict["man"]
vector = subtract_vector("king", "man")
vector = vector+ get_emb("woman")
print(get_closest_vector(vector))

# %%
# save embeddings
path = "embeddings"
model.save_embedding
model.save_embedding(reversed_unique_dict, "embeddings.emb")

# %%
print(model.u_embeddings.mean()))

# %%



