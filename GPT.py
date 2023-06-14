import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4 # change 11 - learning rate decrease 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # we will not call .backward to save memory
def estimate_loss():
    out = {}
    model.eval() # eval phase 
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # setting it back to training phase 
    return out

# Self-attention head # CHANGE 7
class Head(nn.Module):

    """ one-head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query =nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x) #(B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ('affinities')
        wei = q @ k.transpose(-2,-1) * (C**(-0.5)) # (B,T,C) * (B,C,T) == (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim= -1) # (B,T,T)
        wei = self.dropout(wei)
        # perfrom the weighted aggreagation of the values
        v = self.value(x)
        out = wei @ v # [B,T,T] @ [B,T,C] -> (B,T,C)
        return out

#change 13 - Multi Headed attention

class MultiHeadAttention(nn.Module):

    def __init__(self,num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.dropout = nn.Dropout(dropout)
        #CHANGE RESIDUAL
        
        self.proj = nn.Linear(n_embd, n_embd) #projection part
    def forward(self,x):
        #CHANGE RESIUDAL
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out) # CHANGE RESIDUAL
        return out

# implementing the Feed Forward for computation - CHANGE 15
class FeedForward(nn.Module):
    '''a simple linear layer folowed by non-linearity'''

    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4 * n_embd), 
            nn.ReLU(),    
            nn.Linear(4 * n_embd,n_embd), 
            nn.Dropout(dropout),
        )
        #CHANGE RESIDUAL also note the mult of 4 to n_embd
    def forward(self,x):
        return self.net(x)

#CHANGE 16 
class Block(nn.Module):

    '''Transformer block: communication followed by computation'''
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embd)
        #CHANGE - add layer norm but not exactly like the paper
        self.ln1 = nn.LayerNorm(n_embd) # common practice to do before feeding 
        self.ln2 = nn.LayerNorm(n_embd) # # common practice to do before feeding 
        
        # This is the only slight deviation from the paper 'Attention is All you Need'
        # PreNorm Activation Formulation


    def forward(self,x):
        # x = self.sa(x) without forks residual
        # x = self.ffwd(x) # without forks residual
        #also layer norms here
        x = x + self.sa(self.ln1(x)) # CHANGE RESIDUAL - fork off
        x = x + self.ffwd(self.ln2(x)) # CHANGE RESIDUAL - fork off
        return x 
# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # CHANGE 1
        # to go from token to logits we are going to need linear layer
        self.lm_head = nn.Linear(n_embd, vocab_size) # CHANGE 3 lm = language model head

        # Comment (start)
        # so, far we have taken these idx and we have 
        # encoded them based on the identity of the tokens
        #inside idx. So, what is oftenly done in practice -
        # we are not only just encoding the identity of these tokens
        # but also there positions, so we are going to get second
        # position embdeddin table self.position
        #Comment (end)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # we lost this info when we took avg so this reassures that we putting pos_emb back
        # and giving the information back again 
        # CHANGE 5

        #CHANGE 8
        # self.sa_head = Head(n_embd)
        #CHANGE 16 - init ffwd
        self.ffwd = FeedForward(n_embd)
        #CHANGE 14
        self.sa_heads = MultiHeadAttention(4, n_embd//4)
        # i.e. 4 heads of 8 dimensional self_attention
        # #CHANGE 17 increase the number of heads
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head = 4),
        #     Block(n_embd, n_head = 4),
        #     Block(n_embd, n_head = 4),
        #     nn.LayerNorm(n_embd) 
        # )
        # CHANGE - Scaling up 
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.ln_final = nn.LayerNorm(n_embd)

    def forward(self, idx, targets=None):

        B,T = idx.shape # CHANGE 6

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) CHANGE 2
        #CHANGE 6
        pos_emb = self.position_embedding_table(torch.arange(T,device = device)) # T,C
        # torch.arange(T int from 0 to T-1
        x = tok_emb + pos_emb # (B,T,C) # again broadcasted
        #CHANGE 9 - self-attention head
        # x = self.sa_heads(x) # apply one of head self-attn(B,T,C)
        x = self.blocks(x) # B,T,C
        # #CHANGE 17 - adding of ffwd
        # x = self.ffwd(x)
        x = self.ln_final(x) # B,T,C
        logits = self.lm_head(x) # B,T,vocab_size # CHANGE 4
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #CHANGE 10 - crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, "M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
