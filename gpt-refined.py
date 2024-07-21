# import os
# import time
# import inspect
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import time
import inspect
# from deepeval.benchmarks import HellaSwag
# from hellaswag import render_example, iterate_examples
# -----------------------------------------------------------------------------
class Dataloader():
    def __init__(self,filename,B,T):
        self.sequence_start = 0
        self.B = B
        self.T = T
        with open(file=filename,mode='r') as f:
            self.text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        self.tokens = enc.encode(self.text)
        self.tokens = torch.tensor(self.tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch: {len(self.tokens) // (B*T)} batches')
        
    def get_next_batch(self):
        next_chunk = self.tokens[self.sequence_start:self.sequence_start+(self.B*self.T)+1]
        #print(next_chunk) # [1,2,3,4,5,6,7,.....41]
        x = (next_chunk[:-1]).view(self.B,self.T) # [1,2,3,4,5,6,7,8], [9,..], [], [] , [..40]
        y = (next_chunk[1:]).view(self.B,self.T)  # [2,3,4,] ...............[..41]

        self.sequence_start+=(self.B*self.T)
        if self.sequence_start+(self.B*self.T)+1 >= len(self.tokens):
            # 1 epoch completed.
            self.sequence_start = 0

        return x,y

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd,out_features=3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size))
                                    .view(1,1,config.block_size,config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #  (B,T,nh,hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #   (B,T,nh,hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #  (B,T,nh,hs)
        #y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention  (B,T,nh,hs) * (B,T,hs,nh)=(B,T,nh,nh)*(B,T,nh,hs)=(B,T,nh,hs)
        
        # att = (q @ k.transpose(-2,-1)*(1.0/math.sqrt(k.size(-1))))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0,float('-inf'))
        # att = F.softmax(att,dim=-1)
        # y = att @ v

        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)     #x7.6 times faster than normal python compiler (From FastAttention Research Paper by NVIDIA)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(in_features=4 * config.n_embd,out_features=config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd),
            wpe = nn.Embedding(config.block_size,config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)

        #In original GPT-2 Paper,Embedding weights and last linear weights are same.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self.__init_weights__)
    
    def __init_weights__(self,module):
        std = 0.02
        if isinstance(module,nn.Linear):
            if hasattr(module,"NANOGPT_SCALE_INIT"):
                std*=(2*self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight,mean=0.0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)


    def forward(self,idx,device,target=None):
        B,T = idx.size()
        assert T<=self.config.block_size #Tokens for generation less than block_size.

        pos = torch.arange(0,T,device=device) # shape of tensor (T)
        tok = self.transformer.wte(idx)  #(B,T,C)
        pos = self.transformer.wpe(pos) # shape (T,C)
        x = pos+tok # (B,T,C)

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),target=target.view(-1))

        return logits,loss  #(B,T,vocab_size)
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}

        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

#--------------------------------------------------------------------------
# Learning Rate Scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(step):
    # for steps less than warm_up steps.
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # for steps greater than max_steps, use 10% of org lr.
    if step > max_steps:
        return min_lr
    # for in-between steps use cosine-decay.
    decay_Ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0<= decay_Ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_Ratio))
    return  min_lr + coeff*(max_lr - min_lr)
#--------------------------------------------------------------------------

# check if device is cpu or gpu.
if torch.cuda.is_available(): device = 'cuda'
else: device= 'cpu'

torch.set_float32_matmul_precision('high')       #truncating bits (mantissa 10 bits) from last

total_batch_size= 524288
B = 16
T = 1024
assert total_batch_size % (B * T) == 0
grad_accum_steps= total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"caculated grad accum steps: {grad_accum_steps}")
#configure model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# model = torch.compile(model)

#print parameters of model
print(f'Model Parameters: {sum(p.numel() for p in model.parameters())/1e6} M Paramters.')

# Get batch of data.
data = Dataloader(filename='input.txt',B=8,T=512)

# Optimize!
max_steps= 50
# optimizer = torch.optim.AdamW(params=model.parameters(),lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay = 0.1, learning_Rate=6e-4, device=device)
for step in range(max_steps):
    t0 = time.time()
    loss_accum = 0.0
    optimizer.zero_grad()
    for micro_step in range(grad_accum_steps):
        x,y =data.get_next_batch()
        x,y = x.to(device),y.to(device)
      
    # with torch.autocast(device_type=device, dtype=torch.bfloat16):          #bfloat16 is applied to only forward pass activations, not on backward pass or gradiants
        logits,loss = model(x,device,y)
        loss= loss/grad_accum_steps
        loss_accum += loss.detach() 
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

    lr = get_lr(step)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    # torch.cuda.synchronize()    #waiting for GPU to finish work
    t1 = time.time()
    dt = (t1 - t0)*1000          # time difference in ms
    tokens_per_sec = (data.B * data.T )/ (t1 - t0)

    print(f'Step {step} Loss:{loss.item()},lr: {lr: .4f}, dt: {dt: .2f}ms, token_per_sec: {tokens_per_sec: .2f}')

import sys; sys.exit(0)

# import tiktoken
# enc=tiktoken.get_encoding("gpt2")
# encoded=enc.encode("Hello, I'm a language model,")
# tokens = torch.tensor(encoded,dtype = torch.long)

# tokens=tokens.unsqueeze(0).repeat(num_sequences,1) #Repeat 5 times along row axis and 1 time only col axis.
# tokens = tokens.to(device=device)

# prediction phase.
model.eval()
num_sequences = 5
max_length = 30

#Do printing till max Length of Sequence.
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while tokens.size(1) < max_length:

    with torch.no_grad():
        logits=model.forward(tokens,device)
        logits = logits[:,-1,:] # get last token logits for each batch.

        probs=torch.softmax(logits,dim=-1)
        topk , topk_indices=torch.topk(probs,k=50,dim=-1)

        ix = torch.multinomial(topk,1) #select a token from top-k probabilites. #(B,1)
        xcol = torch.gather(topk_indices,dim=-1,index=ix)

        tokens = torch.cat((tokens,xcol),dim=1)

enc = tiktoken.get_encoding('gpt2')
#Print generated text.
for i in range(num_sequences):
    tokens_of_i_row = tokens[i,:max_length].tolist()
    tok=enc.decode(tokens_of_i_row)
    print(">",tok)
