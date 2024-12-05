from transformers import BertModel, AutoTokenizer
from scipy.spatial.distance import cosine
import numpy as np
import torch

model_name = "bert-base-cased"
model = BertModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# text= "when life gives you lemons,"

def predict(text):
    encoded_inputs = tokenizer(text, return_tensors='pt')
    return  model(**encoded_inputs)[0]

sentence1 = "There was a fly drinking from my soup." 
sentence2 = "to become a comercial pilot he had to fly 100 hours."
sentence3 = "a fly sat on my leg today, i was very anoyyed."


token1 =tokenizer.tokenize(sentence1)
token2 =tokenizer.tokenize(sentence2)

out1=predict(sentence1)
out2=predict(sentence2)

emb1 = out1[0:,token1.index("fly"),:].detach()
emb2 = out2[0:,token2.index("fly"),:].detach()

emb1.shape
emb2.shape

cosine(emb1,emb2)
np.sqrt(np.sum())
