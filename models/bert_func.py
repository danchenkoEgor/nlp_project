import numpy as np
import torch
import transformers
import joblib

def load_bert_lr_model(path):   
    
    model_class = transformers.BertModel
    tokenizer_class = transformers.BertTokenizer
    pretrained_weights = 'bert-base-uncased'
    model = model_class.from_pretrained(pretrained_weights)
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    lr = joblib.load(path)
    return model, tokenizer, lr

def prediction(text, model, tokenizer, lr, max_len=256):
    input = tokenizer.encode(text,
                            add_special_tokens=True, 
                            padding='max_length',
                            truncation=True,
                            return_tensors='np',
                            max_length=max_len)
    
    att_mask = np.where(input != 0, 1, 0)
    input = torch.tensor(input)
    att_mask = torch.tensor(att_mask)

    last_hidden_states = model(input, attention_mask=att_mask)
    vector = last_hidden_states[0][:,0,:].detach().numpy()
    pred = lr.predict(vector)[0]

    if pred == 1:
        result = 'Positive review'
    else:
        result = 'Negative review'

    return result