## Prima versione del codice, non racchiusa in un modulo PyTorch, vedere la versione successiva in scaled_dot_product_attention.py

# Codice contenente la funzione scaled dot product attention
from torch import Tensor
import torch.nn.functional as f

def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
# Questa funzione prende in input tre tensor, query, key e value, e restituisce un tensore
    
    temp = query.bmm(key.transpose(1, 2))
    # bmm sta per batch maxtrix-matrix product
    # viene computato il prodotto tra query e key
    # il risultato è un tensore di dimensione `batch_size, query_length, key_length`
    # con: 
    #    - batch_size = grandezza del batch
    #    - query_length = lunghezza della query
    #    - key_length = lunghezza della key

    scale = query.size(-1) ** 0.5
    # viene calcolata la radice quadrata della lunghezza della query

    softmax = f.softmax(temp / scale, dim=-1)
    # viene calcolata la softmax del tensore temp dopo averlo scalato per `scale`

    return softmax.bmm(value)
