# Seconda versione del codice
# Codice contenente la classe ScaledDotProductAttention che contiene la funzione scaled dot product attention
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        temp = query.bmm(key.transpose(1, 2))
        #  bmm sta per batch maxtrix-matrix product
        # viene computato il prodotto tra query e key
        # il risultato è un tensore di dimensione `batch_size, query_length, key_length`
        # con:
        #    - batch_size = grandezza del batch
        #    - query_length = lunghezza della query
        #    - key_length = lunghezza della key

        scale = query.size(-1) ** 0.5
        # viene calcolata la radice quadrata della lunghezza della query

        softmax = nn.functional.softmax(temp / scale, dim=-1)
        # viene calcolata la softmax del tensore temp dopo averlo scalato per `scale`

        output = softmax.bmm(value)
        return output
