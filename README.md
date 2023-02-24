# Progetto di Introduzione all'Intelligenza Aritificiale 2022/2023

## Struttura del repository
 
models  
  attention_head.py `il modulo Attention Head`  
  decoder.py `il modulo Decoder`  
  encoder.py `il modulo Encoder`  
  feed_forward.py `il Point-wise feed-forward network`  
  multi_head_attention.py `il modulo Multi-Head Attention`  
  positional_encoding.py `Lo schema di Positional Encoding per il modello`  
  residual.py `il modulo Residual`  
  transformer.py `il modulo Transformer`  
scripts  
  scaled_dot_product_attention.py `script Scaled Dot-Product Attention`  
  test.py `script per testare il modello`  


## Codice
Il codice è stato scritto in Python 3.9.15 e utilizza le librerie PyTorch 1.13.1.

#### Scaled Dot-Product Attention
Partendo dalle basi, è necessario implementare una funzione di attenzione, che data una query, una chiave e un valore, restituisca un tensore di output. Con la seguente formula:

$$ 
\begin{align}
\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V 
\end{align}
$$

Q, K e V sono batch di matrici, ciascuno con dimensione (batch_size, seq_len, num_features). Moltiplicando Q e K otteniamo un tensore di dimensione (batch_size, seq_len, seq_len), che rappresenta la matrice di attenzione. Per ottenere un valore più stabile, dividiamo ogni elemento per la radice quadrata della dimensione della query. Infine, moltiplichiamo il tensore ottenuto per V, ottenendo il tensore di output.
Visualizza il codice [qui](scripts/scaled_dot_product_attention.py)
