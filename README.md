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

#### AttentionHead e MultiHeadAttention
Possiamo procedere con lo sviluppo dei moduli AttentionHead e MultiHeadAttention.
Consiglio di visualizzare il grafico per capire meglio la composizione di questi moduli:
![alt-text](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/1078abb6-b313-4b34-9393-d5b5581d8a16/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230225%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230225T183322Z&X-Amz-Expires=86400&X-Amz-Signature=82428c1d2c63ef918906d43646e33101221d5c7fcf3ebefc7c61e24c12a8f5bb&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

- AttentionHead è composta da tre strati lineari, che applicano una trasformazione lineare ai dati di input. Il primo strato lineare applica una trasformazione alla query, il secondo alla chiave e il terzo al valore. Il risultato di questi strati lineari viene passato alla funzione di attenzione, che restituisce un tensore di output. Il tensore di output viene passato ad un ulteriore strato lineare, che applica una trasformazione lineare ai dati di input. Il risultato di questo strato lineare viene restituito come output del modulo AttentionHead.

- MultiHeadAttention è composta da un numero di AttentionHead, che applicano una trasformazione lineare ai dati di input. Il risultato di questi strati lineari viene passato ad un ulteriore strato lineare, che applica una trasformazione lineare ai dati di input. Il risultato di questo strato lineare viene restituito come output del modulo MultiHeadAttention.

Visualizza il rispettivo codice [qui](models/attention_head.py) e [qui](models/multi_head_attention.py)
