# Progetto di Introduzione all'Intelligenza Artificiale 2022/2023

Relazione estesa del progetto su [Notion](https://feather-floss-434.notion.site/Progetto-e13990b2b78641fa8b761475bc1c815d)
## Struttura del repository
 
models  
  [attention_head.py](models/attention_head.py) `il modulo Attention Head`  
  [decoder.py](models/decoder.py) `il modulo Decoder`  
  [encoder.py](models/encoder.py) `il modulo Encoder`  
  [multi_head_attention.py](models/multi_head_attention.py) `il modulo Multi-Head Attention`  
  transformer.py `il modulo Transformer`  
utilities 
  [feed_forward.py](utilities/feed_forward.py) `il Point-wise feed-forward network`  
  [positional_encoding.py](utilities/positional_encoding.py) `Lo schema di Positional Encoding per il modello`  
  [residual.py](utilities/residual.py) `il modulo Residual`  
  [scaled_dot_product_attention.py](utilities/scaled_dot_product_attention.py) `script Scaled Dot-Product Attention`  
  test.py `script per testare il modello`  


## Codice
Il codice è stato scritto in Python 3.9.15 e utilizza le librerie PyTorch 1.13.1.

### Scaled Dot-Product Attention
Partendo dalle basi, è necessario implementare una funzione di attenzione, che data una query, una chiave e un valore, restituisca un tensore di output. Con la seguente formula:

$$ 
\begin{align}
\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V 
\end{align}
$$

Q, K e V sono batch di matrici, ciascuno con dimensione (batch_size, seq_len, num_features). Moltiplicando Q e K otteniamo un tensore di dimensione (batch_size, seq_len, seq_len), che rappresenta la matrice di attenzione. Per ottenere un valore più stabile, dividiamo ogni elemento per la radice quadrata della dimensione della query. Infine, moltiplichiamo il tensore ottenuto per V, ottenendo il tensore di output.
Visualizza il codice [qui](scripts/scaled_dot_product_attention.py)

### AttentionHead e MultiHeadAttention
Possiamo procedere con lo sviluppo dei moduli AttentionHead e MultiHeadAttention.
Consiglio di visualizzare il grafico per capire meglio la composizione di questi moduli:  
![alt-text](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/1078abb6-b313-4b34-9393-d5b5581d8a16/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230225%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230225T183322Z&X-Amz-Expires=86400&X-Amz-Signature=82428c1d2c63ef918906d43646e33101221d5c7fcf3ebefc7c61e24c12a8f5bb&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

- AttentionHead è composta da tre strati lineari, che applicano una trasformazione lineare ai dati di input. Il primo strato lineare applica una trasformazione alla query, il secondo alla chiave e il terzo al valore. Il risultato di questi strati lineari viene passato alla funzione di attenzione, che restituisce un tensore di output. Il tensore di output viene passato ad un ulteriore strato lineare, che applica una trasformazione lineare ai dati di input. Il risultato di questo strato lineare viene restituito come output del modulo AttentionHead.

- MultiHeadAttention è composta da un numero di AttentionHead, che applicano una trasformazione lineare ai dati di input. Il risultato di questi strati lineari viene passato ad un ulteriore strato lineare, che applica una trasformazione lineare ai dati di input. Il risultato di questo strato lineare viene restituito come output del modulo MultiHeadAttention.

Visualizza il rispettivo codice [qui](models/attention_head.py) e [qui](models/multi_head_attention.py)

### Positional Encoding
È necessario implementare un modulo che aggiunge un embedding di posizione ai dati di input. Questo embedding viene aggiunto alla somma tra i dati di input e l'embedding di posizione. L'embedding di posizione è calcolato come segue:

$$
\begin{align}
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}}) \\
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})
\end{align}
$$

Visualizza il codice [qui](models/positional_encoding.py)

### Feed Forward
Notiamo che ogni layer nel nostro Encoder e Decoder contiene un Fully Connected Feed-Forward Network, che consiste di due trasformazioni lineari con una attivazione ReLU in mezzo.
Le dimensioni dell'input e dell'output sono 512, e la dimensione del layer interno è di 2048.

Visualizza il codice [qui](models/feed_forward.py)

### Residual
Il modulo Residual possiede tre variabili:
- sublayer, che rappresenta il modulo che viene eseguito all'interno del modulo Residual
- norm, un layer di normalizzazione 
- dropout, un layer di dropout
Nel metodo forward, il modulo Residual prende un numero di tensori di input come argomento, applica a loro il modulo sublayer, a cui aggiunge il tensore originale di input, e applicai il
layer di normalizzazione e infine applica il dropout. Viene restituito il tensore di output.

Visualizza il codice [qui](models/residual.py)

### Encoder
Possiamo finalmente procedere con l'implementazione del layer di encoding e dell'encoder stesso.
Vedi [qua](https://www.notion.so/Progetto-e13990b2b78641fa8b761475bc1c815d?pvs=4#e8fe1b4c907f421ba17653f6d71aad3c) il paragrafo dedicato.

Visualizza il codice [qui](models/encoder.py)

### Decoder
Vedi [qua](https://www.notion.so/Progetto-e13990b2b78641fa8b761475bc1c815d?pvs=4#f9f7bbb08aa94b84ad7b315d91bb1295) il paragrafo dedicato.

Visualizza il codice [qui](models/decoder.py)

### Transformer
Infine è sufficiente unire Encoder e Decoder, come visto in precedenza dal grafico.
![alt-text](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/f26c3908-0b2a-4c70-9740-602231cd23f4/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20230226%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20230226T164029Z&X-Amz-Expires=86400&X-Amz-Signature=a79a744bc95888465f67729cc829d0aebe7180610ceed57426e20ed7b9c045c1&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

Visualizza il codice [qui](models/transformer.py)
