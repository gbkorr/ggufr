
#"especially interesting is tracking the dimensions as it goes trhough the model"
#  warning('in the qmd, step through this process for the input layer, showing graphs of the image during.')
# can I animate this? watching the bar for each layer is quite boring

#please add formulae for each of these
softmax = function(v){x = exp(v - max(v)); x / sum(x)}
attend = function(q,k,v,head_dim,mask){ #qkv heads
  qk = t(apply( #apply softmax to each row
    (q %*% t(k))/sqrt(head_dim) + mask,
  1,softmax))
  # qk: [n_tokens, n_tokens]

  qk %*% v
}
SiLU = function(x) x/(1 + exp(-x))
RMSN = function(v, norm, epsilon) norm * v / sqrt(mean(v^2) + epsilon)
RoPE = function(v, pos, base, dims){
  angles = pos * base ^ -(2*(1:(dims/2) - 1)/dims)

  #interleaved?
  odds = seq(1, dims, by = 2)
  evens = seq(2, dims, by = 2)

  X = v[odds]
  Y = v[evens]

  #treat the first half of the dimensions as x positions, and the second half as y
  #rotate these coordinate pairs, and return the new values

  v[odds] = X * cos(angles) - Y * sin(angles)
  v[evens] = Y * cos(angles) + X * sin(angles)

  v
}


#pass state matrix through a transformation layer
Attention = function(input, layer, model, pos){
  #If loading the model, input has 2048 columns and rows equal to the number of tokens in the prompt.
  #If generating a token, input is 1 x 2048. (2048 being the embedding length). MUST be a matrix, not a vector!

  #pos = In the entire conversation, this is the Nth token.

  #input: [n_tokens, embedding_length] <- indicates dimension of a variable; [rows, columns]

  # ---- Variables ----
  n_tokens = nrow(input)
  embedding_length = ncol(input)

  RMSN_epsilon = model$metadata$smollm3.attention.layer_norm_rms_epsilon$value
  rope_base = model$metadata$smollm3.rope.freq_base$value
  rope_dims = model$metadata$smollm3.rope.dimension_count$value

  heads_q = model$metadata$smollm3.attention.head_count$value #number of Query heads
  heads_kv = model$metadata$smollm3.attention.head_count_kv$value #number of Key/Value heads
  head_dim = embedding_length / heads_q #values per head

    cat('norm, ')

  # ---- Normalize Input ----
  NormIn = t(apply(input, 1, function(x) RMSN(x, layer$attn_norm.weight, RMSN_epsilon))) #normalize by row
  # NormIn: [n_tokens, embedding_length]

    cat('QKV, ')

  # ---- Get New Query/Key/Value Matrices ----
  Q = NormIn %*% t(layer$attn_q.weight)
  K = NormIn %*% t(layer$attn_k.weight)
  V = NormIn %*% t(layer$attn_v.weight)
  # Q: [n_tokens, embedding_length]
  # KV: [n_tokens, head_dim * heads_kv]

    cat('heads, ')

  # ---- Break QKV into Heads ----
  q = vector('list',heads_q)
  k = vector('list',heads_kv)
  v = vector('list',heads_kv)
  for (i in 1:heads_q) {
    q[[i]] = Q[,head_dim * (i - 1) + 1:head_dim,drop=FALSE]
  }
  for (i in 1:heads_kv) {
    k[[i]] = K[,head_dim * (i - 1) + 1:head_dim,drop=FALSE]
    v[[i]] = V[,head_dim * (i - 1) + 1:head_dim,drop=FALSE]
  }
  # q: list of length heads_q, [n_tokens, head_dim]
  # kv: list of length heads_kv, [n_tokens, head_dim]
  #note that while there are more Q heads than KV, all heads have the same dimensions.

  # ---- Apply RoPE ----
  #SmolLM3 skips every fourth layer (and calls this technique "NoPE", but I like to call it "Skipping Rope").
  if (layer$id %% 4 == 3) cat('NoPE, ')
  else {
    cat('RoPE, ')

    #apply RoPE to each Q head, row-by-row
    #(remember, when generating a regular token, there's only one row)
    for (i in 1:heads_q) {
      for (t in 1:n_tokens) q[[i]][t,] = RoPE(q[[i]][t,], pos + t - 1, rope_base, rope_dims) #position increases for each token
    }
    #repeat for the K heads
    for (i in 1:heads_kv) {
      for (t in 1:n_tokens) k[[i]][t,] = RoPE(k[[i]][t,], pos + t - 1, rope_base, rope_dims)
    }
  }

    cat('cache, ')

  # ---- Retrieve and Cache KV ----
  cache = paste0(model$dir,'blk.',layer$id,'.kv.rds')
  kv = readRDS(cache)
  for (i in 1:heads_kv) {
    k[[i]] = rbind(kv$k[[i]], k[[i]]) #append new (rope-rotated) values as the bottom (newest) row
    v[[i]] = rbind(kv$v[[i]], v[[i]]) #v wasn't rotated
    kv$k[[i]] = k[[i]]
    kv$v[[i]] = v[[i]]
  } #this is slow :(
  saveRDS(kv, cache, compress=TRUE)
  # kv: list of length heads_kv, [n_all_tokens_so_far, head_dim]

  # ---- Apply Attention ----
  ctx = vector('list',heads_q)
  mask = matrix(0,nrow(q[[1]]),nrow(k[[1]])) #for regular prediction, don't mask anything
  if (n_tokens != 1) mask[upper.tri(mask)] = -Inf #causal mask for prefill; each token can only respond to previous ones
  for (i in 1:heads_q) { #get ctx for each Q head
    kv_head = ceiling(i/(heads_q/heads_kv)) #which kv head? (each kv head is used for 4 q heads, hence GQA architecture)
    ctx[[i]] = attend(q[[i]], k[[kv_head]], v[[kv_head]], head_dim, mask)
  }
  # ctx: list of length heads_q, [n_tokens, head_dim]

    cat('ctx, ')

  # ---- Reassemble Full Matrix ----
  Context = matrix(0, n_tokens, embedding_length)
  for (i in 1:heads_q) Context[,head_dim * (i - 1) + 1:head_dim] = ctx[[i]] #recombine ctx heads, the inverse of how we broke up Q
  # Context: [n_tokens, embedding_length]

  # ---- Project Output ----
  Out = Context %*% t(layer$attn_output.weight)
  # Out: [n_tokens, embedding_length]

  # ---- Add to Input ----
  input + Out
}

FFN = function(input, layer, model) {
  # ---- Variables ----
  n_tokens = nrow(input)
  embedding_length = ncol(input)

  RMSN_epsilon = model$metadata$smollm3.attention.layer_norm_rms_epsilon$value
  feed_forward_length = model$metadata$smollm3.feed_forward_length$value

    cat('norm, ')

  # ---- Normalize Input ----
  NormIn = t(apply(input, 1, function(x) RMSN(x, layer$ffn_norm.weight, RMSN_epsilon))) #normalize by row
  # NormIn: [n_tokens, embedding_length]

    cat('up, ')

  # ---- Project Up ----
  G = NormIn %*% t(layer$ffn_gate.weight) #Gate
  U = NormIn %*% t(layer$ffn_up.weight) #Up Weights
  # G, U: [n_tokens, feed_forward_length]

    cat('gate, ')

  # ---- Activate Weights ----
  Activation = SiLU(G) * U
  # Activation: [n_tokens, feed_forward_length]

    cat('down.')

  # ---- Project Down ----
  Out = Activation %*% t(layer$ffn_down.weight)
  # Out: [n_tokens, embedding_length]

  # ---- Add to Input
  input + Out
}


