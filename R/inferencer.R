
unembed = function(input, embd, norm, epsilon){
  input = RMSN(input, norm, epsilon) #norm

  logits = input %*% t(embd)
  probs = softmax(logits)

  pick(probs)
}
pick = function(probs){
  top5 = order(probs, decreasing=TRUE)[1:5]
  cat(sep='', paste(reverse_token_lookup(top5), collapse = '\t\t'), '\n', paste(round(probs[top5]*100), collapse = '\t\t'),'\n')

  sample(1:length(probs), 1, prob=probs) #pick based on softmax
}


infer = function(prompt,model){
  n_layers = model$metadata$smollm3.block_count$value
  RMSN_epsilon = model$metadata$smollm3.attention.layer_norm_rms_epsilon$value

  response = prompt
  pos = 0 #token position; Nth token is at pos N

  # ---- Tokenize Prompt ----
  cat('Tokenizing:')
  input_tokens = tokenize(prompt, model)
  embedding = readRDS(paste0(model$dir,'embeddings.rds'))
  input = embedding$token_embd.weight[input_tokens,,drop=FALSE]
  rm(embedding); gc(FALSE) #purge embedding stuff from memory (it's huge!)

  # ---- Init KV Cache ----
  heads_kv = model$metadata$smollm3.attention.head_count_kv$value #number of Key/Value heads
  for (i in 0:(n_layers - 1)){
    filename = paste0(model$dir,'blk.',i,'.kv.rds')
    saveRDS(list(k=vector('list',heads_kv),v=vector('list',heads_kv)),filename,compress=FALSE)
  }

  # ---- Main Loop ----
  for (t in 1:1000){ #max tokens in response

    # ---- Run Layers ----
    #The math is nearly instant; the time taken by each layer is taken purely by loading the .rds.
    for (i in 0:(n_layers - 1)) {
      cat(sep='','\r',progress(i,n_layers),' Transforming... (layer ',i,')')
      layer = readRDS(paste0(model$dir,'blk.',i,'.rds'))
      layer$id = i
      input = Attention(input, layer, model, pos)
      input = FFN(input, layer, model)
      rm(layer); gc(FALSE) #purge layer from memory, since it would get purged later anyway
    }
    pos = pos + 1 #increment position for a new token

    # ---- Finish Prefill ----
    if (t == 1){
      input = input[nrow(input),,drop=FALSE] #cut off the end of input
      pos = length(input_tokens) #set position to end of prompt
      cat('\nFinished prefill. Running model:')
    }

    # ---- Pick Token ----
    cat('\nUnembedding...\n')
    embedding = readRDS(paste0(model$dir,'embeddings.rds')) #reload embedding matrices

    output = unembed(input, embedding$token_embd.weight, embedding$output_norm.weight, RMSN_epsilon) #get token id
    response = paste0(response, reverse_token_lookup(output,model))
    print(response)
    #CHECK FOR EOF

    input = embedding$token_embd.weight[output,,drop=FALSE]

    rm(embedding); gc(FALSE) #purge embedding from memory
  }

}



