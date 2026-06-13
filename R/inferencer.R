
unembed = function(input, embd, norm, epsilon, temperature){
  input = RMSN(input, norm, epsilon) #norm

  logits = input %*% t(embd)

  probs = softmax(logits / temperature)

  visualize(probs,model)

  pick(probs)
}
pick = function(probs) {
  top = order(probs, decreasing=TRUE)[1:10] #need to reduce the number of options, otherwise sample() gets rounding errors
  top[sample(1:length(top), 1, prob=probs[top])] #pick based on softmax
}
visualize = function(probs, model){
  top = order(probs, decreasing=TRUE)
  chances = round(probs[top[1:8]]*100)
  words = reverse_token_lookup(top[1:8],model)
  words = gsub('\n', '\\\\n', words)
  words = gsub('\t', '\\\\n', words)
  cat(sep='', paste(words, collapse = '\t'), '\n', paste(chances,'%',sep='', collapse = '\t'),'\n')
}



infer = function(prompt,model,instructions=NULL,use_template=TRUE,temperature = 1){
  n_layers = model$metadata$smollm3.block_count$value
  RMSN_epsilon = model$metadata$smollm3.attention.layer_norm_rms_epsilon$value

  start_time = as.numeric(Sys.time())
  pos = 0 #token position; Nth token is at pos N


  # ---- Tokenize Prompt ----
  if (!use_template) input_tokens = tokenize(prompt, model)
  else input_tokens = template(prompt, model, instructions)
  response = paste(collapse='',reverse_token_lookup(input_tokens, model))

  cat('Embedding...\n')
  embedding = readRDS(paste0(model$dir,'embeddings.rds'))
  input = embedding$token_embd.weight[input_tokens,,drop=FALSE]
  rm(embedding); gc(FALSE) #purge embedding stuff from memory (it's huge!)

  # ---- Init KV Cache ----
  heads_kv = model$metadata$smollm3.attention.head_count_kv$value #number of Key/Value heads
  for (i in 0:(n_layers - 1)){
    filename = paste0(model$dir,'blk.',i,'.kv.rds')
    saveRDS(list(k=vector('list',heads_kv),v=vector('list',heads_kv)),filename,compress=FALSE)
  }
  cat('Initialized KV cache.\nPrefilling:\n')

  # ---- Main Loop ----
  for (t in 1:1000){ #max tokens in response

    # ---- Run Layers ----
    #The math is nearly instant; the time taken by each layer is taken purely by loading the .rds.
    for (i in 0:(n_layers - 1)) {
      cat(sep='','\r',rep(' ',100),'\r',progress(i+1,n_layers,12),' Layer ',i,': ')
      layer = readRDS(paste0(model$dir,'blk.',i,'.rds'))
      cat('loaded, ')
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
      cat('\nFinished prefill.')
      tokens_start = as.numeric(Sys.time())
      prefill_time = tokens_start - start_time
    }

    # ---- Pick Token ----
    cat('\nUnembedding...\n')
    embedding = readRDS(paste0(model$dir,'embeddings.rds')) #reload embedding matrices

    cat('\fPredictions for the last token:\n') #clear console
    output = unembed(input, embedding$token_embd.weight, embedding$output_norm.weight, RMSN_epsilon, temperature) #get token id

    #check for end of response
    if (output == model$metadata$tokenizer.ggml.eos_token_id$value + 1) return() #R is 1-indexed! dinnae forget!
    else response = paste0(response, reverse_token_lookup(output, model))

    total_token_time = (as.numeric(Sys.time()) - tokens_start)/(t-1)
    cat(sep='',
        '------------\n',
        response,
        '\n------------\n',
        'Prefill in ', round(prefill_time), ' seconds (', round(prefill_time/length(input_tokens)), ' tps).\n',
        'token #', t, ' | ', time_since(start_time), ' elapsed.\n',
        round(total_token_time), ' seconds per token (', round(1/total_token_time,4), ' tps).\n'
    )


    input = embedding$token_embd.weight[output,,drop=FALSE]

    rm(embedding); gc(FALSE) #purge embedding from memory
  }

}



