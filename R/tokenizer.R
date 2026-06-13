


#we need to match the provided template (model$metadata$tokenizer.chat_template$value), because it's strongly baked into RoPE
template = function(prompt){
  #interestingly, it appears that "<|im_start|>" and related control words are not in the vocabulary.
  paste0(
    "<|im_start|>system\n",
    "## Metadata\n\n",
    "Knowledge Cutoff Date: June 2025\n",
    "Today Date: ", strftime(Sys.time(),"%d %B %Y"),"\n",
    "Reasoning Mode: /no_think\n\n",
    "## Custom Instructions\n\n",
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face.\n\n",
    "\n\n",
    "<|im_end|>\n",
    "<|im_start|>user\n",prompt,"<|im_end|>\n"
  )
  #\n is handled properly by the tokenizer, shows up as "Ċ" after ggml_encode
}
#to speed things up, we could actually pre-tokenize most of this, but it doesn't matter much.



BPE = function(str,model){
  tokens = strsplit(str,'') |> unlist() #start with every character as a token
  for (t in 1:length(tokens)) tokens[t] = ggml_encode(tokens[t]) #convert invis characters to something visible, I hate this step

  con = rawConnection(model$raw_merges)
  on.exit(close(con))
  count = model$merge_size

  #merge until finished
  while(TRUE){
    if (length(tokens) == 1) return(tokens) #only one token remains; exit

    #paste raw pairs of tokens (to match the merge format)
    pairs = {output=list(); for (i in 1:(length(tokens) - 1)) output[[length(output)+1]] = c(charToRaw(tokens[i]),charToRaw(' '),charToRaw(tokens[i+1])); output} #merge pairs are separated by a space
    abort = FALSE

    #return to start of merges and read them one by one, applying the first pair that matches the tokens
    seek(con, 0)
    for (i in 1:count) {
      merge = readBin(con,'raw',readBin(con,'integer',size=8)) #gguf_string (raw bytes)
      for (pos in 1:length(pairs)){
        pair = pairs[[pos]]
        if (length(pair) == length(merge) && all(pair == merge)){ #same length of bytes and all bytes match
          tokens[pos] = paste0(tokens[pos],tokens[pos+1]) #actually merge the tokens
          tokens = tokens[-(pos+1)] #merging two tokens means one less total tokens
          abort = TRUE
          break
        }
      }
      if (abort) break
      if (i == count) return(tokens) #no merges found; exit
    }
  }
}

token_lookup = function(tokens,model){
  con = rawConnection(model$raw_tokens)
  on.exit(close(con))
  count = model$vocab_size

  output = rep(NA,length(tokens))
  tokens = lapply(tokens,charToRaw) #convert to a list of raw strings

  for (i in 1:count) {
    raw =  readBin(con,'raw',readBin(con,'integer',size=8)) #gguf_string (raw bytes)
    for (pos in 1:length(tokens)){
      token = tokens[[pos]]
      if (length(token) == length(raw) && all(token == raw)) output[pos] = i #id of matching string
    }
  }

  output
}
reverse_token_lookup = function(tokens, model){
  con = rawConnection(model$raw_tokens)
  on.exit(close(con))
  count = model$vocab_size

  output = rep(NA,length(tokens))

  for (i in 1:count) {
    raw =  readBin(con,'raw',readBin(con,'integer',size=8)) #gguf_string (raw bytes)
    output[i == tokens] = rawToChar(raw) |> ggml_decode()
    if (!sum(is.na(output))) break
  }

  output
}

#tokenize("Hello, how are you?",model) |> reverse_token_lookup(model) |> paste0(collapse='')


tokenize = function(prompt,model){
  library(re) #fancy regex package required

  #common regex (defined by model$metadata$tokenizer.ggml.pre$value, but it's usually this regex)
  #see `class Encoder` in reference
  regex = r"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"
  match = re_match(regex,prompt) |> unlist()

  chunks = match
  tokens = c()
  for (i in 1:length(chunks)) {
    cat(sep='','\r',progress(i,length(chunks)),' Merging tokens... (chunk ',i,')')
    tokens = c(tokens, BPE(chunks[i],model))
  }
  cat('\n')

  print(tokens)
  token_lookup(tokens,model)
}

#tokenizing speed will mainly be limited by how many non-single-token words there are, since every chunk that isn't a full token will have to search the entire merges list to confirm no merge exists



#tokenizing based on original reference: https://github.com/karpathy/minGPT/blob/master/mingpt/bpe.py
#(permalink: https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/bpe.py)

bytes_to_unicode = function(){
  #bad day to be 1-indexed
  clean = c(
    utf8ToInt("!") : (utf8ToInt("~") + 1),
    utf8ToInt("¡") : (utf8ToInt("¬") + 1),
    utf8ToInt("®") : (utf8ToInt("ÿ") + 1)
  )
  #these 188 characters render normally; the rest of the 256-bit chars get mapped upward

  dict = vector('integer',256)
  n = 0
  for (char in 0:255){
    if (char %in% clean) dict[char + 1] = char
    else {
      dict[char + 1] = 256 + n
      n = n + 1
    }
  }

  #intToUtf8(dict[utf8ToInt('!')]) = !
  #intToUtf8(dict[utf8ToInt('ÿ')]) = ÿ
  #intToUtf8(dict[utf8ToInt(' ')]) = Ġ

  dict
}

ggml_unicode = bytes_to_unicode()
ggml_encode = function(char){
    int = utf8ToInt(char)
    if (int < 256){
      int = ggml_unicode[int + 1]
      return(intToUtf8(int))
    }
    else return(char)
}
ggml_decode = function(str) {
  output = ''
  for (char in unlist(strsplit(str,''))){
    int = which(ggml_unicode == utf8ToInt(char)) - 1
    if (!length(int)) int = utf8ToInt(char)
    output = paste0(output,intToUtf8(int))
  }
  output
}



