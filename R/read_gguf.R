
default_path = paste0(tools::R_user_dir('ggufr'),"/SmolLM3-3B-Q8_0.gguf")

download_model = function(path = default_path){
  if (!dir.exists(path)) dir.create(path, recursive = TRUE)
  if (readline("This will download a 3GB file. Is this OK? Y/n ") %in% c('Y','y','Yes','yes')){
    curl::curl_download(url="https://huggingface.co/unsloth/SmolLM3-3B-GGUF/resolve/main/SmolLM3-3B-Q8_0.gguf",path,quiet=FALSE)
  }
}
delete_model = function(path = default_path){
  if (readline(paste0("This will delete the file:\n", path,"\nIs this OK? Y/n ")) %in% c('Y','y','Yes','yes')){
    file.remove(path)
  }
}

#this function can read any GGUF if you set info_only=TRUE
read_gguf = function(path = default_path, info_only=FALSE){
  if (!(readline("This will cache 25GB of tensors, which will be deleted when you close R. Is this OK? Y/n ") %in% c('Y','y','Yes','yes'))) {print('Aborted.'); return()}

  con = file(path,'rb') #start reading file
  on.exit(close(con)) #close connection when done

  #---- General Utils ----
  #progress bar string 60% [======    ]
  progress = function(i,total,steps=10) {
    perc = round(100*i/total)
    sofar = floor(steps * i/total)
    paste0(
      sprintf('%3s',perc),
      "% [",
      paste0(rep('=',sofar),collapse=''),
      paste0(rep(' ',steps-sofar),collapse=''),
      "]"
    )
  }

  #---- Raw-Reading Utils ----
  where = function() seek(con) #get current position
  hex = function(n_bytes) readBin(con, 'raw', n_bytes) #get raw bytes
  bin = function(n_bytes) readBin(con, 'raw', n_bytes) |> rawToBits() #read raw binary
  skip = function(n_bytes) seek(con, n_bytes, 'current') #skip forward from current position
  jump = function(n_bytes) seek(con, n_bytes, 'start') #jump to absolute position
  dump = function(n_bytes=64) {loc = where(); output = readBin(con, 'raw', n_bytes); jump(loc); output} #get raw bytes without advancing

  #---- Numeric Datatype Parsers ----
  #R's readBin isn't great at handling diverse numeric types, so I've implemented some by hand.

  #functions to read the next n bits from the file as a specific numeric type
  uint = function(n_bits) bin(n_bits/8) |> as.uint()
  int = function(n_bits) bin(n_bits/8) |> as.int()
  float = function(n_bits) {
    if (n_bits == 16) bin(2) |> as.float16() #R can't parse float16s natively D;
    else readBin(con, 'numeric', size = n_bits/8)
  }

  #manual bits-to-numeric :(
  as.uint = function(bits) sum(2^(which(bits==1)-1))
  as.int = function(bits) sum(2^(which(bits[-length(bits)]==1)-1)) - (bits[length(bits)]==1) * 2^(length(bits)-1)
  as.float16 = function(bits) {
    #manual implementation of littleendian float16 :sob:
    sign = ifelse(bits[16] == 1,-1,1)
    exponent = as.uint(bits[16 - 5:1])
    fraction = as.uint(bits[16 - 15:6])/1024
    if (exponent == 0) return(sign * 2^-14 * fraction)
    else return(sign * 2^(exponent - 15) * (1 + fraction))
  }

  #---- Other Datatype Parsers ----
  gguf_string = function() hex(uint(64)) |> rawToChar() #uint64: length; hex(length): raw character data
  gguf_type = function() typenames[[1 + uint(32)]] #data type indicator, 0-indexed
  tensor_type = function() tensor_typenames[[1 + uint(32)]] #tensors use a different set of types
  gguf_array = function(){ #type, count, [data]
    #ARRAY:
    #uint32: type
    #uint64: count
    #types[[type]] x count: values

    type = gguf_type()
    count = uint(64)

    val = vector('character', length = count)
    for (i in 1:count) val[i] = types[[type]]()

    val
  }

  #---- Datatype Lookup ----
  typenames = c('int8','uint8','int16','uint16','int32','uint32','float32','bool','string','array','int64','uint64','float64')
  types = list( #function to read each type
    int8    = \() int(8),
    uint8   = \() uint(8),
    int16   = \() int(16),
    uint16  = \() uint(16),
    int32   = \() int(32),
    uint32  = \() uint(32),
    float32 = \() float(32),
    bool    = \() hex(1) != 0, #untested
    string  = \() gguf_string(),
    array   = \() gguf_array(),
    int64   = \() int(64),
    uint64  = \() uint(64),
    float64 = \() float(64)
  )

  #---- Tensor Datatype Lookup ----
  tensor_typenames = c(
    'F32',
    'F16',
    'Q4_0',
    'Q4_1',
    'Q4_2',
    'Q4_3',
    'Q5_0',
    'Q5_1',
    'Q8_0',
    'Q8_1',
    'Q2_K',
    'Q3_K',
    'Q4_K',
    'Q5_K',
    'Q6_K',
    'Q8_K',
    'IQ2_XXS',
    'IQ2_XS',
    'IQ3_0',
    'IQ3_NL_B16',
    'IQ3_NL_B32',
    'IQ3_SQ',
    'IQ4_0',
    'IQ4_NL_B16',
    'IQ4_NL_B32',
    'IQ4_NL_B64',
    'IQ4_SQ',
    'IQ4_K',
    'IQ5_0',
    'IQ5_NL',
    'IQ5_SQ',
    'IQ5_K',
    'IQ6_0',
    'IQ6_NL',
    'IQ6_K',
    'IQ2_K',
    'IQ2_S',
    'IQ2_XS_RW',
    'IQ2_0',
    'IQ3_T',
    'IQ8_0'
  )

  #---- Main Parsing ----
  #maps the gguf metadata

  #initialize objects
  metadata = list() #metadata entries
  tensors = list() #tensor entries
  model = list() #main object

  #Parse Header
    jump(0)
    model$magic = hex(4) |> intToUtf8() #should say "GGUF"
    model$gguf_version = uint(32)
    model$tensor_count = uint(64)
    model$metadata_count = uint(64)
  cat(sep='','Parsing ',path,': ',model$tensor_count,' tensors, ',model$metadata_count, ' metadata keys.\n')

  #Parse Metadata
  for (i in 1:model$metadata_count){
    #METADATA:
    #gguf_string: name
    #uint32: type
    #types[[type]]: value

    name = gguf_string()
    cat(sep='','\r',progress(i,model$metadata_count),' Evaluating key "',name,'"',rep(' ',16))
    type = gguf_type()
    offset = where() #location of data, from start of file. This is really only useful if you want to stream out the tokens
    value = types[[type]]()
    offset_end = where() #end of data

    metadata[[name]] = list(name = name, type = type, start = offset, end = offset_end, value = value)
  }
  cat('\n')

  #Parse Tensors
  for (i in 1:model$tensor_count){
    #TENSOR:
    #gguf_string: name
    #uint32: dimensions (1 or 2)
    #uint64: dimension 1 (columns)
    #(uint64) if dimensions=2: dimension 2 (rows)
    #uint32: type
    #uint64: offset from blob (start of tensor data)

    #GGUF technically supports higher-dimensional tensors, but I've yet to see one.

    name = gguf_string()
    cat(sep='','\r',progress(i,model$tensor_count),' Mapping tensor "',name,'"',rep(' ',16))
    dims = uint(32)
    cols = uint(64)
    rows = ifelse(dims == 2, uint(64), 1)
    type = tensor_type()
    offset = uint(64)

    tensors[[name]] = list(cols=cols,rows=rows,type=type,offset=offset)
  }
  cat('\n')

  #Blob (start of tensor data)
  blob = where()
  blob = ceiling(blob/32)*32 #pad alignment
  model$blob = blob

  #----

  #coalesce variables
  model$metadata = metadata
  model$tensors = tensors

  if (info_only) return(model) #exit before the architecture-specific parsing
  else if (is.null(metadata$general.name) || metadata$general.name$value != 'Smollm3-3B') cat('\nWARNING: This pipeline is only designed to handle SmolLM3!\n')


  #---- Tensor-Reading Utils ----
  read_tensor = function(tensor, verbose){ #tensor = e.g. tensors$token_embd.weights. verbose = text for progress bar
    length = tensor$rows * tensor$cols
    size = tensor_sizes[[tensor$type]] #weights per read, e.g. 32 weights for each Q8_0() call
    jump(model$blob + tensor$offset)

    output = tensor_types[[tensor$type]](length/size, verbose)
    matrix(output, tensor$rows, tensor$cols, byrow=TRUE) #byrow=TRUE: EXTREMELY IMPORTANT! Tensors are stored row-by-row in the raw data, but R defaults to filling matrices column-by-column.
  }

  #only F32 and Q8_0 are needed for this model. All we need to do is parse the quantized data format into R doubles
  tensor_types = list(
    F32 = \(count, verbose) readBin(con, 'numeric', size = 4, n = count),
    Q8_0 = \(count, verbose) {
      output = vector("numeric", count * 32)
      for (i in 1:count){
        d = float(16)
        weights = readBin(con, 'integer', size = 1, n = 32) #32 int8 weights multiplied by one float
        output[32 * (i-1) + 1:32] = d * weights
        if (length(verbose) && i %% 10000 == 0) cat(sep='','\r',progress(i,count),' ',verbose,'                ')
      }
      output
    }
  )
  tensor_sizes = list(#weights per read
    F32 = 1,
    Q8_0 = 32
  )

  #---- Loading Values ----
  #expose a bunch of useful values

  model$embedding_length = model$tensors$token_embd.weight$cols #also in data as model$metadata$smollm3.embedding_length$value
  model$vocab_size = model$tensors$token_embd.weight$rows #also model$metadata$smollm3.vocab_size$value
  model$merge_size = length(model$metadata$tokenizer.ggml.merges$value)

  model$epsilon = model$smollm3.attention.layer_norm_rms_epsilon$value #RMSN epsilon
  model$heads_q = model$smollm3.attention.head_count$value
  model$heads_kv = model$smollm3.attention.head_count_kv$value

  #get raw tokenizer data
  get_raw_array = function(key_name){
    meta = metadata[[key_name]]
    jump(meta$start + 12) #skip array type and count
    hex(meta$end - meta$start)
  }

  #we need the tokens as raw bytes for BPE tokenizing
  model$raw_tokens = get_raw_array('tokenizer.ggml.tokens')
  model$raw_merges = get_raw_array('tokenizer.ggml.merges')

  #---- Caching Tensors ----
  #tensor data is written to an Rds file in tempdir() for each layer, since R can't store every layer at once.

  cache_dir = paste0(tempdir(),'/')
  model$dir = cache_dir

  #cache non-layer tensors
  list(
    output_norm.weight = read_tensor(tensors$output_norm.weight),
    token_embd.weight = read_tensor(tensors$token_embd.weight, verbose='Loading embedding matrix...')
  ) |>
    saveRDS(paste0(cache_dir,"embeddings.rds"),compress=FALSE)

  cat('\n')

  n_blocks = metadata$smollm3.block_count$value
  for (i in 1:n_blocks){
    block = i-1 #0-indexed
    layer = list()
    tensor_names = c(
      "attn_k.weight",
      "attn_norm.weight",
      "attn_output.weight",
      "attn_q.weight",
      "attn_v.weight",
      "ffn_down.weight",
      "ffn_gate.weight",
      "ffn_norm.weight",
      "ffn_up.weight"
    )
    for (t in 1:length(tensor_names)) {
      tensor = tensor_names[t]
      verbose = paste0('Loading tensor "',tensor,'" (',t,'/',length(tensor_names),')')
      layer[[tensor]] = tensors[[paste0('blk.',block,'.',tensor)]] |> read_tensor(verbose)
    }
    cat(sep='','\rCaching layer ',i,' / ',n_blocks,rep(' ',64),'\n')
    saveRDS(layer,paste0(cache_dir,'blk.',block,".rds"),compress=FALSE)
  }

  #----
  model
}


view_metadata = function(model){
  for (i in 1:length(model$metadata)) if (model$metadata[[i]]$type == 'array') model$metadata[[i]]$value = paste0('(',length(model$metadata[[i]]$value),') ',paste0(collapse=', ',na.omit(model$metadata[[i]]$value[1:25])))
  df = list2DF(model$metadata) |> t()
  df = df[,-c(1,4)]
  colnames(df) = c('type','start','value')
  View(df,'Metadata')
  df
}

view_tensors = function(model){
  df = list2DF(model$tensors) |> t()
  colnames(df) = c('cols','rows','type','offset')
  View(df,'Tensors')
  df
}



