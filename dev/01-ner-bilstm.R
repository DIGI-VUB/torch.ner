library(sentencepiece)
library(word2vec)
library(torch)
downloads <- sentencepiece_download_model("english", vocab_size = 25000, dim = 25, model_dir = getwd())
bpemb <- BPEembed(file_sentencepiece = downloads$file_model,
                  file_word2vec = downloads$glove.bin$file_model)

##
## DOCS AT https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
##
ner_bilstm <- nn_module(
  classname = "ner_bilstm",
  initialize = function(bpemb, tagset_size, hidden_size = 32, num_layers = 1, seq_len = 100) {
    self$bpemb          <- bpemb
    self$embedding_dim  <- bpemb$dim
    self$seq_len        <- seq_len
    self$tagset_size    <- as.integer(tagset_size)
    self$num_layers     <- as.integer(num_layers)
    self$hidden_size    <- as.integer(hidden_size)
    
    ## LSTM Layer
    self$rnn       <- nn_rnn(mode = "LSTM", input_size = self$embedding_dim, 
                             hidden_size = self$hidden_size, num_layers = self$num_layers,
                             dropout = 0, bidirectional = TRUE, nonlinearity = "tanh", batch_first = TRUE)
    ## Map of LSTM features to tag space
    self$linear    <- nn_linear(in_features = hidden_size * 2, out_features = tagset_size, bias = TRUE)
  },
  forward = function(tokensequence) {
    # current implementation only works for 1 tokensequence at a time
    batch_size <- 1L 
    
    ## Tokenize and get the embedding of the tokens - sentencepiece splits according to subwords, take the average embedding of the subwords
    emb <- predict(self$bpemb, newdata = tokensequence, type = "encode")
    emb <- lapply(emb, colMeans)
    emb <- do.call(rbind, emb)
    
    ## Pad zeros to the maximum length of the text sequence + put these in a tensor
    #emb <- rbind(emb, matrix(0, nrow = seq_len - nrow(emb), ncol = ncol(emb)))
    n_tokens <- nrow(emb)
    
    ## Forward pass, geting the LSTM features
    emb <- torch_tensor(emb, dtype = torch_float())
    emb <- torch_reshape(emb, list(batch_size, -1, self$embedding_dim)) 
    #rnn_out <- self$rnn(emb)
    rnn_out <- self$rnn(emb, torch_zeros(self$num_layers * 2, batch_size, self$hidden_size, dtype = emb$dtype()))
    rnn_out <- rnn_out[[1]]
    
    ## Put the LSTM feature in 1 line per token and do a Forward pass over the linear layer 
    rnn_out <- torch_reshape(rnn_out, list(n_tokens, -1)) 
    tag_space <- self$linear(rnn_out)
    
    ## Softmax by token, output is 1 line per token with as many columns as there are labels to predict
    tag_scores <- torch_reshape(tag_space, list(-1, self$tagset_size))
    nnf_log_softmax(tag_scores, dim = 1L)
  }
)
traindata <- list(
  data.frame(doc_id = 1, 
             token = c("the", "dog", "ate", "the", "apple"),
             entity = c("DET", "NN", "V", "DET", "NN"),
             entity_nr = c(0, 1, 2, 0, 1), stringsAsFactors = FALSE),
  data.frame(doc_id = 2, 
             token = c("everybody", "read", "that", "book"),
             entity = c("NN", "V", "DET", "NN"), 
             entity_nr = c(1, 2, 0, 1), stringsAsFactors = FALSE))
model <- ner_bilstm(bpemb = bpemb, tagset_size = 3, hidden_size = 8)

optimizer <- optim_sgd(model$parameters, lr = 0.1, momentum = 0.9)

for(epoch in 1:10){
  model$train()
  cat(sprintf("%s epoch %s", Sys.time(), epoch), sep = "\n")
  for(b in traindata){
    optimizer$zero_grad()
    tokens      <- b[["token"]]
    reality     <- b[["entity_nr"]]
    reality     <- torch_tensor(reality, dtype = torch_long())
    tag_scores  <- model(tokens)
    
    loss <- nnf_nll_loss(tag_scores, reality, ignore_index = -1)
    loss$backward()
    optimizer$step()
  }  
}



