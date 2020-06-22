library(sentencepiece) ## using at least version 0.1.3 from https://github.com/bnosac/sentencepiece
library(word2vec)
library(bit)
library(crfsuite)
library(torch)

# bpemb <- BPEembed(file_sentencepiece = "C:/Users/Jan/Dropbox/Work/RForgeBNOSAC/VUB/torch.ner/en.wiki.bpe.vs25000.model",
#                   file_word2vec = "C:/Users/Jan/Dropbox/Work/RForgeBNOSAC/VUB/torch.ner/data/en/en.wiki.bpe.vs25000.d25.w2v.bin")

downloads <- sentencepiece_download_model("dutch", vocab_size = 25000, dim = 25, model_dir = getwd())
bpemb <- BPEembed(file_sentencepiece = downloads$file_model,
                  file_word2vec = downloads$glove.bin$file_model)


##
## DOCS AT https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
##
ner_bilstm <- nn_module(
  classname = "ner_bilstm",
  initialize = function(bpemb, tagset_size, hidden_size = 32, num_layers = 1, nonlinearity = c("tanh", "relu"), dropout = 0) {
    nonlinearity <- match.arg(nonlinearity)
    self$bpemb          <- bpemb
    self$embedding_dim  <- bpemb$dim
    self$tagset_size    <- as.integer(tagset_size)
    self$num_layers     <- as.integer(num_layers)
    self$hidden_size    <- as.integer(hidden_size)
    
    ## LSTM Layer
    self$rnn       <- nn_rnn(mode = "LSTM", input_size = self$embedding_dim, 
                             hidden_size = self$hidden_size, num_layers = self$num_layers,
                             dropout = dropout, bidirectional = TRUE, nonlinearity = nonlinearity, batch_first = TRUE)
    ## Map of LSTM features to tag space
    self$linear    <- nn_linear(in_features = hidden_size * 2, out_features = tagset_size, bias = TRUE)
  },
  forward = function(x, enforce_sorted = FALSE) {
    x <- x[order(sapply(x, length), decreasing = TRUE)]
    ## Tokenize and get the embedding of the tokens - sentencepiece splits according to subwords, take the average embedding of the subwords
    ## Result is a list of matrices with as rows the words and as columns the embeddings
    emb <- lapply(x, FUN=function(tokensequence){
      emb <- predict(self$bpemb, newdata = tokensequence, type = "encode")
      emb <- lapply(emb, colMeans)
      emb <- do.call(rbind, emb)  
      emb <- torch_tensor(emb, dtype = torch_float())
      emb
    })
    ## Order by number of embedded words as that is needed by nn_utils_rnn_pack_sequence
    p <- nn_utils_rnn_pack_sequence(emb, enforce_sorted = FALSE)
    ## Forward pass, getting the LSTM features
    rnn_out <- self$rnn(p)
    rnn_out <- rnn_out[[1]]
    ## Put the LSTM feature in 1 line per token and do a Forward pass over the linear layer 
    tag_space  <- self$linear(rnn_out$data)
    ## Softmax transformation
    tag_scores <- nnf_log_softmax(tag_space, dim = 1L)
    tag_scores
  }
)

##
## Get some training data
##
conll2002 <- ner_download_modeldata("conll2002-nl")
conll2002$label <- factor(conll2002$label, levels = c("B-LOC", "I-LOC", "B-MISC", "I-MISC", "B-ORG", "I-ORG", "B-PER", "I-PER", "O"))
conll2002$label_nr <- as.integer(conll2002$label) - 1L
traindata <- subset(conll2002, data %in% "ned.train")
traindata <- split(traindata, traindata$doc_id)
testdata  <- subset(conll2002, data %in% c("testa", "testb"))
testdata  <- split(testdata, testdata$doc_id)

##
## Setup model (number of entities is 9 location / misc / organisation / person + other)
##
model <- ner_bilstm(bpemb = bpemb, tagset_size = 9, hidden_size = 4, num_layers = 1)
optimizer <- optim_sgd(model$parameters, lr = 0.1, momentum = 0.9, weight_decay = 0.01)


##
## Loop over the training data, update the parameters
##
overview <- list()
for(epoch in 1:10){
  overview[[epoch]] <- list()
  overview[[epoch]]$epoch <- epoch
  cat(sprintf("%s epoch %s", Sys.time(), epoch), sep = "\n")
  ###############################################################################
  ## TRAIN
  ##   - in batches of 50
  ##   - using negative log loss
  ###############################################################################
  model$train()
  ## shuffle training data
  traindata <- sample(traindata)
  chunks <- chunk(from = 1L, to = length(traindata), by = 50L, maxindex = length(traindata))
  for(rangeindex in chunks){
    ## Get a batch of sentences the sentences (list of words)
    train_sentences <- as.which(rangeindex)
    train_sentences <- traindata[train_sentences]
    train_tokens    <- lapply(train_sentences, FUN = function(x) x$token)
    ## Get the known value of the entity
    train_target <- lapply(train_sentences, FUN = function(x) x$label_nr)
    train_target <- unlist(train_target)
    train_target <- torch_tensor(train_target, dtype = torch_long())
    
    ## Forward pass, calculate loss, do backward pass
    optimizer$zero_grad()
    train_scores  <- model(train_tokens)
    loss <- nnf_nll_loss(train_scores, train_target)
    cat(sprintf("  loss on training set: %s", as_array(loss)), sep = "\n")
    loss$backward()
    optimizer$step()
  }  
  overview[[epoch]]$loss_train <- as_array(loss)
  ###############################################################################
  ## Evaluation statistics of the epoch
  ##
  ###############################################################################
  
  ## Stats on training data
  test_sentences <- traindata
  test_tokens <- lapply(test_sentences, FUN = function(x) x$token)
  test_scores <- model(test_tokens)
  
  test_target <- lapply(test_sentences, FUN = function(x) x$label_nr)
  test_target <- unlist(test_target)
  test_target <- torch_tensor(test_target, dtype = torch_long())
  
  loss <- nnf_nll_loss(test_scores, test_target)
  overview[[epoch]]$loss_test <- as_array(loss)
  
  ## Stats on test data
  test_sentences <- testdata
  test_tokens <- lapply(test_sentences, FUN = function(x) x$token)
  test_scores <- model(test_tokens)
  
  test_target <- lapply(test_sentences, FUN = function(x) x$label_nr)
  test_target <- unlist(test_target)
  test_target <- torch_tensor(test_target, dtype = torch_long())
  
  loss <- nnf_nll_loss(test_scores, test_target)
  overview[[epoch]]$loss_test <- as_array(loss)
  
  stats <- list(pred = unlist(lapply(test_sentences, FUN = function(x) x$label_nr))+1,
                obs = apply(exp(as_array(test_scores)), MARGIN=1, FUN=which.max))
  stats$pred <- levels(conll2002$label)[stats$pred]
  stats$obs <- levels(conll2002$label)[stats$obs]
  stats <- crf_evaluation(pred = stats$pred, obs = stats$obs)
  print(stats$overall)
  overview[[epoch]]$stats <- stats
  print(overview[[epoch]])
  
  ##
  ## visualise the evolution of the loss
  ##
  toplot <- list(train = data.frame(type = "train", 
                                    epoch = sapply(overview, FUN=function(x) x$epoch),
                                    loss = sapply(overview, FUN=function(x) x$loss_train)),
                 test = data.frame(type = "test", 
                                    epoch = sapply(overview, FUN=function(x) x$epoch),
                                    loss = sapply(overview, FUN=function(x) x$loss_test)))
  toplot <- do.call(rbind, toplot)
  print(lattice::xyplot(loss ~ factor(epoch), groups = factor(type), data = toplot, type = "b", pch = 20,
        auto.key = list(space = "right", points = FALSE, lines = TRUE), xlab = "Epoch", ylab = "Negative Log Loss", main = "Evolution of loss"))
}


## Small test in English
# downloads <- sentencepiece_download_model("english", vocab_size = 25000, dim = 25, model_dir = getwd())
# bpemb <- BPEembed(file_sentencepiece = downloads$file_model,
#                   file_word2vec = downloads$glove.bin$file_model)
# model <- ner_bilstm(bpemb = bpemb, tagset_size = 3, hidden_size = 8)
# 
# sentences <- c("a dog ate the apple", "everybody read that book")
# sentences <- strsplit(sentences, " ")
# out <- model(sentences, enforce_sorted = FALSE)
# out <- as_array(out)
# out <- exp(out)
# 
# traindata <- list(
#   data.frame(doc_id = 1, 
#              token = c("the", "dog", "ate", "the", "apple"),
#              label = c("DET", "NN", "V", "DET", "NN"),
#              label_nr = c(0, 1, 2, 0, 1), stringsAsFactors = FALSE),
#   data.frame(doc_id = 2, 
#              token = c("everybody", "read", "that", "book"),
#              label = c("NN", "V", "DET", "NN"), 
#              label_nr = c(1, 2, 0, 1), stringsAsFactors = FALSE))

