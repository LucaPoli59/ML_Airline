# Progetto Machine Learning A.A 2022/2023
# Partecipanti:
# Moschi Riccardo 856243
# Poli Luca 852027

#Istallazione delle librerie
install.packages("pacman")
library("pacman")
pacman::p_load(skimr, Boruta, datarium, ggcorrplot, ggplot2, car, sm, GGally,
               corrplot, factoextra, FactoMineR, caret, nnet, kernlab, pROC,
               ROCR,plyr)



######################################  Metodi generici  ######################
### In questa sezione verranno definiti dei metodi generici 
### riusati in tutto il progetto



#' cols_numeric
#'
#' Ottiene le colonne numeriche del dataframe
#'
#' @param df Dataframe d'input
#'
#' @return Nomi delle colonne numeriche
#' 
cols_numeric <- function(df) {
  unlist(lapply(df, is.numeric))
}


#############################  Estrazione ed analisi dei dati  ################
### In questa sezione verranno definiti e usati i metodi per estrarre,
### elaborare e analizzare i dati


#' import_dataset
#'
#' importa il dataset ed estrae n valori casuali
#'
#' @param dataset.name Nome del dataset da importare
#' @param n Numero di valori da estrarre
#'
#' @return Dataframe finale a dimensione n
#' 
import_dataset <- function(dataset.name, n) {
  
  df <- read.csv(dataset.name)
  df <- df[sample(nrow(df), n), ]
  row.names(df) <- NULL
  df
}


#' pre_processing
#' 
#' Realizza operazioni preliminari su esso; cioè: 
#' - eliminazione di colonne inutili
#' - l'eliminazione dei valori nulli;
#' - rinominazione e spostamento della variabile target
#' - conversione in factor delle variabili non numeriche
#'
#' @param df Dataframe contenete i dati
#'
#' @return Dataframe rielaborato
#' 
pre_processing <- function(df) {
  df <- df[, !names(df) %in% c("X", "id")]
  colnames(df)[ncol(df)] <- "target"
  df <- df[,c(ncol(df),1:(ncol(df)-1))]
  df <- na.omit(df)
  
  df$target <- factor(df$target, 
                      levels=c("satisfied", "neutral or dissatisfied") )
  df$target <- revalue(df$target, c("neutral or dissatisfied" = "dissatisfied"))
  
  df$Gender <- factor(df$Gender)
  df$Customer.Type <- factor(df$Customer.Type)
  df$Type.of.Travel <- factor(df$Type.of.Travel)
  df$Class <- factor(df$Class)
  row.names(df) <- NULL
  df
}


#' pair_plot 
#' 
#' funzione che plotta di una riga o colonna con GGally::ggpairs
#'
#' @param df Dataframe da plottare
#' @param filename Nome del file da salvare
#' @param index "row" o "column" indigando riga o colonna
#' 
pair_plot <- function(df, index="row") {
  
  pairs <- ggpairs(df, aes(colour=target))
  if(index == "row")
  {
    plots <- lapply(1:pairs$ncol, function(j) getPlot(pairs, i = 1, j = j))
  }
  else
  {
    plots <- lapply(1:pairs$nrow, function(i) getPlot(pairs, i = i, j = 1))
  }
  
  ggmatrix(
    plots,
    nrow = 1,
    ncol = pairs$ncol,
    xAxisLabels = pairs$xAxisLabels,
    yAxisLabels = "target"
  )
}

#' pair_plot.save
#' 
#' funzione che salva su disco il plot di una riga o colonna con GGally::ggpairs
#'
#' @param df Dataframe da plottare
#' @param filename Nome del file da salvare
#' @param index "row" o "col" indigando riga o colonna
#' 
pair_plot.save <- function(df, filename, index="row") {
  
  pairs <- ggpairs(df, aes(colour=target))
  if(index == "row")
  {
    plots <- lapply(1:pairs$ncol, function(j) getPlot(pairs, i = 1, j = j))
  }
  else
  {
    plots <- lapply(1:pairs$nrow, function(i) getPlot(pairs, i = i, j = 1))
  }
  
  plot_to_save <- ggmatrix(
    plots,
    nrow = 1,
    ncol = pairs$ncol,
    xAxisLabels = pairs$xAxisLabels,
    yAxisLabels = "target"
  )
  ggsave(filename = filename, plot = plot_to_save, dpi=100, scale=1, width = 20)
}

# Import dei dati di train
df.train <- import_dataset("train.csv", 5000)
skim(df.train)

#prima rielaborazione
df.train <- pre_processing(df.train)
skim(df.train)

# Visualizzazione della matrice di correlazione delle variabili numeriche
ggcorrplot(cor(df.train[, cols_numeric(df.train)]))


# Density plot per ogni feature
for (i in colnames(df.train)){
  if(is.numeric(df.train[[i]])){
    d <- density(df.train[, i])
    plot(d, main=paste("Kernel Density of ", i))
    polygon(d, col="green", border="red")
  }
}

# Box plot per ogni feature
for (i in colnames(df.train)){
  if(is.numeric(df.train[[i]])){
    boxplot(df.train[[i]],
            main = paste("Boxplot di ", i),
            xlab = "values",
            ylab =  i,
            col = "green",
            border = "blue",
            horizontal = TRUE,
            notch = FALSE
    )
  }
}


# Visualizzazione del density plot relativo al target
for (i in colnames(df.train)){
  if(is.numeric(df.train[[i]])){
    sm.density.compare(df.train[[i]], df.train$target, xlab=i)
    title(main=paste("distribuzione di ", i, " relativo al target"))
    colfill<-c(2:(2+length(levels(df.train$target))))
    legend("topright", levels(df.train$target), fill=colfill)
  }
}

# Analisi della distribuzione delle feature in relazione con il target
cols_to_print.all <- colnames(df.train[cols_numeric(df.train)])
cols_to_print.1 <- append(cols_to_print.all[1:9], "target", 0)
cols_to_print.2 <- append(cols_to_print.all[9:18], "target", 0)

pair_plot(df.train[cols_to_print.1])
pair_plot(df.train[cols_to_print.2])
pair_plot(df.train[cols_to_print.1], index = "col")
pair_plot(df.train[cols_to_print.2], index = "col")

#pair_plot.save(df.train[cols_to_print.1], "plots/row_1_9.png")
#pair_plot.save(df.train[cols_to_print.2], "plots/row_9_18.png")
#pair_plot.save(df.train[cols_to_print.1], "plots/col_1_9.png", index = "col")
#pair_plot.save(df.train[cols_to_print.2], "plots/col_9_18.png", index = "col")

# visualizzazione importanza delle features, tramite la libreria boruta
boruta <- Boruta(target ~ ., data = df.train)
plot(boruta, cex.axis = 0.8)
attStats(boruta)

#############################  Preprocessing del dataset  ################
### In questa sezione verrà effettuata la PCA, in particolare: sia l'analisi
### delle componenti principali, sia l'effettiva trasformazione.
### In seguito si effettua anche l'operazione di normalizzazione e di encoding.

#' compute_pca
#' 
#' Funzione che addestra il modello della PCA per poter poi effettuare
#'  la trasformazione
#'
#' @param df Dataframe con i dati 
#'
#' @return Modello della PCA per la trasformazione
#' 
compute_pca <- function(df) {
  df.numeric <- df[,cols_numeric(df)]
  pca <- prcomp(df.numeric, center = TRUE, scale. = TRUE)
  pca
}

#' apply_pca
#' 
#' Effettua la trasformazione dei dati
#'
#' @param df Dataframe con i dati
#' @param pca_model Modello della PCA addestrato
#' @param dimension Numero di dimensioni scelte
#'
#' @return Dataframe trasformato
#' 
apply_pca <- function(df, pca_model, dimension) {
  df.numeric <- df[,cols_numeric(df)]
  df.pca <- predict(pca_model, df.numeric)
  
  df.n_numeric <- df[,!cols_numeric(df)]
  df.pr = cbind(df.n_numeric[1], df.pca[, 1:dimension], df.n_numeric[-1])
  df.pr
}

#' encode_factor
#' 
#' Efettua l'encoding delle variabili di tipo factor
#'
#' @param df Dataframe contenente i dati
#'
#' @return Dataframe rielaborato
#' 
encode_factor <- function(df) {
  
  for (col in colnames(df[!cols_numeric(df)][-1])) {
    df[[col]] <- as.numeric(df[[col]])
  }
  df
}


## Analisi delle Componenti Principali

# Calcolo delle componenti della PCA
pca.ris <- PCA(df.train[cols_numeric(df.train)], graph = FALSE)
pca.ris.eig_val <- get_eigenvalue(pca.ris)
head(pca.ris.eig_val, 7)
pca.ris.var <- get_pca_var(pca.ris)

# Plot relativi a cos2, contrib, eigenvalue
fviz_eig(pca.ris, addlabels = TRUE, ylim = c(0, 50))
fviz_pca_var(pca.ris, col.var = "cos2", repel = TRUE,
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"))
fviz_cos2(pca.ris, choice = "var", axes = 1:2) # rispetto alla dim 1-2
fviz_contrib(pca.ris, choice = "var", axes = 2, top = 7) # rispetto alla dim 2

# Plot delle correlazioni, rispetto alle dimensioni, di contrib e cos2
corrplot(pca.ris.var$cos2, is.corr=FALSE)
corrplot(pca.ris.var$contrib, is.corr=FALSE)


## Trasformazione del dataset

# Import dei dati di test
df.test <- pre_processing(import_dataset("test.csv", 500))
row.names(df.test) <- NULL



# PCA trasformazione
pca_model <- compute_pca(df.train)

df.train <- apply_pca(df.train, pca_model = pca_model, dimension = 7)
df.test <- apply_pca(df.test, pca_model = pca_model, dimension = 7)


# Encoding e scaling dei dati
df.train <- encode_factor(df.train)
df.test <- encode_factor(df.test)

df.train[-1] <- scale(df.train[-1])
df.test[-1] <- scale(df.test[-1])



#############################  Modelli  ################
### In questa sezione verranno definiti e addestrati i due modelli utilizzati:
### - Rete Neurale;
### - SVM.
### Essi verranno addestrati tramite la K-Fold con Grid Search sul train set;
### in seguito verranno valutate le relative performance sul test set


#' predict.matrix
#' 
#' Funzione che usa il modello addestrato per predire dei dati e valutare
#' le performance tramite ConfusionMatrix
#'
#' @param model Modello addestrato
#' @param df.x Dati di input per la predizione
#' @param df.y Risultati reali
#'
#' @return ConfusionMatrix della predizione
#' 
predict.matrix <- function(model, df.x, df.y) {
  predicted.probabilities <- predict(model, df.x, type = "prob")
  predicted.classes <- factor(round(predicted.probabilities[, 1]) , 
                              levels = c(1, 0),
                              labels = c("satisfied", "dissatisfied"))
  predicted.table <- confusionMatrix(data=predicted.classes, reference = df.y,
                                     mode = "everything")
  predicted.table
}


#' predict.roc
#' 
#' Funzione che usa il modello addestrato per predire dei dati e valutare
#' le performance tramite roc score
#'
#' @param model Modello addestrato
#' @param df.x Dati di input per la predizione
#' @param df.y Risultati reali
#'
#' @return Roc score della predizione
#' 
predict.roc <- function(model, df.x, df.y) {
  predicted.probabilities <- predict(model, df.x, type = "prob")
  
  roc_score <- roc(predictor = predicted.probabilities[, 1],
                   response = df.y,
                   levels = levels(df.y),
                   direction = '>')
  roc_score
}

#' predict.plot_perf
#' 
#' Funzione che usa il modello addestrato per predire dei dati e valutare
#' le performance tramite il plot del True Positive Rate e False Positive Rate
#'
#' @param model 
#' @param df.x 
#' @param df.y 
#'
predict.plot_perf <- function(model, df.x, df.y) {
  predicted.probabilities <- predict(model, df.x, type = "prob")
  prediction <- prediction(predicted.probabilities[, 1],  df.y)
  
  perf <- performance(prediction, measure = "auc", x.measure = "cutoff")
  perf.tpr_fpr <- performance(prediction, "tpr","fpr")
  
  plot(perf.tpr_fpr, colorize=T,main=paste("AUC:",(perf@y.values)))
  abline(a=0, b=1)
}



## Neural Network

# si addestra la rete su 5 fold, variando la dimensione e il decay
nn.grid <-  expand.grid(size = c(3, 5, 7, 9), decay = c(0, 5e-4))
nn.train_control <- trainControl(method = "cv", number = 5, search = "grid")

nn.model <- train(target~., data = df.train, method = "nnet", trace = FALSE,
                       trControl = nn.train_control, tuneGrid = nn.grid )

plot(nn.model) # Risultati della Grid Search

# Valutazione delle performance su train set e test set
nn.train.confusion_matrix <- predict.matrix(nn.model, 
                                            df.x = df.train[-1],
                                            df.y = df.train$target)
nn.test.confusion_matrix <- predict.matrix(nn.model, 
                                           df.x = df.test[-1],
                                           df.y = df.test$target)

nn.train.confusion_matrix
nn.test.confusion_matrix


## SVM

# Si addestra svm su 5 fold, variando il costo
svm.grid <- expand.grid(C = c(0.5, 1, 10, 100, 1000), sigma = c(0.1))
svm.trainControl <- trainControl(method = "cv", number = 5, search = "grid", 
                                 classProbs = TRUE)
svm.model <- train(target ~ ., data = df.train, method = "svmRadialSigma",
                   trControl = svm.trainControl, tuneGrid = svm.grid,
                   trace = FALSE)

plot(svm.model) # Risultati della Grid Search

# Valutazione delle performance su train set e test set
svm.train.confusion_matrix <- predict.matrix(svm.model, 
                                             df.x = df.train[-1],
                                             df.y = df.train$target)

svm.test.confusion_matrix <- predict.matrix(svm.model, 
                                            df.x = df.test[-1],
                                            df.y = df.test$target)
svm.train.confusion_matrix
svm.test.confusion_matrix




## Valutazione delle performance tramite ROC curve

# Rete Neurale
nn.roc_score <- predict.roc(nn.model, 
                            df.x = df.test[-1],
                            df.y = df.test$target)
plot(nn.roc_score, main=paste("AUC:", (nn.roc_score$auc)))

predict.plot_perf(nn.model, 
                  df.x = df.test[-1],
                  df.y = df.test$target)

# SVM
svm.roc_score <- predict.roc(svm.model, 
                             df.x = df.test[-1],
                             df.y = df.test$target)
plot(svm.roc_score, main=paste("AUC:", (svm.roc_score$auc)))

predict.plot_perf(svm.model, 
                  df.x = df.test[-1],
                  df.y = df.test$target)
