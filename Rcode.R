library(mlbench)
library(e1071)
library(caret)
library(dplyr)
library(corrplot)
library(MASS)

dF = read.csv("C:/Users/91938/Desktop/Fall 2023/PM/Project/bank-full.csv", sep = ';')
dim(dF)
head(dF,5)
colnames(dF)[colnames(dF) == "y"] <- "deposit"
table(dF$deposit)
colSums(is.na(dF))
summary(dF)
######removing oulier###############
max_previous <- max(dF$previous)
dF <- dF[dF$previous != max_previous, ]
##########missing values############
image(is.na(dF), main = "Missing Values", xlab = "Observation", ylab = "Variable", 
      xaxt = "n", yaxt = "n", bty = "n")
axis(1, seq(0, 1, length.out = nrow(dF)), 1:nrow(dF), col = "white")
##############################

lapply(dF, unique)
continuous_Vars <- names(dF)[sapply(dF, is.numeric)]
categorical_Vars <- colnames(dF[, !names(dF)%in%continuous_Vars])
#barplots
par(mfrow=c(3,3))
for (i in categorical_Vars) {
  data <- table(dF[[i]])
  barplot(data, main = paste('Barplot of', i), xlab = i, col = 'powderblue')
}
par(mfrow=c(1,1))
barplot(table(dF[, 'deposit']), main = 'Barplot of Deposit', xlab = 'deposit', col = 'lightseagreen', ylab = 'Frequency')
#########binary#########
binary_columns <- c("default", "housing", "loan")
head(df)
# Use lapply to apply ifelse to each binary column
dF[,binary_columns] <- lapply(dF[,binary_columns], function(x) ifelse(x == "yes", 1, 0))
#dummy variables
dummy_vars <- dummyVars("~job + marital + education + contact + month + poutcome", data = dF)
df_with_dummies <- predict(dummy_vars, newdata = dF)
dummy <- c("job", "marital", "education", "contact", "month", "poutcome")
df <- cbind(dF[, !names(dF)%in%dummy], df_with_dummies)
df <- df[,-which(names(df) == 'deposit')]
dim(df)
nearZeroVar(df, names = T)
columns_to_remove <- c("default","pdays", "jobentrepreneur", "jobhousemaid", "jobself-employed", "jobstudent",
  "jobunemployed", "jobunknown", "educationunknown",
  "monthdec", "monthjan", "monthmar", "monthoct", "monthsep", "poutcomeother",
  "poutcomesuccess"
)
# Remove the specified columns from the data frame
df <- df[, !names(df) %in% columns_to_remove]
dim(df)
continuous_vars <- c("age", "balance","day","duration", "campaign", "previous")
categorical_vars <- colnames(df[, !names(df)%in%continuous_vars])


#barplots
par(mfrow=c(4,4))
for (i in categorical_vars) {
  data <- table(df[[i]])
  barplot(data, main = paste('Barplot of', i), xlab = i, col = 'powderblue')
}
par(mfrow=c(1,1))

nearZeroVar(df, names=T)
############################
#histograms
par(mfrow=c(3,2))
for (i in continuous_vars) {
  data <- df[[i]]
  hist(data, main = paste('Histogram of', i), xlab = i, col = 'sandybrown')
}
par(mfrow=c(1,1))
skewness_values <- apply(df[continuous_vars], 2, skewness)
skewness_values
#boxplots
par(mfrow=c(2,3))
for (i in continuous_vars) {
  data <- df[[i]]
  boxplot(data, main = paste('Boxplot of', i), xlab = i, col = 'sienna')
}
par(mfrow=c(1,1))

###########BOXCOX#############
par(mfrow = c(4, 2))
for (i in continuous_vars){
  hist(df[[i]], main = paste("Before Transformation ",i) , xlab = i, horiz = TRUE)
  bct_ri = BoxCoxTrans(df[[i]])
  trans <- predict(bct_ri, df[[i]])
  hist(trans, main = paste("After Transformation ",i) , xlab = i, horiz = TRUE)
}
par(mfrow = c(1,1))


columns_to_transform <- df[,continuous_vars]
transDf <- preProcess(columns_to_transform, method = c("BoxCox","center", "scale"))
transDf
#apply transformation
transformed <- predict(transDf, columns_to_transform)
transformed
sapply(transformed,skewness)


########histograms after trans##########
par(mfrow=c(3,2))
for (i in colnames(transformed)) {
  hist(transformed[[i]], main = paste('Histogram of', i), xlab = i, col = 'olivedrab')
}
par(mfrow=c(1,1))
#########SS##########
transSS <- spatialSign(transformed[, continuous_vars])
Transform_spa <- as.data.frame(transSS)
sapply(Transform_spa, skewness)
########boxplots after SS#########
par(mfrow=c(2,3))
for (i in continuous_vars) {
  data <- Transform_spa[[i]]
  boxplot(data, main = paste('Boxplot of', i), xlab = i, col = 'goldenrod4')
}
par(mfrow=c(1,1))
#######################
par(mfrow=c(3,2))
for (i in colnames(Transform_spa)) {
  hist(Transform_spa[[i]], main = paste('Histogram of', i), xlab = i, col = 'olivedrab')
}
par(mfrow=c(1,1))
##########correlation############
conPred <- df[, continuous_vars]
hist(conPred$previous)
correlations <- cor(df[, continuous_vars])
corrplot(correlations, method = 'color')
correlations1 <- cor(Transform_spa[, continuous_vars])
corrplot(correlations1, method = 'color')
##########
correlations1 <- cor(dF[, continuous_Vars])
corrplot(correlations1, method = 'color')

##### Final DF #####
final_df <- cbind(df[, categorical_vars],Transform_spa)
dim(final_df)
correlations_1 <- cor(final_df)
corrplot(correlations_1)
highCorr <- findCorrelation(abs(correlations_1), cutoff = .65, names = T)
length(highCorr)
highCorr
final_df[,c(17, 14)]
final <- final_df[, -highCorr]
corrplot(cor(final))
dim(final)
######## Model Building ########
dF$deposit <- as.factor(dF$deposit)
set.seed(5790)
index <- createDataPartition(dF$deposit, p = 0.7, list = FALSE)
x_train <- final[index,]
x_test <- final[-index,]
y_train <- dF$deposit[index]
y_test <- dF$deposit[-index]
dim(x_test)
length(y_test)
ctrl <- trainControl(method = "cv",
                     number= 10,
                     summaryFunction = defaultSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

set.seed(5790)
logistic <- train(x_train,
                y = y_train,
                preProcess = c('center', 'scale'),
                method = "glm",
                metric = "Kappa",
                trControl = ctrl)
logistic
plot(logistic)
pred_logistic <- predict(logistic, x_test)
confusionMatrix(data = pred_logistic,
                reference = y_test)

## LDA
set.seed(5790)
lda <- train(x_train,
             y = y_train,
             method = "lda",
             preProcess = c('center', 'scale'),
             metric = "Kappa",
             trControl = ctrl)
lda
plot(lda)
pred_lda <- predict(lda, x_test)
confusionMatrix(data = pred_lda, y_test)

## PLSDA
set.seed(5790)
plsda <- train(x = x_train,
               y = y_train,
               method = "pls",
               tuneGrid = expand.grid(.ncomp = 1:4),
               preProcess = c("center","scale"),
               metric = "Kappa",
               trControl = ctrl)

plsda

plot(plsda)
pred_plsda <- predict(plsda, x_test)
confusionMatrix(data = pred_plsda, y_test)

## PM
set.seed(5790)
glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(.01, .2, length = 10))
glmn <- train(x=x_train,
              y = y_train,
              method = "glmnet",
              tuneGrid = glmnGrid,
              preProc = c("center", "scale"),
              metric = "Kappa",
              trControl = ctrl)
glmn
plot(glmn)
pred_glmn <- predict(glmn, x_test)
confusionMatrix(pred_glmn, y_test)



 


####### Non linear discriminant analysis ######
library(mda)
set.seed(5790)
mdaFit <- train(x = x_train, 
                y = y_train,
                method = "mda",
                metric = "Kappa",
                preProcess = c('center', 'scale'),
                tuneGrid = expand.grid(.subclasses = 1:10),
                trControl = ctrl)
mdaFit
plot(mdaFit)
pred_MDA <- predict(mdaFit, x_test)
confusionMatrix(pred_MDA, y_test)


############### Neural Networks #############
nnetGrid <- expand.grid(.size = 1:10, .decay = c(0, .1, 1, 2))
maxSize <- max(nnetGrid$.size)
numWts <- (maxSize * (28 + 1) + (maxSize+1)*2) ## 4 is the number of predictors

set.seed(5790)
library(caret)

nnetFit <- train(x=x_train,
                 y = y_train,
                 method = "nnet",
                 metric = "Kappa",
                 preProc = c("center", "scale"),
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 100,
                 MaxNWts = numWts,
                 trControl = ctrl)
nnetFit
plot(nnetFit)
pred_nnet<-predict(nnetFit,x_test)

confusionMatrix(data = pred_nnet,
                reference=y_test)

########## Flexible Discriminant Analysis ############
marsGrid <- expand.grid(degree = 1:3, nprune = 2:15)

# Train the model
fdaTuned <- train(
  x = x_train,
  y = y_train,
  method = "fda",
  metric = "Kappa",
  preProcess = c('center', 'scale'),
  tuneGrid = marsGrid,
  trControl = trainControl(method = "cv"))

fdaTuned
plot(fdaTuned)
pred_fda <- predict(fdaTuned, x_test)
confusionMatrix(data = pred_fda,
                reference=y_test)

############## Support Vector Machines ##########
set.seed(5790)
library(kernlab)
library(caret)
sigmaRangeReduced <- sigest(as.matrix(x_train))

svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1],
                               .C = 2^(seq(-2, 4, by = 2)))
set.seed(5790)
svmRModel <- train(x=x_train,
                   y = y_train,
                   method = "svmRadial",
                   metric = "Kappa",
                   preProc = c("center", "scale"),
                   tuneGrid = svmRGridReduced,
                   fit = FALSE,
                   trControl = ctrl)
svmRModel
plot(svmRModel)
pred_svm <- predict(svmRModel, x_test)
confusionMatrix(data = pred_svm,
                reference=y_test)

############ K-Nearest Neighbors #############
set.seed(5790)
knnFit <- train(x=x_train,
                y = y_train,
                method = "knn",
                metric = "Kappa",
                preProc = c("center", "scale"),
                ##tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)), ## 21 is the best
                tuneGrid = data.frame(.k = 1:50),
                trControl = ctrl)

knnFit
plot(knnFit)
pred_knn<-predict(knnFit, x_test)
confusionMatrix(data = pred_knn,
                reference=y_test)

########## Naive Bayes ##########
install.packages("klaR")
library(klaR)
set.seed(5790)
nbFit <- train(x_train,
               y_train,
               method = "nb",
               metric = "ROC",
               preProc = c("center", "scale"),
               ##tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)), ## 21 is the best
               tuneGrid = data.frame(.fL = 2,.usekernel = TRUE,.adjust = TRUE),
               trControl = ctrl)

nbFit
plot(nbFit) #NO TUNING PARAMETER
pred_nb <- predict(nbFit, x_test)
confusionMatrix(data = pred_nb,
                reference=y_test)


important_pred <- varImp(fdaTuned)
plot(important_pred)
head(x_test)
