# Clear global environment variables
rm(list=ls(all=TRUE))

# Set working directory
setwd("C:/Users/Karmugilan/Downloads/R")

# Read train and test data
train_data=read.csv(file = "traindata.csv",header = T)
test_data=read.csv(file = "testdata.csv",header = T)

#Check summary and convert to appropraite data types
# Drop the variable ID as it does not give any information
str(train_data)
summary(train_data)
train_data$ID=NULL
test_data$ID=NULL
train_data$target=as.factor(as.character(train_data$target))
str(train_data)

# Check NA values
sum(is.na(train_data))
sum(is.na(test_data))

colSums(is.na(train_data))
colSums(is.na(test_data))
train_data$target=as.factor(as.character(train_data$target))
# Split the train data into train and validation sets
library(caret)
rows=createDataPartition(y = train_data$target,p=0.8,list=FALSE)
train_split=train_data[rows,]
val_split=train_data[-rows,]
prop.table(table(train_split$target))
prop.table(table(val_split$target))

# KNN imputation
train_split_knn=knnImputation(data = train_split[,setdiff(names(train_split),"target")],k=10)
val_split_knn=knnImputation(data=val_split[,setdiff(names(train_split),"target")],k=10,distData = train_split_knn)
test_data_knn=knnImputation(data=test_data[,setdiff(names(train_split),"target")],k=10,distData = train_split_knn)

names(train_split_knn)

# fit the XGBoost model
library(xgboost)

dtrain = xgb.DMatrix(data = as.matrix(train_split_knn),
                     label = as.numeric(train_split$target)-1)

model = xgboost(data = dtrain, max.depth = 2, 
                eta = 1, nthread = 2, nround = 2, 
                objective = "binary:logistic", verbose = 1)
# objective = "binary:logistic": we will train a binary classification model ;
# max.deph = 2: the trees won't be deep, because our case is very simple ;
# nthread = 2: the number of cpu threads we are going to use;
# nround = 2: there will be two passes on the data
# eta = 1: It controls the learning rate
# verbose = 1: print evaluation metric
#Use watchlist parameter. It is a list of xgb.DMatrix, each of them tagged with a name.
dtest = xgb.DMatrix(data = as.matrix(val_split_knn),
                    label = as.numeric(val_split$target)-1)

watchlist = list(train=dtrain, test=dtest)

model = xgb.train(data=dtrain, max.depth=15,
                  eta=0.01, nthread = 2, nround=1000, 
                  watchlist=watchlist,
                  eval.metric = "auc", 
                  objective = "binary:logistic",
                  verbose=1,subsample=0.5,early_stopping_rounds=100,colsample_bytree=1)
# eval.metric allows us to monitor two new metrics for each round, logloss and error.

importance <- xgb.importance(feature_names = names(train_split_knn), model = model)
print(importance)
xgb.plot.importance(importance_matrix = importance)

# predict
prob_train_split <- data.frame(predict(model, as.matrix(train_split_knn)))
prob_val_split <- data.frame(predict(model, as.matrix(val_split_knn)))
prob_test <- data.frame(predict(model, as.matrix(test_data_knn)))

print(head(pred))

library(ROCR)
pred_train <- prediction(prob_train_split,train_split$target)
perf <- performance(pred_train,measure = "tpr", x.measure = "fpr")
plot(perf)

# The numbers are probabilities that a datum will be classified as 1. 
# Therefore, will set the rule that if this probability for a specific datum is > 0.5 then the observation is classified as 1 (or 0 otherwise).

prediction_Train <- as.numeric(prob_train_split > 0.14)
prediction_Val <- as.numeric(prob_val_split > 0.14)
prediction_Test <- as.numeric(prob_test > 0.14)
prop.table(table(prediction_Test))

# Build confusion matrix and find accuracy   
cm_Train = table("actual" = train_split$target, "predicted" = prediction_Train);
confusionMatrix(data = prediction_Train,reference = train_split$target,positive = "1")
sensitivity_Train <- cm_Train[2, 2]/sum(cm_Train[2, ])
specificity_Train <- cm_Train[1, 1]/sum(cm_Train[1, ])
accruacy_Train = sum(diag(cm_Train))/sum(cm_Train)
precision_Train= cm_Train[2, 2]/sum(cm_Train[,2])
F1_Train=2*precision_Train*sensitivity_Train/(precision_Train+sensitivity_Train)

cm_Val = table("actual" = val_split$target, "predicted" = prediction_Val);
confusionMatrix(data = prediction_Val,reference = val_split$target,positive = "1")
sensitivity_Val <- cm_Val[2, 2]/sum(cm_Val[2, ])
specificity_Val <- cm_Val[1, 1]/sum(cm_Val[1, ])
accruacy_Val = sum(diag(cm_Val))/sum(cm_Val)
precision_Val= cm_Val[2, 2]/sum(cm_Val[,2])
F1_Val=2*precision_Val*sensitivity_Val/(precision_Val+sensitivity_Val)

write.csv(x = prediction_Test,file = "XGB.csv")

##best result :
## F1 = 62.15%
# Precision=69.37984%
# Recall=56.28931%