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

# KNN imputation for train data : done separately for each class
library(DMwR)
#Split data based on target variable
train_data_0=train_data[train_data$target==0,]
train_data_1=train_data[train_data$target==1,]

train_data_0_knn=knnImputation(data = train_data_0,k=10)
sum(is.na(train_data_0_knn))
train_data_1_knn=knnImputation(data = train_data_1,k=10)
sum(is.na(train_data_1_knn))

# KNN imputation for test data is done using class "1" as recall is more important
test_data_knn=knnImputation(data=test_data,k = 10,distData = train_data_1_knn[,setdiff(names(train_data),"target")])

sum(is.na(test_data_knn))

# combine imputed data
train_data_knn=rbind(train_data_0_knn,train_data_1_knn)

# Check variables where standard deviation =0

sd_train=apply(train_data_knn,2,sd)
sort(sd_train,decreasing = TRUE)

# none of the variables have 0 variables, so none of them are removed

# Split the train data into train and validation sets
library(caret)
rows=createDataPartition(y = train_data_knn$target,p=0.8,list=FALSE)
train_split=train_data_knn[rows,]
val_split=train_data_knn[-rows,]
prop.table(table(train_split$target))
prop.table(table(val_split$target))


## Build model : RANDOM FOREST

library(randomForest)
model = randomForest(target ~ ., data=train_split, 
                     keep.forest=TRUE, ntree=500) 
table(train_split$target)
print(model)

# Important attributes
model$importance  
round(importance(model), 2)   

# Extract and store important variables obtained from the random forest model
rf_Imp_Attr = data.frame(model$importance)
rf_Imp_Attr = data.frame(row.names(rf_Imp_Attr),rf_Imp_Attr[,1])
colnames(rf_Imp_Attr) = c('Attributes', 'Importance')
rf_Imp_Attr = rf_Imp_Attr[order(rf_Imp_Attr$Importance, decreasing = TRUE),]

# plot (directly prints the important attributes) 
varImpPlot(model)

# Predict on Train data 
pred_Train = predict(model, 
                     train_split[,setdiff(names(train_split), "target")],
                     type="response", 
                     norm.votes=TRUE)

# Build confusion matrix and find accuracy   
cm_Train = table("actual" = train_split$target, "predicted" = pred_Train);
confusionMatrix(data = pred_Train,reference = train_split$target,positive = "1")
sensitivity_Train <- cm_Train[2, 2]/sum(cm_Train[2, ])
specificity_Train <- cm_Train[1, 1]/sum(cm_Train[1, ])
accruacy_Train = sum(diag(cm_Train))/sum(cm_Train)
precision_Train= cm_Train[2, 2]/sum(cm_Train[,2])
F1_Train=2*precision_Train*sensitivity_Train/(precision_Train+sensitivity_Train)

# Predicton Test Data
pred_Val = predict(model, val_split[,setdiff(names(val_split),
                                             "target")],
                   type="response", 
                   norm.votes=TRUE)

# Build confusion matrix and find accuracy   
cm_Val = table("actual" = val_split$target, "predicted" = pred_Val);
confusionMatrix(data = pred_Val,reference = val_split$target,positive = "1")
sensitivity_Val <- cm_Val[2, 2]/sum(cm_Val[2, ])
specificity_Val <- cm_Val[1, 1]/sum(cm_Val[1, ])
accruacy_Val = sum(diag(cm_Val))/sum(cm_Val)
precision_Val= cm_Val[2, 2]/sum(cm_Val[,2])
F1_Val=2*precision_Val*sensitivity_Val/(precision_Val+sensitivity_Val)

# Build randorm forest using top 5 important attributes. 
top_Imp_Attr = as.character(rf_Imp_Attr$Attributes[1:5])

# Build the classification model using randomForest
#model_Imp = randomForest(target~.,
#                         data=train_split[,c(top_Imp_Attr,"target")], 
#                         keep.forest=TRUE,ntree=500,mtry=3,cutoff=c(0.5,0.5)) 
model_Imp = randomForest(target~.,
                         data=train_split[,c(top_Imp_Attr,"target")], 
                         keep.forest=TRUE,ntree=500,cutoff=c(0.75,0.25)) 

# Print and understand the model
print(model_Imp)

# Important attributes
model_Imp$importance  

# Predict on Train data 
pred_Train = predict(model_Imp, 
                     train_split[,setdiff(names(train_split), "target")],
                     type="response", 
                     norm.votes=TRUE)

# Build confusion matrix and find accuracy   
cm_Train = table("actual" = train_split$target, "predicted" = pred_Train);
confusionMatrix(data = pred_Train,reference = train_split$target,positive = "1")
sensitivity_Train <- cm_Train[2, 2]/sum(cm_Train[2, ])
specificity_Train <- cm_Train[1, 1]/sum(cm_Train[1, ])
accruacy_Train = sum(diag(cm_Train))/sum(cm_Train)
precision_Train= cm_Train[2, 2]/sum(cm_Train[,2])
F1_Train=2*precision_Train*sensitivity_Train/(precision_Train+sensitivity_Train)
# Predicton Val Data
pred_Val = predict(model_Imp, val_split[,setdiff(names(val_split),
                                                 "target")],
                   type="response", 
                   norm.votes=TRUE)

# Build confusion matrix and find accuracy   
cm_Val = table("actual" = val_split$target, "predicted" = pred_Val);
confusionMatrix(data = pred_Val,reference = val_split$target,positive = "1")
sensitivity_Val <- cm_Val[2, 2]/sum(cm_Val[2, ])
specificity_Val <- cm_Val[1, 1]/sum(cm_Val[1, ])
accruacy_Val = sum(diag(cm_Val))/sum(cm_Val)
precision_Val= cm_Val[2, 2]/sum(cm_Val[,2])
F1_Val=2*precision_Val*sensitivity_Val/(precision_Val+sensitivity_Val)

# Prediction of test data
pred_Test = predict(model_Imp, test_data_knn,
                    type="response", 
                    norm.votes=TRUE)
prop.table(table(pred_Test))

write.csv(x = pred_Test,file = "sub2_RFskew.csv")

F1_Test=2*precision_Test*sensitivity_Test/(precision_Test+sensitivity_Test)

#Precision=46.7354% and Recall=42.7673%
#Your score is 44.66%
# precision_Test=0.46
# sensitivity_Test=0.42