# Clear global environment variables
rm(list=ls(all=TRUE))

# Set working directory
#setwd("C:/Users/Karmugilan/Downloads/R")
setwd("C:/Users/Karmugilan/Downloads/INSOFE/Test Reports/CUTe 4/20171118_Batch32_BLR_CUTe_CSE7305c")
# Read train data
data1 <- read.csv("traindata.csv",header = T)
str(data1)
summary(data1)

# Drop the variable ID as it does not give any information
data1$ID <- NULL

#Checking for NA values
sum(is.na(data1))
colSums(is.na(data1))

# ATTR37 have more than 20% of NA values, Hence removing Attr37 from data
data1$Attr37 <- NULL
data1$target <- as.factor(as.character(data1$target))
sum(is.na(data1))

#Removing rows with NA values greater than 10%
library(DMwR)
manyNAs(data = data1,nORp = 0.1)
nonadata <- data1[-manyNAs(data = data1,nORp = 0.1),]

# KNN imputation for train data
nonadata1 <- knnImputation(data = nonadata,k = 5)
sum(is.na(nonadata1))
View(nonadata1)

write.csv(x = nonadata1,row.names = F,file = "Nonadata.csv")
data <- read.csv(file = "Nonadata.csv",header = T)
data$target <- as.factor(as.character(data$target))


#Model Building

library(ada)
datawot <- data[,-which(names(data) %in% c("target"))]
str(datawot)
model <- ada(x = datawot,y = data$target, iter=10, loss="logistic")
summary(model)

pred_Train  =  predict(model, data)  
write.csv(x = pred_Train,file = "prediction_Train.csv",row.names = F)

# Building confusion matrix and find accuracy   
cm_Train = table(data$target, pred_Train)
accu_Train= sum(diag(cm_Train))/sum(cm_Train)
View(cm_Train)
View(pred_Train)
str(pred_Train)

library(caret)
precision <- posPredValue(pred_Train, data$target, positive="1")
recall <- sensitivity(pred_Train, data$target, positive="1")
F1 <- (2 * precision * recall) / (precision + recall)


#using test data
#Same preprocessing steps for test data
data2 <- read.csv("testdata.csv",header = T)
str(data2)
sum(is.na(data2))
colSums(is.na(data2))
data2$Attr37 <- NULL
data2$id <- NULL
sum(is.na(data2))


manyNAs(data = data2,nORp = 0.1)
nonadata_test <- data2[-manyNAs(data = data2,nORp = 0.1),]
nonadata_test <- knnImputation(data = data2,k = 5)
sum(is.na(nonadata_test))
View(nonadata_test)

write.csv(x = nonadata_test,row.names = F,file = "Nonadata_test.csv")
test_data <- read.csv(file = "Nonadata_test.csv",header = T)
str(test_data)

#Prediction on test data
pred_Test  =  predict(model, test_data) 
View(pred_Test)
write.csv(x = pred_Test,file = "prediction.csv",row.names = F)
head(pred_Test)

## ADA Boost resulted with 24.06 F1 Score.