# ----- ITM 618 - 031 - Final Course Project - Bank Marketing
# ----- Professor: Mehdi Kargar
# ----- Group Members: Ulana Salyk 500700212, Jacqueline Chung 500938284, William Chinnery 500827113, Timothy Nguyen 500709964
# ------------------------------------------------------------ #

# ----- Install packages
install.packages("ISLR")
library(ISLR)

install.packages("partykit")
library(partykit)

install.packages("RWeka") # Packages needed for Information Gain calculations
library(RWeka)

install.packages("party")
library(party)

install.packages("caret") # Packages needed for Confusion Matrix
library(caret)

install.packages("ggplot2") # Packages used for a density plot
library(ggplot2)

install.packages("dplyr")
library(dplyr)

install.packages("rpart")
library(rpart) # Packages needed for Decision Trees

install.packages("rpart.plot")
library(rpart.plot)

# Packages for k-NN methdology
install.packages("class")
library(class)

install.packages("VIM")
library(VIM)

# Load packages for regression model
install.packages("ROCR") # Package for calculating the ROC
library(ROCR)

install.packages("pROC")
library(pROC)

# Load packages for calculating the accuracy of regression model
install.packages("tidyverse")
library(tidyverse)

install.packages("modelr")
library(modelr)

install.packages("broom")
library(broom)

install.packages("GGally")
library(GGally)

install.packages("randomForest")
library(randomForest)

# ----- Preliminary Data Exploration
# ----- Use training dataset to create most accurate model
# Read data from trainset csv file
preTrainData <- read.csv("trainset.csv")
View(preTrainData)

# Read data from testset csv file
testData <- read.csv("testset.csv")
View(testData)

# Check the distribution
pl1 <- ggplot(preTrainData, aes(Subscribed))
pl1 + geom_density(fill = "red", alpha = "0.7")
# We can see that there 9 times as many "no" as there are "yes"
# We can try undersampling overrepresented data, which is "no" in this case

# See summary of attributes 
# This shows us the min, max, and median of numerical attributes, 
#  whereas the frequency of dimensions of categorical attributes
summary(preTrainData) 
# As we can see pdays have has a max of '999', which is significantly higher than the min
# This is because '999' means there was no previous contact with the client
# Thus, it makes sense to eliminate this because it has 

# We can also observe that it makes sense to eliminate poutcome because it has a frequent number of "non-existent" (28015), 
# which can make the data irrelevant with the lack of existent data

# Calculate Information Gains of Attributes
weights <- InfoGainAttributeEval(trainFormula, data = preTrainData)
View(weights)
par(mar=c(5,1,5,1))
barplot(weights, las = 2) 
# Sort barplot
barplot(sort(weights, decreasing = TRUE))

# Based on the sorted barplot and View of weights, it makes sense that 
# marital, education, housing, day_of_week,loan are not very relevant

# ---------- Testing Learning Models and Their Accuracies -------------- #

# Consider the following for the Confusion Matrix:
# No = Positive
# Yes = Negative

# Create Classification (Decision Tree)

# ----- All Attributes

# Create a Decision Tree with rpart
decisionTree <- rpart(Subscribed ~ ., data = preTrainData, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
# Check for overfitting in Training Data vs. Testing Data
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
trainDecisionTree <- predict(decisionTree, preTrainData, type = 'class')
table(preTrainData$Subscribed, trainDecisionTree) # View Confusion Matrix
cmNewTrainDecisionTree <- confusionMatrix(trainDecisionTree, preTrainData$Subscribed)
cmNewTrainDecisionTree$overall['Accuracy'] # Accuracy: 95.87%
# Now use testing data
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 14.28%

# Create Decision Trees with rpart by slowly cutting the attributes
# Have all attributes
# Create Decision Trees with rpart by slowly cutting the attributes
decisionTree <- rpart(Subscribed ~ nr.employed + duration + month + age + job + campaign + marital + education + housing, data = preTrainData, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
# Check for overfitting in Training Data vs. Testing Data
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
trainDecisionTree <- predict(decisionTree, preTrainData, type = 'class')
table(preTrainData$Subscribed, trainDecisionTree) # View Confusion Matrix
cmNewTrainDecisionTree <- confusionMatrix(trainDecisionTree, preTrainData$Subscribed)
cmNewTrainDecisionTree$overall['Accuracy'] # Accuracy: 95.87%
# Now use testing data
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 14.28%

# Now cut 

# Create Decision Trees with rpart by slowly cutting the attributes
decisionTree <- rpart(Subscribed ~ nr.employed + duration + month + age + job + campaign + marital + education + housing, data = preTrainData, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
# Check for overfitting in Training Data vs. Testing Data
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
trainDecisionTree <- predict(decisionTree, preTrainData, type = 'class')
table(preTrainData$Subscribed, trainDecisionTree) # View Confusion Matrix
cmNewTrainDecisionTree <- confusionMatrix(trainDecisionTree, preTrainData$Subscribed)
cmNewTrainDecisionTree$overall['Accuracy'] # Accuracy: 95.87%
# Now use testing data
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 14.28%
# There seems to be even more overfitting

# My new hypothesis: nr.employed has askewed the data, and needs to be cleaned or deleted

# Now exclude nr.employed
decisionTree <- rpart(Subscribed ~ duration + month + contact + age + job + campaign + marital + education + housing, data = preTrainData, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
# Check for overfitting in Training Data vs. Testing Data
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
trainDecisionTree <- predict(decisionTree, preTrainData, type = 'class')
table(preTrainData$Subscribed, trainDecisionTree) # View Confusion Matrix
cmNewTrainDecisionTree <- confusionMatrix(trainDecisionTree, preTrainData$Subscribed)
cmNewTrainDecisionTree$overall['Accuracy'] # Accuracy: 94.83%
# Now use testing data
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 24.13%

# Now cut housing
decisionTree <- rpart(Subscribed ~ duration + month + contact + age + job + campaign + marital + education, data = preTrainData, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
# Check for overfitting in Training Data vs. Testing Data
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
trainDecisionTree <- predict(decisionTree, preTrainData, type = 'class')
table(preTrainData$Subscribed, trainDecisionTree) # View Confusion Matrix
cmNewTrainDecisionTree <- confusionMatrix(trainDecisionTree, preTrainData$Subscribed)
cmNewTrainDecisionTree$overall['Accuracy'] # Accuracy: 94.83%
# Now use testing data
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 24.13%

# Now remove education
decisionTree <- rpart(Subscribed ~ duration + month + contact + age + job + campaign + marital, data = preTrainData, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
# Check for overfitting in Training Data vs. Testing Data
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
trainDecisionTree <- predict(decisionTree, preTrainData, type = 'class')
table(preTrainData$Subscribed, trainDecisionTree) # View Confusion Matrix
cmNewTrainDecisionTree <- confusionMatrix(trainDecisionTree, preTrainData$Subscribed)
cmNewTrainDecisionTree$overall['Accuracy'] # Accuracy: 94.83%
# Now use testing data
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 24.13%

# Remove marital
decisionTree2 <- rpart(Subscribed ~ duration + month + contact + age + job + campaign, data = preTrainData, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree2)
# Check for overfitting in Training Data vs. Testing Data
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
trainDecisionTree <- predict(decisionTree2, preTrainData, type = 'class')
table(preTrainData$Subscribed, trainDecisionTree) # View Confusion Matrix
cmNewTrainDecisionTree <- confusionMatrix(trainDecisionTree, preTrainData$Subscribed)
cmNewTrainDecisionTree$overall['Accuracy'] # Accuracy: 94.83%
# Now use testing data
predict_unseen <- predict(decisionTree2, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 24.13%

# Remove campaign
decisionTree <- rpart(Subscribed ~ duration + month + contact + age + job, data = preTrainData, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
# Check for overfitting in Training Data vs. Testing Data
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
trainDecisionTree <- predict(decisionTree2, preTrainData, type = 'class')
table(preTrainData$Subscribed, trainDecisionTree) # View Confusion Matrix
cmNewTrainDecisionTree <- confusionMatrix(trainDecisionTree, preTrainData$Subscribed)
cmNewTrainDecisionTree$overall['Accuracy'] # Accuracy: 94.83%
# Now use testing data
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 24.13%

# Remove job
decisionTree <- rpart(Subscribed ~ duration + month + contact + age, data = preTrainData, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
# Check for overfitting in Training Data vs. Testing Data
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
trainDecisionTree <- predict(decisionTree, preTrainData, type = 'class')
table(preTrainData$Subscribed, trainDecisionTree) # View Confusion Matrix
cmNewTrainDecisionTree <- confusionMatrix(trainDecisionTree, preTrainData$Subscribed)
cmNewTrainDecisionTree$overall['Accuracy'] # Accuracy: 94.90%
# Now use testing data
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 23.27%

# This shows that job, campaign, marital, and education is not that important

# Remove age
decisionTree <- rpart(Subscribed ~ duration + month + contact, data = preTrainData, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
# Check for overfitting in Training Data vs. Testing Data
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
trainDecisionTree <- predict(decisionTree, preTrainData, type = 'class')
table(preTrainData$Subscribed, trainDecisionTree) # View Confusion Matrix
cmNewTrainDecisionTree <- confusionMatrix(trainDecisionTree, preTrainData$Subscribed)
cmNewTrainDecisionTree$overall['Accuracy'] # Accuracy: 94.74%
# Now use testing data
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 23.87%
# We also notice that the accuracy for testing data increases by very little, while the accuracy for training data decreases

# The accuracy seems to significantly decrease once we add contact
# So we will try excluding contact
decisionTree <- rpart(Subscribed ~ duration + month + age, data = preTrainData, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
# Check for overfitting in Training Data vs. Testing Data
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
trainDecisionTree <- predict(decisionTree, preTrainData, type = 'class')
table(preTrainData$Subscribed, trainDecisionTree) # View Confusion Matrix
cmNewTrainDecisionTree <- confusionMatrix(trainDecisionTree, preTrainData$Subscribed)
cmNewTrainDecisionTree$overall['Accuracy'] # Accuracy: 94.83%
# Now use testing data
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 65.98%


# age seems to be making it a lower accuracy, so age needs to be cleaned and normalized

# So exclude age and contact for now
decisionTree <- rpart(Subscribed ~ duration + month + job + campaign + marital + education + housing, data = preTrainData, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
# Check for overfitting in Training Data vs. Testing Data
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
trainDecisionTree <- predict(decisionTree, preTrainData, type = 'class')
table(preTrainData$Subscribed, trainDecisionTree) # View Confusion Matrix
cmNewTrainDecisionTree <- confusionMatrix(trainDecisionTree, preTrainData$Subscribed)
cmNewTrainDecisionTree$overall['Accuracy'] # Accuracy: 92.66%
# Now use testing data
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 66.74%

# Now cut housing
decisionTree <- rpart(Subscribed ~ duration + month + job + campaign + marital + education, data = preTrainData, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
# Check for overfitting in Training Data vs. Testing Data
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
trainDecisionTree <- predict(decisionTree, preTrainData, type = 'class')
table(preTrainData$Subscribed, trainDecisionTree) # View Confusion Matrix
cmNewTrainDecisionTree <- confusionMatrix(trainDecisionTree, preTrainData$Subscribed)
cmNewTrainDecisionTree$overall['Accuracy'] # Accuracy: 92.66%
# Now use testing data
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 66.74%

# Now remove education
decisionTree <- rpart(Subscribed ~ duration + month + job + campaign + marital, data = preTrainData, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
# Check for overfitting in Training Data vs. Testing Data
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
trainDecisionTree <- predict(decisionTree, preTrainData, type = 'class')
table(preTrainData$Subscribed, trainDecisionTree) # View Confusion Matrix
cmNewTrainDecisionTree <- confusionMatrix(trainDecisionTree, preTrainData$Subscribed)
cmNewTrainDecisionTree$overall['Accuracy'] # Accuracy: 92.66%
# Now use testing data
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 66.74%

# Remove marital
decisionTree2 <- rpart(Subscribed ~ duration + month + job + campaign, data = preTrainData, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree2)
# Check for overfitting in Training Data vs. Testing Data
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
trainDecisionTree <- predict(decisionTree2, preTrainData, type = 'class')
table(preTrainData$Subscribed, trainDecisionTree) # View Confusion Matrix
cmNewTrainDecisionTree <- confusionMatrix(trainDecisionTree, preTrainData$Subscribed)
cmNewTrainDecisionTree$overall['Accuracy'] # Accuracy: 92.66%
# Now use testing data
predict_unseen <- predict(decisionTree2, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 66.74%

# Remove campaign
decisionTree <- rpart(Subscribed ~ duration + month + job, data = preTrainData, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
# Check for overfitting in Training Data vs. Testing Data
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
trainDecisionTree <- predict(decisionTree2, preTrainData, type = 'class')
table(preTrainData$Subscribed, trainDecisionTree) # View Confusion Matrix
cmNewTrainDecisionTree <- confusionMatrix(trainDecisionTree, preTrainData$Subscribed)
cmNewTrainDecisionTree$overall['Accuracy'] # Accuracy: 92.65%
# Now use testing data
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 66.74%


# My hypothesis was correct about contact and age, so we need to either pre-process contact and age or delete it altogether

# It's probably because the data is skewed
# It can also mean it doesn't really matter whether or not some is contacted by Telephone or Cellphone (it seems very similar)
# We can normalize the data later on

# This decision tree with duration and month has the highest accuracy out of all the rpart trees

# Now we have just duration and month left
decisionTree <- rpart(Subscribed ~ duration + month, data = trainData, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 66.74%

# After assessment, it seems that everything other than duration and month give the best accuracy

# ----- Use CTrees and J48Trees
# All Attributes
# Find formula for all values
trainFormula <- Subscribed ~ .
# ------- Test Accuracies with CTrees and J48Trees
# ------- Create a decision tree CTree allCTree from trainFormula
allCTree <- ctree(trainFormula, data = preTrainData)
plot(allCTree) # Plot CTree allCTree
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
allTrainCTree <- predict(allCTree)
table(allTrainCTree, preTrainData$Subscribed) # View Confusion Matrix
cmAllTrainCTree <- confusionMatrix(allTrainCTree, preTrainData$Subscribed)
cmAllTrainCTree$overall['Accuracy'] # Accuracy: 95.97%
# Predict a test dataset testAllCTree with allCTree model and data testData
allTestCTree <- predict(allCTree, newdata = testData)
allTestCTreeTable <- table(allTestCTree, testData$Subscribed)
table(allTestCTree, testData$Subscribed) # View Confusion Matrix
# Calculate the accuracy by creating the Confusion Matrix for the testData
cmAllTestCTree <- confusionMatrix(allTestCTreeTable)
cmAllTestCTree$overall['Accuracy'] # Accuracy: 14.36%

# There is definitely overfitting in the models
# Check if this issue is applicable with other trees

# ------- Create a decision tree J48 Tree allJ48Tree from trainFormula
allJ48Tree <- J48(trainFormula, data = preTrainData)
plot(allJ48Tree) # Plot allJ48Tree
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
allTrainJ48Tree <- predict(allJ48Tree) 
table(allTrainJ48Tree, preTrainData$Subscribed)
cmAllTrainJ48Tree <- confusionMatrix(allTrainJ48Tree, preTrainData$Subscribed)
cmAllTrainJ48Tree$overall['Accuracy'] # Accuracy: 97.06%
# Predict a test dataset allTestJ48Tree with allCTree model and data testData
allTestJ48Tree <- predict(allJ48Tree, newdata = testData)
allTestJ48TreeTable <- table(allTestJ48Tree, testData$Subscribed)
allTestJ48TreeTable # View Confusion Matrix
# Calculate the accuracy by creating the Confusion Matrix for the testData
cmAllTestJ48Tree <- confusionMatrix(allTestJ48TreeTable)
cmAllTestJ48Tree$overall['Accuracy'] # Accuracy: 14.31%

# Overfitting seems to be true so far...

# Top 3 Attributes
# Develop top3FOrmula for the top 3 Information Gains
top3Formula <- Subscribed ~ nr.employed	+ duration + month 

# ------- Build a decision tree CTree called top3CTree from top3Formula
top3CTree <- ctree(top3Formula, data = preTrainData)
plot(top3CTree) # Plot CTree top5CTree
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
top3TrainCTree <- predict(top3CTree) 
table(top3TrainCTree, preTrainData$Subscribed) # View Confusion Matrix
cmtop3TrainCTree <- confusionMatrix(top3TrainCTree, preTrainData$Subscribed)
cmtop3TrainCTree$overall['Accuracy']
# Predict a test dataset top3TestCTree with top3CTree model and data testData
top3TestCTree <- predict(top3CTree, newdata = testData)
top3TestCTreeTable <- table(top3TestCTree, testData$Subscribed)
top3TestCTreeTable # View Confusion Matrix
# Calculate the accuracy by creating the Confusion Matrix
cmTop3CTree <- confusionMatrix(top3TestCTreeTable)
cmTop3CTree$overall['Accuracy']

# ------- Create a decision tree J48 Tree called top3J48Tree from top3Formula
top3J48Tree <- J48(top3Formula, data = preTrainData)
plot(top3J48Tree) # Plot top5J48Tree
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
top3TrainJ48Tree <- predict(top3J48Tree) 
table(top3TrainJ48Tree, preTrainData$Subscribed) # View Confusion Matrix
cmtop3TrainJ48Tree <- confusionMatrix(top3TrainJ48Tree, preTrainData$Subscribed)
cmtop3TrainJ48Tree$overall['Accuracy']
# Predict a test dataset top3TestJ48Tree with top3CTree model and data testData
top3TestJ48Tree <- predict(top3J48Tree, newdata = testData)
top3J48Table <- table(top3TestJ48Tree, testData$Subscribed)
top3J48Table # View Confusion Matrix
# Calculate the accuracy by creating the Confusion Matrix
cmTop3J48Tree <- confusionMatrix(top3J48Table)
cmTop3J48Tree$overall['Accuracy']

# Develop top3FOrmula for the top 3 Information Gains (excluding nr.employed)
top3Formula <- Subscribed ~ duration + month + contact

# ------- Build a decision tree CTree called top3CTree from top3Formula
top3CTree <- ctree(top3Formula, data = preTrainData)
plot(top3CTree) # Plot CTree top5CTree
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
top3TrainCTree <- predict(top3CTree) 
table(top3TrainCTree, preTrainData$Subscribed) # View Confusion Matrix
cmtop3TrainCTree <- confusionMatrix(top3TrainCTree, preTrainData$Subscribed)
cmtop3TrainCTree$overall['Accuracy'] # Accuracy: 94.88%
# Predict a test dataset top3TestCTree with top3CTree model and data testData
top3TestCTree <- predict(top3CTree, newdata = testData)
top3TestCTreeTable <- table(top3TestCTree, testData$Subscribed)
top3TestCTreeTable # View Confusion Matrix
# Calculate the accuracy by creating the Confusion Matrix
cmTop3CTree <- confusionMatrix(top3TestCTreeTable)
cmTop3CTree$overall['Accuracy'] # Accuracy: 24.70%

# ------- Create a decision tree J48 Tree called top3J48Tree from top3Formula
top3J48Tree <- J48(top3Formula, data = preTrainData)
plot(top3J48Tree) # Plot top5J48Tree
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
top3TrainJ48Tree <- predict(top3J48Tree) 
table(top3TrainJ48Tree, preTrainData$Subscribed) # View Confusion Matrix
cmtop3TrainJ48Tree <- confusionMatrix(top3TrainJ48Tree, preTrainData$Subscribed)
cmtop3TrainJ48Tree$overall['Accuracy'] # Accuracy: 95.04%
# Predict a test dataset top3TestJ48Tree with top3CTree model and data testData
top3TestJ48Tree <- predict(top3J48Tree, newdata = testData)
top3J48Table <- table(top3TestJ48Tree, testData$Subscribed)
top3J48Table # View Confusion Matrix
# Calculate the accuracy by creating the Confusion Matrix
cmTop3J48Tree <- confusionMatrix(top3J48Table)
cmTop3J48Tree$overall['Accuracy'] # Accuracy: 24.15%

# With my previous hypothesis about nr.employed, we can observe that nr.employed is definitely a barrier to high accuracy
# But it still shows that CTrees and J48Trees are probably not the best otpion for high accuracy trees
# rpart may be the best option when it comes to splitting trees

# ---- Top 5 Attributes

# As shown from the and table and barplot of the Atributes (Weights)

# ------- Top 5 Attributes based on Information Gain Calculations ------- #
# Develop top5FOrmula for the top 5 Information Gains
top5Formula <- Subscribed ~ nr.employed	+ duration + month + contact + age

# ------- Build a decision tree CTree called top5CTree from top5Formula
top5CTree <- ctree(top5Formula, data = preTrainData)
plot(top5CTree) # Plot CTree top5CTree
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
top5TrainCTree <- predict(top5CTree) 
table(top5TrainCTree, preTrainData$Subscribed) # View Confusion Matrix
cmtop5TrainCTree <- confusionMatrix(top5TrainCTree, preTrainData$Subscribed)
cmtop5TrainCTree$overall['Accuracy']  # Accuracy: 95.99%
# Predict a test dataset top5TestCTree with top5CTree model and data testData
top5TestCTree <- predict(top5CTree, newdata = testData)
top5TestCTreeTable <- table(top5TestCTree, testData$Subscribed)
top5TestCTreeTable # View Confusion Matrix
# Calculate the accuracy by creating the Confusion Matrix
cmTop5CTree <- confusionMatrix(top5TestCTreeTable)
cmTop5CTree$overall['Accuracy'] # Accuracy: 14.37%

# ------- Create a decision tree J48 Tree called top5J48Tree from top5Formula
top5J48Tree <- J48(top5Formula, data = preTrainData)
plot(top5J48Tree) # Plot top5J48Tree
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
top5TrainJ48Tree <- predict(top5J48Tree) 
table(top5TrainJ48Tree, preTrainData$Subscribed) # View Confusion Matrix
cmtop5TrainJ48Tree <- confusionMatrix(top5TrainJ48Tree, preTrainData$Subscribed)
cmtop5TrainJ48Tree$overall['Accuracy'] # Accuracy: 96.11%
# Predict a test dataset top5TestJ48Tree with top5CTree model and data testData
top5TestJ48Tree <- predict(top5J48Tree, newdata = testData)
top5J48Table <- table(top5TestJ48Tree, testData$Subscribed)
top5J48Table # View Confusion Matrix
# Calculate the accuracy by creating the Confusion Matrix
cmTop5J48Tree <- confusionMatrix(top5J48Table)
cmTop5J48Tree$overall['Accuracy'] # Accuracy: 14.28%

# Top 10 Attributes
# ------- Top 10 Attributes based on Information Gain Calculations ------- #
# Develop top10FOrmula for the Top 10 Information Gains
top10Formula <- Subscribed ~ nr.employed	+ duration + month + contact + age + job + campaign + marital + education + housing

# ------- Build a decision tree CTree called top10CTree from top10Formula
top10CTree <- ctree(top10Formula, data = preTrainData)
plot(top10CTree) # Plot top10CTree
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
top10TrainCTree <- predict(top10CTree) 
table(top10TrainCTree, preTrainData$Subscribed) # View Confusion Matrix
cmtop10TrainCTree <- confusionMatrix(top10TrainCTree, preTrainData$Subscribed)
cmtop10TrainCTree$overall['Accuracy'] # Accuracy: 95.97%
# Predict a test dataset top10TestCTree with top10CTree model and data testData
top10TestCTree <- predict(top10CTree, newdata = testData)
top10CTreeTable <- table(top10TestCTree, testData$Subscribed) 
top10CTreeTable # View Confusion Matrix
# Calculate the accuracy by creating the Confusion Matrix
cmTop10CTree <- confusionMatrix(top10CTreeTable)
cmTop10CTree$overall['Accuracy'] # Accuracy: 14.36%

# ------- Create a decision tree J48 Tree called top10J48Tree from top10Formula
top10J48Tree <- J48(top10Formula, data = preTrainData)
plot(top10J48Tree) # Plot top10J48Tree
# Calculate the accuracy by creating the Confusion Matrix for the preTrainData
top10TrainJ48Tree <- predict(top10J48Tree) 
table(top10TrainJ48Tree, preTrainData$Subscribed) # View Confusion Matrix
cmtop10TrainJ48Tree <- confusionMatrix(top10TrainJ48Tree, preTrainData$Subscribed)
cmtop10TrainJ48Tree$overall['Accuracy'] # Accuracy: 14.37%
# Predict a test dataset top10TestJ48Tree with top10J48Tree model and data testData
top10TestJ48Tree <- predict(top10J48Tree, newdata = testData)
top10J48Table <- table(top10TestJ48Tree, testData$Subscribed)
top10J48Table # View Confusion Matrix
# Calculate the accuracy by creating the Confusion Matrix
cmTop10J48Tree <- confusionMatrix(top10J48Table)
cmTop10J48Tree$overall['Accuracy']

# It seems that because the accuracy of CTrees and JTrees are so low and there is a lot of overfitting
# So, these decision trees are not appropriate

# --- Proper Preprocessing

# https://www.rdocumentation.org/packages/caret/versions/3.13/topics/preProcess
# Identify attributes with sentintel values

# Preliminary Data Visulization/exploration using Bar Charts
barplot(table(preTrainData$age),
        main = "Age",
        xlab = "Age",
        ylab = "# of people")
barplot(table(preTrainData$job),
        main = "Job",
        xlab = "Job",
        ylab = "# of people")
barplot(table(preTrainData$marital),
        main = "Marital Status",
        xlab = "status",
        ylab = "# of people")
barplot(table(preTrainData$education),
        main = "Age",
        xlab = "Age",
        ylab = "# of people")
barplot(table(preTrainData$housing),
        main = "Housing",
        xlab = "Housing",
        ylab = "# of people")
barplot(table(preTrainData$loan),
        main = "Has a personal Loan?",
        xlab = "Loan?",
        ylab = "# of people")
barplot(table(preTrainData$contact),
        main = "Contact Communication",
        xlab = "Type",
        ylab = "# of people")
barplot(table(preTrainData$month),
        main = "Last contacted month",
        xlab = "Month",
        ylab = "# of people")
barplot(table(preTrainData$day_of_week),
        main = "Last contacted Day of the week",
        xlab = "Day",
        ylab = "# of people")
barplot(table(preTrainData$duration),
        barplot(table(preTrainData$day_of_week),
                main = "Last contacted Duration",
                xlab = "Duration",
                ylab = "# of people"))
barplot(table(preTrainData$campaign),
        barplot(table(preTrainData$day_of_week),
                main = "Number of contacts dring campaign",
                xlab = "Number of contacts",
                ylab = "# of people"))
barplot(table(preTrainData$pdays),
        barplot(table(preTrainData$day_of_week),
                main = "Last contacted Day of the week",
                xlab = "Day",
                ylab = "# of people"))
# We will likely remove this attribute because of large amount of 999
barplot(table(preTrainData$pdays),
        main = "# of days passed since the client was last contacted",
        xlab = "Days",
        ylab = "# of people")
barplot(table(preTrainData$poutcome),
        main = "Outcome of previous Campaign",
        xlab = "Outcome",
        ylab = "# of people")
barplot(table(preTrainData$nr.employed),
        main = "Number of Employees",
        xlab = "Number",
        ylab = "# of people")
barplot(table(preTrainData$Subscribed),
        main = "Has the Client subscribed to the term deposit?",
        xlab = "Subscribed?",
        ylab = "# of people")

# Histogram for Numeric Attributes
# This shows us the frequency for every dimension in each attribute

# Ignore pdays
par(mar = c(5, 5, 5, 5))
age <- preTrainData$age
hist(age, xlab = "Age", col = "green")

duration <- preTrainData$duration
hist(duration, xlab = "Duration", col = "green")

campaign <- preTrainData$campaign
hist(campaign, xlab = "Campaign", col = "green")

nr.employed <- preTrainData$nr.employed
hist(nr.employed, xlab = "Nr.employed", col = "green")

# Post Data Exploration (We can see there's not much change)
pl1 <- ggplot(pretrainData, aes(Subscribed))
pl1 + geom_density(fill = "red", alpha = "0.7") 
# Consider slowly delete the no's (5%, 10%, 15%, 20%) to undersample "no"

# Delete pdays and poutcome
preTrainData = subset(preTrainData, select = -c(pdays,poutcome))
View(preTrainData)
write.csv(preTrainData, "trainDataLessP.csv")

summary(preTrainData) 

# Calculate the values that be loss to "unknown" values
nrows <- nrow(preTrainData)
nrows # Number of rows in training set: 29271

# Remove "unknown" values and create a new csv file
preTrainData <- preTrainData[!rowSums(preTrainData == "unknown"), ]
View(preTrainData)
trainDataLessUnknown <- sum(complete.cases(preTrainData)) # preTrainData without "unknown" values
trainDataLessUnknown/nrows # 92.85% is being used, while 7.15% is being removed
write.csv(preTrainData, "trainDataLessUn.csv") 

summary(preTrainData) 

# We use Summary tables to signify the correlations

# Slowly remove the outliers
# Find number of each attribute value
# To ensure it's not wrong values
summary(preTrainData$age)
summary(preTrainData$duration) 
summary(preTrainData$campaign) 
summary(preTrainData$nr.employed) 
summary(preTrainData$job) 
summary(preTrainData$marital)
summary(preTrainData$education)
summary(preTrainData$housing)
summary(preTrainData$loan)
summary(preTrainData$contact)
summary(preTrainData$month) 
summary(preTrainData$day_of_week)

# Using Density Plot To Check If Response Variable Is Close To Normal
# Use Distribution Chart to see Sentimental values

# Plotting the dependent variable distribution
pl1 <- ggplot(preTrainData, aes(duration))
pl1 + geom_density(fill = "red", alpha = "0.7")

pl1 <- ggplot(preTrainData, aes(campaign))
pl1 + geom_density(fill = "red", alpha = "0.7")

pl1 <- ggplot(preTrainData, aes(nr.employed))
pl1 + geom_density(fill = "red", alpha = "0.7")

pl1 <- ggplot(preTrainData, aes(job))
pl1 + geom_density(fill = "red", alpha = "0.7")
summary(preTrainData$job)

pl1 <- ggplot(preTrainData, aes(marital))
pl1 + geom_density(fill = "red", alpha = "0.7")
summary(preTrainData$marital)

pl1 <- ggplot(preTrainData, aes(education))
pl1 + geom_density(fill = "red", alpha = "0.7")
summary(preTrainData$education)

pl1 <- ggplot(preTrainData, aes(housing))
pl1 + geom_density(fill = "red", alpha = "0.7")
summary(preTrainData$housing)

pl1 <- ggplot(preTrainData, aes(loan))
pl1 + geom_density(fill = "red", alpha = "0.7")
summary(preTrainData$loan)

pl1 <- ggplot(preTrainData, aes(contact))
pl1 + geom_density(fill = "red", alpha = "0.7")
summary(preTrainData$contact)

pl1 <- ggplot(preTrainData, aes(month))
pl1 + geom_density(fill = "red", alpha = "0.7")
summary(preTrainData$month)

pl1 <- ggplot(preTrainData, aes(day_of_week))
pl1 + geom_density(fill = "red", alpha = "0.7")
summary(preTrainData$day_of_week)

# There doesn't seem to be an easy way to see the correlation between the values

# We see that most of the data is skewed, including Susbcribed
# So we will remove outliers to normalize the data

# Use Boxplot to find outliers
# age
boxplot(preTrainData$age, ylab = "Age", col = "green")
# If you don't need to see the plot again, you can hide it using plot=FALSE
# Now you can assign the outlier values into a vector
outliers <- boxplot(preTrainData$age, plot = FALSE)$out
# Check the results
outliers
# First you need find in which rows the outliers are
preTrainData[which(preTrainData$age %in% outliers),]
# Now you can remove the rows containing the outliers, one possible option is:
trainDataOut <- preTrainData[-which(preTrainData$age %in% outliers),]
# If you check now with boxplot, you will notice that those pesky outliers are gone
boxplot(trainDataOut$age, ylab = "Age")

# campaign
boxplot(preTrainData$campaign, ylab = "Campaign", col = "green")
# If you don't need to see the plot again, you can hide it using plot=FALSE
# Now you can assign the outlier values into a vector
outliers <- boxplot(preTrainData$campaign, plot = FALSE)$out
# Check the results
print(outliers)
# First you need find in which rows the outliers are
preTrainData[which(preTrainData$campaign %in% outliers),]
# Now you can remove the rows containing the outliers, one possible option is:
trainDataOut2 <- preTrainData[-which(preTrainData$campaign %in% outliers),]
# If you check now with boxplot, you will notice that those pesky outliers are gone
boxplot(trainDataOut2$campaign, ylab = "Campaign")

# Although nr.employed has the highest IG value, it is actually not relevant to this context
# It also makes the learning models more inaccurate
# It is best to remove nr.employed altogether
preTrainData = subset(preTrainData, select = -c(nr.employed))
View(preTrainData)
write.csv(preTrainData, "trainDataLessNr.csv")
# Check the summary to if it was removed
summary(preTrainData) 

# duration
boxplot(preTrainData$duration, ylab = "Duration", col = "green")
# Display outliers on boxplot
boxplot(preTrainData$duration)$out
# If you don't need to see the plot again, you can hide it using plot=FALSE
# Now you can assign the outlier values into a vector
outliers <- boxplot(preTrainData$duration, plot = FALSE)$out
# Check the results
outliers
# First you need find in which rows the outliers are
preTrainData[which(preTrainData$duration %in% outliers),]
# Now you can remove the rows containing the outliers, one possible option is:
trainDataOut4 <- preTrainData[-which(preTrainData$duration %in% outliers),]
# If you check now with boxplot, you will notice that those pesky outliers are gone
boxplot(trainDataOut4$duration, ylab = "Duration")

# Remove values that are not relevant
View(preTrainData)
# Delete "dec" from month attribute
trainDataLessOut <- preTrainData[!(preTrainData$month == "dec"), ]
View(trainDataLessOut)
# Delete "sep" from month attribute
trainDataLessOut <- trainDataLessOut[!(trainDataLessOut$month == "sep"), ]
View(trainDataLessOut)
# Delete "illiterate" from education attribute
trainDataLessOut <- trainDataLessOut[!(trainDataLessOut$education == "illiterate"), ]
View(trainDataLessOut)
# Try only deleting "dec" and "illiterate"
# Delete "dec" from month attribute
trainDataLessOut2 <- preTrainData[!(preTrainData$month == "dec"), ]
View(trainDataLessOut2)
# Delete "illiterate" from education attribute
trainDataLessOut2 <- trainDataLessOut2[!(trainDataLessOut2$education == "illiterate"), ]
View(trainDataLessOut2)

# Create a new file without Outliers
write.csv(trainDataLessOut, "trainDataLessOut.csv")

# Normalization

normalize <- function(x) {
        return ((x - min(x)) / (max(x) - min(x))) }
# Normalize age
summary(trainDataLessOut[1])
norm <- as.data.frame(lapply(trainDataLessOut[1], normalize))
summary(trainDataLessOut[1])
# Normalize campaign and duration
summary(trainDataLessOut[,10:11])
norm2 <- as.data.frame(lapply(trainDataLessOut[,10:11], normalize))
summary(trainDataLessOut[,10:11])

# calculate the pre-process parameters from the dataset
# Normalize age
preprocessParams <- preProcess(trainDataLessOut[1], method=c("center"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, testData[1])
summary(transformed)
# Transform duration and campaign
preprocessParams <- preProcess(trainDataLessOut[,10:11], method=c("center"))
# summarize transform parameters
print(preprocessParams)
# summarize the transformed dataset
transformed2 <- predict(preprocessParams, trainDataLessOut[1,10:11])
summary(transformed2)

# Create a new file with center pre-processing
write.csv(trainDataLessOut, "trainDataCenter.csv")

# Test with the most accurate tree
accTree <- rpart(Subscribed ~ age + campaign + duration, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(accTree)
predict_unseen <- predict(accTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 84.59%
# Choose the best complexity parameter "cp" to prune the accTree
cp.optim <- accTree$cptable[which.min(accTree$cptable[,"xerror"]),"CP"]
# accTree prunning using the best complexity parameter. For more in
accTree <- prune(accTree, cp=cp.optim)
rpart.plot(accTree)
pred <- predict(object=accTree,testData,type="class")
t <- table(testData$Subscribed,pred)
a <- confusionMatrix(t)
a$overall['Accuracy'] # Accuracy: 84.59%

# Standardize: combining the scale and center transforms will standardize your data. Attributes will have a mean value of 0 and a standard deviation of 1.

# summarize data
summary(trainDataLessOut[1])
summary(trainDataLessOut[,10:11])

# calculate the pre-process parameters from the dataset
# Normalize age
preprocessParams <- preProcess(trainDataLessOut[1], method=c("center", "scale"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, testData[1])
summary(transformed)
# Transform duration and campaign
preprocessParams <- preProcess(trainDataLessOut[,10:11], method=c("center", "scale"))
# summarize transform parameters
print(preprocessParams)
# summarize the transformed dataset
transformed2 <- predict(preprocessParams, trainDataLessOut[1,10:11])
summary(transformed2)

# Create a new file with Standardized pre-processing
write.csv(trainDataLessOut, "trainDataStand.csv")

# Test with the most accurate tree
accTree <- rpart(Subscribed ~ age + campaign + duration, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(accTree)
predict_unseen <- predict(accTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 84.59%
# Choose the best complexity parameter "cp" to prune the accTree
cp.optim <- accTree$cptable[which.min(accTree$cptable[,"xerror"]),"CP"]
# accTree prunning using the best complexity parameter. For more in
accTree <- prune(accTree, cp=cp.optim)
rpart.plot(accTree)
pred <- predict(object=accTree,testData,type="class")
t <- table(testData$Subscribed,pred)
a <- confusionMatrix(t)
a$overall['Accuracy'] # Accuracy: 84.59%

# Normalize: Data values can be scaled into the range of [0, 1] which is called normalization.
# summarize data
summary(trainDataLessOut[1])
summary(trainDataLessOut[,10:11])

# calculate the pre-process parameters from the dataset
# Normalize age
preprocessParams <- preProcess(trainDataLessOut[1], method=c("range"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, testData[1])
summary(transformed)
# Transform duration and campaign
preprocessParams <- preProcess(trainDataLessOut[,10:11], method=c("range"))
# summarize transform parameters
print(preprocessParams)
# summarize the transformed dataset
transformed2 <- predict(preprocessParams, trainDataLessOut[1,10:11])
summary(transformed2)

# Create a new file with Normalized pre-processing
write.csv(trainDataLessOut, "trainDataNorm.csv")

# Test with the most accurate tree
accTree <- rpart(Subscribed ~ age + campaign + duration, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(accTree)
predict_unseen <- predict(accTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 84.59%
# Choose the best complexity parameter "cp" to prune the accTree
cp.optim <- accTree$cptable[which.min(accTree$cptable[,"xerror"]),"CP"]
# accTree prunning using the best complexity parameter. For more in
accTree <- prune(accTree, cp=cp.optim)
rpart.plot(accTree)
pred <- predict(object=accTree,testData,type="class")
t <- table(testData$Subscribed,pred)
a <- confusionMatrix(t)
a$overall['Accuracy'] # Accuracy: 84.59%

#Box-cox Transform: When an attribute has a Gaussian-like distribution but is shifted, this is called a skew. The distribution of an attribute can be shifted to reduce the skew and make it more Gaussian. The BoxCox transform can perform this operation (assumes all values are positive).

# summarize data
summary(trainDataLessOut[1])
summary(trainDataLessOut[,10:11])

# calculate the pre-process parameters from the dataset
# Normalize age
preprocessParams <- preProcess(trainDataLessOut[1], method=c("BoxCox"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, testData[1])
summary(transformed)
# Transform duration and campaign
preprocessParams <- preProcess(trainDataLessOut[,10:11], method=c("BoxCox"))
# summarize transform parameters
print(preprocessParams)
# summarize the transformed dataset
transformed2 <- predict(preprocessParams, trainDataLessOut[1,10:11])
summary(transformed2)

# Create a new file with BoxCox pre-processing
write.csv(trainDataLessOut, "trainDataBoxCox.csv")

# Test with the most accurate tree
accTree <- rpart(Subscribed ~ age + campaign + duration, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(accTree)
predict_unseen <- predict(accTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 85.03%
# Choose the best complexity parameter "cp" to prune the accTree
cp.optim <- accTree$cptable[which.min(accTree$cptable[,"xerror"]),"CP"]
# accTree prunning using the best complexity parameter. For more in
accTree <- prune(accTree, cp=cp.optim)
rpart.plot(accTree)
pred <- predict(object=accTree,testData,type="class")
t <- table(testData$Subscribed,pred)
a <- confusionMatrix(t)
a$overall['Accuracy'] # Accuracy: 85.05%

# --- DATA MODELLING 

# Use rpart

decisionTree <- rpart(Subscribed ~ duration + month + job, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # 68.22%

decisionTree <- rpart(Subscribed ~ duration + month, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # 68.22%

# Try the accuracy with "sep" in month
decisionTree <- rpart(Subscribed ~ duration + month + job, data = trainDataLessOut2, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # 66.79%

decisionTree <- rpart(Subscribed ~ duration + month, data = trainDataLessOut2, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # 66.79%
# It seems that if I don't remove "sep" from month attribute, the accuracy decreases
# So, I will remove "sep" from month attribute

# Now try with classification decision trees
# Develop top3FOrmula for the top 3 Information Gains
top3Formula <- Subscribed ~ duration + month + job

# ------- Build a decision tree CTree called top3CTree from top3Formula
top3CTree <- ctree(top3Formula, data = trainDataLessOut)
plot(top3CTree) # Plot CTree top5CTree
# Calculate the accuracy by creating the Confusion Matrix for the trainDataLessOut
top3TrainCTree <- predict(top3CTree) 
table(top3TrainCTree, trainDataLessOut$Subscribed) # View Confusion Matrix
cmtop3TrainCTree <- confusionMatrix(top3TrainCTree, trainDataLessOut$Subscribed)
cmtop3TrainCTree$overall['Accuracy']
# Predict a test dataset top3TestCTree with top3CTree model and data testData
top3TestCTree <- predict(top3CTree, newdata = testData)
top3TestCTreeTable <- table(top3TestCTree, testData$Subscribed)
top3TestCTreeTable # View Confusion Matrix
# Calculate the accuracy by creating the Confusion Matrix
cmTop3CTree <- confusionMatrix(top3TestCTreeTable)
cmTop3CTree$overall['Accuracy']

# The accuracy has also increased

# ------- Create a decision tree J48 Tree called top3J48Tree from top3Formula
top3J48Tree <- J48(top3Formula, data = trainDataLessOut)
plot(top3J48Tree) # Plot top5J48Tree
# Calculate the accuracy by creating the Confusion Matrix for the trainDataLessOut
top3TrainJ48Tree <- predict(top3J48Tree) 
table(top3TrainJ48Tree, trainDataLessOut$Subscribed) # View Confusion Matrix
cmtop3TrainJ48Tree <- confusionMatrix(top3TrainJ48Tree, trainDataLessOut$Subscribed)
cmtop3TrainJ48Tree$overall['Accuracy']
# Predict a test dataset top3TestJ48Tree with top3CTree model and data testData
top3TestJ48Tree <- predict(top3J48Tree, newdata = testData)
top3J48Table <- table(top3TestJ48Tree, testData$Subscribed)
top3J48Table # View Confusion Matrix
# Calculate the accuracy by creating the Confusion Matrix
cmTop3J48Tree <- confusionMatrix(top3J48Table)
cmTop3J48Tree$overall['Accuracy']

# Accuracy increased 
# This means my hypothesis is true

# https://www.kaggle.com/jhuno137/classification-tree-using-rpart-100-accuracy
# In order to have an undertanding for the importance of the attributes in identifying the right category ('no' or 'yes') we are going to:
# 1. Create a table for each feature versus class type ('no','yes')
# 2. Report the number of columns where there is a zero in the table created in Step 1
# 3. Reorder the list created in Step 2 by the number of zeroes reported
# 4. Plot the sorted list from step 3

table(trainDataLessOut$Subscribed, trainDataLessOut$month)
table(trainDataLessOut$Subscribed, trainDataLessOut$contact)
table(trainDataLessOut$Subscribed, trainDataLessOut$age) 
table(trainDataLessOut$Subscribed, trainDataLessOut$job) # Why does "unknown" still show up?
table(trainDataLessOut$Subscribed, trainDataLessOut$campaign) 
table(trainDataLessOut$Subscribed, trainDataLessOut$marital)  # Why does "unknown" still show up?
table(trainDataLessOut$Subscribed, trainDataLessOut$education) 
table(trainDataLessOut$Subscribed, trainDataLessOut$housing) 
table(trainDataLessOut$Subscribed, trainDataLessOut$day_of_week)
table(trainDataLessOut$Subscribed, trainDataLessOut$loan) 

# We can instantly spot that month = apr, dec, mar, sep are 'yes'; while others are mized
# We can also see that age = 61-88 is 'yes'; while the rest is mixed
# We can also see that campaign (days since last contact) = 16, 18, 19, 20, 21, 22, 24-43 are 'no'; while the rest is mixed

# This insight will help us evaluating the correctness of the final results.

# Slowly cut down attributes

# Set panelty.matrix, so we can prune the unrealiable branches
penalty.matrix <- matrix(c(0,10,10,0), byrow=TRUE, nrow=2)
tree <- rpart(Subscribed ~ duration + month + age + campaign + job + marital + education + loan + contact + day_of_week,
              data = trainDataLessOut,
              parms = list(loss = penalty.matrix),
              method = "class")

# choosing the best complexity parameter "cp" to prune the tree
cp.optim <- tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]
# tree prunning using the best complexity parameter. For more in
tree <- prune(tree, cp=cp.optim)
rpart.plot(tree)
pred <- predict(object=tree,testData,type="class")
t <- table(testData$Subscribed,pred)
a <- confusionMatrix(t)
a$overall['Accuracy'] # Accuracy: 25.30%

# Remove day_of_week
tree <- rpart(Subscribed ~ duration + month + age + campaign + job + marital + education + loan + contact,
              data = trainDataLessOut,
              parms = list(loss = penalty.matrix),
              method = "class")

# Choosing the best complexity parameter "cp" to prune the tree
cp.optim <- tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]
# Rree prunning using the best complexity parameter. For more in
tree <- prune(tree, cp=cp.optim)
rpart.plot(tree)
pred <- predict(object=tree,testData,type="class")
t <- table(testData$Subscribed,pred)
a <- confusionMatrix(t)
a$overall['Accuracy'] # Accuracy: 25.30%

# Remove contact
tree <- rpart(Subscribed ~ duration + month + age + campaign + job + marital + education + loan,
              data = trainDataLessOut,
              parms = list(loss = penalty.matrix),
              method = "class")

# Shoosing the best complexity parameter "cp" to prune the tree
cp.optim <- tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]
# Tree prunning using the best complexity parameter. For more in
tree <- prune(tree, cp=cp.optim)
rpart.plot(tree)
pred <- predict(object=tree,testData,type="class")
t <- table(testData$Subscribed,pred)
a <- confusionMatrix(t)
a$overall['Accuracy']  # Accuracy: 67.09%
# As suspected, accuracy significantly increases without contact

tree <- rpart(Subscribed ~ duration + month + age + campaign + job + marital + education,
              data = trainDataLessOut,
              parms = list(loss = penalty.matrix),
              method = "class")

# choosing the best complexity parameter "cp" to prune the tree
cp.optim <- tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]
# tree prunning using the best complexity parameter. For more in
tree <- prune(tree, cp=cp.optim)
rpart.plot(tree)
pred <- predict(object=tree,testData,type="class")
t <- table(testData$Subscribed,pred)
a <- confusionMatrix(t)
a$overall['Accuracy']  # Accuracy: 67.09%

# Remove education
tree <- rpart(Subscribed ~ duration + month + age + campaign + job + marital,
              data = trainDataLessOut,
              parms = list(loss = penalty.matrix),
              method = "class")

# choosing the best complexity parameter "cp" to prune the tree
cp.optim <- tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]
# tree prunning using the best complexity parameter. For more in
tree <- prune(tree, cp=cp.optim)
rpart.plot(tree)
pred <- predict(object=tree,testData,type="class")
t <- table(testData$Subscribed,pred)
a <- confusionMatrix(t)
a$overall['Accuracy']  # Accuracy: 67.09%

# Remove marital
tree <- rpart(Subscribed ~ duration + month + age + campaign + job,
              data = trainDataLessOut,
              parms = list(loss = penalty.matrix),
              method = "class")

# choosing the best complexity parameter "cp" to prune the tree
cp.optim <- tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]
# tree prunning using the best complexity parameter. For more in
tree <- prune(tree, cp=cp.optim)
rpart.plot(tree)
pred <- predict(object=tree,testData,type="class")
t <- table(testData$Subscribed,pred)
a <- confusionMatrix(t)
a$overall['Accuracy']  # Accuracy: 67.09%

# Remove job
tree <- rpart(Subscribed ~ duration + month + age + campaign,
              data = trainDataLessOut,
              parms = list(loss = penalty.matrix),
              method = "class")

# Choose the best complexity parameter "cp" to prune the tree
cp.optim <- tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]
# Tree prunning using the best complexity parameter. 
tree <- prune(tree, cp=cp.optim)
rpart.plot(tree)
pred <- predict(object=tree,testData,type="class")
t <- table(testData$Subscribed,pred)
a <- confusionMatrix(t)
a$overall['Accuracy'] # Accuracy: 67.09%

# Remove campaign
tree <- rpart(Subscribed ~ duration + month + age,
              data = trainDataLessOut,
              parms = list(loss = penalty.matrix),
              method = "class")

# Choosing the best complexity parameter "cp" to prune the tree
cp.optim <- tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]
# tree prunning using the best complexity parameter. For more in
tree <- prune(tree, cp=cp.optim)
rpart.plot(tree)
pred <- predict(object=tree,testData,type="class")
t <- table(testData$Subscribed,pred)
a <- confusionMatrix(t)
a$overall['Accuracy'] # Accuracy: 67.09%

# Remove age
tree <- rpart(Subscribed ~ duration + month,
              data = trainDataLessOut,
              parms = list(loss = penalty.matrix),
              method = "class")

# choosing the best complexity parameter "cp" to prune the tree
cp.optim <- tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]
# tree prunning using the best complexity parameter. For more in
tree <- prune(tree, cp=cp.optim)
rpart.plot(tree)
pred <- predict(object=tree,testData,type="class")
t <- table(testData$Subscribed,pred)
t
a <- confusionMatrix(t)
a$overall['Accuracy'] # Accuracy: 68.22%
 
# This has the best accuracy, as suspected

# Now we can test if excluding contact would make the accuracy higher
tree <- rpart(Subscribed ~ duration + month + age + campaign + job + marital + education + loan + day_of_week,
              data = trainDataLessOut,
              parms = list(loss = penalty.matrix),
              method = "class")

# choosing the best complexity parameter "cp" to prune the tree
cp.optim <- tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]
# tree prunning using the best complexity parameter. For more in
tree <- prune(tree, cp=cp.optim)
rpart.plot(tree)
pred <- predict(object=tree,testData,type="class")
t <- table(testData$Subscribed,pred)
a <- confusionMatrix(t)
a$overall['Accuracy'] # Accuracy: 67.98%

# See if removing age does anything
tree <- rpart(Subscribed ~ duration + month + campaign + job + marital + education + loan + day_of_week,
              data = trainDataLessOut,
              parms = list(loss = penalty.matrix),
              method = "class")

# choosing the best complexity parameter "cp" to prune the tree
cp.optim <- tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]
# tree prunning using the best complexity parameter. For more in
tree <- prune(tree, cp=cp.optim)
rpart.plot(tree)
pred <- predict(object=tree,testData,type="class")
t <- table(testData$Subscribed,pred)
a <- confusionMatrix(t)
a$overall['Accuracy'] # Accuracy: 68.22%

# As we suspected, removing age and contact makes the accuracy of the rpart tree a a lot larger


# ----- UNDERSAMPLING TARGET CLASS

# As we mentioned before, we think undersampling the Target Class will help increase the accuracy
ctrl <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats = 10, 
                     verboseIter = FALSE,
                     sampling = "down")
set.seed(42)
model_rf_under <- caret::train(Subscribed ~ .,
                               data = trainDataLessOut,
                               method = "rf",
                               preProcess = c("scale", "center"),
                               trControl = ctrl)
final_under <- data.frame(actual = testData$Subscribed,
                          predict(model_rf_under, newdata = testData, type = "prob"))
final_under$predict <- ifelse(final_under$month > 0.5, "month", "duration")
cm_under <- confusionMatrix(final_under$predict, testData$Subscribed)
cm_under$overall['Accuracy'] # Accuracy: 

# Slowly cut down rows (NORMALIZE SUBSCRIBED RESPONSES)
View(trainDataLessOut) # 27,178 rows
noData <- trainDataLessOut[!(trainDataLessOut$Subscribed == "yes"), ]
View(noData) # 24,208 rows (89.1%)
yesData <- trainDataLessOut[(trainDataLessOut$Subscribed == "yes"), ]
View(yesData) # 2,970 rows (10.93%)

# There is almost 9 times more "no" than "yes"

# Creates a value for dividing the data into train and test -> value is defined as 95% of the number of rows in the dataset
smpSize = floor(0.95*nrow(noData))  
smpSize  # shows the value of the sample size

# Set seed to ensure you always have same random numbers generated
set.seed(123)   
# Randomly identifies the rows equal to sample size (defined in previous instruction) from  all the rows of allData dataset and stores the row number in trainIndex
noIndex = sample(seq_len(nrow(noData)), size = smpSize)  
# Creates the training dataset with row numbers stored in trainIndex
newTrainData = trainDataLessOut[noIndex, ] 
View(newTrainData) # View the training dataset, trainDataLessOut

# Test with the "most accurate" decision tree
accTree <- rpart(Subscribed ~ duration + month, data = newTrainData, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(accTree)
predict_unseen <- predict(accTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 70.87%
# Choose the best complexity parameter "cp" to prune the accTree
cp.optim <- accTree$cptable[which.min(accTree$cptable[,"xerror"]),"CP"]
# accTree prunning using the best complexity parameter. For more in
accTree <- prune(accTree, cp=cp.optim)
rpart.plot(accTree)
pred <- predict(object=accTree,testData,type="class")
t <- table(testData$Subscribed,pred)
a <- confusionMatrix(t)
a$overall['Accuracy'] # Accuracy: 65.71%
# With cutting only 5% of the no's, this  has increase our accuracy, making it the most accurate tree so far
# However, we see that pruning only decreases the accuracy

# Second try at cutting out rows with Subscribed = "no" with 10% reduction

# Creates a value for dividing the data into train and test -> value is defined as 90% of the number of rows in the dataset
smpSize = floor(0.9*nrow(noData))  
smpSize  # shows the value of the sample size

# Set seed to ensure you always have same random numbers generated
set.seed(123)   
# Randomly identifies the rows equal to sample size (defined in previous instruction) from  all the rows of allData dataset and stores the row number in trainIndex
noIndex = sample(seq_len(nrow(noData)), size = smpSize)  
# Creates the training dataset with row numbers stored in trainIndex
newTrainData2 = newTrainData[noIndex, ] 
View(newTrainData2) # View the training dataset, trainDataLessOut

# Test with the "most accurate" decision tree
accTree2 <- rpart(Subscribed ~ duration + month, data = newTrainData2, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(accTree2)
predict_unseen <- predict(accTree2, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 70.15%
# Choose the best complexity parameter "cp" to prune the accTree2
cp.optim <- accTree2$cptable[which.min(accTree2$cptable[,"xerror"]),"CP"]
# accTree2 prunning using the best complexity parameter. For more in
accTree2 <- prune(accTree2, cp=cp.optim)
rpart.plot(accTree2)
pred <- predict(object=accTree2,testData,type="class")
t <- table(testData$Subscribed,pred)
a <- confusionMatrix(t)
a$overall['Accuracy'] # Accuracy: 70.15%

# Third try at cutting out rows with Subscribed = "no" with 20% reduction

# Creates a value for dividing the data into train and test -> value is defined as 80% of the number of rows in the dataset
smpSize = floor(0.8*nrow(noData))  
smpSize  # shows the value of the sample size

# Set seed to ensure you always have same random numbers generated
set.seed(123)   
# Randomly identifies the rows equal to sample size (defined in previous instruction) from  all the rows of allData dataset and stores the row number in trainIndex
noIndex = sample(seq_len(nrow(noData)), size = smpSize)  
# Creates the training dataset with row numbers stored in trainIndex
newTrainData3 = newTrainData2[noIndex, ] 
View(newTrainData3) # View the training dataset, trainData
# Reducing to 80% of the "no" seems to make the tree the most accurate

# Test with the "most accurate" decision tree
accTree3 <- rpart(Subscribed ~ duration + month, data = newTrainData3, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(accTree3)
predict_unseen <- predict(accTree3, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 68.18%
# Choose the best complexity parameter "cp" to prune the accTree3
cp.optim <- accTree3$cptable[which.min(accTree3$cptable[,"xerror"]),"CP"]
# accTree3 prunning using the best complexity parameter. For more in
accTree3 <- prune(accTree3, cp=cp.optim)
rpart.plot(accTree3)
pred <- predict(object=accTree3,testData,type="class")
t <- table(testData$Subscribed,pred)
a <- confusionMatrix(t)
a$overall['Accuracy'] # Accuracy: 68.18%

# Fourth try at cutting out rows with Subscribed = "no" with 30% reduction

# Creates a value for dividing the data into train and test -> value is defined as 70% of the number of rows in the dataset
smpSize = floor(0.7*nrow(noData))  
smpSize  # shows the value of the sample size

# Set seed to ensure you always have same random numbers generated
set.seed(123)   
# Randomly identifies the rows equal to sample size (defined in previous instruction) from  all the rows of allData dataset and stores the row number in trainIndex
noIndex = sample(seq_len(nrow(noData)), size = smpSize)  
# Creates the training dataset with row numbers stored in trainIndex
newTrainData4 = newTrainData3[noIndex, ] 
View(newTrainData4) # View the training dataset, trainData

# Test with the "most accurate" decision tree
accTree4 <- rpart(Subscribed ~ duration + month, data = newTrainData4, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(accTree4)
predict_unseen <- predict(accTree4, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 65.67%
# Choose the best complexity parameter "cp" to prune the accTree4
cp.optim <- accTree4$cptable[which.min(accTree4$cptable[,"xerror"]),"CP"]
# accTree4 prunning using the best complexity parameter. For more in
accTree4 <- prune(accTree4, cp=cp.optim)
rpart.plot(accTree4)
pred <- predict(object=accTree4,testData,type="class")
t <- table(testData$Subscribed,pred)
a <- confusionMatrix(t)
a$overall['Accuracy'] # Accuracy: 65.67%

# This theoretically gives us the most accurate decision tree so far
# However, it seems that balancing it makes it less accurate

# DATA MODELLING 
# ------ METHODLOGY TYPE

# DECISION TREES (Jacqueline)

# CTree & J48Tree
# ---- All Attributes
# Find formula for all values
trainFormula <- Subscribed ~ .

# Pruning https://www.r-bloggers.com/classification-trees-using-the-rpart-function/
# We would then consider whether the tree could be simplified by pruning and make use of the plotcp function:

# Create Decision Trees with rpart by slowly cutting the attributes
decisionTree <- rpart(Subscribed ~ duration + month + contact + age + job + campaign + marital + education + housing, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 23.27%

decisionTree <- rpart(Subscribed ~ duration + month + contact + age + job + campaign + marital + education, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 23.27%

decisionTree <- rpart(Subscribed ~ duration + month + contact + age + job + campaign + marital, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 23.27%

decisionTree2 <- rpart(Subscribed ~ duration + month + contact + age + job + campaign, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree2)
predict_unseen <- predict(decisionTree2, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 23.27%

decisionTree <- rpart(Subscribed ~ duration + month + contact + age + job, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 23.27%

decisionTree <- rpart(Subscribed ~ duration + month + contact + age, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 23.27%

# This shows that job, campaign, marital, and education is not that important

decisionTree <- rpart(Subscribed ~ duration + month + contact, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 23.87%

# The accuracy seems to significantly decrease once we add contact
# So we will try excluding contact

decisionTree <- rpart(Subscribed ~ duration + month + age, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 65.98%

# My new hypothesis: nr.employed has askewed the data, and needs to be cleaned or deleted

decisionTree <- rpart(Subscribed ~ duration + month + age + job + campaign + marital + education + housing, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 65.98%

# age seems to be making it a lower accuracy, so age needs to be cleaned and normalized

decisionTree <- rpart(Subscribed ~ duration + month + age + job + campaign + marital + education, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 65.98%

decisionTree <- rpart(Subscribed ~ duration + month + age + job + campaign, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 65.98%

decisionTree <- rpart(Subscribed ~ duration + month + age + job, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 65.98%

# This shows that campaign, marital, education, and housing is not that important

decisionTree <- rpart(Subscribed ~ duration + month + job, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 66.74%

# Accuracy increased a little without age

# My hypothesis was correct about contact, so we need to either pre-process contact or delete it altogether

# It's probably because the data is skewed
# We will normalize the data later on

# This decision tree has the highest accuracy out of all the rpart trees

decisionTree <- rpart(Subscribed ~ duration + month, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 68.22%

# After assessment, it seems that everything other than duration and month give the best accuracy

# ------- Test Accuracies with CTrees and J48Trees
# ------- Create a decision tree CTree allCTree from trainFormula
allCTree <- ctree(trainFormula, data = trainDataLessOut)
plot(allCTree) # Plot CTree allCTree
# Calculate the accuracy by creating the Confusion Matrix for the trainDataLessOut
allTrainCTree <- predict(allCTree)
table(allTrainCTree, trainDataLessOut$Subscribed) # View Confusion Matrix
cmAllTrainCTree <- confusionMatrix(allTrainCTree, trainDataLessOut$Subscribed)
cmAllTrainCTree$overall['Accuracy'] # Accuracy: 95.97%
# Predict a test dataset testAllCTree with allCTree model and data testData
allTestCTree <- predict(allCTree, newdata = testData)
allTestCTreeTable <- table(allTestCTree, testData$Subscribed)
table(allTestCTree, testData$Subscribed) # View Confusion Matrix
# Calculate the accuracy by creating the Confusion Matrix for the testData
cmAllTestCTree <- confusionMatrix(allTestCTreeTable)
cmAllTestCTree$overall['Accuracy'] # Accuracy: 14.36%

# There is definitely overfitting in the models
# Check if this issue is applicable with other trees

# ------- Create a decision tree J48 Tree allJ48Tree from trainFormula
allJ48Tree <- J48(trainFormula, data = trainDataLessOut)
plot(allJ48Tree) # Plot allJ48Tree
# Calculate the accuracy by creating the Confusion Matrix for the trainDataLessOut
allTrainJ48Tree <- predict(allJ48Tree) 
table(allTrainJ48Tree, trainDataLessOut$Subscribed)
cmAllTrainJ48Tree <- confusionMatrix(allTrainJ48Tree, trainDataLessOut$Subscribed)
cmAllTrainJ48Tree$overall['Accuracy'] # Accuracy: 97.06%
# Predict a test dataset allTestJ48Tree with allCTree model and data testData
allTestJ48Tree <- predict(allJ48Tree, newdata = testData)
allTestJ48TreeTable <- table(allTestJ48Tree, testData$Subscribed)
allTestJ48TreeTable # View Confusion Matrix
# Calculate the accuracy by creating the Confusion Matrix for the testData
cmAllTestJ48Tree <- confusionMatrix(allTestJ48TreeTable)
cmAllTestJ48Tree$overall['Accuracy'] # Accuracy: 14.31%

# This is true so far

# Calculate Information Gains of Attributes
weights <- InfoGainAttributeEval(trainFormula, data = trainDataLessOut)
View(weights)
barplot(weights, las = 2) 
# Sort barplot
barplot(sort(weights, decreasing = TRUE))

# Based on the sorted barplot and View of weights, it makes sense that 
# marital, education, housing, day_of_week,loan are not very relevant

# Top 3 Attributes
# Develop top3FOrmula for the top 3 Information Gains (excluding nr.employed)
top3Formula <- Subscribed ~ duration + month + contact

# ------- Build a decision tree CTree called top3CTree from top3Formula
top3CTree <- ctree(top3Formula, data = trainDataLessOut)
plot(top3CTree) # Plot CTree top5CTree
# Calculate the accuracy by creating the Confusion Matrix for the trainDataLessOut
top3TrainCTree <- predict(top3CTree) 
table(top3TrainCTree, trainDataLessOut$Subscribed) # View Confusion Matrix
cmtop3TrainCTree <- confusionMatrix(top3TrainCTree, trainDataLessOut$Subscribed)
cmtop3TrainCTree$overall['Accuracy'] # Accuracy: 94.88%
# Predict a test dataset top3TestCTree with top3CTree model and data testData
top3TestCTree <- predict(top3CTree, newdata = testData)
top3TestCTreeTable <- table(top3TestCTree, testData$Subscribed)
top3TestCTreeTable # View Confusion Matrix
# Calculate the accuracy by creating the Confusion Matrix
cmTop3CTree <- confusionMatrix(top3TestCTreeTable)
cmTop3CTree$overall['Accuracy'] # Accuracy: 24.70%

# ------- Create a decision tree J48 Tree called top3J48Tree from top3Formula
top3J48Tree <- J48(top3Formula, data = trainDataLessOut)
plot(top3J48Tree) # Plot top5J48Tree
# Calculate the accuracy by creating the Confusion Matrix for the trainDataLessOut
top3TrainJ48Tree <- predict(top3J48Tree) 
table(top3TrainJ48Tree, trainDataLessOut$Subscribed) # View Confusion Matrix
cmtop3TrainJ48Tree <- confusionMatrix(top3TrainJ48Tree, trainDataLessOut$Subscribed)
cmtop3TrainJ48Tree$overall['Accuracy'] # Accuracy: 95.04%
# Predict a test dataset top3TestJ48Tree with top3CTree model and data testData
top3TestJ48Tree <- predict(top3J48Tree, newdata = testData)
top3J48Table <- table(top3TestJ48Tree, testData$Subscribed)
top3J48Table # View Confusion Matrix
# Calculate the accuracy by creating the Confusion Matrix
cmTop3J48Tree <- confusionMatrix(top3J48Table)
cmTop3J48Tree$overall['Accuracy'] # Accuracy: 24.15%

# ---- Top 5 Attributes

# As shown from the and table and barplot of the Atributes (Weights)

# ------- Top 5 Attributes based on Information Gain Calculations ------- #
# Develop top5FOrmula for the top 5 Information Gains
top5Formula <- Subscribed ~ duration + month + contact + age + job

# ------- Build a decision tree CTree called top5CTree from top5Formula
top5CTree <- ctree(top5Formula, data = trainDataLessOut)
plot(top5CTree) # Plot CTree top5CTree
# Calculate the accuracy by creating the Confusion Matrix for the trainDataLessOut
top5TrainCTree <- predict(top5CTree) 
table(top5TrainCTree, trainDataLessOut$Subscribed) # View Confusion Matrix
cmtop5TrainCTree <- confusionMatrix(top5TrainCTree, trainDataLessOut$Subscribed)
cmtop5TrainCTree$overall['Accuracy']  # Accuracy: 95.99%
# Predict a test dataset top5TestCTree with top5CTree model and data testData
top5TestCTree <- predict(top5CTree, newdata = testData)
top5TestCTreeTable <- table(top5TestCTree, testData$Subscribed)
top5TestCTreeTable # View Confusion Matrix
# Calculate the accuracy by creating the Confusion Matrix
cmTop5CTree <- confusionMatrix(top5TestCTreeTable)
cmTop5CTree$overall['Accuracy'] # Accuracy: 14.37%

# ------- Create a decision tree J48 Tree called top5J48Tree from top5Formula
top5J48Tree <- J48(top5Formula, data = trainDataLessOut)
plot(top5J48Tree) # Plot top5J48Tree
# Calculate the accuracy by creating the Confusion Matrix for the trainDataLessOut
top5TrainJ48Tree <- predict(top5J48Tree) 
table(top5TrainJ48Tree, trainDataLessOut$Subscribed) # View Confusion Matrix
cmtop5TrainJ48Tree <- confusionMatrix(top5TrainJ48Tree, trainDataLessOut$Subscribed)
cmtop5TrainJ48Tree$overall['Accuracy'] # Accuracy: 96.11%
# Predict a test dataset top5TestJ48Tree with top5CTree model and data testData
top5TestJ48Tree <- predict(top5J48Tree, newdata = testData)
top5J48Table <- table(top5TestJ48Tree, testData$Subscribed)
top5J48Table # View Confusion Matrix
# Calculate the accuracy by creating the Confusion Matrix
cmTop5J48Tree <- confusionMatrix(top5J48Table)
cmTop5J48Tree$overall['Accuracy'] # Accuracy: 14.28%

# Top 10 Attributes
# ------- Top 10 Attributes based on Information Gain Calculations ------- #
# Develop top10FOrmula for the Top 10 Information Gains
top10Formula <- Subscribed ~ duration + month + contact + age + job + campaign + marital + education + housing + day_of_week

# ------- Build a decision tree CTree called top10CTree from top10Formula
top10CTree <- ctree(top10Formula, data = trainDataLessOut)
plot(top10CTree) # Plot top10CTree
# Calculate the accuracy by creating the Confusion Matrix for the trainDataLessOut
top10TrainCTree <- predict(top10CTree) 
table(top10TrainCTree, trainDataLessOut$Subscribed) # View Confusion Matrix
cmtop10TrainCTree <- confusionMatrix(top10TrainCTree, trainDataLessOut$Subscribed)
cmtop10TrainCTree$overall['Accuracy'] # Accuracy: 95.97%
# Predict a test dataset top10TestCTree with top10CTree model and data testData
top10TestCTree <- predict(top10CTree, newdata = testData)
top10CTreeTable <- table(top10TestCTree, testData$Subscribed) 
top10CTreeTable # View Confusion Matrix
# Calculate the accuracy by creating the Confusion Matrix
cmTop10CTree <- confusionMatrix(top10CTreeTable)
cmTop10CTree$overall['Accuracy'] # Accuracy: 14.36%

# ------- Create a decision tree J48 Tree called top10J48Tree from top10Formula
top10J48Tree <- J48(top10Formula, data = trainDataLessOut)
plot(top10J48Tree) # Plot top10J48Tree
# Calculate the accuracy by creating the Confusion Matrix for the trainDataLessOut
top10TrainJ48Tree <- predict(top10J48Tree) 
table(top10TrainJ48Tree, trainDataLessOut$Approved) # View Confusion Matrix
cmtop10TrainJ48Tree <- confusionMatrix(top10TrainJ48Tree, trainDataLessOut$Approved)
cmtop10TrainJ48Tree$overall['Accuracy'] # Accuracy: 14.37%
# Predict a test dataset top10TestJ48Tree with top10J48Tree model and data testData
top10TestJ48Tree <- predict(top10J48Tree, newdata = testData)
top10J48Table <- table(top10TestJ48Tree, testData$Approved)
top10J48Table # View Confusion Matrix
# Calculate the accuracy by creating the Confusion Matrix
cmTop10J48Tree <- confusionMatrix(top10J48Table)
cmTop10J48Tree$overall['Accuracy']

# rpart

# Try rpart decision trees with two attributes
decisionTree <- rpart(Subscribed ~ duration + month + contact + age + job + campaign + marital + education + housing, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 23.27%
# Calculate Precision, Recall, and F-Score

# It is reversed for this dataset
# Precision: tp/(tp + fp):
prec <- table_mat[1,1]/sum(table_mat[1,1:2])
prec
# Recall: tp/(tp + fn):
recall <- table_mat[1,1]/sum(table_mat)
recall
# F-Score: 2 * precision * recall /(precision + recall):
f_score <- 2 * prec * recall / (prec + recall)
f_score

decisionTree <- rpart(Subscribed ~ duration + month + contact + age + job + campaign + marital + education, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 23.27%
# Calculate Precision, Recall, and F-Score
# It is reversed for this dataset
# Precision: tn/(tn + fn):
prec <- table_mat[1,1]/sum(table_mat[1,1:2])
prec
# Recall: tn/(tn + fp):
recall <- table_mat[1,1]/sum(table_mat)
recall
# F-Score: 2 * precision * recall /(precision + recall):
f_score <- 2 * prec * recall / (prec + recall)
f_score

decisionTree <- rpart(Subscribed ~ duration + month + contact + age + job + campaign + marital, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 23.27%
# Calculate Precision, Recall, and F-Score
# It is reversed for this dataset
# Precision: tn/(tn + fn):
prec <- table_mat[1,1]/sum(table_mat[1,1:2])
prec
# Recall: tn/(tn + fp):
recall <- table_mat[1,1]/sum(table_mat)
recall
# F-Score: 2 * precision * recall /(precision + recall):
f_score <- 2 * prec * recall / (prec + recall)
f_score

decisionTree2 <- rpart(Subscribed ~ duration + month + contact + age + job + campaign, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree2)
predict_unseen <- predict(decisionTree2, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 23.27%
# Calculate Precision, Recall, and F-Score
# It is reversed for this dataset
# Precision: tn/(tn + fn):
prec <- table_mat[1,1]/sum(table_mat[1,1:2])
prec
# Recall: tn/(tn + fp):
recall <- table_mat[1,1]/sum(table_mat)
recall
# F-Score: 2 * precision * recall /(precision + recall):
f_score <- 2 * prec * recall / (prec + recall)
f_score

decisionTree <- rpart(Subscribed ~ duration + month + contact + age + job, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 23.27%
# Calculate Precision, Recall, and F-Score
# It is reversed for this dataset
# Precision: tn/(tn + fn):
prec <- table_mat[1,1]/sum(table_mat[1,1:2])
prec
# Recall: tn/(tn + fp):
recall <- table_mat[1,1]/sum(table_mat)
recall
# F-Score: 2 * precision * recall /(precision + recall):
f_score <- 2 * prec * recall / (prec + recall)
f_score

decisionTree <- rpart(Subscribed ~ duration + month + contact + age, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 23.27%
# Calculate Precision, Recall, and F-Score
# It is reversed for this dataset
# Precision: tn/(tn + fn):
prec <- table_mat[1,1]/sum(table_mat[1,1:2])
prec
# Recall: tn/(tn + fp):
recall <- table_mat[1,1]/sum(table_mat)
recall
# F-Score: 2 * precision * recall /(precision + recall):
f_score <- 2 * prec * recall / (prec + recall)
f_score

# This shows that job, campaign, marital, and education is not that important

decisionTree <- rpart(Subscribed ~ duration + month + contact, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 23.87%
# Calculate Precision, Recall, and F-Score
# It is reversed for this dataset
# Precision: tn/(tn + fn):
prec <- table_mat[1,1]/sum(table_mat[1,1:2])
prec
# Recall: tn/(tn + fp):
recall <- table_mat[1,1]/sum(table_mat)
recall
# F-Score: 2 * precision * recall /(precision + recall):
f_score <- 2 * prec * recall / (prec + recall)
f_score

# The accuracy seems to significantly decrease once we add contact
# So we will try excluding contact

decisionTree <- rpart(Subscribed ~ duration + month + age, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 65.98%
# Calculate Precision, Recall, and F-Score
# It is reversed for this dataset
# Precision: tn/(tn + fn):
prec <- table_mat[1,1]/sum(table_mat[1,1:2])
prec
# Recall: tn/(tn + fp):
recall <- table_mat[1,1]/sum(table_mat)
recall
# F-Score: 2 * precision * recall /(precision + recall):
f_score <- 2 * prec * recall / (prec + recall)
f_score

# Create Decision Trees with rpart by slowly cutting the attributes
decisionTree <- rpart(Subscribed ~ duration + month + age + job + campaign + marital + education + housing, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 65.98%
# Calculate Precision, Recall, and F-Score
# It is reversed for this dataset
# Precision: tn/(tn + fn):
prec <- table_mat[1,1]/sum(table_mat[1,1:2])
prec
# Recall: tn/(tn + fp):
recall <- table_mat[1,1]/sum(table_mat)
recall
# F-Score: 2 * precision * recall /(precision + recall):
f_score <- 2 * prec * recall / (prec + recall)
f_score

decisionTree <- rpart(Subscribed ~ duration + month + age + job + campaign + marital + education, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 65.98%
# Calculate Precision, Recall, and F-Score
# It is reversed for this dataset
# Precision: tn/(tn + fn):
prec <- table_mat[1,1]/sum(table_mat[1,1:2])
prec
# Recall: tn/(tn + fp):
recall <- table_mat[1,1]/sum(table_mat)
recall
# F-Score: 2 * precision * recall /(precision + recall):
f_score <- 2 * prec * recall / (prec + recall)
f_score

decisionTree <- rpart(Subscribed ~ duration + month + age + job + campaign, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 65.98%
# Calculate Precision, Recall, and F-Score
# It is reversed for this dataset
# Precision: tn/(tn + fn):
prec <- table_mat[1,1]/sum(table_mat[1,1:2])
prec
# Recall: tn/(tn + fp):
recall <- table_mat[1,1]/sum(table_mat)
recall
# F-Score: 2 * precision * recall /(precision + recall):
f_score <- 2 * prec * recall / (prec + recall)
f_score

decisionTree <- rpart(Subscribed ~ duration + month + age + job, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 65.98%
# Calculate Precision, Recall, and F-Score
# It is reversed for this dataset
# Precision: tn/(tn + fn):
prec <- table_mat[1,1]/sum(table_mat[1,1:2])
prec
# Recall: tn/(tn + fp):
recall <- table_mat[1,1]/sum(table_mat)
recall
# F-Score: 2 * precision * recall /(precision + recall):
f_score <- 2 * prec * recall / (prec + recall)
f_score

decisionTree <- rpart(Subscribed ~ duration + month + job, data = trainDataLessOut, method = "class")
par(mar=c(1,1,1,1))
rpart.plot(decisionTree)
predict_unseen <- predict(decisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # Accuracy: 66.74%
# Calculate Precision, Recall, and F-Score
# It is reversed for this dataset
# Precision: tn/(tn + fn):
prec <- table_mat[1,1]/sum(table_mat[1,1:2])
prec
# Recall: tn/(tn + fp):
recall <- table_mat[1,1]/sum(table_mat)
recall
# F-Score: 2 * precision * recall /(precision + recall):
f_score <- 2 * prec * recall / (prec + recall)
f_score

# ----- KNN
# https://rpubs.com/njvijay/16444
# kNN requires variables to be normalized or scaled. caret provides facility to preprocess data
# Choose centring and scaling
trainX <- trainDataLessOut[,names(trainDataLessOut) != "Subscribed"]
preProcValues <- preProcess(x = trainX,method = c("center", "scale"))
preProcValues

set.seed(400)
ctrl <- trainControl(method="repeatedcv",repeats = 3) #,classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit <- train(Subscribed ~ ., data = trainDataLessOut, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)
# There is a bunch of Warnings

# Output of kNN fit
knnFit

knnPredict <- predict(knnFit,newdata = testData)
#Get the confusion matrix to see accuracy value and other parameter values
cmKNN <- confusionMatrix(knnPredict, testData$Subscribed)
cmKNN
cmKNN$overall['Accuracy'] # Accuracy: 28.39%

#With twoclasssummary
ctrl <- trainControl(method="repeatedcv",repeats = 3,classProbs=TRUE,summaryFunction = twoClassSummary)
# Random forrest
rfFit <- train(Subscribed ~ ., data = trainDataLessOut, method = "rf", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)
rfFit
plot(rfFit)
#Trying plot with some more parameters
plot(rfFit, print.thres = 0.5, type="S")
#Get the confusion matrix to see accuracy value and other parameter values
cmKNN <- confusionMatrix(knnPredict, testData$Subscribed)
cmKNN
cmKNN$overall['Accuracy'] # Accuracy: 

# Create another knn model
knnFit2 <- kNN(trainDataLessOut, testData, variable = c("duration"), k = 1)
# Output of kNN fit
knnFit2
# Test the testing dataset
knnPredict2 <- predict(knnFit2,newdata = testData)
#Get the confusion matrix to see accuracy value and other parameter values
cmKNN <- confusionMatrix(knnPredict2, testData$Subscribed)
cmKNN
cmKNN$overall['Accuracy'] # Accuracy: 28.39%

# See the summary of training data before using KNN
summary(trainDataLessOut)
knnTry <- kNN(trainDataLessOut, variable = c("duration", "month"), k = 6)
# See summary of training data after using KNN
summary(trainDataLessOut)

# Create another KNN model
trainDataLessOut2 <- kNN(trainDataLessOut)
summary(trainDataLessOut2)
head(trainDataLessOut2)
trainDataLessOut2 <- subset(trainDataLessOut2, select = duration:age)
head(trainDataLessOut2)


# ----- ADVANCED - NEXT STEPS

# ---------- REGRESSION

# Understand and observe attributes
head(trainData)

ggpairs(data = trainData, columns=1:5, title="trees data")

# If you have time:
# https://cran.r-project.org/web/packages/jtools/vignettes/summ.html
# https://www.datacamp.com/community/tutorials/linear-regression-R
# https://www.dataquest.io/blog/statistical-learning-for-predictive-modeling-r/
# https://rpubs.com/cyobero/regression-tree
# https://tutorials.iq.harvard.edu/R/Rstatistics/Rstatistics.html
# http://r-statistics.co/Linear-Regression.html
# https://www.machinelearningplus.com/machine-learning/complete-introduction-linear-regression-r/

# Site: https://rpubs.com/sidTyson92/329310
# Feature Scaling is needed when different features has different ranges, for example age, duration, campaign, and nr.employed
# They have very different ranges but when we training a model, which is basically trying to fit some line(in linear regression) then the error is trying to be minimized,
# to minimize the error the euclidian distance is minimized using some algorithm(gradient descent )
# But if no feature scaling is applied then the training will be highly biased with the feature having large values because the euclidian distance will be large there.

trainData
summary(trainData[1:1])
trainData[1:1] = scale(trainData[1:1])
trainData
summary(trainData[,10:11])
trainData[,10:11] = scale(trainData[,10:11])
trainData
summary(trainData[,14:14])
trainData[,14:14] = scale(trainData[,14:14])
trainData
summary(trainData) # summarize the transformed trainData dataset
testData
testData[1:1] = scale(testData[1:1])
testData[10:11] = scale(testData[10:11])
testData[14:14] = scale(testData[14:14])
testData
summary(testData) # summarize the transformed testData dataset

# Centre transform calculates the mean for an attribute and subtracts it from each value
# summarize data
summary(trainData[1:1])
summary(trainData[,10:11])
summary(trainData[,14:14])
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(trainData[1:1], method=c("center"))
preprocessParams <- preProcess(trainData[,10:11], method=c("center"))
preprocessParams <- preProcess(trainData[14:14], method=c("center"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, trainData[1:1])
# summarize the transformed dataset
summary(transformed)
transformed2 <- predict(preprocessParams, trainData[1:1,10:11])
summary(transformed2)
transformed3 <- predict(preprocessParams, trainData[14:14])
summary(transformed3)

# Standardize: combining the scale and center transforms will standardize your data. Attributes will have a mean value of 0 and a standard deviation of 1.
# summarize data
summary(trainData[1:1])
summary(trainData[,10:11])
summary(trainData[14:14])
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(trainData[1:1], method=c("center", "scale"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, trainData[1:1])
# summarize the transformed dataset
summary(transformed)

# Normalize: Data values can be scaled into the range of [0, 1] which is called normalization.
# Summarize data
summary(trainData[1:1])
summary(trainData[,10:11])
summary(trainData[14:14])
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(trainData[1:1], method=c("range"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, trainData[1:1])
# summarize the transformed dataset
summary(transformed)

#Box-cox Transform: When an attribute has a Gaussian-like distribution but is shifted, this is called a skew. The distribution of an attribute can be shifted to reduce the skew and make it more Gaussian. The BoxCox transform can perform this operation (assumes all values are positive).

# load libraries
install.packages("mlbench")
library(mlbench)

summary(trainData[1:1])
summary(trainData[,10:11])
summary(trainData[14:14])
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(trainData[1:1], method=c("BoxCox"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, trainData[1:1])
# summarize the transformed dataset (note pedigree and age)
summary(transformed)

# We may need the categorical attributes to be numerical attributes
# Encoding categorical data
preTrainData$month = factor(preTrainData$month,
                         levels = c('apr', 'aug', 'dec', 'jul', 'jun', 'mar', 'may', 'nov', 'oct', 'sep'),
                         labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
View(preTrainData)
View(preTrainData$month) 
# However, this strategy is not good because this shows that certain months have more significance than others

# So, use the One Hot Column Encoding

# Otherwise remove categorical columns
regTrainData = subset(trainData2, select = -c(loan, job, marital, education, housing, contact, month, day_of_week))
View(regTrainData)
# summarize the regTrainData dataset
summary(regTrainData)

# Visualize the relationship between two attributes
# Scatter plot is drawn for each one of them against the response, along with the line of best fit as seen below.

scatter.smooth(x = trainData$month, y=trainData$duration, main="Duration vs Month")  # scatterplot
# Not the best method to find the relationship
# So, we'll do this instead:
table(trainDataLessOut$Subscribed, trainDataLessOut$month)

# https://topepo.github.io/caret/pre-processing.html

# Simple Regression
str(trainData) # Structure of the data
par(mar=c(1,1,1,1))
pairs(trainData) # What's the point of this? Margins are too large
data("trainData") # Can't find trainData

plot(trainData$duration, trainData$day_of_week, ylab = "Duration", xlab = "Day of Week", main = "Duration and Day of Week")

# Create a simple linear regression (SLR)
model1 <- lm(duration ~ day_of_week, data = trainData)
abline(model1)
model1
# Because, we can consider a linear model to be statistically significant only when both these p-Values 
# are less than the pre-determined statistical significance level of 0.094.
# This can visually interpreted by the significance stars at the end of the row against each X variable.
# The more the stars beside the variable’s p-Value, the more significant the variable.

# Under statistics
summary(model1)

# Calculate the accuracy of the regression model model1
modelPred1 <- predict(model1, testData)
actuals_preds <- data.frame(cbind(actuals=testData$duration, predicteds=modelPred1))  # make actuals_predicteds dataframe.
correlation_accuracy <- cor(actuals_preds)  # 82.7%
correlation_accuracy
head(actuals_preds)

View(modelPred1)

confusionMatrix(table(testData$Subscribed, modelPred1))

# Under accuracy of regression models
# https://www.machinelearningplus.com/machine-learning/complete-introduction-linear-regression-r/


# http://www.sthda.com/english/articles/38-regression-model-validation/158-regression-model-accuracy-metrics-r-square-aic-bic-cp-and-more/
AIC(model1)
BIC(model1)

# rsquare(), rmse() and mae() [modelr package], computes, respectively, the R2, RMSE and the MAE.
data.frame(
  R2 = rsquare(model1, data = trainData),
  RMSE = rmse(model1, data = trainData),
  MAE = mae(model1, data = trainData)
)

# R2(), RMSE() and MAE() [caret package], computes, respectively, the R2, RMSE and the MAE.
predictions <- model1 %>% predict(trainData)
data.frame(
  R2 = R2(predictions, trainData$Subscribed),
  RMSE = RMSE(predictions, trainData$Subscribed),
  MAE = MAE(predictions, trainData$Subscribed)
)

# glance() [broom package], computes the R2, adjusted R2, sigma (RSE), AIC, BIC.
glance(model1)

# Make predictions and compute the
# R2, RMSE and MAE
trainData %>%
  add_predictions(model1) %>%
  summarise(
    R2 = cor(Subscribed, pred)^2,
    MSE = mean((Subscribed - pred)^2),
    RMSE = sqrt(MSE),
    MAE = mean(abs(Subscribed - pred))
  )

# we'll use the function glance() to simply compare the overall quality of our two models:
  
# Metrics for model 1
glance(model1) %>%
dplyr::select(adj.r.squared, sigma, AIC, BIC, p.value)

sigma(model1)/mean(trainData$Subscribed)

#now we build a model using two attributes
model2 <- lm(duration ~ day_of_week + age, data = trainData)

#now we build a model using three attribute
model3 <- lm(duration ~ day_of_week + age + campaign, data = trainData)
model3

# Linear Regression

# All Attributes (doesn't seem to work)
logitmod <- lm(Subscribed ~ ., data = trainData)
pred <- predict(logitmod, newdata = testData, type = "response")
View(pred)

##the first value is your condition, value if condition is true, value if conditon is false
y_predicted <- ifelse(pred > 0.5, 1, 0)
View(y_predicted)

y_actual <- testData$Subscribed
View(y_actual)

##compares the prediction to the actual, this is the percentage of percision ie 94% accurate
mean(y_predicted == y_actual)

##this is the true positives and true negatives ... from class, used to evaluate the model
##in table function the rowas is the first value and the second value is columns
confusionMatrix(table(y_predicted, y_actual))
pred <- prediction(y_predicted, y_actual)
RP.perf <- performance(pred, "prec", "rec")
plot(RP.perf)
ROC.perf <- performance(pred, "tpr", "fpr")
plot(ROC.perf)
auc.tmp <- performance(pred, "auc")
auc <- as.numeric(auc.tmp@y.values)
View(auc)

# Using Only Numerical Attributes

logitmod <- lm(Subscribed ~ duration + day_of_week, data = trainData)
pred <- predict(logitmod, newdata = testData, type = "response")
View(pred)

plot(pred)

##the first value is your condition, value if condition is true, value if conditon is false
y_predicted <- ifelse(pred > 0.97, 1, 0)
View(y_predicted)

y_actual <- testData$Subscribed
View(y_actual)

##compares the prediction to the actual, this is the percentage of percision ie 94% accurate
mean(y_predicted == y_actual)

##this is the true positives and true negatives ... from class, used to evaluate the model
##in table function the rowas is the first value and the second value is columns
confusionMatrix(table(y_predicted, y_actual))
pred <- prediction(y_predicted, y_actual)
RP.perf <- performance(pred, "prec", "rec")
plot(RP.perf)
ROC.perf <- performance(pred, "tpr", "fpr")
plot(ROC.perf)
auc.tmp <- performance(pred, "auc")
auc <- as.numeric(auc.tmp@y.values)
View(auc)


# ----- INFORMATION GAIN
# https://medium.com/@rishabhjain_22692/decision-trees-it-begins-here-93ff54ef134

# ----- GINI INDEX

gini_process <-function(classes,splitvar = NULL){
  #Assumes Splitvar is a logical vector
  if (is.null(splitvar)){
    base_prob <-table(classes)/length(classes)
    return(1-sum(base_prob**2))
  }
  base_prob <-table(splitvar)/length(splitvar)
  crosstab <- table(classes,splitvar)
  crossprob <- prop.table(crosstab,2)
  No_Node_Gini <- 1-sum(crossprob[,1]**2)
  Yes_Node_Gini <- 1-sum(crossprob[,2]**2)
  return(sum(base_prob * c(No_Node_Gini,Yes_Node_Gini)))
}

gini_process(trainData$Subscribed) 
gini_process(trainData$duration) 
gini_process(trainData$age) 
gini_process(trainData$month) 
gini_process(trainData$contact)
gini_process(trainData$loan) 
gini_process(trainData$nr.employed) 
gini_process(trainData$job) 
gini_process(trainData$campaign) 
gini_process(trainData$marital)
gini_process(trainData$education)
gini_process(trainData$housing)
gini_process(trainData$day_of_week) 
gini_process(trainData$Subscribed,trainData$duration.Length<27178) #0.1946733 # Figure out what the Length is supposed to be
gini_process(trainData$Subscribed,trainData$age<61)
gini_process(trainData$Subscribed,trainData$campaign<16)
gini_process(trainData$Subscribed,trainData$campaign<43)
gini_process(trainData$Subscribed,trainData$duration<10)
# We can instantly spot that month = apr, dec, mar, sep are 'yes'; while others are mized
# We can also see that age = 61-88 is 'yes'; while the rest is mixed
# We can also see that campaign (days since last contact) = 16, 18, 19, 20, 21, 22, 24-43 are 'no'; while the rest is mixed


# Test with iris
data(iris)
gini_process(iris$Species)
iris$Petals.Length
iris$Sepal.Length
gini_process(iris$Petals)
gini_process(iris$Species, iris$Petals)
gini_process(iris$Species, iris$Petals.Length < 2.45)
gini_process(iris$Species, iris$Petals.Length < 5)
gini_process(iris$Species, iris$Sepal)

# ------ PRUNING FOR EACH MODEL
# Simplify the tree based on a cp identified from the graph or printed output threshold
newDecisionTree = prune(decisionTree, cp = 0.5)
# The classification tree can be visualised with the plot function and then the text function adds labels to the graph:
par(mar=c(1,1,1,1))
rpart.plot(newDecisionTree, uniform = TRUE)
# Predict target classes
predict_unseen <- predict(newDecisionTree, testData, type = 'class')
# Calculate accuracy
table_mat <- table(testData$Subscribed, predict_unseen)
table_mat
cmRPart1 <- confusionMatrix(table_mat)
cmRPart1$overall['Accuracy'] # New Accuracy:

# Information Gain Validation

info_process <-function(classes,splitvar = NULL){
  #Assumes Splitvar is a logical vector
  if (is.null(splitvar)){
    base_prob <-table(classes)/length(classes)
    return(-sum(base_prob*log(base_prob,2)))
  }
  base_prob <-table(splitvar)/length(splitvar)
  crosstab <- table(classes,splitvar)
  crossprob <- prop.table(crosstab,2)
  No_Col <- crossprob[crossprob[,1]>0,1]
  Yes_Col <- crossprob[crossprob[,2]>0,2]
  No_Node_Info <- -sum(No_Col*log(No_Col,2))
  Yes_Node_Info <- -sum(Yes_Col*log(Yes_Col,2))
  return(sum(base_prob * c(No_Node_Info,Yes_Node_Info)))
}

data(newTrainData) #####ERROR?????
info_process(newTrainData$loan) 
info_process(newTrainData$loan,newTrainData$Petal.Length<2.45) 

# Convert numerical to categorical attributes(e.g. campaign as a category for classification)

# Replace "unknown" with mean? - only if numerical data is normalized

# For regression, we can convert categorical to numerical

# Try Correlation Analysis (More Complex)
cor(trainData$duration, trainData$age) # -0.02 Weak correlation (close to 0)
cor(preTrainData$duration, preTrainData$month) #
cor(trainData$duration, trainData$campaign) # -0.07 Closer correlation
cor(trainData$campaign, trainData$age)
cor(preTrainData$month, preTrainData$age)
cor(trainData$month, trainData$campaign)