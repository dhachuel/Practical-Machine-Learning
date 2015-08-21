##
## Load packages
##
  library(caret)
  library(rpart)
  library(rpart.plot)
  library(RColorBrewer)
  library(rattle)
  library(randomForest)
  library(reshape2)
  library(ROCR)
  set.seed(1)
  setwd("I:/CRMPO/DEPT/Hachuel/Projects & Requests/Practical Machine Learning")
  

##
## HELPER FUNCTION
##
  pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
      filename = paste0("problem_id_",i,".txt")
      write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
  }

##
## LOADING DATA
##
  training <- read.table(
    file = "pml-training.csv",
    header = TRUE,
    na.strings=c("NA","#DIV/0!",""),
    sep = ","
  )
  testing <- read.table(
    file = "pml-testing.csv",
    header = TRUE,
    na.strings=c("NA","#DIV/0!",""),
    sep = ","
  )


##
## DATA CLEANING
##
  # Remove NA fields equally from both training and test data sets
  training.clean <- training[,colSums(is.na(training)) == 0]
  testing.clean <- testing[,colSums(is.na(testing)) == 0]
  
  # Remove unrelated fields such as timestamp or usernames (fields 1 through 7)
  training.clean <- training.clean[,-c(1:7)]
  testing.clean <- testing.clean[,-c(1:7)]


##
## EXPLORATORY FEATURE ANALYSIS
##
  # Select numeric fields
  training.clean2 <- training.clean
  training.clean2$classe <- as.numeric(training.clean2$classe)
  numeric_cols <- sapply(training.clean2, is.numeric)
  
  # Melt dataset using reshape
  training.clean.lng <- melt(training.clean2[,numeric_cols], id="classe")
  
  # Plot the distribution for each classe in a given variable
  p <- ggplot(aes(x=value, group=classe, colour=factor(classe)), data=training.clean.lng)
  p + geom_density() +
    facet_wrap(~variable, scales="free") +
    ggtitle("Exploratory Feature Distribution Plot")
  rm(training.clean2)
  rm(training.clean.lng)


##
## CALIBRATING RANDOM FORESTS
##
  # Splitting the training set for prior testing (60% train/40% test)
  idx <- runif(nrow(training.clean)) > 0.6
  train <- training.clean[idx==FALSE,]
  test <- training.clean[idx==TRUE,]

  # Growing Forest
  tune.rf <- tuneRF(x = train[,-53],y = train[,53], stepFactor=0.5, mtryStart = 2, ntreeTry = 100)
    # Optimal mtry seems to be 16
  
  rf.fit <- randomForest(factor(classe)~.,data=train, mtry=16, ntree=500, na.action=na.omit, do.trace=5)
  plot(rf.fit) # plotting OOB error rate for each class and global
  print(rf.fit$confusion) # print OOB confusion matrix
  varImpPlot(rf.fit)

  # Test Model
  rf.predict = predict(rf.fit,newdata=test)
  confusionMatrix(rf.predict, test$classe)


##
## GENERATING MODEL
##
  # Train Random Forest on the entire training set
  rf.fit.final <- randomForest(factor(classe)~.,data=training.clean, mtry=16, ntree=500, na.action=na.omit, do.trace = 20)
  # plot(rf.fit.final)
  
  # Generate test cases and corresponding files
  predictions <- predict(rf.fit.final, newdata=testing.clean)
  answers <- as.vector(as.character(as.data.frame(predictions)$predictions))
  pml_write_files(answers)


