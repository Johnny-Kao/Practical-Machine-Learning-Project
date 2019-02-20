# Executive Summary
### Background  
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible 
to collect a large amount of data about personal activity relatively 
inexpensively. These type of devices are part of the quantified self 
movement - a group of enthusiasts who take measurements about themselves 
regularly to improve their health, to find patterns in their behavior, or 
because they are tech geeks. One thing that people regularly do is quantify 
how much of a particular activity they do, but they rarely quantify how well 
they do it. The goal of this project is to use data from accelerometers 
on the belt, forearm, arm, and dumbbell of 6 participants as they 
perform barbell lifts correctly and incorrectly 5 different ways. 

### Data  

The training data for this project are available at: 

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available at: 

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

### Goal

The goal of this project is to predict the manner in which subjects did 
the exercise. This is the "classe" variable in the training set. The model will
use the other variables to predict with. This report describes:  
* How the model is built  
* Use of cross validation  
* An estimate of expected out of sample error  

# Getting and cleaning the Data
The first step is to download the data, load it into R and prepare it for 
the modeling process.  

### Load the functions and static variables
All functions are loaded and static variables are assigned.  Also in this 
section, the seed is set so the pseudo-random number generator operates in a 
consistent way for repeat-ability.  

```{r warning=FALSE, message=FALSE, echo=TRUE}
library(rpart.plot)
library(caret)
library(rpart)
library(rattle)
library(RColorBrewer)
library(randomForest)
library(e1071)

set.seed(1)

# You should download required files first

path <- paste(getwd(),"/","data", sep="")
train.file <- file.path(path, "pml-training.csv")
test.file <- file.path(path, "pml-testing.csv")
```

### Clean Data
Process missing data (i.e., "NA", "#DIV/0!" and ""), and set them all to NA.

```{r}
train.data.raw <- read.csv(train.file, na.strings=c("NA","#DIV/0!",""))
test.data.raw <- read.csv(test.file, na.strings=c("NA","#DIV/0!",""))
```

### Remove unecessary colums
Columns that are not deeded for the model and columns that contain NAs 
are eliminated.  

```{r}
# Drop the first 7 columns as they're unnecessary for predicting.
train.data.clean1 <- train.data.raw[,8:length(colnames(train.data.raw))]
test.data.clean1 <- test.data.raw[,8:length(colnames(test.data.raw))]

# Drop colums with NAs
train.data.clean1 <- train.data.clean1[, colSums(is.na(train.data.clean1)) == 0] 
test.data.clean1 <- test.data.clean1[, colSums(is.na(test.data.clean1)) == 0] 

# Check for near zero variance predictors and drop them if necessary
nzv <- nearZeroVar(train.data.clean1,saveMetrics=TRUE)
zero.var.ind <- sum(nzv$nzv)

if ((zero.var.ind>0)) {
        train.data.clean1 <- train.data.clean1[,nzv$nzv==FALSE]
}

```

### Slice the data for cross validation  
The training data is divided into two sets.  This first is a training set with 70% of the data which is used to train the model.  The second is a validation 
set used to assess model performance.  

```{r}
in.training <- createDataPartition(train.data.clean1$classe, p=0.70, list=F)
train.data.final <- train.data.clean1[in.training, ]
validate.data.final <- train.data.clean1[-in.training, ]
```

# Model Development  
### Train the model  
The training data-set is used to fit a Random Forest model because it 
automatically selects important variables and is robust to correlated 
covariates & outliers in general. 5-fold cross validation is used when 
applying the algorithm. A Random Forest algorithm is a way of averaging 
multiple deep decision trees, trained on different parts of the same data-set,
with the goal of reducing the variance. This typically produces better 
performance at the expense of bias and interpret-ability. The Cross-validation 
technique assesses how the results of a statistical analysis will generalize 
to an independent data set. In 5-fold cross-validation, the original sample 
is randomly partitioned into 5 equal sized sub-samples. a single sample 
is retained for validation and the other sub-samples are used as training 
data. The process is repeated 5 times and the results from the folds are 
averaged.

```{r cache=TRUE}
control.parms <- trainControl(method="cv", 5)
rf.model <- train(classe ~ ., data=train.data.final, method="rf",
                 trControl=control.parms, ntree=251)
rf.model
```

### Estimate performance  
The model fit using the training data is tested against the validation data.
Predicted values for the validation data are then compared to the actual 
values. This allows forecasting the accuracy and overall out-of-sample error,
which indicate how well the model will perform with other data.  

```{r}
rf.predict <- predict(rf.model, validate.data.final)
confusionMatrix(validate.data.final$classe, rf.predict)

accuracy <- postResample(rf.predict, validate.data.final$classe)
acc.out <- accuracy[1]

overall.ose <- 
        1 - as.numeric(confusionMatrix(validate.data.final$classe, rf.predict)
                       $overall[1])
```

### Results  
The accuracy of this model is **`r acc.out`** and the Overall Out-of-Sample 
error is **`r overall.ose`**.

# Run the model
The model is applied to the test data to produce the results.

```{r}
results <- predict(rf.model, 
                   test.data.clean1[, -length(names(test.data.clean1))])
results
```

# Appendix - Decision Tree Visualization

```{r warning=FALSE}
treeModel <- rpart(classe ~ ., data=train.data.final, method="class")
fancyRpartPlot(treeModel)
```

