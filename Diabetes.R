str(diabetes)
summary(diabetes)

library(tidyverse)

db <- diabetes

#Data manipulation

non_zero_median <- median(db$Insulin[db$Insulin != 0], na.rm = TRUE)
db$Insulin[db$Insulin == 0] <- non_zero_median

#Bivariate data exploration

vars <- colnames(db)[-9]

for (i in vars) {
  plot <- ggplot(db, aes(db[,i], fill=factor(Outcome))) +
    geom_bar(position = 'fill') +
    labs(x=i, y='Proportion of diabetic patient') +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
  print(plot)
}

db %>%
  ggplot(aes(x=factor(Outcome), y=Age)) +
  geom_violin(fill='red')

#Data partition

library(caret)
set.seed(123)
partition <- createDataPartition(y=as.factor(db$Outcome), p=0.7, list = FALSE)
db.train <- db[partition,]
db.test <- db[-partition,]

mean(db.train$Outcome)
mean(db.test$Outcome)
summary(db.train)

################################################################################

#RandomForest

ctrl <- trainControl(method = 'repeatedcv', number = 5, repeats = 3,
                     sampling = 'down')

rf.grid <- expand.grid(mtry=1:8)

install.packages('randomForest')

target <- factor(db.train$Outcome)
predictors <- db.train[,-9]

set.seed(1234)
rf1 <- train(y=target, x=predictors, method='rf', ntree=100, importance=TRUE,
             trControl=ctrl, tuneGrid=rf.grid)
rf2 <- train(y=target, x=predictors, method='rf', ntree=300, importance=TRUE,
             trControl=ctrl, tuneGrid=rf.grid)
rf3 <- train(y=target, x=predictors, method='rf', ntree=500, importance=TRUE,
             trControl=ctrl, tuneGrid=rf.grid)

#Confusion Matrices

pred.rf1.class <- predict(rf1, newdata = db.test, type = 'raw')
pred.rf2.class <- predict(rf2, newdata = db.test, type = 'raw')
pred.rf3.class <- predict(rf3, newdata = db.test, type = 'raw')

confusionMatrix(pred.rf1.class, as.factor(db.test$Outcome), positive = '1')
confusionMatrix(pred.rf2.class, as.factor(db.test$Outcome), positive = '1')
confusionMatrix(pred.rf3.class, as.factor(db.test$Outcome), positive = '1')

#ROC & AUC

pred.rf1.prob <- predict(rf1, newdata = db.test, type = 'prob')[,2]
pred.rf2.prob <- predict(rf2, newdata = db.test, type = 'prob')[,2]
pred.rf3.prob <- predict(rf3, newdata = db.test, type = 'prob')[,2]

library(pROC)

roc(db.test$Outcome, pred.rf1.prob)
#Call:
  #roc.default(response = db.test$Outcome, predictor = pred.rf1.prob)

#Data: pred.rf1.prob in 150 controls (db.test$Outcome 0) < 80 cases (db.test$Outcome 1).
#Area under the curve: 0.8243
roc(db.test$Outcome, pred.rf2.prob)
#Call:
  #roc.default(response = db.test$Outcome, predictor = pred.rf2.prob)

#Data: pred.rf2.prob in 150 controls (db.test$Outcome 0) < 80 cases (db.test$Outcome 1).
#Area under the curve: 0.8364
roc(db.test$Outcome, pred.rf3.prob)
#Call:
  #roc.default(response = db.test$Outcome, predictor = pred.rf3.prob)

#Data: pred.rf3.prob in 150 controls (db.test$Outcome 0) < 80 cases (db.test$Outcome 1).
#Area under the curve: 0.8126

imp <- varImp(rf1)
imp
plot(imp)

#rf1 variable importance

#Importance
#Glucose                      100.00
#Age                           47.53
#BMI                           36.17
#SkinThickness                 22.85
#DiabetesPedigreeFunction      20.69
#Insulin                       17.05
#Pregnancies                   13.39
#BloodPressure                  0.00

#ROC curve for rf2

par(pty='s')
plot(roc(db.test$Outcome, pred.rf1.prob))

################################################################################

#XGBoost

xgb.grid <- expand.grid(max_depth=7, min_child_weight=1, gamma=0,
                        nrounds=c(50, 100, 150, 200, 250, 300, 500),
                        eta=c(0.001, 0.002, 0.01, 0.02, 0.1),
                        colsample_bytree=0.6, subsample=0.6)
xgb.grid

xgb.ctrl <- trainControl(method = 'cv', number = 5, sampling = 'down')
install.packages('xgboost')

set.seed(43)
xgb.tuned <- train(as.factor(Outcome) ~ ., data = db.train, method='xgbTree',
                   trControl=xgb.ctrl, tuneGrid=xgb.grid)

xgb.tuned

#Tuning parameter 'max_depth' was held constant at a value of 7
#Tuning parameter 'gamma' was of 0.6
#Tuning parameter 'min_child_weight' was held constant at a value of 1
#Tuning parameter 'subsample' was held constant at a value of 0.6
#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were nrounds = 200, max_depth = 7, eta = 0.02, 
#gamma = 0, colsample_bytree = 0.6, min_child_weight = 1 and subsample = 0.6.

ggplot(xgb.tuned)

pred.xgb.class <- predict(xgb.tuned, newdata = db.test, type = 'raw')
confusionMatrix(pred.xgb.class, as.factor(db.test$Outcome), positive = '1')

pred.xgb.prob <- predict(xgb.tuned, newdata = db.test, type = 'prob')[,2]
roc(as.numeric(db.test$Outcome), pred.xgb.prob)
roc(db.test$Outcome, pred.xgb.prob)

plot(varImp(xgb.tuned))
varImp(xgb.tuned)

#xgbTree variable importance

#Overall
#Glucose                  100.000
#BMI                       72.729
#Age                       59.060
#DiabetesPedigreeFunction  56.674
#BloodPressure             11.508
#SkinThickness              7.441
#Insulin                    7.123
#Pregnancies                0.000

par(pty='s')
plot(roc(db.test$Outcome, pred.xgb.prob))


################################################################################

#Comparison

#Model                Test Accuracy       Test AUC

#Random Forest 1          0.7435            0.8364
#Boosted Tree             0.7304            0.8153

################################################################################

#Source:https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database 

################################################################################

#test

db %>%
  ggplot(aes(y=Glucose, x=factor(Outcome))) +
  geom_boxplot()

################################################################################

db %>%
  ggplot(aes(Glucose)) +
  geom_histogram()

db %>%
  ggplot(aes(Age)) +
  geom_histogram()

