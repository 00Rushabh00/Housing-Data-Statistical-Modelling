## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(fig.pos = "H", out.extra = "")
library(ISLR)
library(ggplot2)
library(GGally)
library(car)
library(gridExtra)
library(MASS)
library(Hmisc)
library(faraway)
library(mice)
library(caret)
library(gbm)
library(tree)
library(e1071)
library(glmnet)
library(randomForest)
library(visdat)
library(dlookr)
library(knitr)
library(kableExtra)
library(rpart)
library(rpart.plot)


setwd("D:\\University\\STAT3040\\FinalProject")

train <- read.csv("train.csv", header = TRUE)
test <- read.csv("test.csv", header = TRUE)



## ---- echo = FALSE------------------------------------------------------------
# Specify categorical variables

train_categorical_variables <- c("id", "school", "hoa", "hvac", "view", "year")
test_categorical_variables <- c("id", "hoa", "hvac", "view", "year")

train[, train_categorical_variables] <- lapply(train[, train_categorical_variables], factor, ordered = FALSE)
train$year <- factor(train$year, ordered = TRUE)

test[, test_categorical_variables] <- lapply(test[, test_categorical_variables], factor, ordered = FALSE)
test$year <- factor(test$year, ordered = TRUE)



## ---- echo=FALSE--------------------------------------------------------------

knitr::kable(t(colSums(is.na(train))), align = "c", caption = "NA Table", format = "latex")


## ---- echo = FALSE------------------------------------------------------------

knitr::kable(summary(train[, c("price", "lat", "lon", "rate", "garage")]), align = "c", caption = " Summary Table", format = "latex")

knitr::kable(summary(train[, c("bath", "bed", "stories", "lot", "living")]), align = "c", caption = " Summary Table", format = "latex")



## ---- echo = FALSE------------------------------------------------------------

vis_miss(train)


## ---- echo=FALSE, warning=FALSE-----------------------------------------------

b1 <- ggplot(train, aes(x = school, y = price)) +
  geom_boxplot(fill = c("tomato","turquoise"), alpha = 0.5) + 
  theme_minimal()

b2 <- ggplot(train, aes(x = school, y = lat)) +
  geom_boxplot(fill = c("tomato","turquoise"), alpha = 0.5) + 
  theme_minimal()

b3 <- ggplot(train, aes(x = school,  y = lon)) +
  geom_boxplot(fill = c("tomato","turquoise"), alpha = 0.5) + 
  theme_minimal()
 
b4 <- ggplot(train, aes(x = school, y = rate)) +
  geom_boxplot(fill = c("tomato","turquoise"), alpha = 0.5) + 
  theme_minimal()

b5 <- ggplot(train, aes(x = school, y = garage)) +
  geom_boxplot(fill = c("tomato","turquoise"), alpha = 0.5) + 
  theme_minimal()

b6 <- ggplot(train, aes(x = school, y = bath)) +
  geom_boxplot(fill = c("tomato","turquoise"), alpha = 0.5) + 
  theme_minimal()

b7 <- ggplot(train, aes(x = school, y = bed)) +
  geom_boxplot(fill = c("tomato","turquoise"), alpha = 0.5) + 
  theme_minimal()

b8 <- ggplot(train, aes(x = school, y = stories)) +
  geom_boxplot(fill = c("tomato","turquoise"), alpha = 0.5) + 
  theme_minimal()

b9 <- ggplot(train, aes(x = school, y = lot)) +
  geom_boxplot(fill = c("tomato","turquoise"), alpha = 0.5) + 
  theme_minimal()

b10 <- ggplot(train, aes(x = school, y = living)) +
  geom_boxplot(fill = c("tomato","turquoise"), alpha = 0.5)+ 
  theme_minimal() 

grid.arrange(b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, ncol = 5, nrow = 2)


## ---- echo =FALSE, warning = FALSE--------------------------------------------

ggpairs(train[, c('price', 'lat', 'lon', 'rate', 'garage', 'bath', 'bed', 'stories', 'lot', 'living')]) + theme_minimal()



## ---- echo = FALSE------------------------------------------------------------

knitr::kable(normality(train), align = "c", caption = "Normality Summary", format = "latex")


## ---- echo =FALSE, include= FALSE---------------------------------------------
old.train <- train

id <- train$id
train <- subset(train, select = -id)
temp <- mice(train, m = 1, maxit=25, seed=580)

data <- complete(temp)

data <- cbind(id, data)

id <- test$id
test <- subset(test, select = -id)
temp2 <- mice(test, m = 1, maxit= 25, seed=500)

data2 <- complete(temp2)

test <- cbind(id, data2)

set.seed(153) 
samp <- sample(c(TRUE, FALSE), nrow(data), replace=TRUE, prob=c(0.8,0.2))

train  <- data[samp, ]
validation   <- data[!samp, ]


## ---- echo = FALSE, warning = FALSE, message=FALSE----------------------------

d1 <- ggplot(data, aes(x = lat)) +
  geom_histogram(aes(fill = "Imputed Data"), color = "black", alpha = 0.5) +
  geom_histogram(data = old.train, aes(x = lat, fill = "Original Data"), color = "black", alpha = 0.5) + labs(title = "Density Plot", x = "lat") + theme_minimal()
  
d2 <- ggplot(data, aes(x = lon)) +
  geom_histogram(aes(fill = "Imputed Data"), color = "black", alpha = 0.5) +
  geom_histogram(data = old.train, aes(x = lon, fill = "Original Data"), color = "black", alpha = 0.5) + labs(title = "Density Plot", x = "lon") + theme_minimal()

grid.arrange(d1, d2, ncol = 2, nrow = 1)



## ---- echo = FALSE, include=FALSE---------------------------------------------

stepAIC(lm(price ~ lat + lon + rate + hoa + hvac + garage + view + year + bath + bed + stories + lot + living, data = train))


## ---- echo = FALSE------------------------------------------------------------

k <- 10
mse_values <- numeric(k)
set.seed(13)
folds <- createFolds(train$price, k = k)

for (i in 1:k) {
  train_fold <- train[-folds[[i]], ]
  validation_fold <- train[folds[[i]], ]
  
  lm_price <- lm(formula = price ~ lon + rate + hoa + garage + 
                   year + bath + bed + stories + living,
                   data = train_fold)
  
  predictions_price_lm <- predict(lm_price, newdata = validation_fold)
  
  mse_values[i] <- mean((predictions_price_lm - validation_fold$price)^2)
}


lm_price_mse <- mean(mse_values)

predictions_price_lm <- predict(lm_price, newdata = validation)
lm_price_mse_valid <- mean((predictions_price_lm - validation$price)^2)

row1 <- data.frame("Multiple Linear Regression MSE (K-fold)", lm_price_mse)
row2 <- data.frame("Multiple Linear Regression MSE (Validation)", lm_price_mse_valid)
colnames(row1) <- c("Metric", "Value")
colnames(row2) <- c("Metric", "Value")
out_table <- rbind(row1,row2)

knitr::kable(out_table, align = "c", caption = "MSE Table", format = "latex")


## ---- echo =FALSE-------------------------------------------------------------
lm_price <- lm(formula = price ~ lon + rate + hoa + garage + 
                   year + bath + bed + stories + living,
                   data = train)

summary_table <- summary(lm_price)

summary_df <- as.data.frame(summary_table$coefficients)

knitr::kable(summary_df, align = "c", caption = "Linear Regression Summary", format = "latex")



## ---- echo=FALSE--------------------------------------------------------------

x <- model.matrix(price ~ . - id - school, data = train)

y <- train$price

ridge_price <- glmnet(x, y, alpha = 0)

cv_ridge_price <-cv.glmnet(x, y, alpha = 0)



## ---- echo = FALSE------------------------------------------------------------

par(mfrow = c(1,2))
plot(ridge_price, xvar = "lambda", label = TRUE)
plot(cv_ridge_price)
par(mfrow = c(1,1))


## ---- echo = FALSE------------------------------------------------------------

best.lam <- cv_ridge_price$lambda.min

predictions_price_ridge_k <- predict(ridge_price, newx = x, s=best.lam)

predictions_price_ridge_valid <- predict(ridge_price, s=best.lam, newx = model.matrix(price ~ . - id - school, data = validation))

mse_ridge_price <- mean((predictions_price_ridge_valid - validation$price)^2)
mse_ridge_price_k <- mean((predictions_price_ridge_k - y)^2)

row1 <- data.frame("Ridge Regression MSE (K-fold)", mse_ridge_price_k)
row2 <- data.frame("Ridge Regression MSE (Validation)", mse_ridge_price)
colnames(row1) <- c("Metric", "Value")
colnames(row2) <- c("Metric", "Value")
out_table <- rbind(row1,row2)

knitr::kable(out_table, align = "c", caption = "MSE Table", format = "latex")


## ---- echo = FALSE------------------------------------------------------------

tree_price <- tree(price ~ ., data = subset(train, select = -c(id, school)))

plot(tree_price)
text(tree_price, pretty = 0)



## ---- echo = FALSE------------------------------------------------------------

set.seed(125)
cv_tree_price <- cv.tree(tree_price, K = 10)
cv_data <- data.frame(size = cv_tree_price$size, dev = cv_tree_price$dev)

ggplot(data = cv_data, aes(x = size, y = dev)) +
  geom_line() +
  geom_point(shape = 21, fill = "tomato", size = 2) +
  labs(x = "Tree Size", y = "Deviance", title = "Cross-Validation Performance") +
  theme_minimal()


## ---- echo = FALSE------------------------------------------------------------
ctrl <- tree.control(nobs = nrow(train), minsize = 5)
tree_price <- tree(price ~ ., data = subset(train, select = -c(id, school)), control = ctrl)

k <- 10
mse_values <- numeric(k)
set.seed(13)
folds <- createFolds(train$price, k = k)

for (i in 1:k) {
  train_fold <- train[-folds[[i]], ]
  validation_fold <- train[folds[[i]], ]
  ctrl_k <- tree.control(nobs = nrow(train_fold), minsize = 5)
  tree_mod <- tree(price ~ . - id -school, data = train_fold, control = ctrl_k)
  
  predictions_price_tree_k  <- predict(tree_mod, newdata = validation_fold)
  
  mse_values[i] <- mean(( predictions_price_tree_k - validation_fold$price)^2)
}

predictions_price_tree <- predict(tree_price, newdata = validation)

mse_tree_price_valid <- mean((predictions_price_tree - validation$price)^2)

mse_tree_price_k <- mean(mse_values)

row1 <- data.frame("Decision Tree MSE (K-fold)", mse_tree_price_k)
row2 <- data.frame("Decision Tree MSE (Validation)", mse_tree_price_valid)
colnames(row1) <- c("Metric", "Value")
colnames(row2) <- c("Metric", "Value")
out_table <- rbind(row1,row2)

knitr::kable(out_table, align = "c", caption = "MSE Table", format = "html")



## ---- echo = FALSE------------------------------------------------------------

set.seed(2)
rf_price <- randomForest(price ~ lat + lon + rate + hoa + hvac + garage + view + year + bath + bed + stories + lot + living , data = train, ntree = 100, mtry = 4)

rf_price


## ---- echo = FALSE------------------------------------------------------------

importance_df <- as.data.frame(importance(rf_price))
names <- row.names(importance_df)
importance_df <- tibble::rownames_to_column(importance_df, "variable")
importance_df$variable <- names

importance_df$IncNodePurity <- as.numeric(importance_df$IncNodePurity)

ggplot(importance_df, aes(x = reorder(variable, IncNodePurity), y = IncNodePurity)) +
  geom_bar(stat = "identity", fill = "lightgreen") +
  coord_flip() +
  theme_minimal() +
  xlab("Variables") +
  ylab("IncNodePurity") +
  ggtitle("Variable Importance Plot")



## -----------------------------------------------------------------------------

fitControl <- trainControl(method = "cv", number = 10)

tuneGrid <- expand.grid(mtry = c(4,5,6,7))

rf_model <- caret::train(price ~ . -id,
                         data = train,
                         method = "rf",
                         trControl = fitControl,
                         tuneGrid = tuneGrid,
                         importance = TRUE,
                         verbose = FALSE)


knitr::kable(rf_model$results, align = "c", caption = "Metric Table", format = "latex")


## ---- echo=FALSE--------------------------------------------------------------

oob.err <- double(10)
test.err <- double(10)
n <- seq(100, 1000, by = 100)
for (i in 1:length(n)) {
    set.seed(2)
    fit = randomForest(price ~ lat + lon + rate + hoa + hvac + garage + view + year + bath + bed + stories + lot + living, data = train, mtry = 8, 
        ntree = n[[i]]) 
    oob.err[i] = fit$mse[n[[i]]]
    pred = predict(fit, data[!samp,])
    test.err[i] = with(data[!samp,], mean((pred - price)^2))
}

df <- data.frame(ntree = n, test.err = test.err, oob.err = oob.err)

ggplot(df) +
  geom_point(aes(x = ntree, y = test.err, color = "Validation Error"), shape = 19) +
  geom_line(aes(x = ntree, y = test.err, color = "Validation Error")) +
  geom_point(aes(x = ntree, y = oob.err, color = "OOB Error"), shape = 19) +
  geom_line(aes(x = ntree, y = oob.err, color = "OOB Error")) +
  labs(y = "Mean Squared Error", color = "Error Type") +
  scale_color_manual(values = c("Validation Error" = "blue", "OOB Error" = "red"))



## ---- echo = FALSE------------------------------------------------------------

fitControl <- trainControl(method = "cv", number = 10)

tuneGrid <- expand.grid(mtry = 8)

rf_model <- caret::train(price ~ . -id,
                         data = train,
                         method = "rf",
                         trControl = fitControl,
                         tuneGrid = tuneGrid,
                         importance = TRUE,
                         ntree = 1000,
                         verbose = FALSE)

rmse <- rf_model$results
rmse <- min(rmse$RMSE)
mse_rf_price_k <- rmse^2
mse_rf_price_valid <- min(test.err)

row1 <- data.frame("Random Forest MSE (K-fold)", mse_rf_price_k)
row2 <- data.frame("Random Forest MSE (Validation)", mse_rf_price_valid)
colnames(row1) <- c("Metric", "Value")
colnames(row2) <- c("Metric", "Value")
out_table <- rbind(row1,row2)

knitr::kable(out_table, align = "c", caption = "MSE Table", format = "html")

rf_price <- randomForest(price ~ lat + lon + rate + hoa + hvac + garage + view + year + bath + bed + stories + lot + living , data = train, ntree = 1000, mtry = 8)



## ---- echo = FALSE------------------------------------------------------------

set.seed(249)
boost_price <- gbm(price ~ lat + lon + rate + hoa + hvac + garage + view +
                  year + bath + bed + stories + lot + living, data =train,
                  distribution = 'gaussian', n.trees = 5000,
                  interaction.depth = 4, shrinkage = 0.01)



## ---- echo = FALSE------------------------------------------------------------

boost_summary <- summary(boost_price, plot = FALSE)

boost_df <- data.frame(boost_summary[1:13,])

ggplot(boost_df, aes(x = boost_df[,2], y = reorder(boost_df[,1], -boost_df[,2], decreasing = TRUE))) +
  geom_bar(stat = "identity", fill = "lightgreen") +
  theme_minimal() +
  ylab("Variables") +
  xlab("Relative Influence") +
  ggtitle("Variable Importance Plot")


## -----------------------------------------------------------------------------

fitControl <- trainControl(method = "cv", number = 10)

tuneGrid <- expand.grid(n.trees = seq(1000, 5000, by = 2000),
                        interaction.depth = c(4,6,8),
                        shrinkage = 0.01,
                        n.minobsinnode = 10)

gbm_model <- caret::train(price ~ . -id,
                          data = train,
                          method = "gbm",
                          trControl = fitControl,
                          tuneGrid = tuneGrid,
                          distribution = "gaussian", 
                          verbose = FALSE)

knitr::kable(gbm_model$results, align = "c", caption = "Metric Table", format = "latex")


## ---- echo=FALSE--------------------------------------------------------------

boost_price <- gbm(price ~ lat + lon + rate + hoa + hvac + garage + view +
                  year + bath + bed + stories + lot + living, data =train,
                  distribution = 'gaussian', n.trees = 1000,
                  interaction.depth = 8, shrinkage = 0.01)

n.trees <- seq(from = 10, to = 1000, by = 10)
predmat <- predict(boost_price, newdata = validation, n.trees = n.trees)

berr <- with(validation, apply((predmat - validation$price)^2, 2, mean))
plot(n.trees, berr, pch = 19, ylab = "Mean Squared Error", xlab = "Trees", 
    main = "Boosting Validation Error")
abline(h = min(test.err), col = "red")



## ---- echo=FALSE--------------------------------------------------------------

rmse <- gbm_model$results
rmse <- min(rmse$RMSE)
mse_gbm_price_k <-  rmse^2
mse_gbm_price_valid <- min(test.err)
row1 <- data.frame("Boosting MSE (K-fold)",mse_gbm_price_k)
row2 <- data.frame("Boosting MSE (Validation)", mse_gbm_price_valid)
colnames(row1) <- c("Metric", "Value")
colnames(row2) <- c("Metric", "Value")
out_table <- rbind(row1,row2)

knitr::kable(out_table, align = "c", caption = "MSE Table", format = "latex")


## ---- echo = FALSE, include=FALSE---------------------------------------------

# Multiple Linear Regression Prediction

kaggle_price_lm <- predict(lm_price, newdata = test)

kaggle_matrix_price_lm <- cbind(as.numeric(levels(test$id))[test$id], kaggle_price_lm)

colnames(kaggle_matrix_price_lm) <- c("id", "price")

write.csv(kaggle_matrix_price_lm, "kaggle_price_lm.csv", row.names = FALSE)


# Ridge Prediction

x <- model.matrix( ~ . -id ,data = test)

kaggle_price_ridge <- predict(ridge_price, s = best.lam, newx = x)

kaggle_matrix_price_ridge <- cbind(as.numeric(levels(test$id))[test$id], kaggle_price_ridge)

colnames(kaggle_matrix_price_ridge) <- c("id", "price")

write.csv(kaggle_matrix_price_ridge, "kaggle_price_ridge.csv", row.names = FALSE)

# Decision Tree Prediction

kaggle_price_tree <- predict(tree_price, newdata = test)

kaggle_matrix_price_tree <- cbind(as.numeric(levels(test$id))[test$id], kaggle_price_tree)

colnames(kaggle_matrix_price_tree) <- c("id", "price")

write.csv(kaggle_matrix_price_tree, "kaggle_price_tree.csv", row.names = FALSE)

# Random Forest Prediction

kaggle_price_rf <- predict(rf_price, newdata = test)

kaggle_matrix_price_rf <- cbind(as.numeric(levels(test$id))[test$id], kaggle_price_rf)

colnames(kaggle_matrix_price_rf) <- c("id", "price")

write.csv(kaggle_matrix_price_rf, "kaggle_price_rf.csv", row.names = FALSE)

# Boosting Prediction

kaggle_price_boost <- predict(boost_price, newdata = test)

kaggle_matrix_price_boost <- cbind(as.numeric(levels(test$id))[test$id], kaggle_price_boost)

colnames(kaggle_matrix_price_boost) <- c("id", "price")

write.csv(kaggle_matrix_price_boost, "kaggle_price_boost.csv", row.names = FALSE)



## ---- echo = FALSE------------------------------------------------------------

models <- c("Multiple Linear Regression", "Ridge Regression", "Decision Tree", "Random Forest", "Boosting")

mse_k <- c(lm_price_mse, mse_ridge_price_k, mse_tree_price_k, mse_ridge_price_k, mse_gbm_price_k)

mse_valid <- c(lm_price_mse_valid, mse_ridge_price, mse_tree_price_valid, mse_ridge_price, mse_gbm_price_valid)

mse_test <- c(0.43900, 0.44807, 0.63099, 0.29468, 0.31265)

df <- data.frame(Model = models, MSE_K_Fold = mse_k, MSE_Validation = mse_valid, MSE_Test = mse_test)

colnames(df) <- c("Model", "K-fold MSE", "Validation MSE", "Test MSE")


