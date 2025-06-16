library(tidyverse)
library(MASS)
library(missRanger)
library(corrplot)
library(caret)
library(car)
library(performance)
library(SHAPforxgboost)
library(pdp)
library(moments)
library(glmnet)

################################################################################

#EXPLORATORY DATA ANALYSIS

summary(train)
sum(is.na(train$Listening_Time_minutes))
sum(is.na(train))
df <- train
str(df)

#Distribution of episodes

df %>%
  ggplot(aes(y = Listening_Time_minutes, x = Genre)) +
  geom_boxplot(fill='lightblue', color='black')

df %>%
  ggplot(aes(y = Episode_Length_minutes)) +
  geom_boxplot(fill='red')

#Columns with NA, Episode_Length_minutes = 87093, Guest_Popularity_percentage =
#146030, Number_of_Ads = 1

#Convert all character columns to factors

df$Podcast_Name <- as.factor(df$Podcast_Name)
df$Episode_Title <- as.factor(df$Episode_Title)
df$Genre <- as.factor(df$Genre)
df$Publication_Day <- as.factor(df$Publication_Day)
df$Publication_Time <- as.factor(df$Publication_Time)
df$Episode_Sentiment <- as.factor(df$Episode_Sentiment)

str(df)

#Remove Podcast_name

df$Podcast_Name <- NULL

#Extract numeric episode number from title

df$Episode_Number <- as.numeric(gsub('Episode ', '', df$Episode_Title))

#Remove Episode Title

df$Episode_Title <- NULL

str(df)
table(df$Genre)

#Percentage of missing values

missing_pct <- sapply(df, function(x) mean(is.na(x)) * 100)
missing_table <- data.frame(Variable = names(missing_pct),
                            Missing_Percentage = round(missing_pct, 2))
missing_table <- missing_table[order(-missing_table$Missing_Percentage), ]

# View the table
print(missing_table)

library(knitr)
kable(missing_table, caption = "Percentage of Missing Values by Variable")

#Imputation of missing data

df_impute <- missRanger(df, num.trees = 20, pmm.k = 0)

#UNIVARIATE ANALYSIS

#Remove observation where Episode_Length_minutes = 0
#Inconsistencies with Number_of_Ads

df_impute_v2 <- df_impute[-150179,]

rows_to_change <- c(283606, 683147, 537705, 602553, 672139, 436577, 495919,
                    211159, 567235)

new_values <- c(3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0)

df_impute_v2$Number_of_Ads[id=rows_to_change] <- new_values

#Univariate Graphs

#Categorical Predictors

df_impute_v2 %>%
  ggplot(aes(Genre)) +
  geom_bar(fill='blue')

#Numeric Predictors

df_impute_v2 %>%
  ggplot(aes(y=Listening_Time_minutes)) +
  geom_boxplot(fill='red')

#Remove Id

df_impute_v3 <- df_impute_v2
df_impute_v3$id <- NULL

#Remove Episode_Length_minutes > 200

df_impute_v3 <- df_impute_v3[-101638,]

#Histogram of Target Variable (Numeric)

df_impute_v3 %>%
  ggplot(aes(Episode_Length_minutes)) +
  geom_histogram(aes(bins = 30, fill='lightblue'))

#BIVARIATE ANALYSIS

df_numeric <- df_impute_v3[sapply(df_impute_v3, is.numeric)]

#Verifying linear relationships between numeric variables

cor_matrix <- cor(df_numeric)
corrplot(cor_matrix, method = 'color', type = 'lower',
         addCoef.col = 'black', tl.srt = 45, diag = FALSE)

df_impute_v3 %>%
  ggplot(aes(Host_Popularity_percentage, Listening_Time_minutes)) +
  geom_point()

#Numeric vs. Categorical

df_impute_v3 %>%
  ggplot(aes(Genre, Listening_Time_minutes)) +
  geom_boxplot(fill='red')

df_impute_v3 %>%
  group_by(Episode_Sentiment) %>%
  summarize(
    mean = mean(Listening_Time_minutes)
  )

df_impute_v3 %>%
  group_by(Publication_Day) %>%
  summarize(
    mean = mean(Listening_Time_minutes)
  )

df_impute_v3 %>%
  group_by(Publication_Time) %>%
  summarize(
    mean = mean(Listening_Time_minutes)
  )

df_impute_v3 %>%
  group_by(Genre) %>%
  summarize(
    mean = mean(Listening_Time_minutes)
  )

df_impute_v3 %>%
  ggplot(aes(Genre, Publication_Time)) +
  stat_summary(fun=mean, geom = 'bar') +
  facet_wrap(~Publication_Day)

df_impute_v3 %>%
  filter(Episode_Length_minutes >= 20 & Episode_Length_minutes <= 80) %>%
  summarize(prop = n() / nrow(df_impute_v3))

table(df_impute_v3$Genre)

################################################################################
################################################################################

#MODEL CONSTRUCTION

#Linear Regression

#Only with numeric data

lm1 <- lm(Listening_Time_minutes ~ ., data = df_numeric)
summary(lm1)

#Linear Regression with all variables without relevel

lm2 <- lm(Listening_Time_minutes ~ .,data=df_impute_v3)
summary(lm2)

#Releveling factor variables

vars_cat <- c('Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment')

for (i in vars_cat) {
  table <-  as.data.frame(table(df_impute_v3[,i]))
  max <- which.max(table[,2])
  level.name <- as.character(table[max,1])
  df_impute_v3[,i] <- relevel(df_impute_v3[,i], ref = level.name)
}

#Train Test Split

partition <- createDataPartition(df_impute_v3$Listening_Time_minutes, p = .75,
                                 list = FALSE)

df_impute_v3_train <- df_impute_v3[partition,]
df_impute_v3_test <- df_impute_v3[-partition,]

mean(df_impute_v3_train$Listening_Time_minutes)
mean(df_impute_v3_test$Listening_Time_minutes)

#Linear Regression

lm_full <- lm(Listening_Time_minutes ~ ., data = df_impute_v3_train)

summary(lm_full)
plot(lm_full)

#MODEL VALIDATION

RMSE(predict(lm_full, newdata = df_impute_v3_test), 
     df_impute_v3_test$Listening_Time_minutes)

vif(lm_full)
check_collinearity(lm_full)

#Binarize factor variables

binarizer <- dummyVars(~ Genre + Publication_Day + Publication_Time + 
                         Episode_Sentiment, data = df_impute_v3, fullRank = TRUE)

binarized_vars <- data.frame(predict(binarizer, newdata = df_impute_v3))

head(binarized_vars)

df_impute_v3_bin <- cbind(df_impute_v3, binarized_vars)
head(df_impute_v3_bin)

df_impute_v3_bin$Genre <- NULL
df_impute_v3_bin$Publication_Day <- NULL
df_impute_v3_bin$Publication_Time <- NULL
df_impute_v3_bin$Episode_Sentiment <- NULL

df_impute_v3_bin %>%
  ggplot(aes((Listening_Time_minutes + 1)^(2/3))) +
  geom_histogram()

#Using Box-Cox to transform target variable

bc <- boxcox(lm(Listening_Time_minutes + 1~., data = df_impute_v3_bin))

lambda <- bc$x[which.max(bc$y)]

#Convert target variable

df_impute_v3_bin$Listening_Time_minutes_bc <- 
  ((df_impute_v3_bin$Listening_Time_minutes + 1)^lambda - 1) / lambda

#Train test split

df_impute_v3_bin_train <- df_impute_v3_bin[partition,]
df_impute_v3_bin_test <- df_impute_v3_bin[-partition,]

#Ful model with binarized variables

lm_full_bin <- lm(Listening_Time_minutes ~ . - Listening_Time_minutes_bc,
                  data = df_impute_v3_bin_train)

summary(lm_full_bin)

lm_full_bin_bc <- lm(Listening_Time_minutes_bc ~ . - Listening_Time_minutes,
                     data = df_impute_v3_bin_train)

pred_bc <- predict(lm_full_bin_bc)

pred_original <- ((lambda * pred_bc + 1)^(1/lambda)) - 1

RMSE(pred_original, df_impute_v3_bin_train$Listening_Time_minutes)

#Stepwise Regression

model_backward_AIC <- stepAIC(lm_full_bin)

lm_null <- lm(Listening_Time_minutes ~ 1, data = df_impute_v3_bin_train)

model_forward_BIC <- stepAIC(lm_null, direction = 'forward',
                             scope = list(upper=lm_full_bin,
                                          lower=lm_null),
                             k=log(nrow(df_impute_v3_bin_train)))

model_forward_BIC$coefficients
model_backward_AIC$coefficients

#MODEL VALIDATION

RMSE(predict(lm_null, newdata = df_impute_v3_bin_test),
     df_impute_v3_bin_test$Listening_Time_minutes)

RMSE(predict(model_backward_AIC, newdata = df_impute_v3_bin_test),
     df_impute_v3_bin_test$Listening_Time_minutes)

RMSE(predict(model_forward_BIC, newdata = df_impute_v3_bin_test),
     df_impute_v3_bin_test$Listening_Time_minutes)

RMSE(predict(lm_full_bin, newdata = df_impute_v3_bin_test),
     df_impute_v3_bin_test$Listening_Time_minutes)

#Regularization

Reg_train <- model.matrix(Listening_Time_minutes ~ . - Listening_Time_minutes_bc,
                          data = df_impute_v3_bin_train)

head(Reg_train)

m_lambda <- glmnet(x = Reg_train, y = df_impute_v3_bin_train$Listening_Time_minutes,
                   family = 'gaussian', lambda = c(0, 10, 100, 500, 1000),
                   alpha = 0.5)

m_lambda$a0
m_lambda$beta
coef(m_lambda)

m_ridge <- glmnet(x = Reg_train, y = df_impute_v3_bin_train$Listening_Time_minutes,
                  family = 'gaussian', lambda = 10, alpha = 0)

m_lasso <- glmnet(x = Reg_train, y = df_impute_v3_bin_train$Listening_Time_minutes,
                  family = 'gaussian', lambda = 10, alpha = 1)

coef(m_ridge)
coef(m_lasso)
coef(m_lambda)

set.seed(1111)

m <- cv.glmnet(x = Reg_train, y = df_impute_v3_bin_train$Listening_Time_minutes,
               family = 'gaussian', alpha = 0.5)
plot(m)

m$lambda.min
m$lambda.1se

m_min <- glmnet(x = Reg_train, y = df_impute_v3_bin_train$Listening_Time_minutes,
                family = 'gaussian', lambda = m$lambda.min, alpha = 0.5)

m_min$beta

m_1se <- glmnet(x = Reg_train, y = df_impute_v3_bin_train$Listening_Time_minutes,
             family = 'gaussian', lambda = m$lambda.1se, alpha = 0.5) 

m_1se$beta

Reg_test <- model.matrix(Listening_Time_minutes ~ . - Listening_Time_minutes_bc,
                         data = df_impute_v3_bin_test)

RMSE(predict(m_min, newx = Reg_test), df_impute_v3_bin_test$Listening_Time_minutes)
RMSE(predict(m_1se, newx = Reg_test), df_impute_v3_bin_test$Listening_Time_minutes)

################################################################################
################################################################################

#XGBoost

xgb_ctrl <- trainControl(method = 'cv', number = 5)
xgb_grid <- expand.grid(max_depth = 7, min_child_weight = 1, gamma = 0,
                        nrounds = c(50, 100, 150, 200, 250, 300),
                        eta = c(0.001, 0.002, 0.01, 0.02, 0.1),
                        colsample_bytree = 0.6, subsample = 0.6)

set.seed(42)

xgb_tuned <- train(Listening_Time_minutes ~ ., data = df_impute_v3_train,
                   method = 'xgbTree', trControl = xgb_ctrl,
                   tuneGrid = xgb_grid)

RMSE(predict(xgb_tuned, newdata = df_impute_v3_test),
     df_impute_v3_test$Listening_Time_minutes)

ggplot(varImp(xgb_tuned), top = 5)

varImp(xgb_tuned)

partial(xgb_tuned, train = df_impute_v3_train, pred.var = 'Episode_Length_minutes',
        plot = TRUE)

partial(xgb_tuned, train = df_impute_v3_train, pred.var = 'Number_of_Ads',
        plot = TRUE)

partial(xgb_tuned, train = df_impute_v3_train, pred.var = 'Host_Popularity_percentage',
        plot = TRUE)

partial(xgb_tuned, train = df_impute_v3_train, pred.var = 'Genre',
        plot = TRUE)

partial(xgb_tuned, train = df_impute_v3_train, pred.var = 'Publication_Day',
        plot = TRUE)

partial(xgb_tuned, train = df_impute_v3_train, pred.var = 'Episode_Number',
        plot = TRUE)

#Extract the final model

xgb_booster <- xgb_tuned$finalModel

#Convert to one hot encode

x_onehot <- model.matrix(~ . -1, data = df_impute_v3_train)
x_mat <- x_onehot[,xgb_booster$feature_names]
#identical(colnames(x_mat), xgb_booster$feature_names)

#Calculate SHAP values

shap_values <- shap.values(xgb_model = xgb_booster,
                           X_train = x_mat)

#Inspect SHAP values

head(shap_values$shap_score)
str(shap_values$shap_score)
#Compute feature importance

shap_contrib <- shap_values$shap_score

#Reshaping for plotting

shap_long <- shap.prep(shap_contrib = shap_contrib, X_train = x_mat)

#Plot SHAP summary

top5_vars <- shap_long %>%
  group_by(variable) %>%
  summarize(mean_abs_shap = mean(abs(mean_value), na.rm = TRUE)) %>%
  arrange(desc(mean_abs_shap)) %>%
  slice_head(n=5) %>%
  pull(variable)

#Sample shap_long

shap_long_small <- shap_long %>% sample_n(10000)

for (var in top5_vars) {
  png(filename = paste0("shap_dependence_", var, ".png"), width = 800, height = 600)
  print(
    shap.plot.dependence(
      data_long = shap_long_small,
      x = var
    )
  )
  dev.off()
}

#Dependence plot

shap.plot.dependence(data_long = shap_long_small, x = "Host_Popularity_percentage",
                     color_feature = 'Episode_Number')

#SHAP values for Test data

x_onehot_test <- model.matrix(~ . -1, data = df_impute_v3_test)
x_mat_test <- x_onehot_test[,xgb_booster$feature_names]

#SHAP values for test data

shap_values_test <- shap.values(xgb_booster, x_mat_test)

#Reshaping for plotting

shap_long_test <- shap.prep(shap_contrib = shap_values_test$shap_score,
                            X_train = x_mat_test)

#Top 5 variables

top5_vars_test <- shap_long_test %>%
  group_by(variable) %>%
  summarize(mean_abs_shap = mean(abs(mean_value), na.rm = TRUE)) %>%
  arrange(desc(mean_abs_shap)) %>%
  slice_head(n = 5) %>%
  pull(variable)

#Sample shap_long_test

shap_long_small_test <- shap_long_test %>% sample_n(10000)

#Dependence plot

shap.plot.dependence(data_long = shap_long_small_test, 
                     x = 'Number_of_Ads')

#Residual diagnosis

xgb_y_pred <- predict(xgb_tuned, newdata = df_impute_v3_test)

xgb_residuals <- df_impute_v3_test$Listening_Time_minutes - xgb_y_pred

plot(xgb_y_pred, xgb_residuals, xlab = 'Predicted Value', ylab = 'Residuals',
     main = 'Residual vs Fitted')
abline(h = 0, col = 'red')

qqnorm(xgb_residuals)
qqline(xgb_residuals, col = 'red')

hist(xgb_residuals, breaks = 30, main = 'Histogram of Residuals',
     xlab = 'Residuals')

ggplot(data.frame(xgb_y_pred, xgb_residuals), aes(x=xgb_y_pred,
                                                  y=xgb_residuals)) +
  geom_point(alpha=0.6) +
  geom_hline(yintercept = 0, color = 'red') +
  labs(title = 'Residual vs Fitted', x = 'Fitted', y = 'Residuals')

#Log transformed target

xgb_tuned_log <- train(log1p(Listening_Time_minutes) ~ ., data = df_impute_v3_train,
                   method = 'xgbTree', trControl = xgb_ctrl,
                   tuneGrid = xgb_grid)


imp <- varImp(xgb_tuned_log)

top5 <- imp$importance %>%
  rownames_to_column("Variable") %>%
  arrange(desc(Overall)) %>%
  slice_head(n=5)

ggplot(top5, aes(x = reorder(Variable, Overall), y = Overall)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 5 Important Variables",
       x = "Variable",
       y = "Importance (Overall)") +
  theme_minimal()





