library(tidyverse)
library(MASS)
library(missRanger)
library(corrplot)
library(caret)
library(car)
library(performance)
library(SHAPforxgboost)

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

#MODEL VALIDATION

RMSE(predict(lm_full, newdata = df_impute_v3_test), 
     df_impute_v3_test$Listening_Time_minutes)

vif(lm_full)
check_collinearity(lm_full)

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

#Extract the final model

xgb_booster <- xgb_tuned$finalModel

#Convert to one hot encode

x_onehot <- model.matrix(~ . -1, data = df_impute_v3_train)
x_mat <- x_onehot[,xgb_booster$feature_names]
#identical(colnames(x_mat), xgb_booster$feature_names)

#Calculate SHAP values

shap_values <- shap.values(xgb_model = xgb_booster, 
                           X_train = x_mat)

head(shap_values$shap_score)


################################################################################




















