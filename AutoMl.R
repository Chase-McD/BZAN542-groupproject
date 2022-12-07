# AutoML Implementaion
# https://docs.h2o.ai/h2o/latest-stable/h2o-docs/downloading.html ## URL Containing how to download H2O and Documentation about how it works. ## Make sure you have Java Installed.

suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(rpart))
suppressPackageStartupMessages(library(doParallel))
suppressPackageStartupMessages(library(rpart.plot))
suppressPackageStartupMessages(library(h2o))
h2o.init(nthreads = -1,
         max_mem_size = '32G')


df = read.csv("UserCarData.csv")
df <- df[,-1]
df[sapply(df,is.character)] <- lapply(df[sapply(df,is.character)],as.factor) # Changes all character types to factors for each column
df$year <- as.factor(df$year) # Changes int year to factor year
df$seats <- as.factor(df$seats) # Changes int seats to factor seats
df$mileage <- as.factor(df$mileage) # Changes int mpg to factor mpg

df <- df %>%
  mutate(selling_price = selling_price * 0.01,
         mi_driven = round(km_driven * 0.6213712)) %>%
  select(-c(km_driven)) # Converts Rubies to Dollars, Converts Km to Miles, and removes the Km column

df$max_power = as.integer(df$max_power)
df$mileage = as.integer(df$mileage)
df = df %>% 
  mutate(selling_price = as.integer(selling_price)) %>%
  select(-torque) # Converts Rubies to Dollars, Converts Km to Miles, and removes the Km column. Removed Torque until we can figure out what to do with it



#infodensity <- nearZeroVar(df, saveMetrics= TRUE); infodensity
#fitControl <- trainControl(method = "cv", number = 10, classProbs = TRUE, allowParallel = TRUE, verboseIter = FALSE)


df <- as.h2o(df)
splits <- h2o.splitFrame(data = df,
                         ratios = 0.70,
                         seed = 1)

train <- splits[[1]]
test <- splits[[2]]
y <- 'sold'
x <- setdiff(names(df),c(y))

glm_fit1 <- h2o.glm(x = x, 
                    y = y, 
                    training_frame = train,
                    model_id = "glm_fit1",
                    family = "binomial",
                    nfolds = 10,
                    keep_cross_validation_fold_assignment = TRUE) 

glm_fit2 <- h2o.glm(x = x, 
                    y = y, 
                    training_frame = train,
                    model_id = "glm_fit2",
                    validation_frame = test,
                    family = "binomial",
                    lambda_search = TRUE,
                    nfolds = 10,
                    keep_cross_validation_fold_assignment = TRUE)

glm_perf1 <- h2o.performance(model = glm_fit1,
                             newdata = test)

glm_perf2 <- h2o.performance(model = glm_fit2,
                             newdata = test)

h2o.auc(glm_perf1)
h2o.auc(glm_perf2)

rf_fit1 <- h2o.randomForest(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "rf_fit1",
                            seed = 1,
                            nfolds = 10,
                            keep_cross_validation_fold_assignment = TRUE)

rf_fit2 <- h2o.randomForest(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "rf_fit2",
                            ntrees = 100,
                            seed = 1,
                            nfolds = 10,
                            keep_cross_validation_fold_assignment = TRUE)

rf_perf1 <- h2o.performance(model = rf_fit1,
                             newdata = test)

rf_perf2 <- h2o.performance(model = rf_fit2,
                             newdata = test)

h2o.auc(rf_perf1)
h2o.auc(rf_perf2)


gbm_fit1 <- h2o.gbm(x = x,
                    y = y,
                    training_frame = train,
                    model_id = "gbm_fit1",
                    seed = 1,
                    nfolds = 10,
                    keep_cross_validation_fold_assignment = TRUE)

gbm_fit2 <- h2o.gbm(x = x,
                    y = y,
                    training_frame = train,
                    model_id = "gbm_fit2",
                    #validation_frame = valid,  #only used if stopping_rounds > 0
                    ntrees = 50,
                    learn_rate = 0.2,
                    seed = 1,
                    nfolds = 10,
                    keep_cross_validation_fold_assignment = TRUE)

gbm_fit3 <- h2o.gbm(x = x,
                    y = y,
                    training_frame = train,
                    model_id = "gbm_fit3",
                    ntrees = 50,
                    learn_rate = 0.18,
                    seed = 1,
                    nfolds = 10,
                    keep_cross_validation_fold_assignment = TRUE)

gbm_perf1 <- h2o.performance(model = gbm_fit1,
                             newdata = test)
gbm_perf2 <- h2o.performance(model = gbm_fit2,
                             newdata = test)
gbm_perf3 <- h2o.performance(model = gbm_fit3,
                             newdata = test)

h2o.auc(gbm_perf1)  
h2o.auc(gbm_perf2)  
h2o.auc(gbm_perf3)


dl_fit1 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit1",
                            seed = 1,
                            nfolds = 10,
                            keep_cross_validation_fold_assignment = TRUE)

dl_fit2 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit2",
                            epochs = 20,
                            hidden= c(10,10),
                            seed = 1,
                            nfolds = 10,
                            keep_cross_validation_fold_assignment = TRUE)

dl_fit3 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit3",
                            epochs = 20,
                            hidden = c(10,10),
                            score_interval = 1,           #used for early stopping
                            stopping_rounds = 3,          #used for early stopping
                            stopping_metric = "AUC",      #used for early stopping
                            stopping_tolerance = 0.0005,  #used for early stopping
                            seed = 1,
                            nfolds = 10,
                            keep_cross_validation_fold_assignment = TRUE)

dl_perf1 <- h2o.performance(model = dl_fit1,
                            newdata = test)
dl_perf2 <- h2o.performance(model = dl_fit2,
                            newdata = test)
dl_perf3 <- h2o.performance(model = dl_fit3,
                            newdata = test)

h2o.auc(dl_perf1)  
h2o.auc(dl_perf2)  
h2o.auc(dl_perf3)  


#### Wasn't able tog et Ensemble to work ##### 
ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                training_frame = train,
                                base_models = list(gbm_perf1, glm_perf2,rf_perf2))


aml <- h2o.automl(y = y,
                  x = x,
                  training_frame = train,
                  leaderboard_frame = test,
                  max_runtime_secs = 100,
                  seed = 1,
                  project_name = "Project_BZAN542")

lb <- aml@leaderboard

# Get model ids for all models in the AutoML Leaderboard
model_ids <- as.data.frame(aml@leaderboard$model_id)[,1]
# Get the "All Models" Stacked Ensemble model
se <- h2o.getModel(grep("StackedEnsemble_AllModels", model_ids, value = TRUE)[1])
# Get the Stacked Ensemble metalearner model
metalearner <- se@model$metalearner_model
h2o.varimp(metalearner)
h2o.varimp_plot(metalearner)
