# Name: 	Denis Stukal
# Date: 	March 23, 2018
# Summary: 	Uses 4*10 pre-trained models to make 4*10 predictions for every unlabeled instance.
#			Input unlabeled data: ~/bot_applications/features.txt

rm(list = ls())

library(caret)
library(doMC)
library(kernlab)
library(adabag)
library(rpart)
library(Matrix)
library(xgboost)
library(Ckmeans.1d.dp)
library(glmnet)


#-------------- LOAD MODELS --------------#
setwd('~/bot_applications/estimated_models')

models <- c("glmnet", "samme", "svm", "xgb")

for (i in 1:10) {
  for (mod in models) {
    load(paste0(mod, "_allper_botplus_bal_list_", i, ".RData"))
    assign(x = paste0(mod, "_list_", i), value = get(paste0(mod, "_list")))
  }
}
Sys.sleep(1)
rm(glmnet_list, samme_list, xgb_list, svm_list)



#-------------- LOAD DATA --------------#
setwd('~/bot_applications/train_data')

for (i in 1:10) {
  load(paste0("allper_list_", i, "_botplus_bal.RData"))
  assign(x = paste0("data_list_for_experiment_", i), value = get("data_list_for_experiment"))
}
Sys.sleep(1)
rm(data_list_for_experiment)

for (i in 1:10) {
  dat <- get(paste0('data_list_for_experiment_', i))
  #--- load model output
  glmneto <- get(paste0("glmnet_list_", i))
  sammeo <- get(paste0("samme_list_", i))
  xgbo <- get(paste0("xgb_list_", i))
  svmo <- get(paste0("svm_list_", i))
  #--- load data for prediction
  newdata <- read.table(file = '~/bot_applications/data/features.txt', header = T, sep = '\t', stringsAsFactors = F)
  names(newdata)[1] <- 'id'
  new_data_ids <- newdata$id
  vars_to_remove <- setdiff(x = colnames(newdata), y = colnames(dat$train_x_trans))
  newdata <- newdata[,-match(x = vars_to_remove, table = colnames(newdata))]
  x_trans <- preProcess(x = dat$train_x_for_trans, method = c("center", "scale"))
  newdata_trans <- predict(x_trans, newdata)
  ddata <- xgb.DMatrix(data = as.matrix(newdata_trans), missing = NA)
  #--- get predictions
  dy_pred_glmnet <- predict(glmneto$model, newx = as.matrix(newdata_trans), type = "class", s = glmneto$best_lambda)
  dprob_pred_glmnet <- predict(glmneto$model, newx = as.matrix(newdata_trans), type = "response", s = glmneto$best_lambda)
  dy_pred_samme <- predict.boosting(object = sammeo$model, newdata = newdata_trans)
  dprob_pred_xgb <- predict(xgbo$model, ddata)
  dy_pred_xgb <- rep("bot", length(dprob_pred_xgb))
  dy_pred_xgb[dprob_pred_xgb < 0.5] <- "not_bot"
  dy_pred_svm <- predict(svmo$model, newdata = newdata_trans)
  doutput <- list("new_data_ids" = new_data_ids, "glmnet_prob" = dprob_pred_glmnet, "glmnet_y" = dy_pred_glmnet, "samme_y" = dy_pred_samme, "xgb_prob" = dprob_pred_xgb, "xgb_y" = dy_pred_xgb, "svm_y" = dy_pred_svm)
  save(x = doutput, file = paste0('~/bot_applications/predictions/prediction_allper_botplus_bal_', i, '.RData'))
  cat('Iter ', i, ' done\n')
}


cat('JOB DONE\n')







