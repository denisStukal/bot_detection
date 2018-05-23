# Name: 	Denis Stukal
# Date: 	March 23, 2018
# Summary: 	Apply the "majority - unanimous" classifier to the output of 2_predict.R

rm(list = ls())

#------------------ FUNCTIONS ----------------#
ensemble <- function(ref_categ = 'bot', rule = "unanim", ...) {
  preds_list <- list(...)
  preds <- sapply(preds_list, function(k) as.character(k))
  preds <- as.matrix(preds)
  refcateg_counts <- sapply(1:nrow(preds), function(k) sum(preds[k,]==ref_categ) )
  if (rule == "unanim") {
    final <- character(length = length(preds_list[[1]]))
    final[refcateg_counts == length(preds_list)] <- ref_categ
    final[final == ""] <- "not"
  } else {
    not_refcateg_counts <- sapply(1:nrow(preds), function(k) sum(preds[k,]=="not") )
    final <- character(length = length(preds_list[[1]]))
    final[refcateg_counts > length(preds_list)/2] <- ref_categ
    final[not_refcateg_counts > length(preds_list)/2] <- "not"
    final[final == ""] <- "unclear"
  }
  return(final)
}

merge_df_from_list <- function(list, id) {
  d <- merge(list[[1]], list[[2]], by = id)
  names(d) <- gsub(pattern = '\\.x', replacement = '', x = names(d))
  names(d) <- gsub(pattern = '\\.y', replacement = '2', x = names(d))
  for (i in 3:length(list)) {
    d <- merge(d, list[[i]], by = id)
    names(d) <- gsub(pattern = '\\.x', replacement = '', x = names(d))
    names(d) <- gsub(pattern = '\\.y', replacement = i, x = names(d))
  }
  return(d)
}

ensemble_prop <- function(data, id_col_name, thres, category) {
  newdata <- as.matrix(data[,which(names(data) != id_col_name)])
  mythres = thres * ncol(newdata)
  pred <- sapply(1:nrow(newdata), function(k) ifelse(sum(newdata[k,] == category) >= mythres, category, 'not'))
  return(data$id[pred == category])
}

#---------------------------------------------------------

#--- get ensemble predictions from each training set
ens_bot <- list()
ens_hum <- list()
for (i in 1:10) {
  doutput <- get(load(paste0('~/bot_applications/predictions/prediction_allper_botplus_bal_', i, '.RData')))
  ens_bot[[i]] <- data.frame('id' = doutput$new_data_ids, 'pred' = ensemble(ref_categ = 'bot', rule = 'unanim', doutput$glmnet_y, doutput$samme_y$class, doutput$xgb_y, doutput$svm_y), stringsAsFactors = F)
  ens_hum[[i]] <- data.frame('id' = doutput$new_data_ids, 'pred' = ensemble(ref_categ = 'not_bot', rule = 'unanim', doutput$glmnet_y, doutput$samme_y$class, doutput$xgb_y, doutput$svm_y), stringsAsFactors = F)
  rm(doutput)
  Sys.sleep(2)
}

#--- merge ensemble predictions into a df
dbot <- merge_df_from_list(list = ens_bot, id = 'id')
dhum <- merge_df_from_list(list = ens_hum, id = 'id')


#--- get estimates, aggregating predictions from individual training sets
predictions <- list()
predictions[['b06']] <- ensemble_prop(data = dbot, id_col_name = 'id', thres = 0.6, category = 'bot')
predictions[['h06']] <- ensemble_prop(data = dhum, id_col_name = 'id', thres = 0.6, category = 'not_bot')

save(x = predictions, file = paste0('~/bot_applications/predictions/predictions_ensemble.RData'))

cat('JOB DONE\n')
