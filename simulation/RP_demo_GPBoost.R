setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
sim_structure <- "friedman1"
likelihood <- "gamma"
if (likelihood == "gaussian") {
  objective <- "regression"
} else if (likelihood == "gamma") {
  objective <- "gamma"
}

# Read data
X_train <- read.csv(paste0("data/sim_", sim_structure, "_X_train.csv"))
X_test <- read.csv(paste0("data/sim_", sim_structure, "_X_test.csv"))
y_train <- read.csv(paste0("data/sim_", sim_structure, "_y_train.csv"))
y_test <- read.csv(paste0("data/sim_", sim_structure, "_y_test.csv"))
y_true_train <- read.csv(paste0("data/sim_", sim_structure, "_y_true_train.csv"))
y_true_test <- read.csv(paste0("data/sim_", sim_structure, "_y_true_test.csv"))

# For compatbility, convert the y columns to matrices
y_train <- as.matrix(y_train)
y_test <- as.matrix(y_test)
y_true_train <- as.matrix(y_true_train)
y_true_test <- as.matrix(y_true_test)

# Run experiment
library(gpboost)

# Scale the data
# Consistent with the Python scripts, I'm using minmax to feature range (0, 1)
MinMaxScaler <- function(x, x_train, a, b, ...) {
  a + (x - min(x_train, ...)) * (b - a) / (max(x_train, ...) - min(x_train, ...))
}
to_norm_features <- colnames(dplyr::select(X_train, -group))
X_test[to_norm_features] <- 
  mapply(MinMaxScaler, X_test[to_norm_features], X_train[to_norm_features],
         a = 0, b = 1)
X_train[to_norm_features] <- 
  mapply(MinMaxScaler, X_train[to_norm_features], X_train[to_norm_features],
         a = 0, b = 1)

# Define model structure: random intercept model
gp_model <- GPModel(group_data = X_train$group, likelihood = likelihood)

# We'll skip parameter tuning and go straight to training
gp_fit <- gpboost(
  data = as.matrix(X_train), label = y_train, gp_model = gp_model,
  nrounds = 50, learning_rate = 0.1, max_depth = 2,
  use_nesterov_acc = TRUE, verbose = 0,
  objective = objective
)

# Check covariance parameters
# Estimated covariance parameters, i.e. sigma and sigma_1
summary(gp_model)

# Output model predictions on both the training and the test sets
y_pred_tr <- predict(
  gp_fit, data = as.matrix(X_train), group_data_pred = X_train$group,
  predict_var = TRUE, pred_latent = FALSE, predict_cov_mat = FALSE
)
# Documentation: result["response_mean"] are the predicted means of the response
# variable (Label) taking into account both the fixed effects (tree-ensemble) 
# and the random effects (gp_model)
y_pred_tr <- data.frame(y = y_pred_tr$response_mean)

# Test set
y_pred_te <- predict(
  gp_fit, data = as.matrix(X_test), group_data_pred = X_test$group,
  predict_var = TRUE, pred_latent = FALSE, predict_cov_mat = FALSE
)
y_pred_te <- data.frame(y = y_pred_te$response_mean)

write.csv(y_pred_tr, file = paste0("results/GPBoost_", sim_structure, "_pred_tr.csv"), row.names = F)
write.csv(y_pred_te, file = paste0("results/GPBoost_", sim_structure, "_pred_te.csv"), row.names = F)

