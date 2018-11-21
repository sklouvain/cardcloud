 # Packages
library(keras)
library(dplyr)
library(ggplot2)
library(tidyr)
library(ggridges)
library(readr)
library(purrr)
library(tensorflow)

# Read data
data <- read.csv("Cardiotocographic.csv", header = T)

# Change to matrix
data <- as.matrix(data)
dimnames(data) <- NULL

# Normalize
data[, 1:21] <- normalize(data[, 1:21])
data[,22] <- as.numeric(data[,22]) -1
#summary(data)
#boxplot(data[, 1:21])

# Data partition
set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(0.75, 0.25))
training <- data[ind==1, 1:21]
test <- data[ind==2, 1:21]
trainingtarget <- data[ind==1, 22]
testtarget <- data[ind==2, 22]

# One Hot Encoding
trainLabels <- to_categorical(trainingtarget)
testLabels <- to_categorical(testtarget)

# Create sequential model
model <- keras_model_sequential()
model %>%
  layer_dense(units=21, activation="tanh", input_shape = ncol(training)) %>%
  layer_dense(units=8, activation = "tanh") %>%
  layer_dense(units = 3, activation = "softmax")
summary(model)

# Compile
model %>%
  compile(loss = "mean_squared_error",
          optimizer = "adam",
          metrics = c("accuracy"))

# Fit model
history <- model %>%
  fit(training,
      trainLabels,
      epoch = 200,
      batch_size = 32,
      validation_split = 0.2
  )

plot(history)