#a = 0.05 # nolint
library("dplyr")
library("ggpubr")

 reddy <- data.frame(
 Accuracy = c(86.46, 85.14, 83.82, 86.13),
 Precision = c(86.5, 85.2, 83.9, 86.2),
 Recall = c(85.2, 85.2, 83.8, 86.1),
 F1 = c(86.4, 85.1, 83.8, 86.1)
 )
 proposed <- data.frame(
 Accuracy = c(89.36, 86.17, 88.29, 87.23),
 Precision = c(85, 86, 88, 87),
 Recall = c(84, 85, 88, 87),
 F1 = c(83, 91, 88, 87)
 )

reddyAcc <- c(86.46, 85.14, 83.82, 86.13)
proposedAcc <- c(89.36, 86.17, 88.29, 87.23)
MODEL_ACCURACY = c(reddyAcc, proposedAcc)
MODEL_TYPE = rep(c("reddy", "proposed"), each = 4)
DATASET <- data.frame(MODEL_TYPE, MODEL_ACCURACY, stringsAsFactors = TRUE)
DATASET

shapiro.test( as.numeric( DATASET$MODEL_ACCURACY ))
#W = 0.87608, p-value = 0.1727
#P>a therfore normally distributed use t-test

group_by(DATASET, MODEL_TYPE) %>%
  summarise(
    count = n(),
    median = median(MODEL_ACCURACY, na.rm = TRUE),
    IQR = IQR(MODEL_ACCURACY, na.rm = TRUE))
res <- t.test(MODEL_ACCURACY ~ MODEL_TYPE)
res