# Import libraries
library(readr)
library(dplyr)

# Read in dataset
df <- read_delim("..//Data/subject-info.csv", delim = ";", escape_double = FALSE, trim_ws = TRUE)

# Preprocessing
# Remove extraneous columns
df <- df[, 1:105]

# Check for columns with missing values
for (i in 1:ncol(df)){
  if (any(is.na(df[,i]))){
    num_missing <- sum(is.na(df[,i]))
    message(paste("The column ", colnames(df)[i], "has ", num_missing, "missing values."))
  } 
}

# Groups of patient features
# R indexes at 1; shift down by 1 for Python
demographics <- seq(6, 28) # 6, 7, ... 28
radiographic <- c(49, 50)
echocardiographic <- seq(51, 60) # 51, 52, ..., 60
laboratory <- seq(29, 48) # 29, 30, ..., 48
ecg <- seq(63, 71) # 63, 64, ..., 71
holter <- seq(74, 92) # 74, 75, ..., 92
medications <- seq(93, 105) # 93, 94, ..., 105

# Limit study to patients with sinus rhythms
df_sinus <- df[df$`ECG rhythm` == 0, ]

# Assign patients with exit of study as NA to values of 0 (assume survivor)
df_sinus$`Exit of the study`[is.na(df_sinus$`Exit of the study`)] <- 0

# Drop columns with more than 10% of values missing
missing_threshold <- nrow(df_sinus) / 10
dropped_columns <- c()
for (i in 1:ncol(df_sinus)){
  if (any(is.na(df_sinus[,i]))){
    num_missing <- sum(is.na(df_sinus[,i]))
    if (num_missing >= missing_threshold){
      dropped_columns <- c(dropped_columns, colnames(df_sinus)[i])
    }
  }
}
df_sinus <- df_sinus %>% select(-all_of(dropped_columns)) # Results in 702 observations

# For first pass, delete rows with missing values
df_sinus <- na.omit(df_sinus) # Results in 535 observations

# Print dataframe to csv file (to later be used in Python)
write.csv(df_sinus, file = "../Data/subject-info-cleaned.csv")

# TODO: If time, perform missing value imputation

# Check for columns with missing values
# for (i in 1:ncol(df_sinus)){
#   if (any(is.na(df_sinus[,i]))){
#     num_missing <- sum(is.na(df_sinus[,i]))
#     message(paste("The column ", colnames(df_sinus)[i], "has ", num_missing, "missing values."))
#   } 
# }

