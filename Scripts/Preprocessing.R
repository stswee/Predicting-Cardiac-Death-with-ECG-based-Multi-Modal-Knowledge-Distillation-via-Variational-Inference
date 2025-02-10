# Import libraries
library(readr)
library(dplyr)

# Read in dataset
df <- read_delim("..//Data/subject-info.csv", delim = ";", escape_double = FALSE, trim_ws = TRUE)

# Preprocessing
# Remove extraneous columns
# df <- df[, 1:105]

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
df_sinus <- df[df$`Holter  rhythm` == 0, ] # 710 patients

# Assign patients with exit of study as NA to values of 0 (assume survivor)
df_sinus$`Exit of the study`[is.na(df_sinus$`Exit of the study`)] <- 0 

# Remove patients who were lost to follow-up or had cardiac transplantation
df_sinus <- df_sinus[df_sinus$`Exit of the study` == 0 | df_sinus$`Exit of the study` == 3,] # 694 patients

# Remove patients with non-cardiac deaths
df_sinus <- df_sinus[df_sinus$`Cause of death` != 1, ] # Down to 663 patients

# Reassign pump failure values to only be 7
df_sinus$`Cause of death`[df_sinus$`Cause of death` == 7] <- 6

# Get patients with Holter ECG
# df_sinus <- df_sinus %>% filter(df_sinus$`Hig-resolution ECG available` != 0)
df_sinus <- df_sinus %>% filter(df_sinus$`Holter available` != 0) # Results in 604 patients

# Sort by class
df_sinus <- df_sinus[order(df_sinus$`Cause of death`),]

# Number in each class
table(df_sinus$`Cause of death`)

# Print dataframe to csv file (to later be used in Python)
write.csv(df_sinus, file = "../Data/subject-info-cleaned.csv")


