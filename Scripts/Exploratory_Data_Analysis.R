# Load in libraries
library(dplyr)
library(tidyverse)

# Load in data
df <- read.csv("../Data/subject-info-cleaned-with-ECGReport.csv")

# Counts of survivor, SCD, and PFD patients 
table(df$Cause.of.death)

# LVEF <= 35%
df <- df %>%
  mutate(LVEF_class = ifelse(LVEF.... <= 35, 1, 0))

# Factor discrete variables
discrete_vars <- c("Gender..male.1.", "NYHA.class", "Diabetes..yes.1.", "History.of.dyslipemia..yes.1.", "Peripheral.vascular.disease..yes.1.", "History.of.hypertension..yes.1.", "Prior.Myocardial.Infarction..yes.1.", "LVEF_class", "Calcium.channel.blocker..yes.1." , "Diabetes.medication..yes.1.", "Amiodarone..yes.1.", "Angiotensin.II.receptor.blocker..yes.1." , "Anticoagulants.antitrombotics...yes.1.", "Betablockers..yes.1.", "Digoxin..yes.1.", "Loop.diuretics..yes.1.", "Spironolactone..yes.1.", "Statins..yes.1.", "Hidralazina..yes.1.", "ACE.inhibitor..yes.1.", "Nitrovasodilator..yes.1.")
for (var in discrete_vars) {
  df[[var]] <- as.factor(df[[var]])
}

# Select ecg impressions and create binarized class
ecg_discrete_vars <- c("Ventricular.Extrasystole", "Ventricular.Tachycardia", "Non.sustained.ventricular.tachycardia..CH.10.", "Paroxysmal.supraventricular.tachyarrhythmia", "Bradycardia")
df <- df %>%
  mutate(across(all_of(ecg_discrete_vars), ~ifelse(. == 0, 0, 1), .names = "{.col}_class_yes.1"))
ecg_discrete_classes <- tail(colnames(df), 5)

# Select continuous variables
continuous_vars <- c("Age", "Weight..kg.", "Height..cm.", "Diastolic.blood..pressure..mmHg.", "Systolic.blood.pressure..mmHg.", "daily.smoking..cigarretes.day.", "alcohol.consumption..standard.units.", "Albumin..g.L.", "ALT.or.GPT..IU.L.", "AST.or.GOT..IU.L.", "Total.Cholesterol..mmol.L.", "Creatinine...mol.L.", "Gamma.glutamil.transpeptidase..IU.L.", "Glucose..mmol.L.", "Hemoglobin..g.L.", "HDL..mmol.L.", "Potassium..mEq.L.", "LDL..mmol.L.", "Sodium..mEq.L.", "Pro.BNP..ng.L.", "Protein..g.L.", "T3..pg.dL.", "T4..ng.L.", "Troponin..ng.mL.", "TSH..mIU.L.", "Urea..mg.dL.", "LVEF....")
# Removed normalized troponin

# Convert to numeric
for (i in 1:length(continuous_vars)){
  df$continuous_vars[i] <- as.numeric(gsub(",", ".", df$continuous_vars[i]))
}

# Make sure continuous variables are numeric
for (col in continuous_vars){
  for (i in 1:nrow(df)) {
    if (df[[col]][i] == "" | is.na(df[[col]][i])) {
      df[[col]][i] <- NA  # Replace empty string with NA
    } else {
      df[[col]][i] <- as.numeric(gsub(",", ".", df[[col]][i]))  # Replace commas with dots and convert to numeric
    }
  }
}

# HDL, LDL, T3, Troponin, TSH, pNN50
df$HDL..mmol.L. <- as.numeric(df$HDL..mmol.L.)
df$LDL..mmol.L. <- as.numeric(df$LDL..mmol.L.)
df$T3..pg.dL. <- as.numeric(df$T3..pg.dL.)
df$Troponin..ng.mL. <- as.numeric(df$Troponin..ng.mL.)
df$TSH..mIU.L. <- as.numeric(df$TSH..mIU.L.)

# Select columns in df in order
clinical_vars <- c(discrete_vars, continuous_vars, ecg_discrete_classes)
clinical_vars_ordered <- clinical_vars[order(match(clinical_vars, colnames(df)))]
df_select <- subset(df, select = c("Cause.of.death", clinical_vars_ordered))

# Split dataset into survivor, SCD, and PFD
survivor_df <- df_select[df_select$Cause.of.death == 0, ]
scd_df <- df_select[df_select$Cause.of.death == 3, ]
pfd_df <- df_select[df_select$Cause.of.death == 6, ]

##### 
# Clinical Features
#####

### survivor_df
# Initialize an empty data frame to store results
discrete_summary <- data.frame(Variable = character(), 
                               Statistic = character(),
                               Value = numeric(), stringsAsFactors = FALSE)

# Loop through each dichotomized factor variable
for (var in discrete_vars) {
  # Get the counts of each level of the factor
  level_counts <- table(survivor_df[[var]], useNA = "ifany")
  
  # For binary dichotomized factors, we expect two levels (e.g., 0 and 1 or two categories)
  # Calculate count for each level
  count <- sum(level_counts, na.rm = TRUE)
  
  # Calculate the percentage of each level (if dichotomized, it will be 0 and 1)
  level_percentages <- (level_counts / count) * 100
  
  # Add the counts for each level (e.g., "0", "1")
  for (level in names(level_counts)) {
    discrete_summary <- rbind(discrete_summary, 
                              data.frame(Variable = var, 
                                         Statistic = paste("count_", level, sep = ""), 
                                         Value = level_counts[level]))
    discrete_summary <- rbind(discrete_summary, 
                              data.frame(Variable = var, 
                                         Statistic = paste("percentage_", level, sep = ""), 
                                         Value = level_percentages[level]))
  }
}

# Now, you can reshape the data frame to have one row per variable
discrete_summary_wide <- discrete_summary %>%
  pivot_wider(names_from = Statistic, values_from = Value)

# View the final summary
discrete_summary_wide

# Compute statistics for continuous variables
cont_summary <- survivor_df %>%
  summarise(across(all_of(continuous_vars), 
                   list(median = ~median(., na.rm = TRUE), 
                        IQR = ~IQR(., na.rm = TRUE)), 
                   .names = "{.col}_{.fn}")) %>%
  pivot_longer(everything(), 
               names_to = c("Variable", "Statistic"), 
               names_sep = "_") %>%
  pivot_wider(names_from = Statistic, values_from = value)

# View final summary
cont_summary

### scd_df
# Initialize an empty data frame to store results
discrete_summary <- data.frame(Variable = character(), 
                               Statistic = character(),
                               Value = numeric(), stringsAsFactors = FALSE)

# Loop through each dichotomized factor variable
for (var in discrete_vars) {
  # Get the counts of each level of the factor
  level_counts <- table(scd_df[[var]], useNA = "ifany")
  
  # For binary dichotomized factors, we expect two levels (e.g., 0 and 1 or two categories)
  # Calculate count for each level
  count <- sum(level_counts, na.rm = TRUE)
  
  # Calculate the percentage of each level (if dichotomized, it will be 0 and 1)
  level_percentages <- (level_counts / count) * 100
  
  # Add the counts for each level (e.g., "0", "1")
  for (level in names(level_counts)) {
    discrete_summary <- rbind(discrete_summary, 
                              data.frame(Variable = var, 
                                         Statistic = paste("count_", level, sep = ""), 
                                         Value = level_counts[level]))
    discrete_summary <- rbind(discrete_summary, 
                              data.frame(Variable = var, 
                                         Statistic = paste("percentage_", level, sep = ""), 
                                         Value = level_percentages[level]))
  }
}

# Now, you can reshape the data frame to have one row per variable
discrete_summary_wide <- discrete_summary %>%
  pivot_wider(names_from = Statistic, values_from = Value)

# View the final summary
discrete_summary_wide

# Compute statistics for continuous variables
cont_summary <- scd_df %>%
  summarise(across(all_of(continuous_vars), 
                   list(median = ~median(., na.rm = TRUE), 
                        IQR = ~IQR(., na.rm = TRUE)), 
                   .names = "{.col}_{.fn}")) %>%
  pivot_longer(everything(), 
               names_to = c("Variable", "Statistic"), 
               names_sep = "_") %>%
  pivot_wider(names_from = Statistic, values_from = value)

# View final summary
cont_summary

### pfd_df
# Initialize an empty data frame to store results
discrete_summary <- data.frame(Variable = character(), 
                               Statistic = character(),
                               Value = numeric(), stringsAsFactors = FALSE)

# Loop through each dichotomized factor variable
for (var in discrete_vars) {
  # Get the counts of each level of the factor
  level_counts <- table(pfd_df[[var]], useNA = "ifany")
  
  # For binary dichotomized factors, we expect two levels (e.g., 0 and 1 or two categories)
  # Calculate count for each level
  count <- sum(level_counts, na.rm = TRUE)
  
  # Calculate the percentage of each level (if dichotomized, it will be 0 and 1)
  level_percentages <- (level_counts / count) * 100
  
  # Add the counts for each level (e.g., "0", "1")
  for (level in names(level_counts)) {
    discrete_summary <- rbind(discrete_summary, 
                              data.frame(Variable = var, 
                                         Statistic = paste("count_", level, sep = ""), 
                                         Value = level_counts[level]))
    discrete_summary <- rbind(discrete_summary, 
                              data.frame(Variable = var, 
                                         Statistic = paste("percentage_", level, sep = ""), 
                                         Value = level_percentages[level]))
  }
}

# Now, you can reshape the data frame to have one row per variable
discrete_summary_wide <- discrete_summary %>%
  pivot_wider(names_from = Statistic, values_from = Value)

# View the final summary
discrete_summary_wide

# Compute statistics for continuous variables
cont_summary <- pfd_df %>%
  summarise(across(all_of(continuous_vars), 
                   list(median = ~median(., na.rm = TRUE), 
                        IQR = ~IQR(., na.rm = TRUE)), 
                   .names = "{.col}_{.fn}")) %>%
  pivot_longer(everything(), 
               names_to = c("Variable", "Statistic"), 
               names_sep = "_") %>%
  pivot_wider(names_from = Statistic, values_from = value)

# View final summary
cont_summary



#####
# ECG Features
#####

# Count 1's in each column ignoring NAs
count_ones <- colSums(survivor_df[, ecg_discrete_classes] == 1, na.rm = TRUE)

# Percentage of 1's
percentage_ones <- colMeans(survivor_df[, ecg_discrete_classes] == 1, na.rm = TRUE) * 100

# Combine results
result <- data.frame(Count = count_ones, Percentage = percentage_ones)

# Display results
result

# Count 1's in each column ignoring NAs
count_ones <- colSums(scd_df[, ecg_discrete_classes] == 1, na.rm = TRUE)

# Percentage of 1's
percentage_ones <- colMeans(scd_df[, ecg_discrete_classes] == 1, na.rm = TRUE) * 100

# Combine results
result <- data.frame(Count = count_ones, Percentage = percentage_ones)

# Display results
result

# Count 1's in each column ignoring NAs
count_ones <- colSums(pfd_df[, ecg_discrete_classes] == 1, na.rm = TRUE)

# Percentage of 1's
percentage_ones <- colMeans(pfd_df[, ecg_discrete_classes] == 1, na.rm = TRUE) * 100

# Combine results
result <- data.frame(Count = count_ones, Percentage = percentage_ones)

# Display results
result
