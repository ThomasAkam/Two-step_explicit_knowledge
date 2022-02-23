# R code for running a 3-way mixed ANOVA to assess whether learning effects on 
# reaction time are different in clinical and control groups. RT data needs to 
# be saved out from Python to a CSV file using the function
# group_comparison.second_step_RT_interaction_test()

# Adapted from https://www.datanovia.com/en/lessons/mixed-anova-in-r/#three-way-bww-b

library(tidyverse)
library(ggpubr)
library(rstatix)

df <- read.csv("reaction_time_data.csv")

ggboxplot(
  df, x="AB", y="RT", color="CR", palette = "jco",
  facet.by = "group")

res.aov <- anova_test(data = df, dv = RT, wid = sID,between = group, within = c(CR, AB), effect.size = 'pes')
get_anova_table(res.aov)