## AlzAware: A Predictive Model for Early Detection of Alzheimer’s Disease Using Social Determinants of Health

### Author: DSFPT07 Phase 5 Group 12
- Branely Ope
- Brian Kipngetich
- Cynthia Atieno
- Geoffrey Mwangi
- Linet Maz'susa
- Maureen Wanjeri
- Mercy Silali

![1000422606](https://github.com/user-attachments/assets/539ea278-8943-4aef-975c-b6946468e684)


## Table of Contents
1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Main Objective](#main-objective)
4. [Specific Objectives](#specific-objectives)
5. [Data Understanding](#data-understanding)
   - [Data Sources](*data-sources)
   - [Data Overview](*data-overview)
6. [Data Preparation](#data-preparation)
7. [Modelling](#modelling)
8. [Evaluation](#evaluation) 
9. [Conclusion](#conclusion)
10. [Recommendations](#recommendations)
11. .[Possible next steps](#possible-next-steps)  

## Overview
The AlzAware Project seeks to harness the power of predictive modeling to identify early signs of Alzheimer’s Disease (AD) and Alzheimer’s Disease-Related Dementias (AD/ADRD) by analyzing social determinants of health. Using data from the Mexican Health and Aging Study (MHAS), this initiative investigates how factors like socioeconomic status, education, and access to healthcare influence cognitive decline. The project aspires to empower early interventions, reduce health disparities, and improve care accessibility for underserved populations.

## Problem Statement
Alzheimer’s Disease affects millions globally, with prevalence expected to rise significantly as populations age. Current diagnostic approaches often fail to detect early cognitive impairment, particularly in marginalized communities where access to healthcare is limited. Social determinants of health, such as education, income, and healthcare access, play a crucial yet underutilized role in understanding and predicting cognitive decline. This project addresses the gap by developing a predictive model that integrates these non-clinical factors, enabling early intervention and reducing disparities in healthcare outcomes.

## Main Objective
To develop a predictive model for the early detection of Alzheimer’s Disease (AD) and Alzheimer’s Disease-Related Dementias (AD/ADRD) by leveraging social determinants of health

## Specific Objectives
- Improved Early Detection: Identify individuals at risk of AD/ADRD based on non-clinical factors, enabling timely intervention.
- Bias Mitigation: Ensure the model provides accurate predictions across diverse demographics, minimizing disparities.
- Enhanced Accessibility: Develop a model that can be applied broadly, requiring only widely available social health data.
- Potential for Generalization: Provide a framework that can be adapted for AD/ADRD prediction in other populations and regions.

## Data Understanding
#### Data Sources:
The dataset used in this project was derived from the Mexican Health and Aging Study (MHAS), a publicly available longitudinal survey focusing on adults aged 50 and above in Mexico. This comprehensive dataset contains detailed information on demographic, socioeconomic, health, and lifestyle factors, making it ideal for exploring the impact of social determinants of health (SDOH) on cognitive outcomes.
#### Dataset Overview:
The dataset includes information collected over multiple years, specifically 2003, 2012, 2016, and 2021. These years were selected to provide historical data (2003 and 2012) for training the predictive model and target outcomes (2016 and 2021) for evaluating cognitive health.

   - The dataset consists of several key variables used in the analysis:

     - Demographics: Age, gender, marital status, and place of residence.
     - Socioeconomic Factors: Education level, income, and employment status.
     - Health Metrics: Self-reported health, chronic conditions, and body mass index (BMI).
     - Lifestyle Behaviors: Physical activity, smoking status, and alcohol consumption.
     - Cognitive Scores: Assessment of cognitive health over time, used as the primary outcome variable.

For download of the datasets, view the [dataset link](https://github.com/geomwangi007/Early-Detection-of-Alzheimer-s-Disease-and-Related-Dementias-Using-Social-Determinants-of-Health/tree/main/Data) and for complete documentation of all the datasets.

## Data Preparation

In this section, we will focus on data cleaning to prepare the dataset for analysis. The following methods will be applied:

- Renaming columns for clarity and consistency
- Handling missing data appropriately
- Identifying and removing duplicate records
- Merging multiple datasets
- Grouping the data for better structure
  
Additionally, we will perform feature engineering, which includes:

- Selecting relevant columns for the analysis
- Dropping irrelevant columns
- Filtering the dataset to include only the relevant rows

## Exploratory Data Analysis
The Exploratory Data Analysis (EDA) focuses on understanding key features within the dataset, including demographic, health, lifestyle, and composite health scores
- **Demographics**:
  - Analyzed key demographic variables such as age, marital status, locality size, education level, number of children, and spouse's gender.
  - These distributions help identify the diversity within the dataset.

- **Health and Lifestyle Variables**:
  - Explored health perceptions, limitations in daily living, depressive symptoms, health coverage, vaccination status, exercise frequency, and tobacco use.
  - These variables are important in understanding the overall health and wellbeing of individuals in the dataset.

- **Composite Score Analysis**:
  - The composite score aggregates various health and lifestyle domains, and its distribution is generally bell-shaped with most values around the middle range.
  - Limitation variables (e.g., daily living activities, mobility, depression) show skewed distributions, with most individuals having fewer limitations.

- **Visual Analysis**:
  - **Histograms**:
    - Each variable’s distribution was visualized using histograms.
    - Composite score: Bell-shaped distribution.
    - Limitation variables: Heavily skewed, indicating fewer individuals report high limitations.
  
  - **Scatter Plot Relationships**:
    - **Composite Score vs. Limitations**: Slight downward trend suggesting a negative correlation—more limitations tend to correlate with lower composite scores.
    - **Limitations Co-occurrence**: Scatter plots show that individuals with limitations in one area often have limitations in other areas (e.g., someone with mobility issues may also have limitations in daily living activities).

- **Notable Observations**:
  - **Low Composite Scores and Higher Limitations**: Individuals with more limitations tend to have lower composite scores, suggesting that physical or mental limitations affect performance.
  - **Clustering at Low Limitation Values**: Most data points are concentrated around low values (0, 1, or 2) for each limitation variable.

- **Implications of the Analysis**:
  - **Impact of Limitations on Performance**: Negative correlation suggests physical and mental limitations could hinder cognitive health assessments and composite scores.
  - **Co-Occurrence of Limitations**: Interrelationships among limitations indicate a need for holistic support for individuals facing multiple challenges, as addressing one limitation may improve others.

This EDA offers valuable insights into the dataset, identifying key factors that may influence cognitive health and early detection of Alzheimer’s Disease and related dementias.

![countplot1](https://github.com/user-attachments/assets/a18bb8f6-3cbf-46fe-aeab-422f26644c99)

The dataset reveals that most individuals fall within the 60–69 age group, followed by the 50–59 group, with fewer in the 70–79 range and the smallest number being those under 49. The majority are married or in a civil union, with "Widowed" being the most common among other marital statuses. There is a near-even split between rural and urban residents, though urban areas slightly outnumber rural ones. Education levels show that most individuals have 1–5 years of education, with fewer having no education or more than 10 years. The majority have 3 or 4 living children, with a small proportion having none. Regarding spouse gender, the distribution is nearly equal, with a slight majority of women.

![countplot2](https://github.com/user-attachments/assets/72930681-1438-4fda-b945-ef274fc68db7)


The dataset shows that most individuals rate their health as "Fair," followed by "Good," with fewer rating their health as "Poor," "Very Good," or "Excellent." Most individuals report no limitations in Activities of Daily Living (ADLs) or Instrumental Activities of Daily Living (IADLs), with only a small number reporting one or more limitations. Depressive symptoms are generally low, with most individuals reporting between 0 to 2 symptoms. A majority have health insurance coverage, and more individuals have received a flu vaccination than those who have not. Exercise frequency is fairly balanced, with a slightly larger portion not exercising three or more times per week. Most individuals do not use tobacco, with a smaller group indicating tobacco use.

![Histplot1](https://github.com/user-attachments/assets/176d763f-6623-47c2-89f3-e4fd6400c56c)

The distribution of the composite score, which aggregates various health and lifestyle domains. Analyzing this score can reveal patterns or trends in overall health across the population in your dataset.

![barplot1](https://github.com/user-attachments/assets/f909f000-1fb9-42b4-84bd-7b1b6a40d3ba)

![lineplot1](https://github.com/user-attachments/assets/9c4a00ec-108e-4848-97d9-7d647bc1128f)

The graph shows a trend of the "Average Composite Score" over the years, with a noticeable decline from 2016 to 2021. The graph shows a consistent downward trend from 2016 to 2021. The average composite score decreased from above 160 in 2016 to around 155 in 2021, indicating a noticeable decline over these five years. The decline appears linear, with no major fluctuations or reversals in the trend, suggesting that this decrease might be a result of systemic or gradual changes rather than sudden or isolated events.

![stacked1](https://github.com/user-attachments/assets/57640427-57b7-40e8-aaff-9c042d75c5d8)

- **Health Insurance by Marital Status**:
  - Most individuals who are "Married or in civil union" have health insurance, with a smaller proportion lacking coverage.
  - Among "Widowed" individuals, many have insurance, though the uninsured proportion is higher compared to married individuals.
  - "Separated or divorced" and "Single" individuals have lower overall counts, with insurance coverage less common in these groups compared to married individuals.
    
![outliers boxplot1](https://github.com/user-attachments/assets/874614fe-5ffd-453b-a599-2e8117fc2664)
![outliers boxplot2](https://github.com/user-attachments/assets/85a1a1dd-b92b-427b-9715-3059f636c868)

The analysis reveals a positive correlation between education level and composite scores, with higher education generally leading to better performance. Individuals with no education have the lowest median scores and high variability, while those with 1-5 years of education show slightly improved but still low scores. Median scores increase significantly at 6 and 7-9 years of education, with the highest scores observed in the 10+ years category, accompanied by a broader range and notable outliers. These outliers, particularly in higher education groups, highlight variability influenced by individual factors such as socio-economic background or quality of education, suggesting that while education is a strong predictor of performance, it is not the sole determinant.

![pairplot1](https://github.com/user-attachments/assets/cca33c99-8f90-41d0-a797-faf692611229)

### Insights from Histograms and Scatter Plot Relationships

- **Variable Distributions**:
  - Histograms reveal that:
    - The composite score follows a bell-shaped distribution, with most values around the middle range.
    - Limitation variables (ADLs, IADLs, mobility, depressive symptoms) are heavily skewed, indicating most individuals experience few limitations.
  
- **Correlations**:
  - Scatter plots suggest:
    - A negative correlation between composite scores and limitations, where higher limitations are associated with lower scores.
    - Interrelationships among limitation variables, showing that limitations often co-occur.

- **Key Observations**:
  - Individuals with higher physical or mental limitations generally have lower composite scores.
  - Most data points cluster at low limitation values, showing that limitations are uncommon for many individuals.

- **Implications**:
  - Limitations negatively affect composite performance, emphasizing the importance of addressing multiple limitations holistically to improve overall outcomes.

## Modelling

## Evaluation

## Conclusion

## Recommendations

## Possible next steps







