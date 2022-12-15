In this project, we looked to answer the question- What Makes a car "sellable"? 

We took our data from Kaggle.com and cleaned the data to make it easier to model. Note, our data was in fact cross-sectional (taken at one point in time).
Reference the 542_Project.Rmd for me details on our models. In this file we added our two best model (GLM and KNN) as well as our two 
worst performing models (Decision Tree and SVM Linear)

For more information on the AutoML- reference the AutoML.R file. Auto ML found GBM (Gradient Boosted Model) to have the best accuracy in predicting 
car sellability (1 for sold or 0 for not sold). Accuracy here is upwards of 96%.
