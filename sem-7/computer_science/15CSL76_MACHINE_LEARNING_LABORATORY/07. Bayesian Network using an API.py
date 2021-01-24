import sys
import numpy as np
import pandas as pd

df = pd.read_csv('Datasets/HeartDisease.csv')
df.drop(['ca','slope','thal','oldpeak'],axis=1,inplace=True)
df.replace('?',np.nan,inplace=True)

from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel

model = BayesianModel([
    ('age','trestbps'),
    ('age'.'fbs'),
    ('sex','trestbps'),
    ('exang','trestbps'),
    ('trestbps','heartdisease'),
    ('fbs','heartdisease'),
    ('heartdisease','restecg'),
    ('heartdisease','thalach'),
    ('heartdisease','chol')
    ])

model.fit(df,estimator=MaximumLikelihoodEstimator)

print(model.get_cpds('age'))
print(model.get_cpds('chol'))
print(model.get_cpds(sex))
model.get_independencies()

from pgmpy.inference import VariableElimination

HeartDisease_infer = VariableElimination(model)

q = HeartDisease_infer.query(variables=['heartdisease'],evidence={'age':28})
print(q['heartdisease'])

q = HeartDisease_infer.query(variables=['heartdisease'],evidence={'chol':100})
print(q['heartdisease'])
