#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('conda', 'install -c conda-forge shap')


# In[ ]:


import pandas as pd


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
submission=pd.read_csv('sample_submission.csv')

train.head()

test.head()

target='Transported'

features=train.columns.drop([target])

from sklearn.model_selection import train_test_split # You are going to be using this repeatedly

train, val = train_test_split(train, random_state=42) # In class example


X_train=train[features]
X_val=val[features]
y_train=train[target]
y_val=val[target]
X_test=test[features]

import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

#set up a pipeline 

pipeline = make_pipeline(
    ce.OneHotEncoder(use_cat_names=True), 
    SimpleImputer(strategy='mean'), 
    DecisionTreeClassifier(random_state=91)
    )
#fit on train
pipeline.fit(X_train, y_train)


#score on train, val 
print('Train Accuracy', pipeline.score(X_train, y_train))
print('Val Accuracy', pipeline.score(X_val, y_val))


#predict on test 
y_pred= pipeline.predict(X_test)




test.columns


y_pred = pipeline.predict(X_test)










#Test Preprocess



y_pred


submission = test[['PassengerId']].copy() #get the id and the prediction 
submission['Transported'] = y_pred
submission.to_csv('titanic-submission-10.csv', index=False)






# In[ ]:


train.to_csv('train.csv',index=False,header=False)
val.to_csv('validation.csv',index=False,header=False)


# In[ ]:


get_ipython().system('pip install --upgrade category_encoders')


# In[ ]:





# In[ ]:


import sagemaker,boto3,os
bucket=sagemaker.Session().default_bucket()
prefix="demo-sagemaker-xgboost-Titanic-Spaceship"

boto3.Session().resource('s3').Bucket(bucket).Object(
    os.path.join(prefix,'train.csv')).upload_file('train.csv')
boto3.Session().resource('s3').Bucket(bucket).Object(
        os.path.join(prefix,'validation.csv')).upload_file('validation.csv')


# In[ ]:


get_ipython().system(' aws s3 ls {bucket}/{prefix}/data --recursive')


# In[ ]:


region=sagemaker.Session().boto_region_name
print("AWS Region :{}",format(region))
role=sagemaker.get_execution_role()
print("RoleArn: {}",format(role))


# In[ ]:


sagemaker.__version__


# In[ ]:


s3_output_location = 's3://{}'.format(bucket)


# In[ ]:


pip install smdebug --upgrade 


# In[ ]:


from sagemaker.debugger import Rule,rule_configs
from sagemaker.session import TrainingInput

s3_ouput_location='s3://{}/{}/{}'.format(bucket,prefix,'model')
container=sagemaker.image_uris.retrieve(framework="xgboost",region=boto3.Session().region_name,version="1.2-1")
print(container)
model=sagemaker.estimator.Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m4.xlarge',
    volume_size=5,
    output_path=s3_output_location,
    sagemaker_session=sagemaker.Session(),
    rules=[Rule.sagemaker(rule_configs.create_xgboost_report())]
)
    
    


# In[ ]:


pip install sagemaker --upgrade


# In[ ]:


model.set_hyperparameters(
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsmple=0.7,
    objective='binary:logistic',
    num_round=1000
)
    


# In[ ]:


from sagemaker.session import TrainingInput
train_input=TrainingInput(
    "s3://{}/{}/{}".format(bucket,prefix,"train.csv"),content_type="csv"
)
validation_input=TrainingInput(
    "s3://{}/{}/{}".format(bucket,prefix,"validation.csv"),content_type="csv"
)
s3_output_location = 's3://{}'.format(bucket)



# In[ ]:


model.fit({"train":train_input,"validation":validation_input},wait=True)


# In[ ]:


rule_output_path=model.output_path + "/" + model.latest_training_job.name + "/rule-output"
get_ipython().system(' aws s3 ls {rule_ouput_path} --recursive')


# In[ ]:


from IPython.display import FileLink,FileLinks
display("Click link below to view the training report",FileLink("CreateXgboostReport/xgboost_report.html"))


# In[ ]:


profiler_report_name=[rule["RuleConfigurationName"]
                      for rule in xgb_model.latest_training_job.rule_job_summary()
                      if "Profiler" in rule["RuleConfigurationName"]][0]
profiler_report_name
display("Click link below to view the profiler report",FileLink(profiler_report_name+"/profiler-output/profiler-report.html"))


# In[ ]:




