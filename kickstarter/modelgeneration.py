import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from category_encoders import OrdinalEncoder
import pickle
import joblib

df=pd.read_csv('kickstarter/ks-projects-201801.csv',
               parse_dates=['deadline','launched'],
               usecols=['name','main_category','launched',
                        'deadline','currency','goal',
                        'pledged','backers','state'])
def wrangle(df):
    df['campaign_length']=((df['deadline'])-(df['launched'])).dt.days
    df = df.rename(columns={'main_category': 'category'})
    # if live and went passed the goal -> successful
    cond=(df['state'] == 'live') & (df['pledged']> df['goal'])
    df.loc[cond,'state']='successful'
    # canceled and suspended are failed
    df.loc[df['state'].str.contains('canceled'),'state']='failed'
    df.loc[df['state'].str.contains('suspended'),'state']='failed'
    df=df.drop(df[(df['state']=='live') | (df['state']=='undefined')].index)
    
    # df.loc[(df['state']=='successful'), 'state']=1
    # df.loc[(df['state']=='failed'), 'state']=0
    
    df['goal']=  df['goal'].astype('int')
    df['pledged']=df['pledged'].astype('int')
    
    df['launched']=df['launched'].dt.month_name()
    df=df.drop(columns='deadline')
    df['name_char_length']=[len(str(x)) for x in df['name']]
    df=df.drop(columns=['name','pledged'])
    df=df.dropna()
    df.reset_index(drop=True)
    df['currency']=df['currency'].str.lower()
    df['category']=df['category'].str.lower()
    df['launched']=df['launched'].str.lower()
    df.loc[df['category'].str.contains('film & video'),'category']='film_and_video'
    return df

df=wrangle(df)
# df.to_csv('cleaned_kickstarter.csv',index=False)

X=df.drop(columns='state')
y=df['state']

X_train, X_test, y_train, y_test=train_test_split(X,y, random_state=42,test_size=0.3)

model=make_pipeline(
                    OrdinalEncoder(),
                    SimpleImputer(strategy="mean"),
                    RandomForestClassifier(max_depth=10,n_estimators=150)
                    )

model.fit(X_train,y_train)

# pickle.dump(model,open('rf_model_pickle_2','wb'))

# training_acc = model.score(X_train, y_train)
# print(training_acc)
# test_acc = model.score(X_test, y_test)
# print(test_acc)

# joblib.dump(model, "./random_forest_compressed.joblib", compress=3)