import pickle 
# import joblib
# import pandas as pd

# Placeholders to initialize model
# category = 'Fashion'
# currency = 'USD'
# goal = 1000
# month = 'july'
# backers= 10
# campaign_length = 20
# campaign_name_length = 20


# input_data = pd.DataFrame({'category':[category],'currency':[currency],
#            'goal':[goal],'launched':[month],'backers':[backers],
#            'campaign_length':[campaign_length],'name_char_length':[campaign_name_length]})

def model(input_data):
    rf = pickle.load(open('rf_model_pickle_2.pkl', 'rb'))
    pred = rf.predict(input_data)
    return pred

# print(model(input_data))
