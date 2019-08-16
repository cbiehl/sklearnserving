import sys
import os
import shutil
import time
import traceback

from flask import Flask, request, jsonify
from functools import wraps
import pandas as pd
import joblib

def authenticate(user, pw):
    if user == 'superuser' and pw == 'marshmallowcake':
        return True

    return False

def login_required(whatever):
    @wraps(whatever)
    def wrapped_stuff(**kwargs):
        if request.authorization and authenticate(request.authorization.username, request.authorization.password):
            return whatever(**kwargs)
        else:
            return 'Unauthorized', 401

    return wrapped_stuff


app = Flask(__name__)

# input data definition
training_data = 'data/data.csv'
columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class']
target = 'Class'

# model storage definition
model_directory = 'model'
model_file_name = '%s/model.pkl' % model_directory
model_columns_file_name = '%s/model_columns.pkl' % model_directory

# global variables
model_columns = None
model = None

@app.route('/modelStatus', methods=['GET'])
@login_required
def model_status():
    if model:
        return 'Model of type %s is trained and ready.' % model.__class__.__name__
    else:
        return 'No model is currently trained.'


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if model:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))

            # https://github.com/amirziai/sklearnflask/issues/3
            # Thanks to @lorenzori
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(model.predict(query))

            return jsonify({"prediction": prediction})

        except Exception as e:

            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print('There is no trained model - call the /training endpoint.')
        return 'No trained model', 400


@app.route('/train', methods=['GET'])
@login_required
def train():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import GradientBoostingRegressor

    modeltype = None
    try:
        modeltype = request.args.get('modeltype')
        if modeltype == 'regression':
            # set regression model class
            modelclass = GradientBoostingRegressor
        else:
            # set classification model class
            modeltype = 'classification'
            modelclass = GradientBoostingClassifier
    except:
        modeltype = 'classification'
        modelclass = GradientBoostingClassifier

    shuffle = None
    try:
        shuffle = request.args.get('shuffle')
    except:
        shuffle = True

    df = pd.read_csv(training_data)
    df_ = df[columns]

    if shuffle is True:
        df_ = df_.sample(frac=1).reset_index(drop=True)

    categoricals = []  # going to one-hot encode categorical variables

    for col, col_type in df_.dtypes.items():
        if col != target:
            if col_type == 'O':
                categoricals.append(col)
            else:
                df_[col].fillna(0, inplace=True)  # fill NA's with 0 for ints/floats, too generic

    # get_dummies effectively creates one-hot encoded variables
    df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=False)

    x = df_ohe[df_ohe.columns.difference([target])]
    y = df_ohe[target]

    # capture a list of columns that will be used for prediction
    global model_columns
    model_columns = list(x.columns)
    joblib.dump(model_columns, model_columns_file_name)

    global model
    model = modelclass()
    start = time.time()
    print('Training model of type %s' % model.__class__.__name__)
    print('Columns:', model_columns)
    print('Target variable:', target)
    model.fit(x, y)

    joblib.dump(model, model_file_name)

    message1 = 'Trained in %.2f seconds' % (time.time() - start)
    message2 = 'Mean training set accuracy: %s' % model.score(x, y)
    return_message = 'Success. \n{0}. \n{1}.'.format(message1, message2)
    return return_message


@app.route('/deleteModel', methods=['DELETE'])
@login_required
def delete_model():
    try:
        shutil.rmtree('model')
        os.makedirs(model_directory)
        return 'Current model successfully deleted.'

    except Exception as e:
        print(str(e))
        return 'Could not remove and recreate the model directory.', 500


# start flask server
try:
    port = int(sys.argv[1])
except Exception as e:
    port = 5000

try:
    model = joblib.load(model_file_name)
    print('Model of type %s loaded' % model.__class__.__name__)
    model_columns = joblib.load(model_columns_file_name)
    print('Model columns loaded')

except Exception as e:
    print("Model not trained")
    model = None

app.run(host='0.0.0.0', port=port, debug=True)
