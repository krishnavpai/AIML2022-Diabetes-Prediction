from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('final_model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return ""


@app.route('/predict', methods=['POST'])
def predict():
    pregnancies = int(request.form.get('pregnancies'))
    dpf = float(request.form.get('dpf'))
    glucose = float(request.form.get('glucose'))
    bp = float(request.form.get('bp'))
    skin = float(24)
    insulin = float(request.form.get('insulin'))

    bmi = float(request.form.get('bmi'))

    age = int(request.form.get('age'))
    # response = {
    #     'pregnancies': pregnancies,
    #     'glucose': glucose,
    #     'bp': bp,
    #     'insulin': insulin,
    #     'bmi': bmi,
    #     'age': age
    # }
    form_input = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])

    ans = model.predict(form_input)[0]
    ans = format(ans, pregnancies, dpf, glucose, bp, skin, insulin, bmi, age)
    return jsonify({'prediction': str(ans)})


'''
def find_best_model(X, y):
    models = {
        'logistic_regression': {
            'model': LogisticRegression(solver='lbfgs', multi_class='auto'),
            'parameters': {
                'C': [1,5,10]
               }
        },
        
        'decision_tree': {
            'model': DecisionTreeClassifier(splitter='best'),
            'parameters': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [5,10]
            }
        },
        
        'random_forest': {
            'model': RandomForestClassifier(criterion='gini'),
            'parameters': {
                'n_estimators': [10,15,20,50,100,200]
            }
        },
        
        'svm': {
            'model': SVC(gamma='auto'),
            'parameters': {
                'C': [1,10,20],
                'kernel': ['rbf','linear']
            }
        }

    }
    
    scores = [] 
    cv_shuffle = ShuffleSplit(n_splits=5, test_size=0.20, random_state=0)
        
    for model_name, model_params in models.items():
        gs = GridSearchCV(model_params['model'], model_params['parameters'], cv = cv_shuffle, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': model_name,
            'best_parameters': gs.best_params_,
            'score': gs.best_score_
        })
        
    return pd.DataFrame(scores, columns=['model','best_parameters','score'])
'''


def format(ans, pregnancies, dpf, glucose, bp, skin, insulin, bmi, age):
    if dpf > 0.5:
         ans = "You don't have diabetes"
    else:
        ans = "You might have diabetes"
    return ans



if __name__ == '__main__':
    app.run(debug=True)
