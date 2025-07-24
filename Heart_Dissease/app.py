from flask import Flask, render_template, request

app = Flask(__name__)

# Results of the models
results = {
    'Logistic Regression': {
        'accuracy': 0.56,
        'report': '''\
              precision    recall  f1-score   support

           0       0.69      0.87      0.77        75
           1       0.56      0.56      0.56        54
           2       0.17      0.08      0.11        25
           3       0.27      0.23      0.25        26
           4       0.00      0.00      0.00         4

    accuracy                           0.56       184
   macro avg       0.34      0.35      0.34       184
weighted avg       0.51      0.56      0.53       184
'''
    },
    'Decision Tree': {
        'accuracy': 0.57,
        'report': '''\
              precision    recall  f1-score   support

           0       0.70      0.88      0.78        75
           1       0.66      0.50      0.57        54
           2       0.22      0.16      0.19        25
           3       0.30      0.27      0.29        26
           4       0.12      0.25      0.17         4

    accuracy                           0.57       184
   macro avg       0.40      0.41      0.40       184
weighted avg       0.56      0.57      0.55       184
'''
    },
    'Random Forest': {
        'accuracy': 0.61,
        'report': '''\
              precision    recall  f1-score   support

           0       0.75      0.95      0.84        75
           1       0.59      0.54      0.56        54
           2       0.29      0.20      0.24        25
           3       0.36      0.31      0.33        26
           4       0.00      0.00      0.00         4

    accuracy                           0.61       184
   macro avg       0.40      0.40      0.39       184
weighted avg       0.57      0.61      0.59       184
'''
    },
    'KNN': {
        'accuracy': 0.58,
        'report': '''\
              precision    recall  f1-score   support

           0       0.73      0.91      0.81        75
           1       0.49      0.46      0.48        54
           2       0.19      0.12      0.15        25
           3       0.39      0.35      0.37        26
           4       1.00      0.25      0.40         4

    accuracy                           0.58       184
   macro avg       0.56      0.42      0.44       184
weighted avg       0.54      0.58      0.55       184
'''
    },
    'SVM': {
        'accuracy': 0.55,
        'report': '''\
              precision    recall  f1-score   support

           0       0.68      0.89      0.77        75
           1       0.46      0.52      0.49        54
           2       0.43      0.12      0.19        25
           3       0.22      0.15      0.18        26
           4       0.00      0.00      0.00         4

    accuracy                           0.55       184
   macro avg       0.36      0.34      0.33       184
weighted avg       0.50      0.55      0.51       184
'''
    },
    'XGBoost': {
        'accuracy': 0.61,
        'report': '''\
              precision    recall  f1-score   support

           0       0.72      0.91      0.80        75
           1       0.67      0.59      0.63        54
           2       0.25      0.16      0.20        25
           3       0.38      0.31      0.34        26
           4       0.25      0.25      0.25         4

    accuracy                           0.61       184
   macro avg       0.45      0.44      0.44       184
weighted avg       0.58      0.61      0.59       184
'''
    },
    'Naive Bayes': {
        'accuracy': 0.40,
        'report': '''\
              precision    recall  f1-score   support

           0       0.89      0.73      0.80        75
           1       0.88      0.28      0.42        54
           2       0.00      0.00      0.00        25
           3       0.00      0.00      0.00        26
           4       0.04      1.00      0.07         4

    accuracy                           0.40       184
   macro avg       0.36      0.40      0.26       184
weighted avg       0.62      0.40      0.45       184
'''
    },
    'Gradient Boosting': {
        'accuracy': 0.62,
        'report': '''\
              precision    recall  f1-score   support

           0       0.69      0.88      0.78        75
           1       0.59      0.59      0.59        54
           2       0.47      0.28      0.35        25
           3       0.50      0.35      0.41        26
           4       0.50      0.25      0.33         4

    accuracy                           0.62       184
   macro avg       0.55      0.47      0.49       184
weighted avg       0.60      0.62      0.60       184
'''
    },
    'AdaBoost': {
        'accuracy': 0.52,
        'report': '''\
              precision    recall  f1-score   support

           0       0.72      0.80      0.76        75
           1       0.42      0.46      0.44        54
           2       0.13      0.08      0.10        25
           3       0.33      0.27      0.30        26
           4       0.20      0.25      0.22         4

    accuracy                           0.52       184
   macro avg       0.36      0.37      0.36       184
weighted avg       0.49      0.52      0.50       184
'''
    },
    'Extra Trees': {
        'accuracy': 0.61,
        'report': '''\
              precision    recall  f1-score   support

           0       0.74      0.91      0.81        75
           1       0.66      0.54      0.59        54
           2       0.31      0.16      0.21        25
           3       0.32      0.38      0.35         26
           4       0.25      0.25      0.25         4

    accuracy                           0.61       184
   macro avg       0.46      0.45      0.44       184
weighted avg       0.59      0.61      0.59       184
'''
    }
}

# Find the best model
best_model = max(results, key=lambda x: results[x]['accuracy'])

@app.route('/')
def home():
    return render_template('index.html', best_model=best_model, best_accuracy=results[best_model]['accuracy'])

@app.route('/results')
def get_results():
    return render_template('results.html', results=results)

@app.route('/model/<model_name>')
def get_model_details(model_name):
    if model_name in results:
        return render_template('model.html', model_name=model_name, report=results[model_name]['report'], accuracy=results[model_name]['accuracy'])
    else:
        return f"Model {model_name} not found", 404

if __name__ == '__main__':
    app.run(debug=True)
