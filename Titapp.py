#import Libraries
import numpy as np
import os
from flask import Flask, request, jsonify, render_template
import pickle


Titapp = Flask(__name__, template_folder='./template', static_folder='../static')

#load saved data
def load_model():
    return pickle.load(open('Titanic.pkl', 'rb'))


model = load_model()

#home page
@Titapp.route('/', methods=['GET', 'POST'])
def home():
    '''
        for rendering results on the HTML GUI
    '''
    if request.method == "POST":
        labels = ["survived", "didn't survive"]
        
        features = [float(str(x)) for x in request.form.values()]
        
        values = [np.array(features)]
        
        prediction = model.predict(values)
        
        result = labels[prediction[0]]
        
        output = 'You {}'.format(result)
        print(output)
        return render_template('index.html', output=output)
    
    return render_template('index.html')

   


if __name__=='__main__':
    #port=int(os.environ.get('PORT',5000))
    Titapp.run(debug=True)
    
    
