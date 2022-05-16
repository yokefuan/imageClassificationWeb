import os 
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

from keras.models import load_model 
from keras.backend import set_session
from skimage.transform import resize 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 

print("Loading model")
# to run tensorflow in cpu
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Load the model 
global model 
model = load_model('LocalFoodRGBWorking.h5')

# This is the default main page
@app.route('/', methods=['GET', 'POST']) 
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        # Save the uploaded image file for prediction
        file.save(os.path.join('static/uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')

@app.route('/prediction/<filename>') 
def prediction(filename):
    #Step 1 - Read the uploaded image file for prediction
    my_image = plt.imread(os.path.join('static/uploads', filename))
    
    #Step 2 - Resize the image to 32 X 32
    my_image_re = resize(my_image, (32,32,3))
    
    #Step 3 - Predict the uploaded image file
    model.run_eagerly=True  
    probabilities = model.predict(np.array( [my_image_re,] ))[0,:]
    print(probabilities)
    
    #Step 4 - Define the categories of classes that we have 
    number_to_class = ['Chicken Rice', 'Hokkien Prawn Noodles', 'Ice Kacang']
    # Sort the probabilities of the prediction
    index = np.argsort(probabilities)
    # Convert the prediction to % and round off to 2 decmial places for display
    predictions = {
      "class1":number_to_class[index[2]],
      "prob1":round(probabilities[index[2]]*100,2),
    }
    
    #Step 5 - return the prediction and the file name of the image as we need to display the
    #uploaded image
    return render_template('predict.html', predictions=predictions, filename=filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
