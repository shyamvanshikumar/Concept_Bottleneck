import json
import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from infer.inference import predict_label, predict_concept
#from infer.arch import MLP


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
most_recent_file = None

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        global most_recent_file
        most_recent_file = filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image successfully uploaded and displayed below')
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        concepts_values, concepts_dist = predict_concept(img_path)
        labels_dist = predict_label(concepts_values)
        return render_template('index.html', filename=filename, concepts=concepts_dist, labels=labels_dist)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/rerun', methods=['POST'])
def rerun():
    concepts_json = request.form.get("features")
    concepts_list = json.loads(concepts_json)

    concepts_dict = dict()
    for i in range(len(concepts_list)):
        try:
            concepts_dict[concepts_list[i].get('original_id')] = concepts_list[i]['probability']
        except KeyError:
            print("table entry not edited correctly")
    
    concepts_dict = {k: v for k, v in sorted(concepts_dict.items(), key=lambda item: item[0])}
    concepts_values = list(concepts_dict.values())
    for i in range(len(concepts_values)):
        concepts_values[i] = float(concepts_values[i])
    labels_dist = predict_label(concepts_values)
    return render_template('index.html', filename=most_recent_file, concepts=concepts_list, labels=labels_dist)



if __name__ == '_main_':
    app.run(debug=True)