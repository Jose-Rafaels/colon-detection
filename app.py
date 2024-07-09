from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from ml.model import inceptionV3 , vgg19


app = Flask(__name__,static_url_path='', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif','jfif'}


global file_uploaded
file_uploaded =  False
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def remove_file() : 
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    

@app.route('/', methods=['POST','GET'])
def index():
    res = {}
    if request.method == 'POST':
        if 'file' not in request.files:
            return ('No file part')
            
        file = request.files['file']
        if file.filename == '':
            return ('No selected file')
           
        if file and allowed_file(file.filename):
            remove_file()
            filename = secure_filename(file.filename)
            ext = filename.rsplit('.', 1)[1].lower()
            new_filename = f"{filename}.{ext}"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], new_filename))
            file_path  = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
            result_vgg19 , percentage_vgg19 = vgg19(file_path)
            result_inceptionV3, percentage_inceptionV3 = inceptionV3(file_path)
            res['image'] = 'uploads/'+ new_filename
            res['percentage_vgg19'] = "{:.2f}".format(percentage_vgg19)+ "%"
            res['result_vgg19'] = result_vgg19
            res['percentage_inceptionV3'] = "{:.2f}".format(percentage_inceptionV3) + "%"
            res['result_inceptionV3'] = result_inceptionV3
            
       
        return render_template('index.html' , data = res)
    else : 
        return render_template('index.html' , data = False )

@app.route('/info', methods=['GET'])
def info():
    return render_template('info.html')
if __name__ == '__main__':
    app.run(debug=True)
