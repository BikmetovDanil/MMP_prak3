import os
import pickle
import pandas as pd
from ensembles import RandomForestMSE, GradientBoostingMSE

from collections import namedtuple
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, flash, request, url_for
from flask import render_template, redirect
from flask import Markup
from flask import send_from_directory
from werkzeug.utils import secure_filename

from wtforms.validators import DataRequired
from wtforms import StringField, SubmitField


app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
upload_path = os.path.join(app.instance_path, 'uploads')
ALLOWED_EXTENSIONS = set(['csv'])
app.config['UPLOAD_FOLDER'] = upload_path
data_path = './../data'
Bootstrap(app)
history = []


class Iteration:
    iteration = 0
    loss = 0


class Data:
    alg = 'rf'
    n = 100
    fss = None
    md = None
    lr = 0.1


train = None
target = None
model = None
cd = Data()


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/model', methods=['GET', 'POST'])
def prepare_model():
    global train, target
    if request.method == 'POST':
        ftrain = request.files['train_file']
        ftarget = request.files['target_file']
        ftrain_name = ''
        ftarget_name = ''
        if ftrain and ftarget:
            if allowed_file(ftrain.filename) and allowed_file(ftarget.filename):
                ftrain_name = secure_filename(ftrain.filename)
                ftarget_name = secure_filename(ftarget.filename)
                ftrain.save(os.path.join(app.config['UPLOAD_FOLDER'], ftrain_name))
                ftarget.save(os.path.join(app.config['UPLOAD_FOLDER'], ftarget_name))
            else:
                flash('Файл должен иметь расширение .csv', category='error')
                return render_template('model.html', train=train, target=target)
        else:
            flash('Надо заполнить поля', category='error')
            return render_template('model.html', train=train, target=target)
        train = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], ftrain_name))
        target = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], ftarget_name))
        target = target.iloc[:, -1]
        if type(train) != 'numpy.ndarray':
            train = train.to_numpy()
        if type(target) != 'numpy.ndarray':
            target = target.to_numpy()
        feat_col = []
        for j in range(train.shape[1]):
            t = str
            try:
                t = type(float(train[1, j]))
            except:
                pass
            if t == float:
                feat_col += [j]
        train = train[:, feat_col]
        cd.alg = request.form['alg']
        if cd.alg == 'rf':
            return render_template('rf.html', train=train, target=target)
        else:
            return render_template('gb.html', train=train, target=target)
    return render_template('model.html', train=train, target=target)


@app.route('/rf', methods=['GET', 'POST'])
def random_forest():
    global model, train, target
    if request.method == 'POST':
        cd.n = int(request.form['n_estimators'])
        if 'fss_default' in request.form and request.form['fss_default'] == 'on':
            cd.fss = None
        else:
            cd.fss = int(request.form['fss'])
        if 'wo_depth' in request.form and request.form['wo_depth'] == 'on':
            cd.md = None
        else:
            cd.md = int(request.form['max_depth'])
        model = RandomForestMSE(n_estimators=cd.n, max_depth=cd.md,
                                feature_subsample_size=None)
        loss = model.fit(train, target)
        history = []
        for i, elem in enumerate(loss):
            message = Iteration()
            message.iteration = i
            message.loss = round(elem, 3)
            history.append(message)
        return render_template('predict.html', cd=cd, train=train, target=target, model=model, history=history)
    return render_template('rf.html')


@app.route('/gb', methods=['GET', 'POST'])
def gradient_boosting():
    global model, train, target
    if request.method == 'POST':
        cd.n = int(request.form['n_estimators'])
        if 'fss_default' in request.form and request.form['fss_default'] == 'on':
            cd.fss = None
        else:
            cd.fss = int(request.form['fss'])
        if 'wo_depth' in request.form and request.form['wo_depth'] == 'on':
            cd.md = None
        else:
            cd.md = int(request.form['max_depth'])
        cd.lr = float(request.form['learning_rate'])
        model = GradientBoostingMSE(n_estimators=cd.n, max_depth=cd.md,
                                    feature_subsample_size=cd.fss, learning_rate=cd.lr)
        loss = model.fit(train, target)
        history = []
        for i, elem in enumerate(loss):
            message = Iteration()
            message.iteration = i
            message.loss = round(elem, 3)
            history.append(message)
        return render_template('predict.html', cd=cd, train=train, target=target, model=model, history=history)
    return render_template('gb.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return render_template('predict.html', cd=cd, train=train, target=target, model=model, history=history)


@app.route('/prediction.csv', methods=['GET', 'POST'])
def download():
    global model
    if request.method == 'POST':
        fpred = request.files['predict_file']
        fpred_name = ''
        if fpred:
            if allowed_file(fpred.filename):
                fpred_name = secure_filename(fpred.filename)
                fpred.save(os.path.join(app.config['UPLOAD_FOLDER'], fpred_name))
            else:
                #flash('Файл должен иметь расширение .csv', category='error')
                return render_template('model.html', train=train, target=target)
        else:
            #flash('Надо заполнить поле', category='error')
            return render_template('model.html', train=train, target=target)
        pred = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], fpred_name))
        if type(pred) != 'numpy.ndarray':
            pred = pred.to_numpy()
        feat_col = []
        for j in range(pred.shape[1]):
            t = str
            try:
                t = type(float(train[1, j]))
            except:
                pass
            if t == float:
                feat_col += [j]
        pred = pred[:, feat_col]
        p = model.predict(pred)
        submit = pd.DataFrame({"Answer": p})
        submit.to_csv("prediction.csv", index=True)
        path = os.path.join(app.root_path, "prediction.csv")
        return send_from_directory(app.root_path, "./prediction.csv")
    return render_template('predict.html', cd=cd, train=train, target=target, model=model, history=history)
