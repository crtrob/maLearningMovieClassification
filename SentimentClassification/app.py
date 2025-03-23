# Script containing Flask code for main review-predict website
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import numpy as np
import sqlite3 
import pickle
import os
# take vect var from vectorizer.py hashingvectorizer storage
from vectorizer import vect

# __name__ lets Flask know the HTML template folder is in the same directory as this program
app = Flask(__name__)

#### prepare classifier
# store current file directory into variable
cur_dir = os.path.dirname(__file__)
# open classifier with this stored directory and store in clf
clf = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'classifier.pkl'), 'rb'))
# create variable db as the path to the sqlite review database
db = os.path.join(cur_dir, 'reviews.sqlite')

# method to hashingvectorizer-transform a review, predict its review result and test its accuracy
def classify(document):
    # establishes what the integers of the review result mean
    label = {0: 'negative', 1: 'positive'}
    # uses hashingvectorizer transform on entered review to process 
    X = vect.transform([document])
    # use classifier-holder clf to predict the review result w/ label dict
    y = clf.predict(X)[0]
    # see probability of correct prediction using classifier-holder
    proba = np.max(clf.predict_proba(X))
    # return the predicted result and the probability 
    return label[y], proba

# method to update the classifier with the given review and result
def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])

# method to place any given review and result into the review_db database
def sqlite_entry(path, document, y):
    # establish connection into sqlite base in path, then create cursor variable
    conn = sqlite3.connect(path)
    c = conn.cursor()
    # insert the review and result into the review_db database
    c.execute("INSERT INTO review_db (review, sentiment, date) VALUES (?, ?, DATETIME('now'))",
              (document, y))
    # commit change to connection and close it
    conn.commit()
    conn.close()

#### Flask section of code
# creates text field to enter review and necessitates data entry of at least 15 chars for string
class ReviewForm(Form):
    moviereview = TextAreaField('', [validators.DataRequired(), validators.length(min = 15)])

# specifies URL which triggers index method,
@app.route('/')
# which calls out the text field ReviewForm and gives back the review form html
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form = form)

# specifies URL which triggers results method,
@app.route('/results', methods = ['POST'])
# which classifies the data in the validated review, then generates results html 
def results():
    # identical instantiation to that in index()
    form = ReviewForm(request.form)
    # if the review form is validated
    if request.method == 'POST' and form.validate():
        # store the text from the form
        review = request.form['moviereview']
        # classify the review
        y, proba = classify(review)
        # returns results.html template with review, review results and correctness
        # probability as extra parameters
        return render_template('results.html', content = review, prediction = y,
                               probability = round(proba*100, 2))
    # otherwise case, where it just retrieves the same old reviewform.html template & text form
    return render_template('reviewform.html', form = form)

# specifies URL which triggers feedback method,
@app.route('/thanks', methods = ['POST'])
# which processes the received review and prediction and adds it to classifier model and database
def feedback():
    # instantiate received review, the predicted result, and
    # whether the user clicked the 'correct' or 'incorrect' buttons
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']

    # backwards-traces the dictionary for review results
    inv_label = {'negative': 0, 'positive': 1}
    # use that bw-trace on the prediction
    y = inv_label[prediction]
    # if that doesn't work with the feedback,
    if feedback == 'Incorrect':
        # flips the binary values of the results with a binary-NOT
        y = int(not(y))
    # update classifier with received review and prediction
    train(review, y)
    # insert it into the review_db sqlite database
    sqlite_entry(db, review, y)
    # retrieves 'thanks.html' file
    return render_template('thanks.html')

# only run if file is directly run & not imported as script
if __name__ == '__main__':
    app.run()