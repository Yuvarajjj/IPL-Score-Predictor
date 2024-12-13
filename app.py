from flask import Flask, render_template, request, send_file
from flask_sqlalchemy import SQLAlchemy
import pickle
import numpy as np
import matplotlib.pyplot as plt
import io

# Load the Linear Regression model
filename = 'imp.pkl'
regressor = pickle.load(open(filename, 'rb'))

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
db = SQLAlchemy(app)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    batting_team = db.Column(db.String(100))
    bowling_team = db.Column(db.String(100))
    venue = db.Column(db.String(100))
    overs = db.Column(db.Float)
    runs = db.Column(db.Integer)
    wickets = db.Column(db.Integer)
    runs_in_prev_5 = db.Column(db.Integer)
    wickets_in_prev_5 = db.Column(db.Integer)
    predicted_score = db.Column(db.Integer)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    temp_array = []

    if request.method == 'POST':
        # Encoding batting team
        batting_team = request.form['batting-team']
        batting_teams = ['Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab', 'Kolkata Knight Riders', 
                         'Mumbai Indians', 'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad']
        temp_array += [1 if team == batting_team else 0 for team in batting_teams]

        # Encoding bowling team
        bowling_team = request.form['bowling-team']
        temp_array += [1 if team == bowling_team else 0 for team in batting_teams]

        # Encoding venue
        venue = request.form['venue']
        venues = ['M Chinnaswamy Stadium', 'Eden Gardens', 'Feroz Shah Kotla', 'MA Chidambaram Stadium, Chepauk',
                  'Punjab Cricket Association Stadium, Mohali', 'Wankhede Stadium', 'Sawai Mansingh Stadium',
                  'Rajiv Gandhi International Stadium, Uppal']
        temp_array += [1 if v == venue else 0 for v in venues]

        # Adding remaining inputs
        overs = float(request.form['overs'])
        runs = int(request.form['runs'])
        wickets = int(request.form['wickets'])
        runs_in_prev_5 = int(request.form['runs_in_prev_5'])
        wickets_in_prev_5 = int(request.form['wickets_in_prev_5'])

        temp_array += [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5]

        # Ensure correct number of features
        if len(temp_array) != 29:
            return "Feature length mismatch. Please check the input data."

        data = np.array([temp_array])
        my_prediction = int(regressor.predict(data)[0])

        # Save the prediction to the database
        new_prediction = Prediction(
            batting_team=batting_team,
            bowling_team=bowling_team,
            venue=venue,
            overs=overs,
            runs=runs,
            wickets=wickets,
            runs_in_prev_5=runs_in_prev_5,
            wickets_in_prev_5=wickets_in_prev_5,
            predicted_score=my_prediction
        )
        db.session.add(new_prediction)
        db.session.commit()

        return render_template('result.html', lower_limit=my_prediction - 10, upper_limit=my_prediction + 5, prediction_id=new_prediction.id)

@app.route('/plot/<int:prediction_id>')
def plot_prediction(prediction_id):
    prediction = Prediction.query.get(prediction_id)
    if not prediction:
        return {'message': 'Prediction not found'}, 404
    
    # Create the plot
    overs = np.array([prediction.overs])
    predicted_score = np.array([prediction.predicted_score])

    plt.figure()
    plt.plot(overs, predicted_score, 'bo-', label='Predicted Score')
    plt.xlabel('Overs') 
    plt.ylabel('Predicted Score')
    plt.title(f'Prediction for {prediction.batting_team} vs {prediction.bowling_team} at {prediction.venue}')
    plt.legend()

    # Save the plot to a bytes object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Return the image
    return send_file(img, mimetype='image/png')
@app.route('/history')
def history():
    predictions = Prediction.query.all()
    return render_template('history.html', predictions=predictions)
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create database tables
    app.run(debug=True)


