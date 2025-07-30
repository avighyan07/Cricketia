import os
import re
import pandas as pd
import joblib
import nltk
import numpy as np
import networkx as nx

from flask import Flask, render_template, request
from backend.forms import InputForm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "avighyan"

# Load model and data
model = joblib.load("modelnew.joblib")
df = pd.read_csv("IPL dataset final.csv")

# ------------------------- HOME -------------------------
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html", title="Home")

# ------------------------- SUMMARIZER -------------------------
@app.route('/summarize', methods=['GET', 'POST'])
def summarize_text():
    summary = ""
    if request.method == 'POST':
        text = request.form.get("text")
        if text.strip():
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LsaSummarizer()
            summary_sentences = summarizer(parser.document, 3)
            summary = ' '.join(str(sentence) for sentence in summary_sentences)
        else:
            summary = "Please enter valid text to summarize."
    return render_template("summarize.html", summary=summary)

# 
@app.route("/predict", methods=["GET", "POST"])
def predict():
    form = InputForm()
    message = ""

    if form.validate_on_submit():
        x_new = pd.DataFrame(dict(
            batting_team=[form.batting_team.data],
            bowling_team=[form.bowling_team.data],
            city=[form.city.data],
            runs_left=[form.runs_left.data],
            balls_left=[form.balls_left.data],
            wickets=[form.wickets.data],
            total_runs_x=[form.total_runs_x.data],
            crr=[form.crr.data],
            rrr=[form.rrr.data]
        ))

        prediction_prob = model.predict_proba(x_new)[0]
        win_percentage = round(prediction_prob[1] * 100)
        winning_team = form.batting_team.data if prediction_prob[1] > prediction_prob[0] else form.bowling_team.data
        message = f"{winning_team} has a {win_percentage}% chance of winning."

    return render_template("predict.html", form=form, output=message)
# ------------------------- PLAYER COMPARISON -------------------------
@app.route('/compare', methods=['GET', 'POST'])
def compare():
    player_names = sorted(df['Player'].dropna().unique())

    p1_stats = None
    p2_stats = None
    p1 = p2 = None

    if request.method == 'POST':
        p1 = request.form.get('player1')
        p2 = request.form.get('player2')

        p1_data = df[df['Player'] == p1]
        p2_data = df[df['Player'] == p2]

        if not p1_data.empty and not p2_data.empty:
            p1_stats = p1_data.iloc[0].to_dict()
            p2_stats = p2_data.iloc[0].to_dict()

    return render_template('compare.html', players=player_names, p1_stats=p1_stats, p2_stats=p2_stats, p1=p1, p2=p2)

# ------------------------- PLAYER INFO -------------------------
@app.route("/cricketer_info")
def cricketer_info():
    search_query = request.args.get("search", "").lower()
    if search_query:
        filtered_df = df[df["Player"].str.lower().str.contains(search_query, na=False)]
    else:
        filtered_df = df

    players = filtered_df.to_dict(orient="records")
    return render_template("players.html", players=players, search_query=search_query)
@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if request.method == "POST":
        name = request.form.get("name")
        message = request.form.get("message")
        # For now, just print to console or store in a text/db
        print(f"Feedback received from {name}: {message}")
        return render_template("feedback.html", success=True)
    return render_template("feedback.html")

# ------------------------- IPL TRIVIA -------------------------
@app.route('/quiz_trivia')
def quiz_trivia():
    questions = [
        {"question": "Which team won the first IPL title in 2008?", "options": ["CSK", "RCB", "RR", "KKR"]},
        {"question": "Who has scored the most runs in IPL history?", "options": ["Virat Kohli", "Suresh Raina", "David Warner", "Rohit Sharma"]},
        {"question": "Which player has taken the most wickets in IPL history?", "options": ["Lasith Malinga", "Amit Mishra", "Dwayne Bravo", "Yuzvendra Chahal"]},
        {"question": "Which IPL team has won the most titles?", "options": ["CSK", "MI", "KKR", "SRH"]},
        {"question": "Who hit the fastest century in IPL history?", "options": ["Chris Gayle", "AB de Villiers", "David Warner", "Yusuf Pathan"]},
        {"question": "Which player has the highest individual score in an IPL match?", "options": ["Brendon McCullum", "Chris Gayle", "Virender Sehwag", "KL Rahul"]},
        {"question": "Which team holds the record for the highest total in an IPL match?", "options": ["RCB", "MI", "CSK", "SRH"]},
        {"question": "Who was the captain of MI when they won their first IPL title?", "options": ["Sachin Tendulkar", "Ricky Ponting", "Harbhajan Singh", "Rohit Sharma"]},
        {"question": "Which bowler has the best bowling figures in a single IPL match?", "options": ["Sohail Tanvir", "Alzarri Joseph", "Anil Kumble", "Sunil Narine"]},
        {"question": "Which Indian player won the first-ever Orange Cap?", "options": ["Sachin Tendulkar", "Virat Kohli", "Gautam Gambhir", "Shaun Marsh"]}
    ]

    correct_answers = ["RR", "Virat Kohli", "Yuzvendra Chahal", "CSK", "Chris Gayle", "Chris Gayle", "RCB", "Rohit Sharma", "Alzarri Joseph", "Shaun Marsh"]
    return render_template("trivia.html", questions=questions, correct_answers=correct_answers)

# ------------------------- RUN -------------------------
if __name__ == "__main__":
    app.run(debug=True)
