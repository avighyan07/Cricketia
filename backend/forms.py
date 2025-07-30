import pandas as pd
from flask_wtf import FlaskForm
from wtforms import (
    SelectField,
    DecimalField,
    
    IntegerField,
    SubmitField,
   
)
from wtforms.validators import DataRequired, NumberRange


import os
import pandas as pd

# Adjust path to go one level up
csv_path = os.path.join(os.path.dirname(__file__), '..', 'filtered_final_df.csv')

X_data = pd.read_csv(csv_path).drop(columns="result")


class InputForm(FlaskForm):
    batting_team = SelectField(
        label="batting_team",
        choices=X_data.batting_team.unique().tolist(),
        validators=[DataRequired()]
    )
    bowling_team = SelectField(
        label="bowling_team",
        choices=X_data.bowling_team.unique().tolist(),
        validators=[DataRequired()]
    )
   
    city = SelectField(
        label="city",
        choices=X_data.city.unique().tolist(),
        validators=[DataRequired()]
    )
 
    runs_left = IntegerField(
        label="runs_left",
        validators=[DataRequired(),NumberRange(min=1, message="Runs left must be greater than 0.")]
    )
    balls_left = IntegerField(
        label="balls_left ",
        validators=[DataRequired(),NumberRange(min=1, message="Runs left must be greater than 0.")]
    )
    wickets = IntegerField(
        label="wickets ",
        validators=[DataRequired(),NumberRange(min=1, message="Runs left must be greater than 0.")]
    )
    total_runs_x = IntegerField(
        label="total_runs_scored ",
        validators=[DataRequired(),NumberRange(min=1, message="Runs left must be greater than 0.")]
    )
    crr = DecimalField(
        label="crr",
        validators=[DataRequired(), NumberRange(min=0.01, message="CRR must be greater than 0.")]
    )
    rrr = DecimalField(
        label="rrr",
        validators=[DataRequired(), NumberRange(min=0.01, message="CRR must be greater than 0.")]
    )
   
    submit = SubmitField("Predict")