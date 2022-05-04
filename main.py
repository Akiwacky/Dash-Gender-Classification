from dash import html, dcc, Dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pickle
from dash.exceptions import PreventUpdate
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Layout

header = dbc.Container(
    [
        html.H1("Gender Classification Model Example"),
        html.Hr()
    ]
)

parameters = dbc.Container(
    [
        html.Br(),
        html.Div(
            html.P("Please enter details in all fields", className="card-text")
        ),
        dbc.Card(
            [
                html.Div(
                    [
                        dcc.Dropdown(
                            id='drink',
                            options=[
                                {'label': 'Wine', 'value': 'Wine'},
                                {'label': 'Beer', 'value': 'Beer'},
                                {'label': 'Spirits', 'value': 'Spirits'},
                                {'label': 'Cocktails', 'value': 'Cocktails'},
                                {'label': 'Do Not Drink', 'value': 'Do Not Drink'},
                            ],
                            placeholder="Favorite Alcoholic Drink",
                        )
                    ]
                ),

                html.Div(
                    [
                        dcc.Dropdown(
                            id='footwear',
                            options=[
                                {'label': 'Trainers', 'value': 'Trainers'},
                                {'label': 'Heels', 'value': 'Heels'},
                                {'label': 'Shoes', 'value': 'Shoes'},
                            ],
                            placeholder="Favorite Footwear",
                        )
                    ], className="mt-3"),

                html.Div(
                    [
                        dcc.Dropdown(
                            id='music',
                            options=[
                                {'label': 'Afro Beats', 'value': 'Afro Beats'},
                                {'label': 'Hip Hop', 'value': 'Hip Hop'},
                                {'label': 'Pop', 'value': 'Pop'},
                                {'label': 'RnB', 'value': 'RnB'},
                                {'label': 'Rock', 'value': 'Rock'},
                                {'label': 'Dance & Electronic', 'value': 'Dance & Electronic'},
                                {'label': 'Latin', 'value': 'Latin'},
                                {'label': 'Other', 'value': 'Other'},
                            ],
                            placeholder="Favorite Music Genre",
                        )
                    ], className="mt-3"),

                html.Div(
                    [
                        dcc.Dropdown(
                            id='color',
                            options=[
                                {'label': 'Warm', 'value': 'Warm'},
                                {'label': 'Neutral', 'value': 'Neutral'},
                                {'label': 'Cool', 'value': 'Cool'},

                            ],
                            placeholder="Favorite Color",
                        )
                    ], className="mt-3"),

                html.Div(
                    [
                        dcc.Dropdown(
                            id='exercise',
                            options=[
                                {'label': 'Weights', 'value': 'Weights'},
                                {'label': 'Cycling', 'value': 'Cycling'},
                                {'label': 'Dance', 'value': 'Dance'},
                                {'label': 'Running', 'value': 'Running'},
                                {'label': 'Sports', 'value': 'Sports'},

                            ],
                            placeholder="Favorite Exercise",
                        )
                    ], className="mt-3"),

                html.Div(
                    [
                        dcc.Dropdown(
                            id='holiday',
                            options=[
                                {'label': 'Adventure', 'value': 'Adventure'},
                                {'label': 'Beach', 'value': 'Beach'},
                                {'label': 'Girls Trip', 'value': 'Girls Trip'},
                                {'label': 'Guys Trip', 'value': 'Guys Trip'},

                            ],
                            placeholder="Favorite Holiday",
                        )
                    ], className="mt-3"),
                html.Div(
                    [
                        dbc.InputGroup(
                            [
                                dbc.Input(id='number', placeholder="Avg Hours Watching Sports Per Week", type="number"),
                                dbc.InputGroupText("Hrs"),
                            ],
                        )
                    ], className="mt-3"
                ),
            ], className="my-5"),

        html.Div(
            [
                dbc.Button("Submit", color="primary", id='button'),
            ],
            className="d-grid gap-2 col-4 mx-auto",
        ),

        html.Br(),
        html.Div(id='display-selected-values')

    ])

tab1_content = parameters

tab2_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("""Hey, I go by Akiwacky, and this is a simple Gender Classification 
            Machine Learning model which i created to further understand Machine Learning, 
            The dataset was created and then bootstrapped to 5000 samples. It is only created 
            to understand the concept further. Please see my GitHub for more details...""",
                   className="card-text col-8 mx-auto my-3"),

            html.A("Link to external site", href='https://plot.ly', target="_blank", )
        ]
    ),
    className="mt-3",
)

tabs = dbc.Tabs(
    [
        dbc.Tab(tab1_content, label="Home"),
        dbc.Tab(tab2_content, label="About"),
    ]
)

app.layout = dbc.Container(
    [
        header,
        tabs
    ],
    fluid=True,
)

# Functions

# Load Pickle Model
with open('test-model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# Single Gender Function
def predict_single(gender, dv, model):
    X = dv.transform([gender])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


@app.callback(
    Output("display-selected-values", "children"),
    [Input("button", "n_clicks"),
     State("drink", "value"), State("footwear", "value"),
     State("music", "value"), State("color", "value"),
     State("exercise", "value"), State("holiday", "value"),
     State("number", "value")]
)
def test_model(n_clicks, drink, footwear, music, color, exercise, holiday, number=0):
    if not n_clicks:
        raise PreventUpdate

    customer = {
        'favorite_alcoholic_drink': drink,
        'favorite_footwear': footwear,
        'favorite_music_genre': music,
        'favorite_color': color,
        'favorite_exercise': exercise,
        'type_of_holiday': holiday,
        'avg_time_spent_watching_sports': int(number)}

    prediction = predict_single(customer, dv, model)
    if prediction >= 0.5:
        return f"Prediction: {round(prediction, 2)}, verdict: Male"
    else:
        return f"Prediction: {round(prediction, 2)}, verdict: Female"


if __name__ == '__main__':
    app.run_server(debug=True)
