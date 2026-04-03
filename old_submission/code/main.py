from flask import Flask, render_template, redirect, url_for,jsonify
from flask_cors import CORS
import os
import src.dual_simulation_runner as no
import json
import ast

username = os.environ.get("USERNAME").title() if os.environ.get("USERNAME") else "Guest"

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html", username=username)

@app.route("/about")
def about():
    return render_template("about.html", username=username)

@app.route("/features")
def features():
    return render_template("features.html", username=username)

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", username=username)

@app.route("/contact")
def contact():
    return render_template("contact.html", username=username)

@app.route("/simulation")
def simulation():
    result = {"Default SUMO Control": ["0", "0", "0", "0",  "0", "0"], "Random Action Baseline": ["0", "0", "0", "0",  "0", "0"], "QMIX AI Control": ["0", "0", "0", "0",  "0", "0"], "Best Method": "N/A", "Avg Vehicles": "0", "Avg Wait Time": "0"}
    return render_template("simulation.html", data=result, username=username)

@app.route("/simulationresults")
def runsimulation():
    # run python file
    no.run_comparison_simulations()
    # read results from text file
    with open("results.txt", "r") as f:
        result = f.read()
    
    return render_template("simulation.html", data=ast.literal_eval(result), username=username)

if __name__ == "__main__":
    app.run(debug=False )