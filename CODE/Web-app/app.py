from flask import Flask, render_template, url_for, request, redirect, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load("randomfs.pkl") 

cols = ["mileage", "year","listing_color_BLACK","listing_color_BLUE","listing_color_BROWN","listing_color_GOLD","listing_color_GRAY","listing_color_GREEN","listing_color_ORANGE","listing_color_PINK","listing_color_PURPLE","listing_color_RED","listing_color_SILVER","listing_color_TEAL","listing_color_UNKNOWN","listing_color_WHITE","listing_color_YELLOW","make_name_Acura","make_name_BMW","make_name_Buick","make_name_Cadillac","make_name_Chevrolet","make_name_Chrysler","make_name_Dodge","make_name_Ford","make_name_GMC","make_name_Honda","make_name_Hyundai","make_name_Jeep","make_name_Kia","make_name_Mazda","make_name_Mercedes-Benz","make_name_Nissan","make_name_RAM","make_name_Subaru","make_name_Toyota","make_name_Volkswagen","make_name_other","model_name_1500","model_name_2500","model_name_3 Series","model_name_4Runner","model_name_Acadia","model_name_Accord","model_name_Altima","model_name_Blazer","model_name_C-Class","model_name_CR-V","model_name_CX-5","model_name_Camry","model_name_Challenger","model_name_Charger","model_name_Cherokee","model_name_Civic","model_name_Colorado","model_name_Compass","model_name_Corolla","model_name_Cruze","model_name_Durango","model_name_EcoSport","model_name_Edge","model_name_Elantra","model_name_Enclave","model_name_Encore","model_name_Encore GX","model_name_Envision","model_name_Equinox","model_name_Escape","model_name_Expedition","model_name_Explorer","model_name_F-150","model_name_F-250 Super Duty","model_name_Focus","model_name_Forester","model_name_Forte","model_name_Fusion","model_name_Fusion Hybrid","model_name_Gladiator","model_name_Grand Caravan","model_name_Grand Cherokee","model_name_HR-V","model_name_Highlander","model_name_Impala","model_name_Jetta","model_name_Journey","model_name_Kona","model_name_MDX","model_name_Malibu","model_name_Murano","model_name_Mustang","model_name_Odyssey","model_name_Optima","model_name_Outback","model_name_Pacifica","model_name_Passat","model_name_Pathfinder","model_name_Pilot","model_name_Prius","model_name_RAV4","model_name_Ranger","model_name_Renegade","model_name_Rogue","model_name_Rogue Sport","model_name_Santa Fe","model_name_Sentra","model_name_Sierra 1500","model_name_Silverado 1500","model_name_Silverado 2500HD","model_name_Sonata","model_name_Sorento","model_name_Soul","model_name_Sportage","model_name_Suburban","model_name_Tacoma","model_name_Tahoe","model_name_Terrain","model_name_Tiguan","model_name_Traverse","model_name_Trax","model_name_Tucson","model_name_Versa","model_name_Wrangler Unlimited","model_name_XT5"]

def format_cols(columns):
    cols_copy = [0] * len(cols)
    for x in columns[1:3]:
        if x != "mileage" or x != "year":
            cols_copy[cols.index(x)] = 1
    
    cols_copy[1] = float(columns[0])
    cols_copy[0] = float(columns[4])

    return cols_copy



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    int_features = [x for x in request.form.values()]

    int_features[3] = int_features[3].upper()
    final = [""] * 5
    final[0] = int_features[0]
    final[1] = "listing_color_" + int_features[3]
    final[2] = "make_name_" + int_features[1]
    final[3] = "model_name_" + int_features[2]
    final[4] = int_features[4]

    final = format_cols(final)

    data_unseen = pd.DataFrame([final], columns=cols)

     # Make prediction using model loaded from disk as per the data.
    prediction = model.predict(data_unseen)
    # Take the first value of prediction
    output = prediction[0]

    return render_template('home.html', pred='Expected Car Value Will be ${:.2f}'.format(output))


@app.route('/api',methods=['POST'])
def predict2():
    # Get the data from the POST request.
    data = request.get_json(force=True)

    dict = data[0]

    int_features = [dict[x] for x in dict.keys()]

    final = np.array(int_features)


    final[0] = float(final[0])
    final[1] = "listing_color_" + final[1]
    final[2] = "make_name_" + final[2]
    final[3] = "model_name_" + final[3]

    print("This is the final:" + str(format_cols(final)))

    final = format_cols(final)

    data_unseen = pd.DataFrame([final], columns=cols)
    print(data_unseen)
   
    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict(data_unseen)
    # Take the first value of prediction
    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    
    app.run(port=5001, debug=True)