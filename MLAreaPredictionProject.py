import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


def to_feet(value,unit):
        "convert units to feet"
        conversions = {'ft': 1.0, 'in': 1/12, 'cm': 0.0328084 }
        return value * conversions.get(unit.lower(),1.0)


#training data
#example ([10ft,5ft]=50sqft, [20ft,10ft] =200sqft,etc.)
x_train = np.array([[10,5],[20,10],[8,12],[15,15],[5,5],[2,2]])
#target data of area and perimeter
y_train = np.array([[50,30],[200,60],[96,40],[225,60],[25,20],[4,8]])

#polynomial pipeline
model = Pipeline([
    ('polynomial', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('linear', LinearRegression())])

#train the model                                               
model.fit(x_train,y_train)

#input
print("--- ML Dimension Calculator ---")
print("Units supported: ft, in, cm")

#query inputs
try:
    #side 1
    val1= float(input("Enter Side 1 value: "))
    unit1= input("Enter Side 1 unit (ft/in/cm):").strip().lower()

    #side 2
    val2 = float(input("Enter Side 2 value: "))
    unit2 = input("Enter Side 2 unit (ft/in/cm): ").strip().lower()

    #negative input handling
    if val1 <=0 or val2 <= 0:
           print("\nError: dimensional input cannot be 0 or negative")
    else: 
    #convert dimensions
        side1_Feet = to_feet(val1,unit1)
        side2_Feet = to_feet(val2,unit2)

        #predict
        prediction = model.predict([[side1_Feet,side2_Feet]])
        area , perimeter = prediction[0]
    
        print("\n--- Model Predictions ---")
        print(f"Area:      {area:.2f} sq ft")
        print(f"Perimeter: {perimeter:.2f} ft")
        
except ValueError:
        print("Error: Please enter numbers for the dimensions. ")