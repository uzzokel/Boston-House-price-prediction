
# Load model
model = jb.load('boston_rf.pkl')
# Single prediction
# App title
image = Image.open("pexels-photo-2911260.jpeg")
new_width = 800
new_height = 200
resized_image= image.resize((new_width,new_height))
st.image(resized_image, caption="Resized Image")
st.set_page_config(layout="wide")
st.markdown("<h1 style ='text-align:center;'> Boston Housing Prices ML Model Deployment</h1>", unsafe_allow_html=True)

#main Container
st.markdown("""<style> .stApp{background-color:white;}</style>""", unsafe_allow_html=True)

fd = pd.read_csv(r"C:\Users\UZO\Documents\python_class\fd.csv")
st.dataframe(fd.head())
# Input options
st.sidebar.header("Input Features")
st.sidebar.image(r"C:\Users\UZO\Documents\python_class\house.jpg")
html_temp = """
<div style = "color;red, padding:10px>"
<h4 style ='text-align:center;'>The Price is the MEDV in $1000</h4>"
</div>
"""
html_pred = """
<div style = "background-color:white; padding:10px>"
<h4 style ='text-align:center;'>Housing Price prediction is shown above in $1000 ! </h4>
</div>
"""

st.set_page_config("Housing Price prediction", layout = "centered")
st.markdown(html_temp, unsafe_allow_html=True)
def user_input():
        CRIM = st.sidebar.number_input("Crime Rate (CRIM )", min_value = 0.0000, max_value=0.2000, value=0.00)
        ZN = st.sidebar.number_input("Land Zone (ZN)", min_value = 0, max_value=18, step=1,value=1)
        INDUS= st.sidebar.number_input("Retail Business(INDUS)", min_value = 0, max_value=12,step=1, value=0)
        NOX= st.sidebar.number_input("Conc. Nitric Oxide( NOX)", min_value = 0.0, max_value=0.6, value=0.0)
        RM = st.sidebar.number_input("Avg Numb of Rooms(RM)", min_value = 0, max_value=8,step=1, value=0)
        AGE = st.sidebar.number_input("Age of House(AGE)", min_value = 0, max_value=99,step =1, value=0)
        DIS= st.sidebar.number_input("Weighed Distance( DIS)", min_value = 0.0, max_value=7.0, value=0.0)
        RAD= st.sidebar.number_input("Radial Highway( RAD)", min_value = 0, max_value=4,step=1, value=0)
        TAX = st.sidebar.number_input("Propert Tax Rate(TAX)", min_value = 0, max_value=300,step=1, value=0)
        PTRATIO = st.sidebar.number_input("Pupil Teach. Ratio(PTRATIO)", min_value = 0, max_value=25,step=1, value=0)
        LSTAT= st.sidebar.number_input("Population Lower Status(LSTAT)", min_value = 0, max_value=10,step=1, value=0)

        return pd.DataFrame([[CRIM, ZN, INDUS, NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,LSTAT]])
    
input_df = user_input()
# Display input
st.write("### Input Features:")
st.write(input_df)
# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write("This is :blue[Price Prediction]:", prediction)
    st.markdown(html_pred, unsafe_allow_html=True)


st.subheader("Model Visualization")
x = fd.drop ('MEDV',axis=1) # Features
y = fd['MEDV']              # Target variable (house prices in $1000s)

from sklearn.model_selection import train_test_split

# We invoke the splitting

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.3,random_state = 42)

# Import StandardScaler.
from sklearn.preprocessing import StandardScaler

# Instantiate StandardScaler.
scaler = StandardScaler()

# Fit and transform training data.
xtrain_scaled = scaler.fit_transform(xtrain)

# Also transform test data.
xtest_scaled = scaler.transform(xtest)

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go

# prediction using the RandomForest Regressor

rf = RandomForestRegressor(n_estimators=100,oob_score=True, random_state=42)

# to train the model

rf.fit(xtrain,ytrain)

# To evaluate the model

rf_pred = rf.predict(xtest)

# Compute residuals.
residuals = ytest - rf_pred

df = pd.DataFrame({'Actual': ytest, 'Predicted': rf_pred})


# # Plot 2: Regression Fit (Actual vs Predicted).

df = pd.DataFrame({'Actual': ytest, 'Predicted': rf_pred})

# Create scatter plot of Actual vs Predicted

fig = px.scatter(df, x='Actual', y='Predicted', 
                 title='Actual vs. Predicted Values',
                 labels={'Actual': 'Actual Value', 'Predicted': 'Predicted Value'})

#Add diagonal line for perfect prediction reference
min_val = min(df['Actual'].min(), df['Predicted'].min())
max_val = max(df['Actual'].max(), df['Predicted'].max())
fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
              line=dict(color="Red", dash="dash"))

fig.show()
st.plotly_chart(fig,width='stretch')
