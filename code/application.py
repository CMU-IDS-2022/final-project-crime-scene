import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import altair as alt
import streamlit as st
from PIL import Image
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
#from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
import cleanup as cp
import os
import requests
import io


def load_data():
    # cp.prep_fbi_dataset()
    df = cp.prep_fbi_dataset()
    return df
    
st.title("Application")
with st.spinner(text="Loading data..."):
    df = load_data()
st.write(df.head())

selection = st.selectbox("Select feature",options = ['Bias','Crime','Location'])

if selection:
    selection = [selection]
    st.write(selection)




