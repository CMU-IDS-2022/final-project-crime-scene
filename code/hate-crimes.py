import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import cleanup as cp
import umap
from vega_datasets import data
from sklearn.cluster import DBSCAN


@st.cache(allow_output_mutation=True)
def load_data():
    df = cp.prep_fbi_dataset()
    df_city = cp.prep_city_dataset()
    return df,df_city

@st.cache(allow_output_mutation=True)
def plot_cluster(selection):
    #partition dataframe depending on selection
    bias = df_hate.iloc[:,80:115]
    bias_labels = df_hate.columns[80:115]

    crime = df_hate.iloc[:,23:71]
    crime_labels = df_hate.columns[23:71]

    victim_type = df_hate.iloc[:,71:80]
    victim_type_labels = df_hate.columns[71:80]

    location = df_hate.iloc[:,110:]
    location_labels = df_hate.columns[110:]

    offender = df_hate.iloc[:,15:23]
    offender_labels = df_hate.columns[15:23]
    offender_labels = np.array([i[14:] for i in offender_labels])


    num_indicators = len(selection)
    if num_indicators == 1:
        if selection[0] == 'Bias':
            X = bias.to_numpy()
            labels = bias_labels
        elif selection[0] == 'Crime':
            X = crime.to_numpy()
            labels = crime_labels
        elif selection[0] == 'Location':
            X = location.to_numpy()
            labels = location_labels
        elif selection[0] == 'Offender Race':
            X = offender.to_numpy()
            labels = offender_labels
        elif selection[0] == 'Victim Type':
            X = victim_type.to_numpy()
            labels = victim_type_labels
    else:
        for idx,s in enumerate(selection):
            if idx == 0:
                if s == 'Bias':
                    X = bias.to_numpy()
                    labels = bias_labels
                elif s == 'Crime':
                    X = crime.to_numpy()
                    labels = crime_labels
                elif s == 'Location':
                    X = location.to_numpy()
                    labels = location_labels
                elif s == 'Offender Race':
                    X = offender.to_numpy()
                    labels = offender_labels
                elif s == 'Victim Type':
                    X = victim_type.to_numpy()
                    labels = victim_type_labels
            else:
                if s == 'Bias':
                    X = np.concatenate((X,bias.to_numpy()),axis=1)
                    labels =  np.concatenate((labels,bias_labels))
                elif s == 'Crime':
                    X = np.concatenate((X,crime.to_numpy()),axis=1)
                    labels =  np.concatenate((labels,crime_labels))
                elif s == 'Location':
                    X = np.concatenate((X,location.to_numpy()),axis=1)
                    labels =  np.concatenate((labels,location_labels))
                elif s == 'Offender Race':
                    X = np.concatenate((X,offender.to_numpy()),axis=1)
                    labels =  np.concatenate((labels,offender_labels))
                elif s == 'Victim Type':
                    X = np.concatenate((X,victim_type.to_numpy()),axis=1)
                    labels =  np.concatenate((labels,victim_type_labels))
    

    np.random.seed(0)
    random_sample = np.random.permutation(len(X))[:1000]
    X = X[random_sample]

    #Dimensionality Reduction
    X_tsne2d = umap.UMAP(densmap=True,n_components=2,n_neighbors=2,random_state=0).fit_transform(X)

    #Clustering
    dbscan = DBSCAN(min_samples=10, eps=1)
    dbscan_cluster_assignments = dbscan.fit_predict(X_tsne2d)
    inliers = dbscan_cluster_assignments != -1

    #plotting the cluster
    data = pd.concat([pd.DataFrame(X_tsne2d[:,:][inliers],columns=['X','Y']), pd.DataFrame(dbscan_cluster_assignments[inliers],columns=['Assign'])], axis=1)
    total_counts = data['Assign'].value_counts()
    assign = data['Assign']
    data['total_count'] = [total_counts.get(a) for a in assign]
    if len(selection) == 1:
        data['common_indicator'] = [labels[np.argmax(X[dbscan_cluster_assignments == i].sum(axis=0))] for i in dbscan_cluster_assignments[inliers]]
    else:
        data['common_indicator'] = [labels[np.sort(np.argsort(-X[dbscan_cluster_assignments == i].sum(axis=0))[0:(num_indicators)])].tolist() for i in dbscan_cluster_assignments[inliers]]
    chart = alt.Chart(data,title='{}'.format(selection)).mark_point().encode(
        alt.X("X"),
        alt.Y("Y"),
        alt.Color("Assign:N",legend=alt.Legend(title="Clusters")),
        alt.Tooltip(["Assign",'total_count','common_indicator'])
    ).properties(
    width=500,
    height=500)

    return chart
    
st.title("Hate Crimes in the United States")

with st.spinner(text="Loading data..."):
    df_hate,df_city = load_data()

st.write("FBI Hate Crimes Dataset")    
st.write(df_hate.head())

st.write("Cities Dataset")
st.write(df_city.head())

additional = st.checkbox('Would you like to view additional data?')

if additional:
    st.selectbox("Select your features",options=["Wellness Factor"])

#US MAP
# Title 
st.header("US MAP")
# df = pd.read_csv('hate_crime.csv');
alt.data_transformers.disable_max_rows()

#Drawing up the UP MAP
st.subheader("US MAP")

#Adding a new column Frequency, that holds the number of cases in each state.
df_hate['Frequency']=df_hate['STATE_NAME'].map(df_hate['STATE_NAME'].value_counts())
df_hate = df_hate.drop_duplicates(subset=['STATE_NAME'], keep='first')
df_hate = df_hate.rename({'STATE_NAME': 'state'}, axis = 1)

#Alinging the state values
ansi = pd.read_csv('https://www2.census.gov/geo/docs/reference/state.txt', sep='|')
ansi.columns = ['id', 'abbr', 'state', 'statens']
ansi = ansi[['id', 'abbr', 'state']]
df_hate = pd.merge(df_hate, ansi, how='left', on='state')
states = alt.topo_feature(data.us_10m.url, 'states')

#Defining selection criteria 
click = alt.selection_multi(fields=['state'])

# Building and displaying the US MAP
displayUSMap = alt.Chart(states).mark_geoshape().encode(
    color=alt.Color('Frequency:Q', title = "No. of cases"),         
    tooltip=['Frequency:Q', alt.Tooltip('Frequency:Q')], 
    opacity = alt.condition(click, alt.value(1), alt.value(0.2)),
).properties(
    title = "Choropleth Map of the US on the number of recorded cases per State"  
).transform_lookup(
    lookup='id',
    from_=alt.LookupData(df_hate, 'id', ['Frequency', 'state'])
).properties(
    width=500,
    height=300
).add_selection(click).project(
    type='albersUsa'
)


bars = (
    alt.Chart(
        df_hate.nlargest(15, 'Frequency'),
        title='Top 15 states by population').mark_bar().encode(
    x='Frequency',
    opacity=alt.condition(click, alt.value(1), alt.value(0.2)),
    color='Frequency',
    y=alt.Y('state', sort='x'))
.add_selection(click))
            
st.write(displayUSMap & bars)    
#Reference for Interaction: https://stackoverflow.com/questions/63751130/altair-choropleth-map-color-highlight-based-on-line-chart-selection

# Plotting the Offender Race vs Victim's Hate Crime 
df_HeatMap = df_hate[['BIAS_DESC','OFFENDER_RACE']].copy()
df_HeatMap = df_HeatMap.dropna()
df_HeatMap['VictimHateCrime'] = pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Indian"), 'Anti-Indian',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Arab"), 'Anti-Arab',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Asian"), 'Anti-Asian',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Bisexual"), 'Anti-Sexual Orientation',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Gay"), 'Anti-Sexual Orientation',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Gender"), 'Anti-Sexual Orientation',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Heterosexual"), 'Anti-Sexual Orientation',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Lesbian"), 'Anti-Sexual Orientation',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Black"), 'Anti-Black',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Islamic"), 'Anti-Islamic',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Female"), 'Anti-sexism',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Male"), 'Anti-sexism',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Hispanic"), 'Anti-hispanic',
                                pd.np.where(df_HeatMap.BIAS_DESC.str.contains("Jewish"), 'Anti-Jewish' 
                                , 'Multiple Groups'))))))))))))))
heatmap1 = alt.Chart(df_HeatMap).mark_rect(
    tooltip=True
).encode(
    alt.X("VictimHateCrime", scale=alt.Scale(zero=False), axis=alt.Axis(labelAngle=-0), title='Victim Hate Crime Type'),
    alt.Y("OFFENDER_RACE", scale=alt.Scale(zero=False), title='Offender Race'),
    alt.Color("count():Q", title = "No. of cases")
).properties(
    width=1480,
    height=300,
    title = "Frequency of attacks per Offender's race to Victim's Hate Crime Type"
).configure_title(
    fontSize=20
).configure_axis(
    titleFontSize=18    
)
st.write(heatmap1)
    
 #Clustering   
st.header("Clustering on Hate Crimes")
selection = st.multiselect("Select your features",options = ['Bias','Crime','Location','Offender Race','Victim Type'])

#make selection for clustering
if selection:
    cluster = plot_cluster(selection)
    st.write(cluster)



#Feature Importance
st.header("Feature Importance")
def load_features_final(name):
    return pd.read_csv(name)

df_features_final = load_features_final("https://raw.githubusercontent.com/CMU-IDS-2022/final-project-crime-scene/main/data/features_final.csv")

state = st.text_input("Enter 2 letter state abbreviation")

#City Visualization
#selecting only the cities in the selected state
features_final = df_features_final
features_final.drop(features_final[features_final['STATE_ABBR'] != state].index, inplace = True) 
st.write("Enter one of the following cities in " + state)
st.write(features_final['City'])

#select the city for which you want to see the factors
city = st.text_input("Enter city name")
if( city in features_final.values  ):
    features_final.drop(features_final[features_final['City'] != city].index, inplace = True)
    print(features_final.head())
    source = pd.DataFrame({
        'X' : ['High School Completion','Income Inequality','Life Expectancy','Racial Segregation','Racial Diversity','Unemployment','Uninsured'],
        'Y': [features_final['High School Completion'].tolist()[0],features_final['Income Inequality'].tolist()[0], features_final['Life Expectancy'].tolist()[0], features_final['Neighborhood racial/ethnic segregation'].tolist()[0], features_final['Racial/ethnic diversity'].tolist()[0], features_final['Unemployment - annual, neighborhood-level'].tolist()[0], features_final['Uninsured'].tolist()[0] ],
        })
    hist = alt.Chart(source).mark_bar().encode(
        x='X',
        y='Y'
    ).properties(
        width=500,
        height=800)
        
    st.subheader("Visualizing features in a city")
    st.write(hist)
else:
    st.write("Please choose a city from the list provided")
    