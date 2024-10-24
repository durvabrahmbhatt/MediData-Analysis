import streamlit as st
from copy import deepcopy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from typing import Iterable
import textwrap
import pickle
import os
import google.generativeai as genai

import folium
from streamlit_folium import st_folium
import branca.colormap as cm
import requests
import json

genai.configure(api_key="AIzaSyB1x-cE9Ja9szrA5byKe7dPVU62OAxKfYk")
model = genai.GenerativeModel('gemini-pro')

# Initialize prediction, predetermined_question, and response as None at the beginning of your script
prediction = None
predetermined_question = None
response = None

# Define helper functions
def string_preparation(st: str) -> str:
    return (st.replace('(', ' ')
              .replace(')', ' ')
              .strip()
              .replace(' ', '_')
              .replace('__', '_'))

def row_to_string(row: Iterable['str']) -> str:
    return ' '.join(row.values)

def load_data(file_path):
    try:
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            st.error(f"File not found: {file_path}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def get_symptoms(df):
    symptom_columns = [col for col in df.columns if col.startswith('Symptom')]
    symptoms = set()
    for col in symptom_columns:
        symptoms.update(df[col].dropna().unique())
    return sorted(symptoms)

def process_symptoms(symptoms_experiencing, vectorizer, my_df, X):
    input_symptoms = [string_preparation(symptom.strip()) for symptom in symptoms_experiencing]
    input_symptoms_string = ' '.join(input_symptoms)
    
    input_vector = vectorizer.transform([input_symptoms_string])
    similarity_scores = np.sum(input_vector.toarray() * X.toarray(), axis=1)
    
    # Normalize similarity scores to get probabilities
    if np.sum(similarity_scores) > 0:
        probabilities = similarity_scores / np.sum(similarity_scores)
    else:
        probabilities = similarity_scores  # In case of no symptoms match, probabilities will be zero
    
    # Scale probabilities by multiplying with 1000
    probabilities = probabilities * 10000
    
    # Create a DataFrame for top diseases with scaled probabilities
    top_diseases = my_df.copy()
    top_diseases['Probability'] = probabilities
    
    # Sort by Probability and drop duplicates
    top_diseases = top_diseases.sort_values(by='Probability', ascending=False).drop_duplicates(subset='Disease')
    
    # Get the top 5 diseases with the highest probabilities
    top_diseases = top_diseases.head(5)
    
    return top_diseases

# Streamlit app title
st.title("Symptom Checker and Disease Prediction")

# File paths for datasets
dataset_path = 'D:/inventory-tracker/dataset.csv'
severity_data_path = 'D:/inventory-tracker/Symptom-severity.csv'
descriptions_path = 'D:/inventory-tracker/symptom_Description.csv'
precautions_path = 'D:/inventory-tracker/symptom_precaution.csv'

# Load and preprocess dataset
df = load_data(dataset_path)
if not df.empty:
    df_new = (df.drop('Disease', axis=1).fillna('')).applymap(string_preparation)
    simptoms_list = df_new.apply(row_to_string, axis=1).values.tolist()

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(simptoms_list)

    my_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    symptoms_list = vectorizer.get_feature_names_out()
    my_df["Disease"] = df["Disease"].map(str.strip)

    # Save the trained model using pickle
    model_filename = 'trained_model_LR.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(vectorizer, file)

    # User input for symptoms
    st.header("Enter Symptoms for Disease Prediction")

    symptoms = get_symptoms(df)

    # Initialize session state if it does not exist
    if 'selected_symptoms' not in st.session_state:
        st.session_state.selected_symptoms = []

    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None

    if 'chat' not in st.session_state:
        st.session_state.chat = model.start_chat(history=[])
        st.session_state.last_response = None

    if 'predicted_diseases' not in st.session_state:
        st.session_state.predicted_diseases = []

    # Function to clear chat history
    def clear_chat_history():
        st.session_state.chat.history = []
        st.session_state.last_response = None

    # Use a multiselect dropdown to select symptoms
    selected_symptoms = st.multiselect(
        'Select Symptoms',
        options=symptoms,
        default=st.session_state.selected_symptoms,
        key='symptom_multiselect'
    )

    # Update session state with new selections
    if selected_symptoms != st.session_state.selected_symptoms:
        st.session_state.selected_symptoms = selected_symptoms
        st.session_state.last_prediction = None  # Reset last prediction
        st.session_state.predicted_diseases = []  # Reset predicted diseases
        clear_chat_history()  # Clear chat history when symptoms change
        st.rerun()

    # st.write('Selected Symptoms:', st.session_state.selected_symptoms)

    # Clear all selected symptoms
    if st.button('Clear All Symptoms'):
        st.session_state.selected_symptoms = []
        st.session_state.last_prediction = None  # Reset last prediction
        st.session_state.predicted_diseases = []  # Reset predicted diseases
        clear_chat_history()  # Clear chat history when symptoms are cleared
        st.rerun()

    if st.session_state.selected_symptoms:
        top_diseases = process_symptoms(st.session_state.selected_symptoms, vectorizer, my_df, X)
        
        if not top_diseases.empty:
            st.session_state.predicted_diseases.extend(top_diseases['Disease'].tolist())
            
            st.title("Top 5 Predicted Diseases with Probabilities:")
            st.dataframe(top_diseases[['Disease', 'Probability']])

            # Check if the probabilities of the top 2 diseases are the same
            if top_diseases.iloc[0]['Probability'] == top_diseases.iloc[1]['Probability']:
                st.warning("The probabilities for the top 2 diseases are the same. Please enter additional symptoms for more precise prediction.")


            # Plotting bar chart for top diseases with scaled probabilities
            fig = go.Figure(data=[go.Bar(
                x=top_diseases['Disease'],
                y=top_diseases['Probability'],
                # text=top_diseases['Probability'],
                text=[f"{prob:.2f}" for prob in top_diseases['Probability']],  # Format numbers to 2 decimal places
                texttemplate='%{text}',
                textposition='auto'
            )])
            fig.update_layout(
                title="Top 5 Predicted Diseases with Scaled Probabilities",
                xaxis_title="Disease",
                yaxis_title="Probability",
                title_x=0.5
            )
            st.plotly_chart(fig)
            
            # Plotting severity data for the first disease
            first_disease = top_diseases.iloc[0]['Disease']
            st.title(f"Severity of symptoms for {first_disease}:")
            
            severity_data = load_data(severity_data_path)
            symptom_weights = dict(zip(severity_data['Symptom'], severity_data['weight']))
            
            disease_row = my_df[my_df["Disease"] == first_disease].drop("Disease", axis=1)
            symptoms_with_value_1 = disease_row.columns[disease_row.iloc[0] == 1].tolist()
            severity_values = [symptom_weights.get(symptom, 0) for symptom in symptoms_with_value_1]

            fig = go.Figure(data=[go.Pie(labels=symptoms_with_value_1, values=severity_values, hole=0.3)])
            fig.update_layout(title_text=f"Severity of Symptoms for {first_disease}", title_x=0.5)
            st.plotly_chart(fig)
            
            # Load descriptions and precautions
            descriptions = load_data(descriptions_path)
            precautions = load_data(precautions_path)
            
            description_text = descriptions[descriptions["Disease"] == first_disease]["Description"].values[0]
            wrapped_text = textwrap.fill(description_text, width=80)
            st.title(f"Disease Predicted: {first_disease}")
            st.title("Description:")
            st.write(wrapped_text)
            
            precautions_list = precautions[precautions["Disease"] == first_disease]
            if not precautions_list.empty:
                precautions_list = precautions_list.values.tolist()[0][1:]
                st.title("Precautions:")
                for i, precaution in enumerate(precautions_list, start=1):
                    st.write(f"{i}. {precaution}")
            else:
                st.write("No precautions available for the predicted condition.")


        prediction=first_disease
        if prediction:

            if os.path.exists(f'./who_dataset/{prediction}.csv'):
                who_df = pd.read_csv(f'./who_dataset/{prediction}.csv')

                # Create a Streamlit app
                st.title("Death Rate Analysis")

                # Add a selectbox to select the region
                region_names = who_df['Country Name'].unique()
                selected_region = st.selectbox("Select a region:", region_names)

                # Filter data for the selected region
                region_data = who_df[who_df['Country Name'] == selected_region]

                # Add a selectbox to select the Sex
                Sex_names = region_data['Sex'].unique()
                selected_Sex = st.selectbox("Select a Sex:", Sex_names)

                # Filter data for the selected Sex
                Sex_data = region_data[region_data['Sex'] == selected_Sex]

                # Add a slider to select a range of years
                year_range = st.slider("Select year range:", min_value=Sex_data['Year'].min(), max_value=Sex_data['Year'].max(), value=(Sex_data['Year'].min(), Sex_data['Year'].max()))
                filtered_data = Sex_data[(Sex_data['Year'] >= year_range[0]) & (Sex_data['Year'] <= year_range[1])]

                # Create a line graph with the filtered data
                st.write("Number of deaths per year in ", selected_region, " for ", selected_Sex, " from ", year_range[0], " to ", year_range[1])
                st.line_chart(data=filtered_data, x='Year', y='Number')
            else:
                print(f"No data exist for {prediction}. ")

            predetermined_question = f"What is the timeline for different age groups to recover from {prediction} in tabluar format?"
            st.title("Recovery Timings")
            
            # Send the predetermined question and get the response
            if 'chat' in st.session_state:
                if st.session_state.last_response:
                    # st.session_state.chat.history.remove(st.session_state.last_response)
                    st.session_state.last_response = None
                
                response = st.session_state.chat.send_message(predetermined_question)
                st.write(response.text)
                st.session_state.last_response = response
            else:
                st.write("No specific question available for the predicted condition.")
            
            # Ask Gemini about nearby doctors
            nations_data = f"""give me realdata of differnet nations for {prediction} with Estimated Number of Adults who are suffering {prediction} and Estimated Adult Population the output should be Formated as a JSON string with the following structure:
    {{
        "country name": {
            "estimated_number_of_adults_suffering",
            "estimated_adult_population"
        },
    }}estimated_number_of_adults_suffering and estimated_adult_population this keys should be shown in the response keys should be same for all disease and all in lower case and the numbers should be in long integer format top 10 countries?"""
            st.title("Top Nations with highest youth patients ")
            # st.write(nations_data)
                
            nations_data_response = st.session_state.chat.send_message(nations_data)
            # Convert the data to a DataFrame
            data=nations_data_response.text
            if data:
                # remove the ```json and ```
                data_str = data.strip('```json\n')
                data_str = data_str.strip('```')
                data_str = data_str.strip('JSON')
                # parse the JSON data
                data = data_str
                data_string = data.replace('(', '{"estimated_number_of_adults_suffering": ').replace(')', '}').replace(', ', ', "estimated_adult_population": ')
                # Load the JSON string into a Python dictionary
                try:
                    data_dict = json.loads(data_string)
                    # Change "united states" to "united states of america"
                    if "united states" in data_dict:
                        data_dict["united states of america"] = data_dict.pop("united states")
                    # Print the result to verify
                    # print(json.dumps(data_dict, indent=2))
                    # print(data)
                    # st.write(type(data_dict))
                    # st.write(data_dict)
                    # data = json.loads(data)
                    # st.write(type(data))
                    # st.write(data)
                    # Convert the data to a DataFrame
                    df = pd.DataFrame.from_dict(data_dict, orient='index').reset_index()
                    df.rename(columns={'index': 'country'}, inplace=True)
                    predicted_disease=prediction.lower()
                    # Calculate suffering percentage
                    df['suffering_percentage'] = (df['estimated_number_of_adults_suffering'] / df['estimated_adult_population']) * 100
                    # Create a base map
                    m = folium.Map(location=[20, 0], zoom_start=2)
                    # Create a color scale
                    colormap = cm.linear.YlOrRd_09.scale(df['suffering_percentage'].min(), df['suffering_percentage'].max())
                    # Load GeoJSON data for countries
                    geojson_url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json'
                    # Add suffering percentage to GeoJSON data properties
                    def add_suffering_percentage(feature):
                        country = feature['properties']['name'].lower()
                        suffering_percentage = df.loc[df['country'] == country, 'suffering_percentage']
                        if not suffering_percentage.empty:
                            feature['properties']['suffering_percentage'] = f"{suffering_percentage.values[0]:.2f}%"
                        else:
                            feature['properties']['suffering_percentage'] = "N/A"
                        return feature
                    # Function to get color based on country name
                    def style_function(feature):
                        suffering_percentage = feature['properties'].get('suffering_percentage', "N/A")
                        if suffering_percentage != "N/A":
                            return {
                                'fillOpacity': 0.7,
                                'weight': 0.1,
                                'fillColor': colormap(float(suffering_percentage[:-1]))
                            }
                        else:
                            return {
                                'fillOpacity': 0.1,
                                'weight': 0.1,
                                'fillColor': 'grey'
                            }
                    # Function to highlight the country on hover
                    def highlight_function(feature):
                        return {
                            'fillOpacity': 1,
                            'weight': 1,
                            'color': 'black'
                        }
                    # Fetch GeoJSON data and add suffering percentage
                    geojson_data = requests.get(geojson_url).json()
                    geojson_data['features'] = [add_suffering_percentage(feature) for feature in geojson_data['features']]
                    # Add GeoJSON overlay with tooltips and highlight
                    geojson = folium.GeoJson(
                        geojson_data,
                        name='geojson',
                        style_function=style_function,
                        highlight_function=highlight_function,
                        tooltip=folium.GeoJsonTooltip(
                            fields=['name', 'suffering_percentage'],
                            aliases=['Country:', 'Suffering Percentage:'],
                            localize=True
                        )
                    ).add_to(m)
                    # Add the colormap to the map
                    colormap.add_to(m)
                    # Display the map in Streamlit
                    st_folium(m, width=700, height=500)
                except json.JSONDecodeError as e:
                    st.write("No data received from the chat.")
                    st.stop()
            else:
                st.write("No data received from the chat.")

            # Nearby_doctors=f"Give me doctors in windsor canada for {prediction} disease in google map format?"
            # doctor_response = st.session_state.chat.send_message(Nearby_doctors)
            # st.write(doctor_response.text)

        else:
            st.write("Please select symptoms to get a disease prediction.")


else:
    st.error("No data to display")


