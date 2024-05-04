#import libraries
import streamlit as st  # for creating web apps
from streamlit_option_menu import option_menu
import numpy as np  # for numerical computing
import pandas as pd # for dataframe and manipulation
import seaborn as sns #for graphs and data visualization
from matplotlib import pyplot as plt 
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
import plotly.express as px #for graphs and data visualization
sns.set() #setting seaborn as default for plots
import pickle

# CSS Stylingstre

# Load the contents of CSS file
with open('style.css') as f:
    css = f.read()

# Use the st.markdown function to apply the CSS to Streamlit app
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
# Loading the dataset
@st.cache_resource # Cache the data loading step to enhance performance
def load_data():
    return pd.read_csv('framingham_clean.csv')

df = load_data()


def show_home_page():
    st.markdown('<h1 class="my-title ">Framingham Heart Study</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">1- Distribution of Age & Gender among the dataset. </h3>', unsafe_allow_html=True)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    most_frequent_age = df['age'].mode().values[0]
    sns.histplot(df['age'], bins=20, kde=True, ax=ax1)
    ax1.set_title('Age Distribution (Most Frequent Age)')
    ax1.set(xlabel='Age', ylabel='Frequency')
    ax1.annotate(f'Most Frequent Age: {most_frequent_age}', xy=(most_frequent_age, 0), xytext=(most_frequent_age, 50),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='red'), color='red')

    # Gender distribution plot
    sns.countplot(data=df, x='gender', ax=ax2)
    ax2.set_title('Gender Distribution')
    ax2.set_xlabel('Gender')
    ax2.set_ylabel('Count')
    total_count = len(df)
    for p in ax2.patches:
        percentage = f'{100 * p.get_height() / total_count:.1f}%'
        ax2.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
    ax2.set_xticklabels(['Female', 'Male'])
    st.pyplot(fig)

    ### 4.2 "What is the distribution of key cardiovascular risk factors in the dataset, and how does their skewness impact the understanding of the data's central tendency and spread? 

    with st.expander(" 2. Distribution of key Cardiovascular Risk Factors with Skewness Labels"):
        # Define colors for each variable
        colors = ['orange', 'green', 'red', 'purple', 'blue', 'grey']

        # Create a figure to contain all the subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.subplots_adjust(wspace=0.3, hspace=0.3)

        variables = ['glucose', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate']

        # Define skewness labels based on your criteria
        def get_skew_label(skewness):
            if skewness < -1 or skewness > 1:
                return 'Highly Skewed'
            elif (-1 <= skewness <= -0.5) or (0.5 <= skewness <= 1):
                return 'Moderately Skewed'
            else:
                return 'Approximately Symmetric'

        for i, var in enumerate(variables):
            row, col = i // 3, i % 3
            ax = axes[row, col]

            # Calculate skewness
            skewness = df[var].skew()
            skew_label = get_skew_label(skewness)

            sns.histplot(df[var], color=colors[i], kde=True, ax=ax)
            ax.set_title(f'Distribution of {var}\nSkewness: {skewness:.2f} ({skew_label})')

        # Display the entire figure in Streamlit
        st.pyplot(fig)

    ### 4.3 "How does the distribution of heart rate and age groups in the dataset provide insights into the demographics and health characteristics of the surveyed population, and is there any noticeable correlation between these factors? 

    # Define the columns for the Streamlit app
    col1, col2 = st.columns(2)

    # Create a DataFrame for Plotly
    plotly_df = df.copy()
    plotly_df['heart_rate_groups'] = plotly_df['heart_rate_groups'].map({0: 'Low', 1: 'Normal', 2: 'High'})
    plotly_df['age_groups'] = plotly_df['age_groups'].map({0: 'Adults', 1: 'Middle-Aged', 2: 'Senior'})

    st.markdown('<P class="sub-header">2. Heart Rate and Age Group Analysis </p>', unsafe_allow_html=True)
    # Create subplots
    fig3 = make_subplots(rows=1, cols=2, subplot_titles=('Count by HeartRate Group', 'Count by Age Group'))

    # Create the first plot (HeartRate Grouped)
    fig3.add_trace(go.Bar(x=plotly_df['heart_rate_groups'].value_counts().index, y=plotly_df['heart_rate_groups'].value_counts(), marker_color='lightcoral'), row=1, col=1)
    fig3.update_xaxes(title_text='Heart Rate Group', row=1, col=1)
    fig3.update_yaxes(title_text='Count', row=1, col=1)

    # Create the second plot (Count by Age Group)
    fig3.add_trace(go.Bar(x=plotly_df['age_groups'].value_counts().index, y=plotly_df['age_groups'].value_counts(), marker_color='lightblue'), row=1, col=2)
    fig3.update_xaxes(title_text='Age Group', row=1, col=2)
    st.plotly_chart(fig3)

    ### 4.4 Is there a difference in the number of male and female patients with coronary heart disease? 
    ### 4.5 How does the prevalence of diabetes vary across different age groups, and what percentage of people in each age group have diabetes? 


    st.markdown('<h3 class="sub-header">3. Diabetes and Coronary Heart Disease by Age Group and Gender</h2>', unsafe_allow_html=True)
    # Define age group labels
    age_group_labels = ['Adults', 'Middle-Aged', 'Senior']

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # First subplot: Diabetes by Age Group
    plt.sca(axes[0])  # Set the current axes to the first subplot
    ax1 = sns.countplot(x='age_groups', hue='diabetes', data=df, palette='rainbow')
    plt.xlabel('Age Group')
    plt.ylabel('No. of Patients')
    plt.xticks(ticks=[0, 1, 2], labels=age_group_labels)
    plt.legend(title='Diabetes', labels=['Negative', 'Positive'])
    plt.title('Diabetes by Age Group')
    total_count1 = len(df)
    for p in ax1.patches:
        height = p.get_height()
        percentage = height / total_count1 * 100
        ax1.text(p.get_x() + p.get_width() / 2, height + 5, f'{percentage:.1f}%', ha='center')

    # Second subplot: Coronary Heart Disease by Gender
    plt.sca(axes[1])  # Set the current axes to the second subplot
    sns.set_style("whitegrid")  # Add grid lines
    ax2 = sns.countplot(x='gender', hue='TenYearCHD', data=df, palette='Paired')
    plt.xlabel('Gender', fontsize=14)
    plt.xticks(ticks=[0, 1], labels=['Female', 'Male'], fontsize=12)
    plt.ylabel('No. of Patients', fontsize=14)
    plt.yticks(fontsize=12)
    plt.legend(['Neg.', 'Pos.'], title='CHD Status', title_fontsize=12, fontsize=12)
    plt.title('Coronary Heart Disease (CHD) by Gender', fontsize=16)
    total_count2 = len(df)
    for p in ax2.patches:
        height = p.get_height()
        percentage = height / total_count2 * 100
        ax2.text(p.get_x() + p.get_width() / 2, height + 5, f'{percentage:.1f}%', ha='center')

    # Adjust plot aesthetics
    sns.despine(left=True, ax=axes[1])
    axes[1].set_axisbelow(True)  # Move grid lines behind the bars

    # Display the subplots in Streamlit
    st.pyplot(fig)

    ### 4.6 "How do systolic and diastolic blood pressures vary across different age groups and genders, and are there any noticeable patterns or differences that could indicate potential health trends or risk factors? 
    ### 4.7 How do glucose and total cholesterol vary across different age groups and genders, and are there any noticeable patterns or differences that could indicate potential health trends or risk factors? 

    # Create age group labels
    age_group_labels = ['Adults', 'Middle-Aged', 'Senior']

    # Box plots for Sys. BP by Age Group & Gender
    fig1 = px.box(df, x='age_groups', y='sysBP', color='gender', title='Sys. BP vs Age Group by Gender')
    fig1.update_xaxes(categoryorder='array', categoryarray=age_group_labels)

    # Boxen plots for Dia. BP by Age Group & Gender
    fig2 = px.box(df, x='age_groups', y='diaBP', color='gender', title='Dia. BP vs Age Group by Gender')
    fig2.update_xaxes(categoryorder='array', categoryarray=age_group_labels)

    # Box plots for Sys. BP by Age Group & Gender
    fig3 = px.box(df, x='age_groups', y='glucose', color='gender', title='Glucose vs Age Group by Gender')
    fig3.update_xaxes(categoryorder='array', categoryarray=age_group_labels)

    # Boxen plots for Dia. BP by Age Group & Gender
    fig4 = px.box(df, x='age_groups', y='totChol', color='gender', title='Total Cholesterol vs Age Group by Gender')
    fig4.update_xaxes(categoryorder='array', categoryarray=age_group_labels)

    # Create a checkbox in the sidebar for selecting the figure to display
    selected_figure = st.sidebar.radio("Health Metrics Analysis by Age Group and Gender", [None,'Sys. BP', 'Dia. BP', 'Glucose', 'Total Cholesterol'])

    # Display the selected figure in the main area
    if selected_figure == 'Sys. BP':
        st.plotly_chart(fig1)
    elif selected_figure == 'Dia. BP':
        st.plotly_chart(fig2)
    elif selected_figure == 'Glucose':
        st.plotly_chart(fig3)
    elif selected_figure == 'Total Cholesterol':
        st.plotly_chart(fig4)

    ### 4.8 How does the number of cigarettes smoked per day ('cigsPerDay') vary across different age groups? 

    # Define age group labels
    age_group_labels = ['Adults', 'Middle-Aged', 'Senior']

    # Create a density plot for 'cigsPerDay' by age group
    plt.figure(figsize=(10, 7))
    sns.set(style="whitegrid")

    # Create a list of colors
    colors = ['turquoise', 'coral', 'gold']

    # Create a custom color palette
    palette = sns.color_palette(colors, as_cmap=True)

    # Plot the density plot for 'cigsPerDay' by age group
    sns.kdeplot(data=df, x='cigsPerDay', hue='age_groups', common_norm=False, fill=True, palette=palette)
    st.markdown('<h3 class="sub-header"> 4- Cigs. per day by Age Group (Density Plot)</h3>', unsafe_allow_html=True)
    plt.xlabel('Cigs. / Day')
    plt.ylabel('Density')
    # Display the plot in Streamlit
    st.pyplot(plt.gcf())

    ### 4.9 How are systolic blood pressure, diastolic blood pressure, total cholesterol, and 10-year coronary heart disease risk related to each other?

    # Sidebar Filters
    with st.sidebar:
      st.markdown('<h2 style="color: orange; text-align: center;font-family:Times New Roman;">Scatter Plot:</h2>', unsafe_allow_html=True)

    #st.sidebar.title("Scatter Plot")
    age_group = st.sidebar.slider("Select Age Group", min_value=30, max_value=70, value=(30, 70))


    # Apply filters to the data
    filtered_data = df.copy()
    # Add color variable based on gender and smoker status
    if age_group[0] != 30 or age_group[1] != 70:
        filtered_data = filtered_data[(filtered_data["age"] >= age_group[0]) & (filtered_data["age"] <= age_group[1])]
    # Scatter plot


    # Select features from the predefined list
    x_feature = st.sidebar.selectbox("X-axis Feature", ["totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"])
    y_feature = st.sidebar.selectbox("Y-axis Feature", ["glucose", "heartRate", "BMI", "diaBP", "sysBP", "totChol"])


    # Select color variable
    color_variable = st.sidebar.selectbox("Color Variable", ["None", "Gender", "Smoking Status", "Ten Year CHD"])
    size_feature = "BMI"
    if color_variable == "Gender":
        fig = px.scatter(
            filtered_data,
            x=x_feature,
            y=y_feature,
            size=size_feature,  # Specify the size variable
            color="gender",
            labels={x_feature: x_feature, y_feature: y_feature},
            title=f"{x_feature} vs. {y_feature} by Gender"
        )
    elif color_variable == "Smoking Status":
        fig = px.scatter(
            filtered_data,
            x=x_feature,
            y=y_feature,
            color="currentSmoker",
            size=size_feature,  # Specify the size variable
            labels={x_feature: x_feature, y_feature: y_feature},
            title=f"{x_feature} vs. {y_feature} by Smoking Status"
        )
    elif color_variable == "Ten Year CHD":
        fig = px.scatter(
            filtered_data,
            x=x_feature,
            y=y_feature,
            color="TenYearCHD",
            size=size_feature,  # Specify the size variable

            labels={x_feature: x_feature, y_feature: y_feature},
            title=f"{x_feature} vs. {y_feature} by Ten Year CHD"
        
        )
    else:
        fig = px.scatter(
            filtered_data,
            x=x_feature,
            y=y_feature,
            size=size_feature,  # Specify the size variable
            labels={x_feature: x_feature, y_feature: y_feature},
            title=f"{x_feature} vs. {y_feature}"
        )

    st.plotly_chart(fig)



    # close the container div with the specified CSS class
    st.markdown('</div>', unsafe_allow_html=True)





# Page for displaying Prediction content
def show_prediction_page():
    st.markdown('<h1 class="my-title ">Coronary Heart Disease Prediction</h1>', unsafe_allow_html=True)   

        
    st.markdown('<h3 class="sub-header">1- Please enter your details to predict the risk of developing heart disease in the next ten years. </h3>', unsafe_allow_html=True)
    #st.markdown('Please enter your details to predict the risk of developing heart disease in the next ten years.')
    # Function to load the model and scaler

    @st.cache_data
    def load_model_and_scaler():
        with open('rf_hyper_model.pkl', 'rb') as model_file, open('scaler.pkl', 'rb') as scaler_file:
            model = pickle.load(model_file)
            scaler = pickle.load(scaler_file)
        # Deepcopy the model and scaler if needed to avoid mutation warning
        from copy import deepcopy
        return deepcopy(model), deepcopy(scaler)
    model, scaler = load_model_and_scaler()
    
    # Sidebar input for prediction
    with st.sidebar:
        #st.subheader("Enter Your Details")
        st.markdown('<h2 style="color: orange; text-align: center;">Enter Your Details:</h2>', unsafe_allow_html=True)
        sysBP = st.number_input('Systolic Blood Pressure', 80, 200, 120)
        glucose = st.number_input('Glucose Level', 40, 400, 100)
        age = st.number_input('Age', 30, 80, 50)
        cigsPerDay = st.number_input('Cigarettes per Day', 0, 60, 10)
        totChol = st.number_input('Total Cholesterol', 100, 600, 250)
        diaBP = st.number_input('Diastolic Blood Pressure', 60, 140, 80)
    
    
    # Improved "Additional Details" section with columns
    # Using markdown to create a styled and centered subheader
    #st.markdown('<h2 style="color: orange; text-align: center;">Additional Details:</h2>', unsafe_allow_html=True)
    st.markdown("""
                <style>
                    #main-container h4 {
                        color: orange;
                        text-align: center;
                        font-family: 'Times New Roman', Times, serif; /* Setting Times New Roman font */
                    }
                </style>
                <div id="main-container">
                    <h4>Additional Details:</h4>
                """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        # Adding tooltips for better user guidance
        prevalentHyp_value = st.selectbox('Prevalent Hypertension', ['No', 'Yes'],
                                        help='Select "Yes" if you have been diagnosed with hypertension.')
        BPMeds_value = st.selectbox('Blood Pressure Medications', ['No', 'Yes'],
                                    help='Select "Yes" if you are currently on medication for blood pressure.')

    with col2:
        diabetes_value = st.selectbox('Diabetes', ['No', 'Yes'],
                                    help='Select "Yes" if you have been diagnosed with diabetes.')
        gender_value = st.selectbox('Gender', ['Female', 'Male'],
                                    help='Select your gender.')

    #Optional: Custom styling can be added to make the select boxes visually appealing
    st.markdown("""
    <style>
    div[role="listbox"] {
        background-color: #f1f3f6;  /* Light grey background */
        border-radius: 10px;  /* Rounded borders */
        border: 1px solid #ccc;  /* Light grey border */
    }
    </style>
    """, unsafe_allow_html=True)


    # Prediction button and results
    if st.button('Predict Risk'):
        st.write("")  # Spacer
        st.write("")  # Spacer
        st.write("")  # Spacer
        user_data = [[sysBP, glucose, age, cigsPerDay, totChol, diaBP,
                    1 if prevalentHyp_value == 'Yes' else 0,
                    1 if BPMeds_value == 'Yes' else 0,
                    1 if diabetes_value == 'Yes' else 0,
                    1 if gender_value == 'Male' else 0]]
        user_data_scaled = scaler.transform(user_data)
        prediction = model.predict(user_data_scaled)
        probability = model.predict_proba(user_data_scaled)[0][1]
        st.write("")
        st.write("")

        def load_custom_css():
            css = """
            /* Base styles for the main container */
            #main-container {
                width: 60%;  /* Set the desired width of the main content */
                margin: 20px auto;  /* Center the container */
                padding: 0px; /* Padding around the content */
                border: 0px solid #ccc; /* Adding a border for better visibility */
                border-radius: 10px; /* Rounded corners */
                background-color: #f9f9f9; /* Light background for the container */
            }

            /* Custom styles for error and success messages */
            .stAlert {
                margin: 20px 0; /* Space above and below the alert */
                padding: 10px; /* Padding inside the alert */
                text-align: center;  /* Center the text */
                
            }

            /* Style adjustments for headers and probabilities */
            h2, h3, h1 {
                text-align: center;  /* Centering all text headers */
                color: #333; /* Dark color for better readability */
            }
            h1 {
                margin: 5px 0; /* Tighter margin for the probability header */
            }
            """
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

        load_custom_css()
        st.markdown("""
                <style>
                    #main-container h4 {
                        color: orange;
                        padding: auto;
                        text-align: center;
                        font-family: 'Times New Roman', Times, serif; /* Setting Times New Roman font */
                    }
                </style>
                <div id="main-container">
                    <h4>Probability of Risk:</h4>
                """, unsafe_allow_html=True)
        risk_color = 'red' if prediction[0] == 1 else 'green'
        alert_type = "High Risk of CHD 🚨" if prediction[0] == 1 else "Low Risk of CHD 🍀"
        alert_color = "#ffa1a1" if prediction[0] == 1 else "#a1ffad"  # Light red for error, light green for success

        # First, display the metric
        st.markdown(f"""
            <div style="text-align: center; margin: 20px 0;">
                <h2 style="margin: 0; color: {risk_color};">{probability:.2%}</h2>
            </div>
            """, unsafe_allow_html=True)

        # Then, simulate an alert box using custom HTML/CSS for centering and styling
        st.markdown(f"""
            <div style="background-color: {alert_color}; padding: 10px; border-radius: 5px; text-align: center; width:60%;margin:auto;font-family: 'Times New Roman';">
                {alert_type}
            </div>
            """, unsafe_allow_html=True)


       
        
        st.write("")
        st.write("")
      

     
        # Visualization of feature importances
        features = ['Systolic BP', 'Glucose', 'Age', 'Cigarettes per Day', 'Total Cholesterol',
                'Diastolic BP', 'Prevalent Hypertension', 'BP Meds', 'Diabetes', 'Gender']
        feature_importances = model.feature_importances_

        # Create a DataFrame for Plotly
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importances
        })

        # Calculating percentage of each feature importance
        importance_df['Percentage'] = 100 * importance_df['Importance'] / importance_df['Importance'].sum()
        importance_df.sort_values('Percentage', ascending=False, inplace=True)

        # Plotly Express bar chart for feature importances
        fig = px.bar(importance_df, x='Feature', y='Percentage', 
                    text='Percentage',
                    title="Feature Importance in Predicting CHD", 
                    color='Percentage',
                    color_continuous_scale=px.colors.sequential.Viridis)

        # Customize hover data
        fig.update_traces(texttemplate='%{text:.2f}%', hoverinfo='text+name')

        # Update layout for better readability and aesthetics
        fig.update_layout(
            title={
                'text': "Feature Importance in Predicting CHD",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            title_font=dict(
                color="orange",  # Set the color of the title
                family="Times New Roman",  # Optionally set the font family to Times New Roman
                size=20
            ),
            coloraxis_showscale=False,
            yaxis_title='Importance (%)',
            xaxis_title="Features",
            xaxis={'categoryorder':'total descending'},
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
        )

        # Make the graph more interactive with hover effects
        fig.update_traces(marker_line_color='rgb(8,48,107)',
                        marker_line_width=1.5, opacity=0.6)

        # Display the figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)
       # Dynamic Key Insights based on feature importances
        
        #st.markdown("### Key Insights")
        st.markdown("""
                <style>
                    #main-container h4 {
                        color: orange;
                        padding: auto;
                        text-align: center;
                        font-family: 'Times New Roman', Times, serif; /* Setting Times New Roman font */
                    }
                </style>
                <div id="main-container">
                    <h4>Key Insights:</h4>
                """, unsafe_allow_html=True)
        st.markdown("""
<style>
.centered-text {
    text-align: center; /* Center align text */
    margin-bottom: 10px; /* Spacing between items */
}
.centered-text span {
    display: block; /* Make span a block to take full width */
    color: #0074e4;;
    font-weight: bold;
    margin-bottom: 5px; /* Spacing between title and description */
}
</style>
<div class="centered-text">
    <span>Top Influential Factors:</span>
    <div>The graph highlights which factors are most important for predicting heart disease.</div>
    <span>High-Risk Indicators:</span>
    <div>Higher values in features like Systolic Blood Pressure and Glucose levels indicate a greater influence on risk.</div>
    <span>Actionable Advice:</span>
    <div>Focus on managing the top risk factors to potentially lower your overall risk of CHD.</div>
</div>
""", unsafe_allow_html=True)

  

def main():
    

    # Define the sidebar and ensure it remains consistent
    with st.sidebar:
        st.markdown('<h1 class="sidebar-title">Coronary Heart Disease</h1>', unsafe_allow_html=True)
        st.image('./heart.jpeg' ,caption="CHD")
        # Create a collapsible container for the project overview
        
        selected = option_menu(None, ["CHD Dashboard", "CHD Prediction"], 
                              
                               icons=["bar-chart", "robot"], 
                               default_index=0, orientation="horizontal",
                               styles={
                                   "container": {"padding": "0!important", "background-color": "#fafafa", "width": "100%"},
                                   "icon": {"color": "orange", "font-size": "10px"}, 
                                   "nav-link": {"font-size": "10px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                                   "nav-link-selected": {"background-color": "#3380FF"},
                               })
       
        with st.expander("Overview"):
                st.write("""
            This dataset contains information related to cardiovascular disease risk factors for a group of individuals. Cardiovascular diseases, including heart disease and stroke, are significant health concerns worldwide. Understanding the risk factors associated with these diseases is essential for prevention and management.

            """)    
    # Conditionally display pages based on sidebar selection
    if selected == "CHD Dashboard":
        show_home_page()
    elif selected == "CHD Prediction":
        show_prediction_page()

if __name__ == "__main__":
    main()
st.write("")
st.write("")
st.sidebar.markdown("<h4 style='color: blue; font-size: 16px;' margin-top:20px; >Made with 💙 Eng. Sameera Al-khalifi</h4>", unsafe_allow_html=True)
