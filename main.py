import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def main():
    # Set a wide layout and page title
    st.set_page_config(page_title='Data Exploration Tool', layout='wide')

    
   # Injecting custom CSS for styling
    st.markdown("""
        <style>
        body {
            color: white !important;
            background-color: black;
        }
        .stApp {
            background-color: black;
        }
        .stMarkdown, .stText, .stCode {
            color: white !important;
        }
        /* Styling for dropdown lists and input boxes */
        .stSelectbox, .stMultiSelect, .stTextInput, .stNumberInput {
            color: white !important;
        }
        .stSelectbox > div > div, 
        .stMultiSelect > div > div,
        .stTextInput > div > div,
        .stNumberInput > div > div {
            background-color: #333 !important;
            color: white !important;
            border: 1px solid #555 !important;
        }
        /* Ensure text color is white for selected options */
        .stSelectbox > div > div > div[data-baseweb="select"] > div,
        .stMultiSelect > div > div > div[data-baseweb="select"] > div {
            color: white !important;
        }
        /* Style for the dropdown arrow */
        .stSelectbox > div > div > div[data-baseweb="select"] > div:last-child {
            color: white !important;
        }
        h1 {
            font-size: 40px;
            font-weight: 700;
            color: #4db8ff !important;
        }
        h2, h3 {
            color: #80ccff !important;
        }
        .stButton > button {
            background-color: #004080;
            color: white !important;
            border-radius: 8px;
        }
        .custom-card {
            background-color: #004080;
            color: white !important;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 2px 2px 10px rgba(255, 255, 255, 0.1);
        }
        .element-container, .stDataFrame {
            color: white !important;
        }
        /* Ensure sidebar text is also white */
        .css-1d391kg, .css-1d391kg .stMarkdown {
            color: white !important;
        }
        /* Style for the file uploader */
        .stFileUploader > div > div {
            background-color: #333 !important;
            color: white !important;
        }
        /* Style for the sidebar */
        .css-1d391kg {
            background-color: #1a1a1a;
        }
        /* Style for the main content area */
        .css-1d391kg {
            background-color: black;
        }
        /* Style for the plots */
        .js-plotly-plot .plotly {
            background-color: #1a1a1a !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.markdown("""
        <h1 style="text-align: center;">📊 Data Visualization Tool</h1>
        <p style="text-align: center; font-size: 18px; color: white;">
        Welcome to the Data Visualization Tool! This app helps you visualize data from CSV or Excel files.<br>
        You can choose between various chart types and customize them for deeper insights.
        </p>
        <hr style="border: 1px solid #004080;">
    """, unsafe_allow_html=True)
    
    
    
    
    
    # Sidebar Section - File Upload
    
    st.sidebar.title("📁 Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            
            # Display Data Preview with styled card
            st.markdown("""
                <div class="custom-card">
                    <h2>Data Preview</h2>
                </div>
            """, unsafe_allow_html=True)
            st.dataframe(df.head(10))  # Display first 10 rows

            # Sidebar menu with buttons for sections
            st.sidebar.title("🛠️ Menu")

            # Use st.session_state to keep track of button clicks
            if 'data_operations' not in st.session_state:
                st.session_state['data_operations'] = False
            if 'visualization' not in st.session_state:
                st.session_state['visualization'] = False

            # Data Operations Button
            if st.sidebar.button("🛠️ Data Operations"):
                st.session_state['data_operations'] = True
                st.session_state['visualization'] = False

            # Visualization Button
            if st.sidebar.button("📊 Visualization"):
                st.session_state['data_operations'] = False
                st.session_state['visualization'] = True

            if st.session_state['data_operations']:
                data_operations(df)

            if st.session_state['visualization']:
                visualization(df)

        except Exception as e:
            st.error(f"Error: {e}")

def data_operations(df):
    st.sidebar.title("🛠️ Data Operations")
    
    
            #basic stats section
        
#1.details of data set (some stat)

    if st.sidebar.button("check Basic Statistics"):
       st.write(" ###Basic Stats")
       st.write(df.describe())
       
                       #SECTION DATA TYPES

# 1. Check Data Types of Columns

    if st.sidebar.button("Check Data Types"):
        st.write("### Data Types of Columns")
        st.write(df.dtypes)
        
# 2. Check Unique Values
    if st.sidebar.button("Check Unique Values"):
        st.write("### Unique Values")
        st.write(df.nunique())

          #section oulier 
           
# 1.Detect Outliers

    if st.sidebar.button("Detect Outliers"):
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_columns) > 0:
            st.write("### Outlier Detection")
            for col in numeric_columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                st.write(f"Column '{col}': {len(outliers)} outliers detected.")
        else:
            st.warning("No numeric columns available for outlier detection.")
            
    #2. Handling out-liers 
    



            #MISSING VALUE SECTION 
## 1. Find Null Values
    if st.sidebar.button("Find Missing Values"):
        null_summary = df.isnull().sum()
        null_columns = null_summary[null_summary > 0]
        if not null_columns.empty:
            st.write("### Null Values Summary")
            null_df = pd.DataFrame({
                'Column Name': null_columns.index,
                'Null Count': null_columns.values,
                'Data Type': df[null_columns.index].dtypes
            })
            st.dataframe(null_df)
        else:
            st.success("No null values found in the dataset.")

# 2. Handle Missing Values
    st.sidebar.subheader("Handle Missing Values")
    column_to_handle = st.sidebar.selectbox("Select column to handle missing values", df.columns)
    missing_value_option = st.sidebar.selectbox(
        "Select a method to handle missing values",
        ["Drop Rows with Missing Values", "Drop Column with Missing Values", "Fill Missing Values"]
    )

    if missing_value_option == "Drop Rows with Missing Values":
        if st.sidebar.button("Drop Rows"):
            df.dropna(subset=[column_to_handle], inplace=True)
            st.success(f"Dropped rows with missing values in column '{column_to_handle}'")
    elif missing_value_option == "Drop Column with Missing Values":
        if st.sidebar.button("Drop Column"):
            df.drop(columns=[column_to_handle], inplace=True)
            st.success(f"Dropped column '{column_to_handle}' containing missing values")
    elif missing_value_option == "Fill Missing Values":
        imputation_method = st.sidebar.selectbox(
            "Choose imputation method",
            ["Mean", "Median", "Mode", "Fill with Zero", "Fill with Custom Value", "Forward Fill", "Backward Fill"]
        )
        
        if imputation_method == "Fill with Custom Value":
            custom_value = st.sidebar.text_input("Enter custom value to fill missing values")
        
        if st.sidebar.button("Apply Imputation"):
            if imputation_method == "Mean":
                df[column_to_handle].fillna(df[column_to_handle].mean(), inplace=True)
            elif imputation_method == "Median":
                df[column_to_handle].fillna(df[column_to_handle].median(), inplace=True)
            elif imputation_method == "Mode":
                df[column_to_handle].fillna(df[column_to_handle].mode()[0], inplace=True)
            elif imputation_method == "Fill with Zero":
                df[column_to_handle].fillna(0, inplace=True)
            elif imputation_method == "Fill with Custom Value" and custom_value:
                df[column_to_handle].fillna(custom_value, inplace=True)
            elif imputation_method == "Forward Fill":
                df[column_to_handle].fillna(method='ffill', inplace=True)
            elif imputation_method == "Backward Fill":
                df[column_to_handle].fillna(method='bfill', inplace=True)
            else:
                st.warning("Please enter a custom value to fill missing data.")
            st.success(f"Filled missing values in '{column_to_handle}' using {imputation_method}")

    
        
# 2.New feature: Change Data Types

    st.sidebar.subheader("Change Data Types")
    col_to_change = st.sidebar.selectbox("Select column to change data type", df.columns)
    new_dtype = st.sidebar.selectbox("Select new data type", ["int64", "float64", "string", "datetime64"])
    if st.sidebar.button("Change Data Type"):
        try:
            if new_dtype == "datetime64":
                df[col_to_change] = pd.to_datetime(df[col_to_change])
            else:
                df[col_to_change] = df[col_to_change].astype(new_dtype)
            st.success(f"Changed data type of '{col_to_change}' to {new_dtype}")
        except Exception as e:
            st.error(f"Error changing data type: {e}")
    

    # 4. Rename Columns
    rename_column = st.sidebar.selectbox("Select a column to rename", df.columns)
    new_name = st.sidebar.text_input("New name for the column", "")
    if st.sidebar.button("Rename Column"):
        if new_name:
            df.rename(columns={rename_column: new_name}, inplace=True)
            st.success(f"Column '{rename_column}' renamed to '{new_name}'")
        else:
            st.warning("Please enter a new name for the column.")

    # 5. Drop Columns
    columns_to_drop = st.sidebar.multiselect("Select columns to drop", df.columns)
    if st.sidebar.button("Drop Selected Columns"):
        df.drop(columns=columns_to_drop, inplace=True)
        st.success(f"Dropped columns: {', '.join(columns_to_drop)}")
    
    # New feature: Add New Column
    st.sidebar.subheader("Add New Column")
    new_col_name = st.sidebar.text_input("Enter new column name")
    new_col_value = st.sidebar.text_input("Enter value or formula (use 'df' for existing columns)")
    if st.sidebar.button("Add New Column"):
        if new_col_name and new_col_value:
            try:
                df[new_col_name] = eval(new_col_value)
                st.success(f"Added new column '{new_col_name}'")
            except Exception as e:
                st.error(f"Error adding new column: {e}")
        else:
            st.warning("Please enter both column name and value/formula")
    
    # New feature: Add Duplicate Column
    st.sidebar.subheader("Add Duplicate Column")
    col_to_duplicate = st.sidebar.selectbox("Select column to duplicate", df.columns)
    new_col_name = st.sidebar.text_input("Enter name for the duplicate column")
    if st.sidebar.button("Add Duplicate Column"):
        if new_col_name:
            df[new_col_name] = df[col_to_duplicate]
            st.success(f"Created duplicate column '{new_col_name}' from '{col_to_duplicate}'")
        else:
            st.warning("Please enter a name for the duplicate column")
        

# 1. Remove Duplicates
    if st.sidebar.button("Remove Duplicates"):
        duplicate_count = df.duplicated().sum()
        df.drop_duplicates(inplace=True)
        st.success(f"Removed {duplicate_count} duplicate rows from the dataset.")
        
 
    # Download updated DataFrame
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Download Updated File", data=csv, file_name='updated_data.csv', mime='text/csv')

def visualization(df):
    st.sidebar.title("📊 Visualization Options")

    # Single Column Plot
    st.sidebar.subheader('Single Column Plot')
    single_col = st.sidebar.selectbox("Select a column for single-column plot", df.columns)
    single_plot_type = st.sidebar.selectbox("Select plot type", ['Histogram', 'Box Plot'])

    if single_plot_type == 'Histogram':
        bins = st.sidebar.number_input("Number of bins", min_value=1, value=10)
        color = st.sidebar.color_picker("Select color for histogram", "#1f77b4")

    if st.sidebar.button("Generate Single Column Plot"):
        st.write(f"### {single_plot_type}: {single_col}")
        fig, ax = plt.subplots()
        if single_plot_type == 'Histogram':
            sns.histplot(df[single_col], bins=bins, color=color, ax=ax)
        elif single_plot_type == 'Box Plot':
            sns.boxplot(y=df[single_col], ax=ax, color=color)
        st.pyplot(fig)




    # X vs Y Column Plot
    st.sidebar.subheader('X vs Y Column Plot')
    x_col = st.sidebar.selectbox("Select X-axis column", df.columns)
    y_col = st.sidebar.selectbox("Select Y-axis column", df.columns)
    plot_type = st.sidebar.selectbox("Select plot type", ['Scatter Plot', 'Line Plot', 'Bar Plot', 'Box Plot', 'Histogram'])

    if plot_type == 'Histogram':
        bins = st.sidebar.number_input("Number of bins for histogram", min_value=1, value=10)
    color = st.sidebar.color_picker("Select color for plot", "#1f77b4")

    if st.sidebar.button("Generate X vs Y Plot"):
        st.write(f"### {plot_type}: {x_col} vs {y_col}")
        if plot_type == 'Scatter Plot':
            fig = px.scatter(df, x=x_col, y=y_col, color_discrete_sequence=[color])
        elif plot_type == 'Line Plot':
            fig = px.line(df, x=x_col, y=y_col, line_shape='linear', color_discrete_sequence=[color])
        elif plot_type == 'Bar Plot':
            fig = px.bar(df, x=x_col, y=y_col, color_discrete_sequence=[color])
        elif plot_type == 'Box Plot':
            fig = px.box(df, x=x_col, y=y_col, color_discrete_sequence=[color])
        elif plot_type == 'Histogram':
            fig = px.histogram(df, x=x_col, nbins=bins, color_discrete_sequence=[color])
        st.plotly_chart(fig)






    # Multi-Column Visualization
    st.sidebar.subheader('Multi-Column Visualization')
    multi_columns = st.sidebar.multiselect("Select columns for multi-column visualizations", df.columns)
    multi_plot_type = st.sidebar.selectbox("Select plot type for multi-columns", 
                                           ['Pair Plot', 'Scatter Plot', 'Box Plot', 'Histogram'])

    bins = st.sidebar.number_input("Number of bins (for applicable plots)", min_value=1, value=10)

    if st.sidebar.button("Generate Multi-Column Plot"):
        if len(multi_columns) < 2:
            st.warning("Please select at least two columns.")
        else:
            st.write(f"### {multi_plot_type} of {', '.join(multi_columns)}")
            if multi_plot_type == 'Pair Plot':
                pair_plot = sns.pairplot(df[multi_columns])
                st.pyplot(pair_plot.fig)
            elif multi_plot_type == 'Scatter Plot':
                fig, ax = plt.subplots()
                sns.scatterplot(data=df, x=multi_columns[0], y=multi_columns[1], ax=ax)
                plt.title(f'Scatter Plot: {multi_columns[0]} vs {multi_columns[1]}')
                st.pyplot(fig)
            elif multi_plot_type == 'Box Plot':
                fig, ax = plt.subplots()
                sns.boxplot(data=df[multi_columns], ax=ax)
                plt.title(f'Box Plot of {", ".join(multi_columns)}')
                st.pyplot(fig)
            elif multi_plot_type == 'Histogram':
                fig, ax = plt.subplots()
                for col in multi_columns:
                    sns.histplot(df[col], bins=bins, label=col, kde=True, ax=ax)
                plt.title('Histogram of Selected Columns')
                plt.legend()
                st.pyplot(fig)

if __name__ == "__main__":
    main()