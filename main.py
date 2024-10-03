import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import io
from sklearn.preprocessing import LabelEncoder

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
            font-size: 50px;
            font-weight: 700;
            color: #4db8ff !important;
        }
        h3 {
            color: white !important;
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
    <h1 style="text-align: center;">📊 DATA REFINERY </h1>
    <p style="text-align: center; font-size: 18px; color: white; background-color: #004080; padding: 10px; border-radius: 5px;">
    Welcome Data Refinery Tool! This app helps you to perform Basic preprocessing & visualize data from CSV or Excel files.<br>
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
                    <h2>Preview Raw Data</h2>
                </div>
            """, unsafe_allow_html=True)
             # Radio button for Show/Hide data preview
                    # Radio button for Show/Hide data preview
            preview_option = st.radio("Select an option:", ("Hide Data Preview", "Show Data Preview"))
            num_rows = 5
            if preview_option == "Show Data Preview":
            # Ask user how many rows to display
             num_rows = st.number_input("Enter number of rows to preview", min_value=1, max_value=len(df), value=5)
             st.dataframe(df.head(num_rows))  # Display the chosen number of rows
            else:
             st.write("Data preview is hidden.")


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
def is_normalized(df):
    # Check if data is approximately in the range [0, 1] or [-1, 1]
    return np.all((df.min() >= -1.001) & (df.max() <= 1.001))

def data_operations(df):
    st.sidebar.title("🛠️ Data Operations")
    
def is_normalized(series):
    # Check if data is approximately in the range [0, 1] or [-1, 1]
    return np.all((series.min() >= -1.001) & (series.max() <= 1.001))

def display_preview(dataframe, num_rows=5):
    st.subheader("Dataset Preview (First 5 rows)")
    st.dataframe(dataframe.head(num_rows))
    st.write(f"Total rows: {len(dataframe)}, Total columns: {len(dataframe.columns)}")

def data_operations(df):
    st.sidebar.title("🛠️ Data Operations")
    operation_type = st.sidebar.radio("Select Operation Type", ["Column Handling", "Pre-Processing", "Advanced Operations", "Feature Engeenering"])

    display_preview(df)

    if operation_type == "Column Handling":
        st.sidebar.subheader("Column Handling")
        column_operation = st.sidebar.selectbox("Select Column Operation", 
            ["Column Preview", "Column Statistics", "Rename Column", "Add New Column", 
             "Drop Column", "Add Duplicate Column", "Check Duplicate Columns", 
             "Change Data Type", "Check Unique Values"])

        if column_operation == "Column Preview":
            preview_columns = st.sidebar.multiselect("Select columns to preview", df.columns)
            if preview_columns:
                st.write("### Column Preview")
                st.write(df[preview_columns])

        elif column_operation == "Column Statistics":
            stat_column = st.sidebar.selectbox("Select a column for statistics", df.columns)
            stat_operations = st.sidebar.multiselect("Select operations", ["Mode", "Median", "Mean"])
            if stat_operations:
                st.write(f"### Statistics for {stat_column}")
                if "Mode" in stat_operations:
                    mode_value = df[stat_column].mode().values
                    st.write(f"Mode: {mode_value}")
                if "Median" in stat_operations:
                    median_value = df[stat_column].median()
                    st.write(f"Median: {median_value}")
                if "Mean" in stat_operations:
                    mean_value = df[stat_column].mean()
                    st.write(f"Mean: {mean_value}")

        elif column_operation == "Rename Column":
            rename_column = st.sidebar.selectbox("Select a column to rename", df.columns)
            new_name = st.sidebar.text_input("New name for the column", "")
            if st.sidebar.button("Rename Column"):
                if new_name:
                    df.rename(columns={rename_column: new_name}, inplace=True)
                    st.success(f"Column '{rename_column}' renamed to '{new_name}'")
                    display_preview(df)
                else:
                    st.warning("Please enter a new name for the column.")

        elif column_operation == "Add New Column":
            new_col_name = st.sidebar.text_input("Enter new column name")
            new_col_value = st.sidebar.text_input("Enter value or formula (use 'df' for existing columns)")
            if st.sidebar.button("Add New Column"):
                if new_col_name and new_col_value:
                    try:
                        df[new_col_name] = eval(new_col_value)
                        st.success(f"Added new column '{new_col_name}'")
                        display_preview(df)
                    except Exception as e:
                        st.error(f"Error adding new column: {e}")
                else:
                    st.warning("Please enter both column name and value/formula")

        elif column_operation == "Drop Column":
            columns_to_drop = st.sidebar.multiselect("Select columns to drop", df.columns)
            if st.sidebar.button("Drop Selected Columns"):
                df.drop(columns=columns_to_drop, inplace=True)
                st.success(f"Dropped columns: {', '.join(columns_to_drop)}")
                display_preview(df)

        elif column_operation == "Add Duplicate Column":
            col_to_duplicate = st.sidebar.selectbox("Select column to duplicate", df.columns)
            new_col_name = st.sidebar.text_input("Enter name for the duplicate column")
            if st.sidebar.button("Add Duplicate Column"):
                if new_col_name:
                    df[new_col_name] = df[col_to_duplicate]
                    st.success(f"Created duplicate column '{new_col_name}' from '{col_to_duplicate}'")
                    display_preview(df)
                else:
                    st.warning("Please enter a name for the duplicate column")

        elif column_operation == "Check Duplicate Columns":
            st.write("### Duplicate Columns Check")
            duplicate_cols = df.columns[df.columns.duplicated()].unique()
            if len(duplicate_cols) > 0:
                st.write("Duplicate columns found:")
                st.write(duplicate_cols)
            else:
                st.write("No duplicate columns found.")

        elif column_operation == "Change Data Type":
            col_to_change = st.sidebar.selectbox("Select column to change data type", df.columns)
            new_dtype = st.sidebar.selectbox("Select new data type", ["int64", "float64", "string", "datetime64"])
            if st.sidebar.button("Change Data Type"):
                try:
                    if new_dtype == "datetime64":
                        df[col_to_change] = pd.to_datetime(df[col_to_change])
                    else:
                        df[col_to_change] = df[col_to_change].astype(new_dtype)
                    st.success(f"Changed data type of '{col_to_change}' to {new_dtype}")
                    display_preview(df)
                except Exception as e:
                    st.error(f"Error changing data type: {e}")

        elif column_operation == "Check Unique Values":
            unique_col = st.sidebar.selectbox("Select column to check unique values", df.columns)
            if st.sidebar.button("Show Unique Values"):
                unique_values = df[unique_col].unique()
                st.write(f"### Unique Values in {unique_col}")
                st.write(unique_values)
                
    #///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                     #<       preprocessing    >

    elif operation_type == "Pre-Processing":
        st.sidebar.subheader("Pre-Processing")
        preprocessing_operation = st.sidebar.selectbox("Select Pre-Processing Operation", 
            ["Check Basic Stats", "Data Summary", "Find Missing Values", "Handle Missing Values", 
             "Detect Outliers", "Check Data Types", "Normalize Data"])
        
        if preprocessing_operation == "Data Summary":
            st.write("### Data Summary")
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

        elif preprocessing_operation == "Check Basic Stats":
            st.write("### Basic Statistics")
            st.write(df.describe())

        elif preprocessing_operation == "Find Missing Values":
            st.write("### Missing Values Summary")
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            st.write(missing_data)
            
            if not missing_data.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                missing_data.plot(kind='bar', ax=ax)
                plt.title('Missing Values by Column')
                plt.ylabel('Count of Missing Values')
                st.pyplot(fig)

        elif preprocessing_operation == "Handle Missing Values":
            column_to_handle = st.sidebar.selectbox("Select column to handle missing values", df.columns)
            handling_method = st.sidebar.selectbox("Select handling method", 
                ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with custom value"])
            
            if handling_method == "Fill with custom value":
                custom_value = st.sidebar.text_input("Enter custom value")
            
            if st.sidebar.button("Apply Handling"):
                if handling_method == "Drop rows":
                    df.dropna(subset=[column_to_handle], inplace=True)
                elif handling_method == "Fill with mean":
                    df[column_to_handle].fillna(df[column_to_handle].mean(), inplace=True)
                elif handling_method == "Fill with median":
                    df[column_to_handle].fillna(df[column_to_handle].median(), inplace=True)
                elif handling_method == "Fill with mode":
                    df[column_to_handle].fillna(df[column_to_handle].mode()[0], inplace=True)
                elif handling_method == "Fill with custom value":
                    df[column_to_handle].fillna(custom_value, inplace=True)
                
                st.success(f"Applied {handling_method} to column {column_to_handle}")
                display_preview(df)

        elif preprocessing_operation == "Detect Outliers":
            column_to_check = st.sidebar.selectbox("Select column to check for outliers", df.select_dtypes(include=[np.number]).columns)
            
            Q1 = df[column_to_check].quantile(0.25)
            Q3 = df[column_to_check].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[column_to_check] < lower_bound) | (df[column_to_check] > upper_bound)]
            
            st.write(f"### Outliers in {column_to_check}")
            st.write(f"Number of outliers: {len(outliers)}")
            st.write("Outlier rows:")
            st.write(outliers)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=df[column_to_check], ax=ax)
            plt.title(f'Boxplot of {column_to_check}')
            st.pyplot(fig)

        elif preprocessing_operation == "Check Data Types":
            st.write("### Data Types")
            st.write(df.dtypes)

        elif preprocessing_operation == "Normalize Data":
            col_to_normalize = st.sidebar.selectbox("Select column to normalize", df.select_dtypes(include=[np.number]).columns)
            normalization_method = st.sidebar.selectbox("Select normalization method", ["Min-Max Scaling", "Standardization"])
            
            if st.sidebar.button("Normalize"):
                if normalization_method == "Min-Max Scaling":
                    scaler = MinMaxScaler()
                    df[col_to_normalize] = scaler.fit_transform(df[[col_to_normalize]])
                elif normalization_method == "Standardization":
                    scaler = StandardScaler()
                    df[col_to_normalize] = scaler.fit_transform(df[[col_to_normalize]])
                
                st.success(f"Normalized column '{col_to_normalize}' using {normalization_method}")
                display_preview(df)
                
                
                
    #////////////////////////////////////////////////////////////////////////////////////////////////////
                        #///  ADVANCE OPERATIONS ///
                        
                        
                        
    elif operation_type == "Advanced Operations":
        st.sidebar.subheader("Advanced Operations")
        advanced_operation = st.sidebar.selectbox("Select Advanced Operation", 
            ["Group By","Pivot Table", "Merge DataFrames","Compare Columns","Encode categorical data"])
        
        if advanced_operation == "Group By":
            group_by_col = st.sidebar.selectbox("Select column to group by", df.columns)
            agg_function = st.sidebar.selectbox("Select aggregation function", ["mean", "sum", "count", "min", "max"])
            if st.sidebar.button("Apply Group By"):
                grouped_data = df.groupby(group_by_col).agg(agg_function)
                st.write("### Grouped Data")
                st.write(grouped_data)

        elif advanced_operation == "Pivot Table":
            index_col = st.sidebar.selectbox("Select index column", df.columns)
            columns_col = st.sidebar.selectbox("Select columns for pivot", df.columns)
            values_col = st.sidebar.selectbox("Select values column", df.columns)
            agg_function = st.sidebar.selectbox("Select aggregation function", ["mean", "sum", "count", "min", "max"])
            if st.sidebar.button("Create Pivot Table"):
                pivot_data = pd.pivot_table(df, index=index_col, columns=columns_col, values=values_col, aggfunc=agg_function)
                st.write("### Pivot Table")
                st.write(pivot_data)

        elif advanced_operation == "Merge DataFrames":
            st.write("### Merge DataFrames")
            merge_type = st.sidebar.selectbox("Select Merge Type", ["inner", "outer", "left", "right"])
            st.sidebar.text("DataFrame to merge")
            uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file:
                new_df = pd.read_csv(uploaded_file)
                merge_on = st.sidebar.selectbox("Select column to merge on", df.columns.intersection(new_df.columns))
                if st.sidebar.button("Merge DataFrames"):
                    merged_df = pd.merge(df, new_df, on=merge_on, how=merge_type)
                    st.write("### Merged DataFrame")
                    display_preview(merged_df)
                     
        elif advanced_operation == "Compare Columns":
           st.write("### Compare Two Columns Note : Please before doing any filteration fill the missing values first")
    
    # Select the first column for comparison
        column1 = st.selectbox("Select the first column for comparison", df.columns)
    
    # Select the second column for comparison
        column2 = st.selectbox("Select the second column for comparison", df.columns)
        unique_values_column2 = df[column2].unique()
    # Select values from the first column
    #selected_values_column1 = st.multiselect(
        #f"Select values from {column1}",
        #unique_values_column1
    #)
    
    # Select values from the second column
        selected_values_column2 = st.multiselect(
        f"Select values from {column2}",
        unique_values_column2
    )

    # Button to compare the columns
    if st.button("Compare Columns"):
        # Filter the DataFrame based on selected values
        filtered_data = df[
            #(df[column1].isin(selected_values_column1)) & 
            (df[column2].isin(selected_values_column2))
        ]

        # Create a DataFrame to show the comparison
        comparison_data = filtered_data[[column1, column2]].copy()

        # Create a new column to indicate whether the values in the two columns are equal or not
        comparison_data['Comparison Result'] = comparison_data[column1] == comparison_data[column2]
        
        # Display the comparison DataFrame
        st.write("### Comparison Result")
        st.write(comparison_data)
        
        
        
# Only show the feature engineering options when "Feature Engineering" is selected
# Assuming 'data_operations' is selected by the user in a previous step:
    elif data_operations == "Feature Engineering":
     st.write("### Feature Engineering")
    
    # Select the operation for feature engineering
    feature_operation = st.selectbox(
        "Select Feature Engineering Operation",
        ["Create New Column", "Transform Existing Column", "Feature Scaling", "Feature Encoding", "Binning"]
    )

    # Only show the options and perform operations when the user selects an operation
    if feature_operation == "Create New Column":
        column1 = st.selectbox("Select the first column", df.columns)
        column2 = st.selectbox("Select the second column", df.columns)
        operation = st.selectbox("Select the operation", ["Add", "Subtract", "Multiply", "Divide"])
        
        if st.button("Create New Column"):
            if operation == "Add":
                df['New Column'] = df[column1] + df[column2]
            elif operation == "Subtract":
                df['New Column'] = df[column1] - df[column2]
            elif operation == "Multiply":
                df['New Column'] = df[column1] * df[column2]
            elif operation == "Divide":
                df['New Column'] = df[column1] / df[column2].replace(0, np.nan)
            
            st.success("New column created successfully!")
            st.write(df[['New Column']].head())

    elif feature_operation == "Transform Existing Column":
        column_to_transform = st.selectbox("Select the column to transform", df.columns)
        transformation = st.selectbox("Select transformation", ["Log", "Square Root", "Square"])
        
        if st.button("Transform Column"):
            if transformation == "Log":
                df['Transformed Column'] = np.log(df[column_to_transform].replace(0, np.nan))
            elif transformation == "Square Root":
                df['Transformed Column'] = np.sqrt(df[column_to_transform].replace(0, np.nan))
            elif transformation == "Square":
                df['Transformed Column'] = df[column_to_transform] ** 2
            
            st.success("Column transformed successfully!")
            st.write(df[['Transformed Column']].head())

    # Feature scaling: standardization or normalization
    elif feature_operation == "Feature Scaling":
        column_to_scale = st.selectbox("Select the column to scale", df.columns)
        scaling_method = st.selectbox("Select Scaling Method", ["Standardization", "Normalization"])
        
        if st.button("Scale Column"):
            if scaling_method == "Standardization":
                df['Scaled Column'] = (df[column_to_scale] - df[column_to_scale].mean()) / df[column_to_scale].std()
            elif scaling_method == "Normalization":
                df['Scaled Column'] = (df[column_to_scale] - df[column_to_scale].min()) / (df[column_to_scale].max() - df[column_to_scale].min())
            
            st.success("Column scaled successfully!")
            st.write(df[['Scaled Column']].head())

    # Feature encoding: label encoding or one-hot encoding
    elif feature_operation == "Feature Encoding":
        column_to_encode = st.selectbox("Select the column to encode", df.columns)
        encoding_method = st.selectbox("Select Encoding Method", ["Label Encoding", "One-Hot Encoding"])
        
        if st.button("Encode Column"):
            if encoding_method == "Label Encoding":
                le = LabelEncoder()
                df['Encoded Column'] = le.fit_transform(df[column_to_encode])
            elif encoding_method == "One-Hot Encoding":
                df = pd.get_dummies(df, columns=[column_to_encode], drop_first=True)
                
            st.success("Column encoded successfully!")
            st.write(df.head())

    # Binning: equal-width or custom binning
    elif feature_operation == "Binning":
        column_to_bin = st.selectbox("Select the column to bin", df.select_dtypes(include=[np.number]).columns)
        
        binning_method = st.selectbox("Select Binning Method", ["Equal-Width Binning", "Custom Binning"])
        
        if binning_method == "Equal-Width Binning":
            num_bins = st.slider("Select number of bins", 2, 10, 5)
            if st.button("Bin Column"):
                df['Binned Column'] = pd.cut(df[column_to_bin], bins=num_bins)
                st.success(f"Column binned into {num_bins} bins!")
                st.write(df[['Binned Column']].head())
        
        elif binning_method == "Custom Binning":
            bin_edges = st.text_input("Enter custom bin edges (comma-separated values)", "0, 10, 20, 30, 50")
            labels = st.text_input("Enter labels for the bins (comma-separated)", "Low, Medium, High")
            
            if st.button("Bin Column with Custom Ranges"):
                try:
                    bin_edges = [float(x) for x in bin_edges.split(',')]
                    labels = labels.split(',')
                    df['Binned Column'] = pd.cut(df[column_to_bin], bins=bin_edges, labels=labels)
                    st.success("Column binned with custom ranges successfully!")
                    st.write(df[['Binned Column']].head())
                except:
                    st.error("Please ensure the bin edges and labels are correctly entered.")


   
    # Download updated DataFrame
    st.sidebar.subheader("Download Updated Data")
    file_format = st.sidebar.radio("Choose file format", ["CSV", "Excel"])
    if file_format == "CSV":
        csv = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("Download Updated CSV", data=csv, file_name='updated_data.csv', mime='text/csv')
    else:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_data = output.getvalue()
        st.sidebar.download_button("Download Updated Excel", data=excel_data, file_name='updated_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        

    return df  # Return the dataframe in case it was updated
    
    ## plotting/////////////////////////////////////////////////////////////////////////////////

def visualization(df):
    st.sidebar.title("📊 Visualization Options")
    
    plot_type = st.sidebar.radio("Select Plot Type", ["Single Column", "X vs Y", "Multiple Columns"])
    
    if plot_type == "Single Column":
        single_column_plot(df)
    elif plot_type == "X vs Y":
        x_vs_y_plot(df)
    else:
        multiple_column_plot(df)

def single_column_plot(df):
    st.sidebar.subheader('Single Column Plot')
    
    single_col = st.sidebar.selectbox("Select a column for single-column plot", df.columns)
    single_plot_type = st.sidebar.selectbox("Select plot type", [
        'Histogram', 
        'Box Plot',
        'Pie Chart', 
        'Bar Chart', 
        'Heatmap', 
        'Dot Plot', 
        'Radar Chart', 
        'Density Plot'
    ])

    color = st.sidebar.color_picker("Select color for plot", "#1f77b4")

    # Number of bins input for Histogram and Density Plot
    bins = st.sidebar.number_input("Number of bins", min_value=1, value=10)

    if st.sidebar.button("Generate Single Column Plot"):
        st.write(f"### {single_plot_type}: {single_col}")
        fig, ax = plt.subplots()

        if single_plot_type == 'Histogram':
            sns.histplot(df[single_col], bins=bins, color=color, ax=ax)
        
        elif single_plot_type == 'Box Plot':
            sns.boxplot(y=df[single_col], ax=ax, color=color)

        elif single_plot_type == 'Pie Chart':
            # Prepare data for pie chart
            if df[single_col].dtype == 'object':  # Ensure it's categorical data
                counts = df[single_col].value_counts()
                ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
                ax.axis('equal')  # Equal aspect ratio ensures pie chart is circular.

        elif single_plot_type == 'Bar Chart':
            # Ensure the column is categorical
            if df[single_col].dtype == 'object':
                counts = df[single_col].value_counts()
                sns.barplot(x=counts.index, y=counts.values, ax=ax, color=color)
                ax.set_xticklabels(counts.index, rotation=45)

        elif single_plot_type == 'Heatmap':
            # Create a simple heatmap for a correlation matrix
            corr = df.corr()  # Calculate correlation matrix
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)

        elif single_plot_type == 'Dot Plot':
            # Create a dot plot
            sns.stripplot(y=df[single_col], ax=ax, color=color, jitter=True)

        elif single_plot_type == 'Radar Chart':
            # Prepare data for radar chart
            if df[single_col].dtype == 'object':
                counts = df[single_col].value_counts()
                categories = counts.index
                values = counts.values
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()

                # Complete the loop
                values = np.concatenate((values,[values[0]]))
                angles += angles[:1]

                ax.fill(angles, values, color=color, alpha=0.25)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)
                ax.set_yticklabels([])  # Hide y-tick labels

        elif single_plot_type == 'Density Plot':
            sns.kdeplot(df[single_col], ax=ax, color=color, fill=True, bw_adjust=bins)

        st.pyplot(fig)
       
def x_vs_y_plot(df):
    st.sidebar.subheader('X vs Y Column Plot')
    x_col = st.sidebar.selectbox("Select X-axis column", df.columns)
    y_col = st.sidebar.selectbox("Select Y-axis column", df.columns)
    plot_type = st.sidebar.selectbox("Select plot type", ['Scatter Plot', 'Line Plot', 'Bar Plot', 'Box Plot', 'Histogram','funnel'])

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
            fig = px.histogram(df, x=x_col, y=y_col, nbins=bins, color_discrete_sequence=[color])
        elif plot_type == 'funnel':
            fig = px.funnel(df, x=x_col, y=y_col,  color_discrete_sequence=[color])
        st.plotly_chart(fig)

def multiple_column_plot(df):
    st.sidebar.subheader('Multi-Column Visualization')
    multi_columns = st.sidebar.multiselect("Select columns for multi-column visualizations", df.columns)
    multi_plot_type = st.sidebar.selectbox("Select plot type for multi-columns", 
                                           ['Pair Plot', 'Scatter Plot', 'Box Plot', 'Histogram', 'funnel chart'])

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

# You can call this function in your main app
# visualization(df)
if __name__ == "__main__":
    main()
    
    