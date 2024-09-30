import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def main():
    # Set a wide layout and page title
    st.set_page_config(page_title='Data Visualization Tool', layout='wide')

    # Injecting custom CSS and JavaScript for styling and animations
    st.markdown("""
        <style>
        body {
            background-color: #f5f5f5;
        }
        .css-1d391kg { 
            background-color: #e6f7ff;
        }
        .css-10trblm {
            font-size: 40px;
            font-weight: 700;
            color: #006699;
            animation: fadeIn 3s ease-in;
        }
        .css-145kmo2, .css-1avcm0n {
            color: #003366;
        }
        button {
            background-color: #004d80;
            color: white;
            border-radius: 8px;
        }
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        select {
            background-color: #f0f0f5;
            border-radius: 8px;
        }
        .custom-card {
            background-color: #004d80;
            color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.markdown("""
        <h1 style="text-align: center; color: #006699;">📊 Data Visualization Tool</h1>
        <p style="text-align: center; font-size: 18px;">
        Welcome to the Data Visualization Tool! This app helps you visualize data from CSV or Excel files.<br>
        You can choose between various chart types and customize them for deeper insights.
        </p>
        <hr style="border: 1px solid #004d80;">
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
            """, unsafe_allow_html=True)
            st.dataframe(df.head(10))  # Display first 10 rows
            st.markdown("</div>", unsafe_allow_html=True)

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
                # Data Operations Functionalities
                st.sidebar.title("🛠️ Data Operations")

                # 1. Find Null Values
                if st.sidebar.button("Find Null Values"):
                    null_summary = df.isnull().sum()
                    null_columns = null_summary[null_summary > 0]
                    if not null_columns.empty:
                        st.write("### Null Values Summary")
                        null_df = pd.DataFrame({
                            'Column Name': null_columns.index,
                            'Null Count': null_columns.values,
                            'Data Type': df.dtypes[null_columns.index].values
                        })
                        st.dataframe(null_df)  # Show null value summary
                    else:
                        st.success("No null values found in the dataset.")

                # 2. Check Data Types of Columns
                if st.sidebar.button("Check Data Types"):
                    st.write("### Data Types of Columns")
                    st.write(df.dtypes)

                # 3. Rename Columns
                rename_column = st.sidebar.selectbox("Select a column to rename", df.columns)
                new_name = st.sidebar.text_input("New name for the column", "")
                if st.sidebar.button("Rename Column"):
                    if new_name:
                        df.rename(columns={rename_column: new_name}, inplace=True)
                        st.success(f"Column '{rename_column}' renamed to '{new_name}'")
                    else:
                        st.warning("Please enter a new name for the column.")

                # 4. Drop Columns
                columns_to_drop = st.sidebar.multiselect("Select columns to drop", df.columns)
                if st.sidebar.button("Drop Selected Columns"):
                    df.drop(columns=columns_to_drop, inplace=True)
                    st.success(f"Dropped columns: {', '.join(columns_to_drop)}")

                # 5. Remove Duplicates
                if st.sidebar.button("Remove Duplicates"):
                    duplicate_count = df.duplicated().sum()
                    df.drop_duplicates(inplace=True)
                    st.success(f"Removed {duplicate_count} duplicate rows from the dataset.")

                # 6. Detect Outliers
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

                # 7. Download updated DataFrame
                def convert_df_to_csv(dataframe):
                    return dataframe.to_csv(index=False).encode('utf-8')

                csv = convert_df_to_csv(df)
                st.sidebar.download_button("Download Updated File", data=csv, file_name='updated_data.csv', mime='text/csv')

            if st.session_state['visualization']:
                # Visualization Functionalities
                st.sidebar.title("📊 Visualization Options")

                # Section for Plotting Options
                st.sidebar.subheader('Single Column Plot')
                single_col = st.sidebar.selectbox("Select a column for single-column plot", df.columns)
                single_plot_type = st.sidebar.selectbox("Select plot type", ['Histogram', 'Box Plot'])

                # New: Bins and Color for Single Column Plot
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

                # 2. X vs Y Visualizations (Scatter, Line, etc.)
                st.sidebar.subheader('X vs Y Column Plot')
                x_col = st.sidebar.selectbox("Select X-axis column", df.columns)
                y_col = st.sidebar.selectbox("Select Y-axis column", df.columns)
                plot_type = st.sidebar.selectbox("Select plot type", ['Scatter Plot', 'Line Plot', 'Bar Plot', 'Box Plot', 'Histogram'])

                # New: Bins and Color for X vs Y Plot
                if plot_type == 'Histogram':
                    bins = st.sidebar.number_input("Number of bins for histogram", min_value=1, value=10)
                    color = st.sidebar.color_picker("Select color for histogram", "#1f77b4")
                else:
                    color = st.sidebar.color_picker("Select color for X vs Y plots", "#1f77b4")

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

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

if __name__ == "__main__":
    main()
