import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import io

def main():
    # Set a wide layout and page title
    st.set_page_config(page_title='Data Visualization Tool', layout='wide')

    # Injecting custom CSS and JavaScript for styling and animations
    st.markdown("""
        <style>
        /* Customize the main background color */
        body {
            background-color: #f5f5f5;
        }
        /* Sidebar background and header */
        .css-1d391kg { 
            background-color: #e6f7ff;
        }
        /* Title styling */
        .css-10trblm {
            font-size: 40px;
            font-weight: 700;
            color: #006699;
            animation: fadeIn 3s ease-in;
        }
        /* Sidebar text styling */
        .css-145kmo2, .css-1avcm0n {
            color: #003366;
        }
        /* Button styles */
        button {
            background-color: #004d80;
            color: white;
            border-radius: 8px;
        }
        /* Animation for title */
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        /* Customized select box */
        select {
            background-color: #f0f0f5;
            border-radius: 8px;
        }
        /* Custom card component */
        .custom-card {
            background-color: #004d80;
            color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and description with custom HTML and animation
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

            # Sidebar menu for DataFrame operations
            st.sidebar.title("🛠️ Data Operations")

            # 1. Rename Columns
            rename_column = st.sidebar.selectbox("Select a column to rename", df.columns)
            new_name = st.sidebar.text_input("New name for the column", "")
            if st.sidebar.button("Rename Column"):
                if new_name:
                    df.rename(columns={rename_column: new_name}, inplace=True)
                    st.success(f"Column '{rename_column}' renamed to '{new_name}'")
                else:
                    st.warning("Please enter a new name for the column.")

            # 2. Drop Columns
            columns_to_drop = st.sidebar.multiselect("Select columns to drop", df.columns)
            if st.sidebar.button("Drop Selected Columns"):
                df.drop(columns=columns_to_drop, inplace=True)
                st.success(f"Dropped columns: {', '.join(columns_to_drop)}")

            # 3. Download updated DataFrame
            def convert_df_to_csv(dataframe):
                return dataframe.to_csv(index=False).encode('utf-8')

            csv = convert_df_to_csv(df)
            st.sidebar.download_button("Download Updated File", data=csv, file_name='updated_data.csv', mime='text/csv')

            # Section for Plotting Options
            st.sidebar.title("📊 Plotting Options")

            # 1. Single Column Visualizations (Histograms, Box Plots, etc.)
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

            # 3. Multi-column visualizations
            st.sidebar.subheader('Multi-Column Visualization')
            multi_columns = st.sidebar.multiselect("Select columns for multi-column visualizations", df.columns)

            # New: Plot types for Multi-Column Visualization
            multi_plot_type = st.sidebar.selectbox("Select plot type for multi-columns", 
                                                   ['Pair Plot', 'Scatter Plot', 'Box Plot', 'Histogram'])

            # New: Bins option for Multi-column Plot
            bins = st.sidebar.number_input("Number of bins (for applicable plots)", min_value=1, value=10)

            if st.sidebar.button("Generate Multi-Column Plot"):
                if len(multi_columns) < 2:
                    st.warning("Please select at least two columns.")
                else:
                    st.write(f"### {multi_plot_type} of {', '.join(multi_columns)}")
                    if multi_plot_type == 'Pair Plot':
                        pair_plot = sns.pairplot(df[multi_columns])
                        st.pyplot(pair_plot)
                    elif multi_plot_type == 'Scatter Plot':
                        fig = sns.scatterplot(data=df, x=multi_columns[0], y=multi_columns[1])
                        plt.title(f'Scatter Plot: {multi_columns[0]} vs {multi_columns[1]}')
                        st.pyplot(fig)
                    elif multi_plot_type == 'Box Plot':
                        fig = plt.figure()
                        sns.boxplot(data=df[multi_columns])
                        plt.title(f'Box Plot of {", ".join(multi_columns)}')
                        st.pyplot(fig)
                    elif multi_plot_type == 'Histogram':
                        fig = plt.figure()
                        for col in multi_columns:
                            sns.histplot(df[col], bins=bins, label=col, kde=True)
                        plt.title('Histogram of Selected Columns')
                        plt.legend()
                        st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == '__main__':
    main()
