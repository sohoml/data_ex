import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def main():
    st.title('Data Visualization Web Tool')

    # Sidebar for plot and column selection
    st.sidebar.header('User Input Features')

    # File uploader for CSV/Excel
    uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])
    
    if uploaded_file:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)

            st.write("## Preview of Data")
            st.dataframe(df.head())  # Show a preview of the data

            # Select columns for X and Y axes in the sidebar
            all_columns = df.columns.tolist()
            x_col = st.sidebar.selectbox("Select X-axis column", all_columns)
            y_col = st.sidebar.selectbox("Select Y-axis column", all_columns)

            # Allow multiple column selection
            multi_columns = st.sidebar.multiselect("Select columns for multi-column visualizations", all_columns)

            # Select the type of plot in the sidebar
            plot_type = st.sidebar.selectbox(
                "Select Plot Type", 
                ['Scatter Plot', 'Line Plot', 'Bar Plot', 'Box Plot', 'Correlation Matrix', 'Pair Plot']
            )

            st.write(f"## {plot_type}: {x_col} vs {y_col}")

            # Create the plot based on user's choice
            if plot_type == 'Scatter Plot':
                fig = px.scatter(df, x=x_col, y=y_col)
                st.plotly_chart(fig)
            
            elif plot_type == 'Line Plot':
                fig = px.line(df, x=x_col, y=y_col)
                st.plotly_chart(fig)
            
            elif plot_type == 'Bar Plot':
                fig = px.bar(df, x=x_col, y=y_col)
                st.plotly_chart(fig)
            
            elif plot_type == 'Box Plot':
                fig = px.box(df, x=x_col, y=y_col)
                st.plotly_chart(fig)

            elif plot_type == 'Correlation Matrix':
                if len(multi_columns) > 1:
                    corr_matrix = df[multi_columns].corr()
                    st.write("## Correlation Matrix")
                    fig, ax = plt.subplots()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                    st.pyplot(fig)
                else:
                    st.warning("Please select at least two columns for the Correlation Matrix.")

            elif plot_type == 'Pair Plot':
                if len(multi_columns) > 1:
                    st.write("## Pair Plot")
                    pair_plot = sns.pairplot(df[multi_columns])
                    st.pyplot(pair_plot)
                else:
                    st.warning("Please select at least two columns for the Pair Plot.")
            
            else:
                st.error("Invalid plot type selected.")

            # Option for additional visualizations (e.g., histograms)
            st.write("## Histogram")
            selected_column = st.selectbox("Select a column for histogram", all_columns)
            fig, ax = plt.subplots()
            sns.histplot(df[selected_column], ax=ax)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == '__main__':
    main()
