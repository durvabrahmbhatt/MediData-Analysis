import streamlit as st
def main():
    st.title("MediData")

    # App Description
    st.write(
        """
    Welcome to MediData, an interactive tool for analyzing and visualizing Health analytics over 40 diseases.

    ### About MediData:
    MediData provides various features for exploring Health analytics data, including:

    - Prediction of Disease based on symptoms provided by user including description and precautions of the disease.
    - Clustering top nations based on Disease from which youth are suffering in the world.
    - Visualizing Death ratio in different nations for the disease.

    **Get started:** Use the sidebar to navigate through different sections and explore the features of MediData.

    #### Developer Information:
    MediData is developed using Streamlit, a powerful library for building interactive web applications with Python.

    **Developers:** Dhrumil Shah, Durva Brahmbhatt, Dhwani Sheth, Devansh Mehta
    """
    )




if __name__ == "__main__":
    main()