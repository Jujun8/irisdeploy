# app.py
import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd # Optional: for displaying probabilities nicely

# --- Page Configuration (Optional but Recommended) ---
st.set_page_config(
    page_title="Model Deployment App",
    page_icon="🚀",
    layout="wide" # or "centered"
)

# --- Global Variables / Constants (Placeholder - ADAPT THESE!) ---
# You MUST adapt these based on your model.pkl
EXPECTED_NUM_FEATURES = 4  # How many input features does your model expect?
# If you know your feature names, define them:
# FEATURE_NAMES = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
FEATURE_NAMES = [f"Feature {i+1}" for i in range(EXPECTED_NUM_FEATURES)]

# If you know your target class names (for classification models):
# CLASS_NAMES = ["Class A", "Class B", "Class C"]
CLASS_NAMES = None # Set this if you have a classification model and know the names

# --- Load Model ---
@st.cache_resource # Caches the model loading
def load_model_from_file(model_path='model.pkl'):
    """Loads a model from a .pkl file."""
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. "
                 f"Please ensure it's in the same directory as app.py.")
        return None
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        # If your .pkl file is a dictionary, e.g., {'model': actual_model_object, ...}
        # you might need: model = model['model']
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model_from_file()

# --- Page Functions ---

def home_page():
    """Displays the Home Page content."""
    st.title("🏠 Welcome to the Model Deployment App!")
    st.markdown("""
    This application allows you to interact with a deployed machine learning model.
    Use the navigation sidebar to:
    - **Predict:** Input data to get predictions from the model.
    - **About:** Learn more about this application and the model.

    **Note:** This is a template. You'll need to adapt the 'Predict' page
    based on the specific input features your `model.pkl` expects.
    """)
    if model is None:
        st.warning("Model could not be loaded. Please check the `model.pkl` file and errors above.")
    else:
        st.success("Model loaded successfully! Navigate to the 'Predict' page to use it.")

def predict_page():
    """Displays the Prediction Page content."""
    st.title("🔮 Make a Prediction")

    if model is None:
        st.error("Model is not loaded. Cannot make predictions.")
        st.info("Please check the Home page for error messages regarding model loading.")
        return

    st.markdown(f"Enter the values for the **{EXPECTED_NUM_FEATURES} features** your model expects:")

    input_features = []
    # Create columns for better layout if many features
    # num_cols = 2
    # cols = st.columns(num_cols)

    for i in range(EXPECTED_NUM_FEATURES):
        # For simplicity, using number_input. You might want sliders, selectboxes, etc.
        # with cols[i % num_cols]: # Use this if using columns
        feature_val = st.number_input(
            label=FEATURE_NAMES[i],
            value=0.0, # Default value
            step=0.1,
            key=f"feature_{i}" # Unique key for each widget
        )
        input_features.append(feature_val)

    if st.button("✨ Predict", type="primary"):
        if len(input_features) == EXPECTED_NUM_FEATURES:
            # Convert to NumPy array and reshape for a single sample
            try:
                input_array = np.array(input_features).reshape(1, -1)

                # --- Make Prediction ---
                prediction = model.predict(input_array)
                prediction_text = f"{prediction[0]}" # Basic display

                st.subheader("Prediction Result:")
                st.success(f"Predicted Output: **{prediction_text}**")

                # --- (Optional) Display Probabilities for Classification Models ---
                if hasattr(model, "predict_proba"):
                    try:
                        probabilities = model.predict_proba(input_array)[0]
                        st.subheader("Prediction Probabilities:")
                        if CLASS_NAMES and len(CLASS_NAMES) == len(probabilities):
                            proba_df = pd.DataFrame({'Class': CLASS_NAMES, 'Probability': probabilities})
                            proba_df = proba_df.sort_values(by='Probability', ascending=False).reset_index(drop=True)
                            st.dataframe(proba_df.style.format({"Probability": "{:.2%}"}))
                            st.bar_chart(proba_df.set_index('Class'))
                        else:
                            st.write("Probabilities per class:")
                            for i, proba in enumerate(probabilities):
                                st.write(f"  - Class {i}: {proba*100:.2f}%")
                    except Exception as e:
                        st.warning(f"Could not get probabilities: {e}")

            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.error("Ensure your input features match what the model was trained on.")
        else:
            st.error(f"Please provide all {EXPECTED_NUM_FEATURES} feature values.")

    st.markdown("---")
    st.sidebar.subheader("Current Inputs:")
    for i in range(EXPECTED_NUM_FEATURES):
        st.sidebar.write(f"{FEATURE_NAMES[i]}: {input_features[i]}")


def about_page():
    """Displays the About Page content."""
    st.title("ℹ️ About This Application")
    st.markdown("""
    This Streamlit application serves as a deployment interface for a pre-trained machine learning model (`model.pkl`).

    **Functionality:**
    - **Load Model:** Attempts to load `model.pkl` from the application's root directory.
    - **Prediction Interface:** Provides input fields for the model's features and displays the prediction.
    - **Multi-Page Structure:** Uses a sidebar for navigation between different sections of the app.

    **To Customize This Template:**
    1.  Place your `model.pkl` file in the same directory as this `app.py` script.
    2.  **Crucially, update the global variables at the top of `app.py`:**
        *   `EXPECTED_NUM_FEATURES`: Set this to the number of input features your model requires.
        *   `FEATURE_NAMES` (optional but recommended): A list of names for your input features. This will make the UI more user-friendly.
        *   `CLASS_NAMES` (optional, for classification): If your model is a classifier, provide a list of your class labels in the correct order. This helps in displaying probabilities.
    3.  Adjust the input widgets (e.g., `st.number_input`, `st.slider`) in the `predict_page()` function to best suit your feature types (numerical, categorical, etc.) and their expected ranges.
    4.  Modify how the `prediction` and `probabilities` are displayed if needed.
    """)
    if model:
        st.subheader("Loaded Model Information:")
        st.text(f"Model Type: {type(model).__name__}")
        # You can try to display more model info if available and safe
        # For scikit-learn models, you might access parameters:
        # if hasattr(model, 'get_params'):
        #     st.json(model.get_params())
    else:
        st.warning("Model is not currently loaded.")

# --- Main App Logic (Sidebar Navigation) ---
st.sidebar.title("Navigation")
PAGES = {
    "🏠 Home": home_page,
    "🔮 Predict": predict_page,
    "ℹ️ About": about_page
}
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# Call the selected page function
page_function = PAGES[selection]
page_function()

st.sidebar.markdown("---")
st.sidebar.info("App created with Streamlit.")
