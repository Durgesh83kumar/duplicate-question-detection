import streamlit as st
import helper
import pickle
import numpy as np
import nltk

nltk.download('stopwords')
# Load model
model = pickle.load(open('model_hybrid_tfidf.pkl','rb'))

st.title("Quora Question Pairs Duplicate Detector")
st.markdown("Detect whether two questions have the same meaning using NLP.")

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

if q1 and q2 and st.button('Find'):
    with st.spinner("Analyzing questions..."):
        try:
            # Create feature vector (ensure 2D)
            query = helper.query_point_creator(q1, q2).reshape(1, -1)
            prob = model.predict_proba(query)[0][1]
            result = 1 if prob > 0.5 else 0

            # Display probability as text
            st.write(f"Duplicate Probability: {prob:.2f}")

            # Dynamic color based on probability
            bar_color = "#4CAF50" if prob > 0.5 else "#f44336"  # green or red
            bar_width = int(prob * 100)

            # Custom HTML progress bar
            progress_html = f"""
            <div style="background-color:#ddd; border-radius:5px; width:100%; height:25px;">
                <div style="
                    width:{bar_width}%;
                    height:100%;
                    background-color:{bar_color};
                    text-align:center;
                    color:white;
                    border-radius:5px;">
                    {bar_width}%
                </div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)

            # Colored message for duplicate vs not duplicate
            if result:
                st.markdown(f"<p style='color:green;font-weight:bold;'>Duplicate Questions</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='color:red;font-weight:bold;'>Not Duplicate</p>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")