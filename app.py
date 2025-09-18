import streamlit as st
from fastai.vision.all import *
import plotly.express as px

# Title
st.title("Door, Drink or Telephone?")

st.markdown(
    """
    **Bu ilova eshik, ichimlik yoki telefon rasmni klassifikatsiya qiladi.**  
    Quyidagi tugma orqali rasm yuklang va natijani kuting.
    """
)

# Rasmni joylash
file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif', 'svg'])
if file:
    st.image(file)
    # PIL convert
    img = PILImage.create(file) 

    # Model
    model = load_learner('mixture.pkl')

    # Predict
    pred, pred_id, probs = model.predict(img) 
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}")

    # Plotting
    fig = px.bar(x=probs*100, 
                 y=model.dls.vocab,
                 orientation='h')
    st.plotly_chart(fig)
