import streamlit as st
from fastai.vision.all import *
import plotly.express as px

#title
st.title("Door, Drink or Telefhone?")

st.markdown(
    """
    **Bu ilova eshik, ichimlik yoki telefon rasmni klassifikatsiya qiladi.**  
    Quyidagi tugma orqali rasm yuklang va natijani kuting.
    """
)

#rasmni joylash
file = st.file_uploader('Rasm yuklash',  type=['png', 'jpeg', 'gif', 'svg'])
if file:
    st.image(file)
    # PIL convert
    img = PILImage.create(file) 

    #model
    model = load_learner('mixture_new.pkl')

    #predict
    pred, pred_id, probs = model.predict(img) 
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}")

    #plotting
    fig = px.bar(x=probs*100, 
                 y=model.dls.vocab,
                 orientation='h')
    st.plotly_chart(fig)
