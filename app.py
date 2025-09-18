from pathlib import Path
import io
import streamlit as st
from fastai.vision.all import *

st.title("Door, Drink or Telephone?")
st.markdown("""
**Bu ilova eshik, ichimlik yoki telefon rasmni klassifikatsiya qiladi.**  
Quyidagi tugma orqali rasm yuklang va natijani kuting.
""")

MODEL_PATH = Path(__file__).parent / "mixture.pkl"

file = st.file_uploader("Rasm yuklash", type=["png","jpeg","jpg","gif","svg"])
if file:
    st.image(file, caption="Yuklangan rasm", use_container_width=True)

    try:
        img = PILImage.create(io.BytesIO(file.read()))
    except Exception as e:
        st.error("Rasmni o‘qishda xatolik.")
        st.exception(e)
        st.stop()

    try:
        learn = load_learner(MODEL_PATH)
        pred, pred_id, probs = learn.predict(img)
        st.success(f"Bashorat: {pred}")
        st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

        import plotly.express as px
        st.plotly_chart(px.bar(x=probs*100, y=learn.dls.vocab, orientation="h"),
                        use_container_width=True)
    except Exception as e:
        st.error("Predict bosqichida xato yuz berdi.")
        st.exception(e)
