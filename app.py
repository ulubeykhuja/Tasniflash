from pathlib import Path
import io
import streamlit as st
from fastai.vision.all import *
from PIL import Image  # PIL bilan qayta o‘lchash uchun

st.title("Door, Drink or Telephone?")
st.markdown("""
**Bu ilova eshik, ichimlik yoki telefon rasmni klassifikatsiya qiladi.**  
Quyidagi tugma orqali rasm yuklang va natijani kuting.
""")

MODEL_PATH = Path(__file__).parent / "mixture.pkl"

file = st.file_uploader("Rasm yuklash", type=["png", "jpeg", "jpg", "gif", "svg"])
if file:
    # 240x240 pikselga ko‘rinishini moslab chiqarish
    st.image(file, caption="Yuklangan rasm (240×240)", width=240)

    try:
        # UploadedFile -> PIL.Image -> RGB -> 240×240
        raw = Image.open(io.BytesIO(file.read())).convert("RGB")
        raw_resized = raw.resize((240, 240))  # Model kirish o‘lchami
        img = PILImage.create(raw_resized)
    except Exception as e:
        st.error("Rasmni o‘qishda yoki 240×240 ga o‘lchashda xatolik.")
        st.exception(e)
        st.stop()

    try:
        learn = load_learner(MODEL_PATH)
    except Exception as e:
        st.error("Model fayli (mixture.pkl) topilmadi yoki o‘qilmadi.")
        st.exception(e)
        st.stop()

    try:
        pred, pred_id, probs = learn.predict(img)
        st.success(f"Bashorat: {pred}")
        st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

        import plotly.express as px
        fig = px.bar(x=probs * 100,
                     y=learn.dls.vocab,
                     orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error("Predict bosqichida xato yuz berdi.")
        st.exception(e)
