import streamlit as st
import pandas as pd

# (gunakan kode sebelumnya untuk menghasilkan df_result)

st.title("Perbandingan Klasifikasi Dataset Iris")
st.write("Hasil klasifikasi sebelum dan sesudah diskritisasi dengan K-Means.")

st.dataframe(df_result)