import streamlit as st
import pickle
import time
import pandas as pd

st.set_page_config(page_title= 'ML Portfolio', page_icon=":female-technologist:", layout='wide')

st.markdown("# Welcome to My ML Portfolio App!")
st.write("Portfolio by anggiasusanti7@gmail.com")

select_var=st.sidebar.selectbox("Want to open about?", ("Home", "Iris Species", "Heart Disease Risk"))

# HOME PAGE
def home():
    st.write("""
             App ini memprediksi **Iris Species** and **Heart Disease Risk** with ML
             
    """)

# Iris Species
def iris():
    st.write("""
             App ini memprediksi **Iris Species**
             
             Data diperoleh dari [Iris dataset](https://www.kaggle.com/uciml/iris) by UCIML. 
    """)
    st.sidebar.header('Fitur Input Pengguna:')
    
    uploaded_file = st.sidebar.file_uploader("Upload file CSV input anda", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            st.sidebar.header('Input Manual')
            SepalLengthCm = st.sidebar.slider('Panjang Sepal (cm)', min_value=4.3, value=6.5, max_value=10.0)
            SepalWidthCm = st.sidebar.slider('Lebar Sepal (cm)', min_value=2.0, value=3.3, max_value=5.0)
            PetalLengthCm = st.sidebar.slider('Panjang Petal (cm)',min_value= 1.0, value=4.5, max_value=9.0)
            PetalWidthCm = st.sidebar.slider('Lebar Petal (cm)',min_value= 0.1, value=1.4, max_value=5.0)
            data = {'Panjang Sepal (cm)': SepalLengthCm,
                    'Lebar Sepal (cm)': SepalWidthCm,
                    'Panjang Petal (cm)': PetalLengthCm,
                    'Lebar Petal (cm)': PetalWidthCm}
            features = pd.DataFrame(data, index=[0])
            return features
        input_df = user_input_features()

    # img = Image.open("iris.JPG")
    st.image("https://www.easytogrowbulbs.com/cdn/shop/products/BeardedIrisColorfullMix_VIS-sqWeb_8a293612-7bc0-4a9f-89ac-917e820d0ccb.jpg?v=1664472481&width=1920", width=500)
    
    button_var = st.sidebar.button('Prediksi!')

    if button_var:
        df = input_df
        st.write(df)

        with open("generate_iris.pkl", 'rb') as file:  
            loaded_model = pickle.load(file)    # load previous load model
        prediction = loaded_model.predict(df)   # do prediction according to input value
        result = ['Iris-setosa' if prediction == 0 else ('Iris-versicolor' if prediction == 1 else 'Iris-virginica')]
        
        st.subheader('Hasil Prediksi: ')
        output = str(result[0])
        with st.spinner('Wait for it...'):
            time.sleep(4)
            st.success(f"Prediksi App ini adalah: {output}")



# Heart Disease
def heart():
    st.write("""
    App ini memprediksi **Risiko Penyakit Jantung**
    
    Data diperoleh dari [Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML. 
    
    """)
    st.sidebar.header('Fitur Input Pengguna:')

    uploaded_file = st.sidebar.file_uploader("Upload file CSV input anda", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            st.sidebar.header('Manual Input')
            cp = st.sidebar.slider('Jenis nyeri dada', min_value=0, value=1, max_value=3, step=1, help="Jenis nyeri dada yang dialami")
            if cp == 0:
                wcp = "Typical Angina"
            elif cp == 1:
                wcp = "Atypical Angina"
            elif cp == 2:
                wcp = "Nyeri Non-Angina"
            else:
                wcp = "Tanpa Gejala"
            st.sidebar.write(f"Jenis Nyeri Dada: {wcp}")

            thalach = st.sidebar.slider('Denyut Jantung Maksimum Tercapai', min_value=60, value=150, max_value=220, step=1, help="Maximum heart rate achieved during exercise")
            slope = st.sidebar.selectbox('Kemiringan Segmen ST', options=[0, 1, 2], index=0, help="Slope of the peak exercise ST segment")
            oldpeak = st.sidebar.slider('Oldpeak (Segmen ST Menurun/Depresi)', min_value=0.0, value=1.0, max_value=6.2, step=0.1, help="ST depression induced by exercise relative to rest")
            exang = st.sidebar.radio('Angina Yang Dipicu Olahraga', options=['Yes', 'No'], index=0, help="Whether exercise induced angina is present")
            if exang == 'Yes':
                exang = 1
            else:
                exang = 0
            ca = st.sidebar.selectbox('Jumlah Pembuluh Darah Utama', options=[0, 1, 2, 3], index=0, help="Number of major vessels colored by fluoroscopy")
            thal = st.sidebar.selectbox('Thalassemia (Hasil tes thalium scan)', options=[1, 2, 3], index=0, help="Thalassemia result")
            sex= st.sidebar.radio('Sex', options=['Laki-laki', 'Perempuan'], index=0)
            if sex=="Perempuan":
                sex=0
            else :
                sex=1
            age = st.sidebar.number_input('Umur', min_value=29, max_value=77, value=30, step=1, help="Usia pasien pada tahun ini")

            data = {'cp': cp,
                    'thalach': thalach,
                    'slope': slope,
                    'oldpeak': oldpeak,
                    'exang': exang,
                    'ca': ca,
                    'thal': thal,
                    'sex' : sex,
                    'age': age}
            
            # Create a DataFrame from the input data
            features = pd.DataFrame(data, index=[0])
            return features

        input_df=user_input_features()
        st.image("https://drramjimehrotra.com/wp-content/uploads/2022/09/Women-Heart-Disease-min-resize.png")

        if st.sidebar.button('Prediksi!'):
            df= input_df
            st.write(df)
            with open("generate_heart_disease.pkl", 'rb') as file:  
                loaded_model = pickle.load(file)

            prediction_proba = loaded_model.predict_proba(df)
            if prediction_proba[:,1] >= 0.4:
                prediction = 1
            else: 
                prediction = 0
            
            result = ['Tidak Ada Risiko Penyakit Jantung' if prediction == 0 else 'Terdeteksi Risiko Penyakit Jantung']
            
            # Print the prediction result
            st.subheader('Hasil Prediksi: ')
            output = str(result[0])
            with st.spinner('Wait for it...'):
                time.sleep(4)
                if output == "Tidak Ada Risiko Penyakit Jantung":
                    st.success(f"Hasil Prediksi : {output}")
                if output == "Terdeteksi Risiko Penyakit Jantung":
                    st.error(f"Hasil Prediksi : {output}")
                    st.info("Silahkan berkonsultasi dengan Dokter untuk Evaluasi dan Saran lebih lanjut")
            
if select_var == "Home":
    home()
elif select_var == "Iris Species":
    iris()
elif select_var == "Heart Disease Risk":
    heart()
