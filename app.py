import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import zscore
import pickle

# Konfigurasi halaman
st.set_page_config(
    page_title="ML Model Evaluation & Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-result {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk load dan preprocess data
@st.cache_data
def load_and_preprocess_data():
    try:
        # Ganti path sesuai dengan lokasi file Anda
        df = pd.read_csv('ObesityDataSet.csv')
        
        # Data cleaning
        df_cleaned = df.copy()
        df_cleaned = df_cleaned.drop_duplicates()
        
        # Konversi kolom numerik
        num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        for col in num_cols:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        
        # Handle missing values
        for col in num_cols:
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
        
        cat_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        for col in cat_cols:
            if col in df_cleaned.columns:
                df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
        
        return df_cleaned, num_cols, cat_cols
    except FileNotFoundError:
        st.error("File ObesityDataSet.csv tidak ditemukan. Pastikan file berada di direktori yang sama dengan aplikasi ini.")
        return None, None, None

# Fungsi untuk preprocessing features - DIPERBAIKI
def preprocess_features(df_cleaned):
    X = df_cleaned.drop('NObeyesdad', axis=1)
    y = df_cleaned['NObeyesdad']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Pisahkan kolom numerik dan kategorikal
    numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    
    # Filter kolom yang benar-benar ada
    numerical_cols = [col for col in numerical_cols if col in X_train.columns]
    categorical_cols = [col for col in categorical_cols if col in X_train.columns]
    
    # Preprocessing numerik
    num_imputer = SimpleImputer(strategy='median')
    X_train_num = pd.DataFrame(num_imputer.fit_transform(X_train[numerical_cols]), columns=numerical_cols)
    X_test_num = pd.DataFrame(num_imputer.transform(X_test[numerical_cols]), columns=numerical_cols)
    
    # Standardisasi data numerik
    scaler = StandardScaler()
    X_train_num_scaled = pd.DataFrame(scaler.fit_transform(X_train_num), columns=numerical_cols)
    X_test_num_scaled = pd.DataFrame(scaler.transform(X_test_num), columns=numerical_cols)
    
    # Preprocessing kategorikal
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train_cat = pd.DataFrame(cat_imputer.fit_transform(X_train[categorical_cols]), columns=categorical_cols)
        X_test_cat = pd.DataFrame(cat_imputer.transform(X_test[categorical_cols]), columns=categorical_cols)
        
        # One-hot encoding
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit(X_train_cat)
        
        X_train_cat_encoded = pd.DataFrame(encoder.transform(X_train_cat), columns=encoder.get_feature_names_out(categorical_cols))
        X_test_cat_encoded = pd.DataFrame(encoder.transform(X_test_cat), columns=encoder.get_feature_names_out(categorical_cols))
        
        # Gabungkan
        X_train_final = pd.concat([X_train_num_scaled.reset_index(drop=True), X_train_cat_encoded.reset_index(drop=True)], axis=1)
        X_test_final = pd.concat([X_test_num_scaled.reset_index(drop=True), X_test_cat_encoded.reset_index(drop=True)], axis=1)
    else:
        X_train_final = X_train_num_scaled
        X_test_final = X_test_num_scaled
        encoder = None
        cat_imputer = None
    
    # Simpan nama fitur untuk konsistensi
    feature_names = X_train_final.columns.tolist()
    
    return X_train_final, X_test_final, y_train, y_test, num_imputer, cat_imputer, encoder, scaler, feature_names

# Fungsi evaluasi model
def evaluate_model(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

# Fungsi untuk train model
@st.cache_resource
def train_models(X_train, y_train):
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'SVC': SVC()
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

# Fungsi untuk hyperparameter tuning
@st.cache_resource
def tune_models(X_train, y_train):
    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'KNN': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        },
        'SVC': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    }
    
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'SVC': SVC(probability=True)
    }
    
    tuned_models = {}
    best_params = {}
    
    for name, model in models.items():
        with st.spinner(f'Tuning {name}...'):
            grid = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train, y_train)
            tuned_models[name] = grid.best_estimator_
            best_params[name] = grid.best_params_
    
    return tuned_models, best_params

# Fungsi untuk preprocessing input prediksi - DIPERBAIKI TOTAL
def preprocess_user_input(user_data, preprocessors):
    """
    Preprocessing input user dengan memastikan konsistensi dengan training data
    """
    try:
        # Buat DataFrame dari input user
        input_df = pd.DataFrame([user_data])
        
        # Definisikan kolom berdasarkan yang digunakan saat training
        numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        
        # Filter kolom yang ada
        numerical_cols = [col for col in numerical_cols if col in input_df.columns]
        categorical_cols = [col for col in categorical_cols if col in input_df.columns]
        
        # Preprocessing numerik
        if numerical_cols and preprocessors['num_imputer'] is not None:
            input_num = input_df[numerical_cols].copy()
            # Imputation
            input_num_imputed = pd.DataFrame(
                preprocessors['num_imputer'].transform(input_num), 
                columns=numerical_cols
            )
            # Standardisasi
            input_num_scaled = pd.DataFrame(
                preprocessors['scaler'].transform(input_num_imputed), 
                columns=numerical_cols
            )
        else:
            input_num_scaled = pd.DataFrame()
        
        # Preprocessing kategorikal
        if categorical_cols and preprocessors['cat_imputer'] is not None and preprocessors['encoder'] is not None:
            input_cat = input_df[categorical_cols].copy()
            # Imputation
            input_cat_imputed = pd.DataFrame(
                preprocessors['cat_imputer'].transform(input_cat), 
                columns=categorical_cols
            )
            # One-hot encoding
            input_cat_encoded = pd.DataFrame(
                preprocessors['encoder'].transform(input_cat_imputed), 
                columns=preprocessors['encoder'].get_feature_names_out(categorical_cols)
            )
        else:
            input_cat_encoded = pd.DataFrame()
        
        # Gabungkan fitur
        if not input_num_scaled.empty and not input_cat_encoded.empty:
            input_final = pd.concat([
                input_num_scaled.reset_index(drop=True), 
                input_cat_encoded.reset_index(drop=True)
            ], axis=1)
        elif not input_num_scaled.empty:
            input_final = input_num_scaled
        elif not input_cat_encoded.empty:
            input_final = input_cat_encoded
        else:
            raise ValueError("Tidak ada data yang dapat diproses")
        
        # Pastikan urutan dan keberadaan fitur sesuai dengan training
        expected_features = preprocessors['feature_names']
        
        # Tambahkan kolom yang hilang dengan nilai 0
        for feature in expected_features:
            if feature not in input_final.columns:
                input_final[feature] = 0
        
        # Reorder kolom sesuai urutan training
        input_final = input_final[expected_features]
        
        return input_final
        
    except Exception as e:
        st.error(f"Error dalam preprocessing: {str(e)}")
        raise e

# Header aplikasi
st.markdown('<h1 class="main-header">📊 Evaluasi Model ML & Prediksi Obesitas</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🔧 Pengaturan")
page = st.sidebar.selectbox("Pilih Halaman", ["Model Evaluation", "Prediksi Obesitas"])

# Load data
df_cleaned, num_cols, cat_cols = load_and_preprocess_data()

if df_cleaned is not None:
    
    if page == "Model Evaluation":
        show_data_info = st.sidebar.checkbox("Tampilkan Info Dataset", value=True)
        show_correlation = st.sidebar.checkbox("Tampilkan Korelasi Fitur", value=False)
        run_tuning = st.sidebar.button("🚀 Jalankan Hyperparameter Tuning", type="primary")
        
        # Info dataset
        if show_data_info:
            st.subheader("📋 Informasi Dataset")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Jumlah Baris", df_cleaned.shape[0])
            with col2:
                st.metric("Jumlah Kolom", df_cleaned.shape[1])
            with col3:
                st.metric("Fitur Numerik", len(num_cols))
            with col4:
                st.metric("Fitur Kategorikal", len(cat_cols))
            
            # Distribusi target
            st.subheader("📊 Distribusi Kelas Target")
            fig, ax = plt.subplots(figsize=(12, 6))
            target_counts = df_cleaned['NObeyesdad'].value_counts()
            ax.bar(target_counts.index, target_counts.values, color='skyblue')
            ax.set_title('Distribusi Kelas Target (Obesity Level)')
            ax.set_xlabel('Obesity Level')
            ax.set_ylabel('Jumlah')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        # Preprocessing
        X_train_final, X_test_final, y_train, y_test, num_imputer, cat_imputer, encoder, scaler, feature_names = preprocess_features(df_cleaned)
        
        # Korelasi matrix
        if show_correlation and num_cols:
            st.subheader("🔗 Heatmap Korelasi Fitur Numerik")
            numerical_data = X_train_final.select_dtypes(include=['int64', 'float64'])
            if not numerical_data.empty:
                correlation_matrix = numerical_data.corr()
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5, ax=ax)
                ax.set_title("Heatmap Korelasi Fitur Numerik")
                st.pyplot(fig)
        
        # Train model awal
        st.subheader("🎯 Evaluasi Model Awal")
        
        with st.spinner('Training model...'):
            models = train_models(X_train_final, y_train)
        
        # Evaluasi model awal
        results_before = {}
        predictions_before = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test_final)
            predictions_before[name] = y_pred
            results_before[name] = evaluate_model(y_test, y_pred)
        
        results_df_before = pd.DataFrame(results_before).T
        
        # Tampilkan hasil sebelum tuning
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Tabel Hasil Evaluasi:**")
            st.dataframe(results_df_before.style.format("{:.4f}"))
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(results_df_before, annot=True, cmap="Blues", fmt=".3f", ax=ax)
            ax.set_title("Heatmap - Sebelum Tuning")
            st.pyplot(fig)
        
        # Hyperparameter tuning
        if run_tuning:
            st.subheader("🛠️ Hyperparameter Tuning")
            
            tuned_models, best_params = tune_models(X_train_final, y_train)
            
            # Store best model untuk prediksi
            st.session_state['best_model'] = tuned_models['RandomForest']
            st.session_state['preprocessors'] = {
                'num_imputer': num_imputer,
                'cat_imputer': cat_imputer,
                'encoder': encoder,
                'scaler': scaler,
                'feature_names': feature_names
            }
            
            # Tampilkan parameter terbaik
            st.write("**Parameter Terbaik:**")
            for model_name, params in best_params.items():
                st.write(f"- **{model_name}**: {params}")
            
            # Evaluasi model setelah tuning
            results_after = {}
            predictions_after = {}
            
            for name, model in tuned_models.items():
                y_pred = model.predict(X_test_final)
                predictions_after[name] = y_pred
                results_after[name] = evaluate_model(y_test, y_pred)
            
            results_df_after = pd.DataFrame(results_after).T
            
            # Tampilkan hasil setelah tuning
            st.subheader("✨ Hasil Setelah Tuning")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Tabel Hasil Evaluasi:**")
                st.dataframe(results_df_after.style.format("{:.4f}"))
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(results_df_after, annot=True, cmap="Greens", fmt=".3f", ax=ax)
                ax.set_title("Heatmap - Setelah Tuning")
                st.pyplot(fig)
            
            # Perbandingan
            st.subheader("📈 Perbandingan Sebelum vs Sesudah Tuning")
            
            # Gabungkan hasil
            combined_df = pd.concat([
                results_df_before.add_suffix(' (Before)'),
                results_df_after.add_suffix(' (After)')
            ], axis=1)
            
            # Heatmap perbandingan
            fig, ax = plt.subplots(figsize=(14, 6))
            sns.heatmap(combined_df, annot=True, cmap="YlGnBu", fmt=".3f", ax=ax)
            ax.set_title("Perbandingan Performa: Sebelum vs Sesudah Tuning")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Bar chart untuk setiap metrik
            st.subheader("📊 Detail Perbandingan per Metrik")
            
            metrics = results_df_before.columns
            cols = st.columns(2)
            
            for i, metric in enumerate(metrics):
                with cols[i % 2]:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    
                    comparison_data = pd.DataFrame({
                        'Before Tuning': results_df_before[metric],
                        'After Tuning': results_df_after[metric]
                    })
                    
                    comparison_data.plot(kind='bar', ax=ax, color=['lightblue', 'lightgreen'])
                    ax.set_title(f"Perbandingan {metric}")
                    ax.set_ylabel(metric)
                    ax.set_ylim(0, 1.05)
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    ax.legend()
                    plt.xticks(rotation=45)
                    
                    st.pyplot(fig)
            
            # Summary improvement
            st.subheader("📋 Ringkasan Peningkatan")
            
            improvement_data = []
            for model in results_df_before.index:
                for metric in results_df_before.columns:
                    before = results_df_before.loc[model, metric]
                    after = results_df_after.loc[model, metric]
                    improvement = ((after - before) / before) * 100
                    improvement_data.append({
                        'Model': model,
                        'Metric': metric,
                        'Before': before,
                        'After': after,
                        'Improvement (%)': improvement
                    })
            
            improvement_df = pd.DataFrame(improvement_data)
            st.dataframe(improvement_df.style.format({
                'Before': '{:.4f}',
                'After': '{:.4f}',
                'Improvement (%)': '{:.2f}%'
            }))
    
    elif page == "Prediksi Obesitas":
        st.subheader("🔮 Prediksi Level Obesitas")
        st.write("Masukkan data berikut untuk memprediksi level obesitas:")
        
        # Form input
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Data Demografis**")
                age = st.number_input("Umur", min_value=10, max_value=100, value=25)
                gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
                height = st.number_input("Tinggi Badan (m)", min_value=1.0, max_value=2.5, value=1.7, format="%.2f")
                weight = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=70.0, format="%.1f")
            
            with col2:
                st.write("**Kebiasaan Makan**")
                family_history = st.selectbox("Riwayat Keluarga Overweight", ["yes", "no"])
                favc = st.selectbox("Konsumsi Makanan Tinggi Kalori", ["yes", "no"])
                fcvc = st.number_input("Konsumsi Sayuran (per hari)", min_value=0.0, max_value=5.0, value=2.0, format="%.1f")
                ncp = st.number_input("Jumlah Makan Utama", min_value=1.0, max_value=5.0, value=3.0, format="%.1f")
                caec = st.selectbox("Konsumsi Makanan di Antara Waktu Makan", 
                                  ["no", "Sometimes", "Frequently", "Always"])
            
            with col3:
                st.write("**Gaya Hidup**")
                smoke = st.selectbox("Merokok", ["yes", "no"])
                ch2o = st.number_input("Konsumsi Air (liter/hari)", min_value=0.0, max_value=5.0, value=2.0, format="%.1f")
                scc = st.selectbox("Monitor Kalori", ["yes", "no"])
                faf = st.number_input("Aktivitas Fisik (hari/minggu)", min_value=0.0, max_value=7.0, value=2.0, format="%.1f")
                tue = st.number_input("Waktu Teknologi (jam/hari)", min_value=0.0, max_value=24.0, value=2.0, format="%.1f")
                calc = st.selectbox("Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"])
                mtrans = st.selectbox("Transportasi", 
                                    ["Walking", "Bike", "Motorbike", "Public_Transportation", "Automobile"])
            
            submitted = st.form_submit_button("🔍 Prediksi", type="primary")
        
        if submitted:
            # Persiapkan data untuk prediksi
            user_data = {
                'Age': age,
                'Gender': gender,
                'Height': height,
                'Weight': weight,
                'family_history_with_overweight': family_history,
                'FAVC': favc,
                'FCVC': fcvc,
                'NCP': ncp,
                'CAEC': caec,
                'SMOKE': smoke,
                'CH2O': ch2o,
                'SCC': scc,
                'FAF': faf,
                'TUE': tue,
                'CALC': calc,
                'MTRANS': mtrans
            }
            
            # Train model sederhana untuk prediksi (jika belum ada best_model)
            if 'best_model' not in st.session_state:
                with st.spinner('Training model untuk prediksi...'):
                    # Preprocessing data
                    X_train_final, X_test_final, y_train, y_test, num_imputer, cat_imputer, encoder, scaler, feature_names = preprocess_features(df_cleaned)
                    
                    # Train model sederhana
                    best_model = RandomForestClassifier(random_state=42, n_estimators=100)
                    best_model.fit(X_train_final, y_train)
                    
                    st.session_state['best_model'] = best_model
                    st.session_state['preprocessors'] = {
                        'num_imputer': num_imputer,
                        'cat_imputer': cat_imputer,
                        'encoder': encoder,
                        'scaler': scaler,
                        'feature_names': feature_names
                    }
        
            try:
                # Preprocessing input user
                preprocessors = st.session_state['preprocessors']
                processed_input = preprocess_user_input(user_data, preprocessors)
                
                # Prediksi
                model = st.session_state['best_model']
                prediction = model.predict(processed_input)[0]
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(processed_input)[0]
                    classes = model.classes_
                else:
                    probabilities = None
                    classes = None
                
                # Tampilkan hasil
                st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                st.success(f"## 🎯 Hasil Prediksi: **{prediction}**")
                
                # Interpretasi hasil
                interpretations = {
                    'Insufficient_Weight': "Berat badan kurang - Disarankan untuk meningkatkan asupan nutrisi",
                    'Normal_Weight': "Berat badan normal - Pertahankan pola hidup sehat",
                    'Overweight_Level_I': "Kelebihan berat badan tingkat I - Mulai diet seimbang dan olahraga",
                    'Overweight_Level_II': "Kelebihan berat badan tingkat II - Konsultasi dengan ahli gizi",
                    'Obesity_Type_I': "Obesitas tipe I - Perlu program penurunan berat badan terstruktur",
                    'Obesity_Type_II': "Obesitas tipe II - Konsultasi medis dan program intensif",
                    'Obesity_Type_III': "Obesitas tipe III - Segera konsultasi dengan dokter spesialis"
                }
                
                st.info(f"**Interpretasi:** {interpretations.get(prediction, 'Konsultasi dengan ahli kesehatan')}")
                
                # Tampilkan probabilitas jika tersedia
                if probabilities is not None and classes is not None:
                    st.write("**Tingkat Kepercayaan Prediksi:**")
                    prob_df = pd.DataFrame({
                        'Kategori': classes,
                        'Probabilitas': probabilities
                    }).sort_values('Probabilitas', ascending=False)
                    
                    # Bar chart probabilitas
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(range(len(prob_df)), prob_df['Probabilitas'], 
                                 color=['red' if cat == prediction else 'lightblue' for cat in prob_df['Kategori']])
                    ax.set_xlabel('Kategori Obesitas')
                    ax.set_ylabel('Probabilitas')
                    ax.set_title('Distribusi Probabilitas Prediksi')
                    ax.set_xticks(range(len(prob_df)))
                    ax.set_xticklabels(prob_df['Kategori'], rotation=45, ha='right')
                    
                    # Highlight prediksi tertinggi
                    max_idx = prob_df.reset_index(drop=True)['Probabilitas'].idxmax()
                    bars[max_idx].set_color('red')
                    
                    for i, (cat, prob) in enumerate(zip(prob_df['Kategori'], prob_df['Probabilitas'])):
                        ax.text(i, prob + 0.01, f'{prob:.3f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Hitung BMI
                bmi = weight / (height ** 2)
                st.write(f"**BMI Anda:** {bmi:.2f}")
                
                if bmi < 18.5:
                    bmi_cat = "Underweight"
                elif bmi < 25:
                    bmi_cat = "Normal"
                elif bmi < 30:
                    bmi_cat = "Overweight"
                else:
                    bmi_cat = "Obese"
                
                st.write(f"**Kategori BMI:** {bmi_cat}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Terjadi kesalahan dalam prediksi: {str(e)}")

else:
    st.error("Tidak bisa memuat dataset. Pastikan file 'ObesityDataSet.csv' tersedia.")

# Footer
st.markdown("---")
st.markdown("*Aplikasi Evaluasi Model ML & Prediksi Obesitas dibuat dengan Streamlit* 🚀")