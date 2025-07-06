import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ==================================
# Konfigurasi Halaman & Judul
# ==================================
st.set_page_config(page_title="YouTube Playlist Analyzer", layout="wide")

st.title("🚀 Analisis Pola Engagement Playlist YouTube")
st.markdown("Masukkan **Playlist ID** dari kanal YouTube mana pun untuk mengelompokkan video berdasarkan pola engagement audiens menggunakan K-Means Clustering.")

# ==================================
# Sidebar untuk Input Pengguna
# ==================================
st.sidebar.header("⚙️ Pengaturan Analisis")
playlist_id_input = st.sidebar.text_input("Masukkan Playlist ID YouTube", "PL_K9e2LM-il7CzfCt82ABBbfuqf2trmI")
use_sample_data = st.sidebar.checkbox(
    "Gunakan Dataset Contoh (Podhub Deddy Corbuzier)", 
    value=False,
    help="Jika dicentang, aplikasi akan menggunakan data CSV yang sudah ada tanpa memanggil API."
)

api_key = st.secrets.get("YOUTUBE_API_KEY", "")

# Validasi kunci API jika data sample tidak digunakan
if not use_sample_data and not api_key:
    st.error("Kunci API YouTube tidak ditemukan di secrets.toml. Silakan tambahkan untuk melanjutkan.")
    st.stop()

# ==================================
# Fungsi Pengambilan & Caching Data
# ==================================
# Cache data selama 1 jam untuk mengurangi panggilan API
@st.cache_data(ttl=3600)
def get_videos_from_playlist(playlist_id, api_key):
    video_ids = []
    next_page_token = None
    while True:
        url = f"https://www.googleapis.com/youtube/v3/playlistItems?part=contentDetails&playlistId={playlist_id}&maxResults=50&key={api_key}"
        if next_page_token:
            url += f"&pageToken={next_page_token}"
        
        response = requests.get(url).json()
        if 'error' in response:
            raise Exception(f"Error API: {response['error']['message']}")
            
        items = response.get("items", [])
        for item in items:
            video_ids.append(item["contentDetails"]["videoId"])
        
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return video_ids

@st.cache_data(ttl=3600)
def get_video_statistics(video_ids, api_key):
    stats_list = []
    for i in range(0, len(video_ids), 50):
        video_id_chunk = ",".join(video_ids[i:i+50])
        url = f"https://www.googleapis.com/youtube/v3/videos?part=statistics,snippet&id={video_id_chunk}&key={api_key}"
        response = requests.get(url).json()
        for item in response.get("items", []):
            stats = item.get("statistics", {})
            snippet = item.get("snippet", {})
            stats_list.append({
                "video_id": item['id'],
                "title": snippet.get("title", "N/A"),
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "comments": int(stats.get("commentCount", 0))
            })
    return pd.DataFrame(stats_list)

@st.cache_data
def load_data(is_sample, playlist_id, api_key):
    """Memuat data dari API atau dari file CSV contoh."""
    if is_sample:
        try:
            df = pd.read_csv("youtube_engagement_data.csv")
            return df
        except FileNotFoundError:
            st.error("File youtube_engagement_data.csv tidak ditemukan. Harap nonaktifkan mode data contoh atau sediakan file tersebut.")
            return None
    else:
        if not playlist_id:
            st.warning("Harap masukkan Playlist ID yang valid.")
            return None
        video_ids = get_videos_from_playlist(playlist_id, api_key)
        if not video_ids:
            st.warning("Tidak ada video yang ditemukan di playlist ini atau ID tidak valid.")
            return None
        df = get_video_statistics(video_ids, api_key)
        return df

@st.cache_data
def process_and_cluster(df):
    """Fungsi untuk melakukan preprocessing dan clustering data."""
    df_selected = df[['views', 'likes', 'comments']]
    
    Q1 = df_selected.quantile(0.25)
    Q3 = df_selected.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_cleaned = df[~((df_selected < lower_bound) | (df_selected > upper_bound)).any(axis=1)].copy()
    
    if df_cleaned.empty:
        return None, 0
    
    df_features = df_cleaned[['views', 'likes', 'comments']].copy()
    
    # Rekayasa fitur engagement rate
    df_features['engagement_rate'] = (df_features['likes'] + df_features['comments']) / df_features['views']
    df_features['engagement_rate'] = df_features['engagement_rate'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Normalisasi
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_features)
    df_transformed = pd.DataFrame(scaled_data, columns=df_features.columns, index=df_features.index)

    # K-Means (Gunakan k=3 berdasarkan analisis notebook Anda)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_transformed['cluster'] = kmeans.fit_predict(df_transformed)
    
    # PCA untuk visualisasi
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(df_transformed)
    df_transformed['pc1'] = pca_result[:, 0]
    df_transformed['pc2'] = pca_result[:, 1]
    
    return df_transformed, df_cleaned.shape[0], len(df)


# ==================================
# Alur Utama Aplikasi
# ==================================
if st.sidebar.button("🚀 Analisis Playlist"):
    st.session_state.analysis_triggered = True
else:
    st.info("Masukkan Playlist ID di sidebar dan klik 'Analisis Sekarang'.")
    st.stop()
    
if 'analysis_triggered' in st.session_state and st.session_state.analysis_triggered:
    try:
        with st.spinner("Memuat dan memproses data..."):
            df_raw = load_data(use_sample_data, playlist_id_input, api_key)
            
            if df_raw is None:
                st.stop()
                
            df_clustered, cleaned_rows, raw_rows = process_and_cluster(df_raw.copy())

            if df_clustered is None:
                st.warning("Tidak cukup data untuk melakukan clustering setelah pembersihan outlier.")
                st.stop()
            
            # Gabungkan hasil cluster dengan data asli untuk analisis
            df_final = df_raw.loc[df_clustered.index].copy()
            df_final['cluster'] = df_clustered['cluster']

        st.success(f"Analisis selesai! Dari {raw_rows} video, {cleaned_rows} video digunakan untuk clustering.")

        # --- Tampilan Hasil ---
        st.header("1. Visualisasi Cluster")
        st.markdown("Video-video dikelompokkan ke dalam 3 segmen utama berdasarkan kemiripan pola engagement-nya.")
        
        fig_pca, ax_pca = plt.subplots(figsize=(10, 7))
        sns.scatterplot(data=df_clustered, x='pc1', y='pc2', hue='cluster', palette='viridis', s=100, ax=ax_pca)
        ax_pca.set_title('Visualisasi Cluster Video dengan PCA')
        ax_pca.set_xlabel('Principal Component 1')
        ax_pca.set_ylabel('Principal Component 2')
        st.pyplot(fig_pca)

        st.header("2. Analisis & Insight per Cluster")
        st.markdown("Setiap klaster memiliki karakteristik unik. Memahami ini dapat membantu dalam strategi konten.")

        # Ringkasan statistik per cluster
        cluster_summary = df_final.groupby('cluster')[['views', 'likes', 'comments']].mean().reset_index()
        cluster_summary['Jumlah Video'] = df_final['cluster'].value_counts().sort_index().values
        cluster_summary.columns = ['Cluster', 'Rata-rata Views', 'Rata-rata Likes', 'Rata-rata Komentar', 'Jumlah Video']
        
        st.subheader("Ringkasan Statistik per Cluster")
        st.dataframe(cluster_summary)
        
        # Interpretasi Cluster
        st.subheader("Interpretasi Klaster")
        highest_perf_cluster = cluster_summary.sort_values(by='Rata-rata Views', ascending=False).iloc[0]['Cluster']
        
        st.markdown(f"""
        Berdasarkan rata-rata metrik, kita dapat menginterpretasikan klaster sebagai berikut:
        - **Klaster Performa Tinggi (Contoh: Cluster {int(highest_perf_cluster)})**: Video dalam klaster ini memiliki *views*, *likes*, dan *comments* yang jauh di atas rata-rata. Ini adalah konten "juara" Anda.
        - **Klaster Performa Menengah**: Video dengan metrik yang moderat, mencerminkan performa standar kanal Anda.
        - **Klaster Performa Rendah**: Video dengan engagement paling rendah.
        """)

        st.header("3. Detail Video per Cluster")
        st.markdown("Jelajahi video-video yang termasuk dalam setiap klaster untuk menemukan pola topik atau format.")
        
        selected_cluster = st.selectbox("Pilih Cluster untuk dilihat detailnya:", sorted(df_final['cluster'].unique()))
        
        st.dataframe(
            df_final[df_final['cluster'] == selected_cluster][['title', 'views', 'likes', 'comments']],
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengambil atau memproses data: {e}")
        st.warning("Pastikan Playlist ID yang Anda masukkan benar dan kunci API Anda valid serta memiliki kuota.")