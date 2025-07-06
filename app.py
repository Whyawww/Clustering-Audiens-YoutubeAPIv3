import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="YouTube Playlist Analyzer", layout="wide")

st.title("Analisis Pola Engagement Playlist YouTube")
st.markdown("Masukkan **Playlist ID** dari kanal YouTube mana pun untuk mengelompokkan video berdasarkan pola engagement audiens menggunakan K-Means Clustering.")

st.sidebar.header("Pengaturan Analisis")
playlist_id_input = st.sidebar.text_input("Masukkan Playlist ID YouTube", "PL_K9e2LM-il7CzfCt82ABBbfuqf2trmI")
use_sample_data = st.sidebar.checkbox(
    "Gunakan Dataset Contoh (Podhub Deddy Corbuzier)",
    value=False,
    help="Jika dicentang, aplikasi akan menggunakan data CSV yang sudah ada tanpa memanggil API."
)

api_key = st.secrets.get("YOUTUBE_API_KEY", "")

if not use_sample_data and not api_key:
    st.error("Kunci API YouTube tidak ditemukan di secrets.toml. Harap tambahkan untuk melanjutkan.")
    st.stop()

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
    df_selected = df[['views', 'likes', 'comments']]
    
    Q1 = df_selected.quantile(0.25)
    Q3 = df_selected.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_cleaned = df[~((df_selected < lower_bound) | (df_selected > upper_bound)).any(axis=1)].copy()
    
    if df_cleaned.empty or df_cleaned['views'].sum() == 0:
        return None, 0, len(df)
    
    df_features = df_cleaned[['views', 'likes', 'comments']].copy()
    
    df_features['engagement_rate'] = np.divide(
        (df_features['likes'] + df_features['comments']),
        df_features['views'],
        out=np.zeros_like((df_features['likes'] + df_features['comments']), dtype=float),
        where=df_features['views']!=0
    )

    features_to_scale = ['views', 'likes', 'comments', 'engagement_rate']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_features[features_to_scale])
    df_transformed = pd.DataFrame(scaled_data, columns=features_to_scale, index=df_features.index)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_transformed['cluster'] = kmeans.fit_predict(df_transformed)
    
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(df_transformed)
    df_transformed['pc1'] = pca_result[:, 0]
    df_transformed['pc2'] = pca_result[:, 1]
    
    return df_transformed, df_cleaned.shape[0], len(df)

if st.sidebar.button("Analisis Playlist"):
    try:
        with st.spinner("Memuat dan memproses data..."):
            df_raw = load_data(use_sample_data, playlist_id_input, api_key)
            
            if df_raw is None:
                st.stop()
                
            df_clustered, cleaned_rows, raw_rows = process_and_cluster(df_raw.copy())

            if df_clustered is None:
                st.warning("Tidak cukup data untuk melakukan clustering setelah pembersihan outlier.")
                st.stop()
            
            df_final = df_raw.loc[df_clustered.index].copy()
            df_final['cluster'] = df_clustered['cluster']
            df_final['engagement_rate_scaled'] = df_clustered['engagement_rate']

        st.success(f"Analisis selesai! Dari {raw_rows} video, {cleaned_rows} video digunakan untuk clustering.")

        st.header("1. Visualisasi Cluster")
        st.markdown("Video-video dikelompokkan ke dalam 3 segmen utama berdasarkan kemiripan pola engagement-nya.")
        
        fig_pca, ax_pca = plt.subplots(figsize=(10, 7))
        sns.scatterplot(data=df_clustered, x='pc1', y='pc2', hue='cluster', palette='viridis', s=100, ax=ax_pca)
        ax_pca.set_title('Visualisasi Cluster Video dengan PCA')
        ax_pca.set_xlabel('Principal Component 1')
        ax_pca.set_ylabel('Principal Component 2')
        st.pyplot(fig_pca)

        st.header("2. Analisis & Insight per Cluster")
        st.markdown("Menganalisis karakteristik dan segmen video berdasarkan hasil clustering untuk memberikan insight yang dapat ditindaklanjuti.")
        
        cluster_summary = df_final.groupby('cluster')[['views', 'likes', 'comments', 'engagement_rate_scaled']].mean()
        cluster_summary['Jumlah Video'] = df_final['cluster'].value_counts()
        cluster_summary.reset_index(inplace=True)

        # Mengganti nama kolom untuk tampilan yang lebih baik
        cluster_summary.rename(columns={
            'cluster': 'Cluster',
            'views': 'Rata-rata Views',
            'likes': 'Rata-rata Likes',
            'comments': 'Rata-rata Komentar',
            'engagement_rate_scaled': 'Rata-rata Engagement Rate (Scaled)'
        }, inplace=True)
        
        st.subheader("Ringkasan Statistik per Cluster")
        st.dataframe(cluster_summary)
        
        # Visualisasi Engagement Rate
        st.subheader("Perbandingan Engagement Rate per Cluster")
        fig_engagement, ax_engagement = plt.subplots(figsize=(8, 5))
        # Menggunakan kolom 'Cluster' yang sudah diganti namanya
        sns.barplot(data=cluster_summary, x='Cluster', y='Rata-rata Engagement Rate (Scaled)', ax=ax_engagement)
        plt.ylabel('Rata-rata Engagement Rate (Scaled)')
        plt.xlabel('Cluster')
        st.pyplot(fig_engagement)
        
        # Interpretasi Cluster
        st.subheader("Interpretasi Klaster")
        highest_engagement_cluster = cluster_summary.sort_values(by='Rata-rata Engagement Rate (Scaled)', ascending=False).iloc[0]['Cluster']
        
        st.markdown(f"""
        Berdasarkan `engagement_rate` yang telah diskalakan:
        - **Klaster Engagement Tinggi (Contoh: Cluster {int(highest_engagement_cluster)})**: Video di klaster ini, meskipun mungkin tidak selalu memiliki *views* tertinggi, namun mampu memancing *likes* dan *comments* yang sangat tinggi secara proporsional. Ini adalah konten yang paling resonan dengan audiens inti Anda.
        - **Klaster Lainnya**: Perlu dianalisis lebih lanjut apakah klaster lain menunjukkan pola 'views tinggi tapi engagement rendah' (populer tapi pasif) atau 'views rendah dan engagement rendah' (kurang diminati).
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