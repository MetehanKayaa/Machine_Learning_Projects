# 🎬 Movie Recommendation System (Film Tavsiye Sistemi)

Bu proje, kullanıcıların izledikleri ve beğendikleri filmlere benzer içeriktekileri bulmasını sağlayan, **İçerik Tabanlı Filtreleme (Content-Based Filtering)** mantığıyla geliştirilmiş bir tavsiye sistemi projesidir. 

Metin tabanlı film detayları (özetler, türler, oyuncular vb.) doğal dil işleme teknikleriyle vektörleştirilmiş ve filmler arasındaki benzerlik dereceleri hesaplanmıştır.

---

## 💾 Büyük Dosya ve Model Erişimi (Kaggle Integration)

Projenin kalbini oluşturan ve filmler arasındaki ilişkileri tutan kosinüs benzerliği matrisi (`similarity.pkl`), **~190 MB** boyutunda olması ve GitHub'ın 100 MB'lık dosya yükleme sınırını aşması nedeniyle doğrudan bu repoya yüklenmemiştir.

Projenin yerelde (local) sorunsuz çalışabilmesi için gerekli olan bu dosyayı aşağıdaki Kaggle bağlantısından indirebilirsiniz:

🔗 **Kaggle Dataset:** [Movie Recommendation Similarity Matrix](https://www.kaggle.com/datasets/metehankayaaaa/movie-recommendation-similarity-matrix)

### 🚀 Nasıl Çalıştırılır?
1. Yukarıdaki Kaggle linkine giderek `similarity.pkl` dosyasını bilgisayarınıza indirin.
2. İndirdiğiniz dosyayı, bu projenin yer aldığı klasörün içine (notebook dosyası ile aynı dizine) yerleştirin.
3. Proje notebook'unu (`.ipynb`) yukarıdan aşağıya çalıştırdığınızda, kod otomatik olarak bu dosyayı algılayacak ve tavsiye sistemini ayağa kaldıracaktır.

---

## 🛠️ Uygulanan Teknikler ve Teknolojiler

- **Veri Ön İşleme & Temizleme:** Eksik ve çoklu girilmiş veri formatlarının (JSON/Dictionary yapıları) Pandas ve NumPy kullanılarak temizlenmesi ve analize hazır hale getirilmesi.
- **Metin Ön İşleme (NLP):** Film özetleri, etiketleri ve oyuncu kadrolarının birleştirilerek metin havuzu oluşturulması.
- **Vektörleştirme (Vectorization):** Metinlerin sayısal verilere dönüştürülmesi için `CountVectorizer` veya `TF-IDF` kullanımı.
- **Benberlik Analizi:** Öznitelik vektörleri arasındaki geometrik mesafeyi ölçmek için `Cosine Similarity` (Kosinüs Benzerliği) metriği.
