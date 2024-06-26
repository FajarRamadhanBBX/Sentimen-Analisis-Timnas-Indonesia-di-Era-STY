# Submission 2: Sentimen Analisis Timnas Sepak Bola Indonesia pada Masa Kepelatihan Shin-Tae Yong
Nama: Fajar Ramadhan

Username dicoding: fajar_ramadhan_bbk

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Analisis Sentimen Timnas Sepakbola di Era STY](https://www.kaggle.com/datasets/pajarbebek/analisis-sentimen-timnas-sepakbola-di-era-sty) |
| Masalah | Sepak bola merupakan olahraga paling populer di Indonesia.Supporter sepak bola Indonesia merupakan salah satu supporter terbesar di dunia. Saat ini, performa timnas sepak bola Indonesia meningkat tajam sejak kehadiran pelatih kepala Shin-Tae Young. Hal ini tak lepas dari beberapa kebijakannya yang cukup berbeda dari pelatih lainnya, seperti naturalisasi hingga pemotongan 2 generasi. Hal ini menimbulkan perdebatan di kalangan masyarakat, ada yang mendukung, dan ada pula yang menyayangkannya. maka dari itu dibuatkan sebuah pemodelan untuk melakukan analisis sentimen terhadap komentar masyarakat pada media sosial X|
| Solusi machine learning | Membuat sebuah model yang dapat mengetahui sentimen masyakarakat lewat komentar. Hal ini dapat dimanfaatkan bagi *stakeholder* yang terlibat dalam timnas sepakbola untuk mengambil keputusan tentang perpanjangan kontrak kepala pelatih STY.
| Metode pengolahan | Data diolah pada komponen transform dan trainer dengan cara membersihkan komentar dari noise seperti perubahan huruf besar menjadi huruf kecil, serta penghapusan punctuation. Lalu label juga diubah dalam tipe tensor agar dapat diproses oleh pipeline. |
| Arsitektur model | model terdiri dari beberapa layer dengan tugas berbeda, seperti layer pertama terdapat layer TextVectorization untuk mengubah string menjadi bentuk numerik, lalu dilanjut dengan embedding, global average pooling, serta dense |
| Metrik evaluasi | metrik yang digunakan yaitu *Area Under Curve*, *False Positive* dan *Negative*, *True Positive* dan *Negative,* *Binary Accuracy*, serta *binary crossentropy*|
| Performa model | Model yang dilatih mendapatkan hasil yang baik dengan persentase binary accuracy 95% dengan total data validasi (*example count*) 76. Namun, jika ditinjau dari hasil evaluasi masih dibutuhkan peningkatan kualitas model. Pada metriks seperti AUC, *false negative*, dan *binary accuracy* dinilai masih kurang baik. Nilai AUC tidak lebih dari 0.6, yang artinya performa model masih mendekati model random guessing. Lalu pada *binary Accuracy* nilai yang dihasilkan juga tidak lebih dari 0.5, serta jumlah *false negative* yang sama *true negatif*, yaitu 23. Sedangkan untuk nilai *false positive* mendapat 11 dan *true positive* mendapat 19, yang artinya model lebih banyak memprediksi nilai yang benar. Terakhir, untuk nilai *binary crossentropy* mendapatkan nilai 1.39626  
| Opsi deployment | Proyek ini di deploy menggunakan Railway App, yaitu salah satu platform yang menyediakan layanan deploy proyek termasuk model machine learning secara gratis |
| Web app |  dapat diakses disini: https://analisis-sentimen-sty-production-ab80.up.railway.app/v1/models/sentimen-analysis-sty-model/metadata |
| Monitoring| Monitoring dilakukan menggunakan Prometheus. Salah satu hal yang dapat dimonitoring adalah *request count* untuk mengetahui jumlah *request* |

## Note:
saat ini model dalam keadaan non-aktif, sehingga model tidak dapat diakses
