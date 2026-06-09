# uni
stock predict project
# 📈 Makine Öğrenmesi Algoritmaları ile Hisse Senedi Tahmini (Apple Inc.)

## 📝 Özet
Bu proje, Apple Inc. hisse senedi fiyatlarının gelecekteki değerlerini tahmin etmek amacıyla geleneksel zaman serisi yöntemleri ile modern Derin Öğrenme (Deep Learning) algoritmalarının karşılaştırmalı bir analizini sunmaktadır. Çalışma kapsamında **ARIMA**, **LSTM**, **GRU** ve **SimpleRNN** modelleri geliştirilmiş; model performansları MAE, RMSE ve MAPE metrikleriyle değerlendirilmiştir.

Sonuçlar, **GRU (Gated Recurrent Unit)** modelinin düşük hata oranları ile finansal zaman serisi tahmininde en yüksek performansı sergilediğini göstermiştir.

---

## 🛠️ Kullanılan Teknolojiler ve Araçlar
* **Veri Ön İşleme & Analiz:** Python (Pandas, Scikit-Learn, NumPy) & R (Readr)
* **Zaman Serisi Analizi:** R (Tseries, Forecast) - KPSS Testi, Otokorelasyon (ACF/PACF)
* **Derin Öğrenme Modellemesi:** Python (TensorFlow / Keras)
* **Veri Görselleştirme:** Matplotlib, ggplot2
* **Veri Kaynağı:** Yahoo Finance (12.06.2000 - 10.06.2024 arası Apple Hisse Verisi)

---

## 1. Problem Tanımı ve Yaklaşım
Finansal piyasaların yüksek volatilitesi ve karmaşık yapısı, geleneksel tahmin yöntemlerini yetersiz kılmaktadır. Bu çalışmada, büyük veri setlerindeki gizli örüntüleri (pattern) öğrenme kapasitesine sahip derin öğrenme modellerinin, hisse senedi gibi doğrusal olmayan (non-linear) verilerdeki başarısı test edilmiştir.

### 🔬 Metodoloji:
1.  **Veri Ön İşleme:** Yahoo Finance üzerinden çekilen veri seti temizlenmiş, eksik veriler giderilmiş ve `MinMaxScaler` ile (0,1) aralığında normalize edilmiştir. Model verimi için işlem hacmi (Volume) değişkeni düşük korelasyon sebebiyle dışarıda bırakılmıştır.
2.  **Veri Şekillendirme (Time Series to Supervised):** Zaman serisi verisi, denetimli öğrenmeye uygun hale getirilmiş; girdi sekansı (n_day) **5 gün** olarak belirlenip 1 günlük (Close) çıktı tahmin edilmiştir. Verinin %80'i eğitim, %20'si test olarak ayrılmıştır.
3.  **Durağanlık Analizi:** ARIMA modeli için KPSS birim kök testi uygulanmış, durağan olmayan seri fark alma (differencing) yöntemiyle durağan hale getirilmiştir.
4.  **Model Eğitimi:** Geleneksel yaklaşımla **ARIMA(5,2,0)** modeli kurulurken, derin öğrenme yaklaşımlarıyla **LSTM**, **GRU** ve **SimpleRNN** modelleri Keras üzerinde aşağıdaki mimari ile eğitilmiştir.

---

## 2. Derin Öğrenme Model Mimarisi & Hiperparametreler

Tüm RNN tabanlı modellerde (LSTM, GRU, SimpleRNN) adil bir karşılaştırma yapabilmek adına standart bir mimari ve aynı hiperparametreler kullanılmıştır.

| Parametre | Değer |
| :--- | :--- |
| **Ağ Tipi** | Sequential (Sıralı) |
| **Katmanlar** | 2 Gizli Katman (RNN/LSTM/GRU) + 1 Dense Çıktı |
| **Nöron Sayısı (Units)** | 50 (Her bir gizli katman için) |
| **Aktivasyon Fonksiyonu** | ReLU |
| **Dropout Oranı** | %1 (0.01) |
| **Optimizasyon (Optimizer)** | Adam (Learning Rate: 0.01) |
| **Kayıp Fonksiyonu (Loss)** | MAE (Mean Absolute Error) |
| **Eğitim Turu (Epochs)** | 50 |
| **Yığın Boyutu (Batch Size)** | 128 |

**Keras Kod Akışı Örneği:**
```python
model = Sequential()
model.add(SimpleRNN(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.01))
model.add(SimpleRNN(units=50, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(1))

model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
history = model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_test, y_test), verbose=1)



📊 Temel Çıktılar:
En İyi Performans (Genel): GRU modeli, hem MAE hem de MAPE değerlerinde en düşük sapmayı göstererek hisse senedi fiyatlarını öngörmede en başarılı mimari olmuştur.

Büyük Hataların Minimizasyonu: LSTM modeli, RMSE skorunun en düşük çıkmasıyla ani fiyat dalgalanmalarındaki büyük sapmaları minimize etmede başarılı olmuştur.

Geleneksel vs. Modern: Klasik zaman serisi modeli olan ARIMA, derin öğrenme tabanlı RNN varyantlarına kıyasla oldukça yüksek hata payı ile çalışmış, yüksek boyutlu finansal verilerde yetersiz kalmıştır.

4. Gerçekçi Kısıtlar ve Gelecek Çalışmalar
Bu proje, makine öğrenmesi modellerinin piyasa öngörülerindeki yüksek potansiyelini kanıtlasa da, finansal piyasaların makroekonomik kararlar (faiz, enflasyon), jeopolitik krizler ve yatırımcı psikolojisi gibi dış etkenlerden bağımsız olmadığı unutulmamalıdır. Gelecek çalışmalarda, hisse fiyatı parametrelerine ek olarak duygu analizi (Sentiment Analysis) ve makroekonomik göstergelerin (NLP tabanlı haber analizleri vb.) modele entegre edilmesi hedeflenmektedir.

Projenin detaylı analizleri, grafiksel sonuçları ve akademik temelleri için dizinde bulunan makale/tez dokümanını inceleyebilirsiniz.
