# 🎤 Gelişmiş Ses Analizi ve PyAnnote Sistemi

Bu proje, pyannote.audio kütüphanesi kullanarak gelişmiş ses analizi, konuşmacı diyarizasyonu ve duygu analizi yapabilen kapsamlı bir uygulamadır.

## ✨ Özellikler

### 🎯 Temel Özellikler
- **Ses Kaydı**: Mikrofon girişinden gerçek zamanlı ses kaydı
- **Dosya Desteği**: Çoklu ses formatları (WAV, MP3, FLAC, M4A, OGG)
- **Konuşmacı Diyarizasyonu**: PyAnnote.audio 3.1 ile gelişmiş konuşmacı tanıma
- **Transkripsiyon**: OpenAI Whisper API ile konuşma-metin dönüştürme

### 🔬 Gelişmiş Analiz Özellikleri
- **Ses Aktivitesi Tespiti (VAD)**: Konuşma ve sessizlik bölgelerini tespit etme
- **Örtüşen Konuşma Tespiti**: Aynı anda konuşan kişileri tespit etme
- **Konuşmacı Tanıma**: Konuşmacı embedding'leri ve benzerlik analizi
- **Gürültü Azaltma**: NoiseReduce kütüphanesi ile ses iyileştirme
- **Ses Ayrıştırma**: Karışık seslerden ayrı kanallar çıkarma (geliştirilme aşamasında)
- **Duygu Analizi**: Ses özelliklerine dayalı duygu tahmini
- **Canlı Analiz**: Gerçek zamanlı ses analizi (kayıt sırasında)

### 📊 Görselleştirme Özellikleri
- **Dalga Formu**: Ses sinyalinin zaman alanında görüntülenmesi
- **Spektogram**: Frekans analizi ve görselleştirme
- **Konuşmacı Zaman Çizelgesi**: Renkli konuşmacı segmentleri
- **Duygu Dağılımı**: Pasta grafiği ile duygu oranları
- **İstatistik Grafikleri**: Konuşmacı süreleri, embedding'ler, segment analizi

### 📄 Raporlama
- **Detaylı Rapor**: Kapsamlı analiz sonuçları
- **PDF/Metin Çıktı**: Sonuçları dosya olarak kaydetme
- **İstatistik Özetleri**: Konuşmacı ve duygu istatistikleri

## 🛠️ Kurulum

### Gereksinimler
- Python 3.8 veya üzeri
- HuggingFace hesabı ve erişim tokeni
- OpenAI API anahtarı (transkripsiyon için)

### 1. Depoyu Klonlayın
```bash
git clone <repo-url>
cd pyannote-audio
```

### 2. Sanal Ortam Oluşturun
```bash
python -m venv pyannote-env
source pyannote-env/bin/activate  # Linux/Mac
# veya
pyannote-env\Scripts\activate     # Windows
```

### 3. Gerekli Kütüphaneleri Yükleyin
```bash
pip install -r requirements.txt
```

### 4. GPU Desteği (İsteğe Bağlı)
CUDA destekli GPU kullanmak için:
```bash
pip install torch[cuda] torchaudio[cuda]
```

### 5. API Anahtarlarını Yapılandırın
`basit_ses_kayit_ve_analiz.py` dosyasında aşağıdaki değerleri güncelleyin:

```python
# HuggingFace erişim tokeni
HF_TOKEN = "your_huggingface_token_here"

# OpenAI API anahtarı
OPENAI_API_KEY = "your_openai_api_key_here"
```

## 🚀 Kullanım

### Uygulamayı Başlatma
```bash
python basit_ses_kayit_ve_analiz.py
```

### Temel Kullanım Adımları

1. **Ses Girişi**:
   - 🎙️ Mikrofon ile kayıt yapın
   - 📁 Mevcut ses dosyası yükleyin

2. **Analiz Seçenekleri**:
   - ✅ İstediğiniz analiz türlerini seçin
   - ⚡ Hızlı analiz veya 🔍 tam analiz seçin

3. **Sonuçları İnceleyin**:
   - 📊 Görselleştirme sekmelerini kontrol edin
   - 📋 Detaylı raporu okuyun
   - 📄 Rapor dosyası oluşturun

### Analiz Seçenekleri

| Özellik | Açıklama | Önerilen Kullanım |
|---------|----------|-------------------|
| 🎯 Ses Aktivitesi Tespiti | Konuşma/sessizlik bölgeleri | Tüm analizler |
| 🗣️ Örtüşen Konuşma | Aynı anda konuşma tespiti | Grup konuşmaları |
| 😊 Duygu Analizi | Konuşmacı duygusal durumu | Müşteri hizmetleri |
| 🔇 Gürültü Azaltma | Ses kalitesi iyileştirme | Gürültülü ortamlar |
| 🎼 Ses Ayrıştırma | Kanal separasyonu | Karışık kayıtlar |
| 🔴 Canlı Analiz | Gerçek zamanlı işlem | Kayıt sırasında |

## 📊 Desteklenen Dosya Formatları

- **WAV**: Yüksek kalite, önerilen format
- **MP3**: Sıkıştırılmış, yaygın kullanım
- **FLAC**: Kayıpsız sıkıştırma
- **M4A**: Apple formatı
- **OGG**: Açık kaynak format

## 🎨 Arayüz Özellikleri

### Ana Sekmeler
- 🌊 **Dalga Formu**: Ses sinyali ve spektogram
- 📊 **Spektogram**: Detaylı frekans analizi
- 😊 **Duygu Analizi**: Duygu dağılım grafikleri
- 📈 **İstatistikler**: Konuşmacı ve analiz istatistikleri

### Bilgi Panelleri
- 📋 **İşlem Günlüğü**: Gerçek zamanlı işlem takibi
- 🎯 **Diyarizasyon Sonuçları**: Konuşmacı segmentleri
- 💬 **Konuşma İçeriği**: Transkripsiyon sonuçları
- 🔬 **Detaylı Analiz**: Kapsamlı analiz raporu
- 😊 **Duygu Raporu**: Duygu analizi detayları

## ⚙️ Konfigürasyon

### Ses Kayıt Parametreleri
```python
SAMPLE_RATE = 44100  # Örnekleme hızı
CHANNELS = 1         # Mono kayıt
DTYPE = np.float32   # Veri tipi
```

### Model Ayarları
```python
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
VAD_MODEL = "pyannote/voice-activity-detection"
OVERLAP_MODEL = "pyannote/overlapped-speech-detection"
```

## 🔧 Sorun Giderme

### Yaygın Sorunlar

1. **Model İndirme Hatası**:
   ```
   HuggingFace tokeninizin geçerli olduğundan emin olun
   İnternet bağlantınızı kontrol edin
   ```

2. **Ses Cihazı Hatası**:
   ```
   Mikrofon izinlerini kontrol edin
   Ses cihazı sürücülerini güncelleyin
   ```

3. **GPU Hatası**:
   ```
   CUDA sürümünüzü kontrol edin
   CPU moduna geçiş yapın
   ```

4. **Transkripsiyon Hatası**:
   ```
   OpenAI API anahtarınızı kontrol edin
   API kullanım limitinizi kontrol edin
   ```

## 📈 Performans İpuçları

- **GPU Kullanımı**: CUDA destekli GPU ile 10x hızlı işlem
- **Ses Kalitesi**: 44.1kHz örnekleme hızı önerilen
- **Dosya Boyutu**: Büyük dosyalar için segment işleme
- **Bellek**: RAM kullanımını izleyin (büyük dosyalar için)

## 🤝 Katkıda Bulunma

1. Depoyu fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'e push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 🙏 Teşekkürler

- [PyAnnote.audio](https://github.com/pyannote/pyannote-audio) ekibi
- [HuggingFace](https://huggingface.co) platformu
- [OpenAI](https://openai.com) Whisper API
- Python ses işleme topluluğu

## 📞 İletişim

Sorularınız ve önerileriniz için:
- Issue açın
- Pull request gönderin
- Dokumentasyonu inceleyin

---

**🚀 Keyifli analiz yapmalarınızı diliyoruz!** 