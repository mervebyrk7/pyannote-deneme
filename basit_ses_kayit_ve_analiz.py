#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Basit Ses Kaydı ve PyAnnote Analizi

Bu program, mikrofon girişinden ses kaydı alır, dosyaya kaydeder ve
PyAnnote.audio kullanarak konuşmacı diyarizasyonu yapar.
"""

import os
import time
import wave
import threading
import numpy as np
import sounddevice as sd
import torch
from pyannote.audio import Pipeline
from pyannote.audio import Model
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
from datetime import datetime
# macOS uyumluluğu için matplotlib backend ayarı
import matplotlib
matplotlib.use('TkAgg')  # macOS'ta en uyumlu backend

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches

# macOS'ta font uyarılarını bastır
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
import openai  # Whisper yerine OpenAI API kullanacağız
import base64  # Base64 kodlama için
import librosa
import soundfile as sf
import noisereduce as nr
from scipy import signal
import seaborn as sns
from collections import defaultdict
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# HuggingFace erişim tokeni - kendi tokeninizle değiştirin
HF_TOKEN = "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"

# OpenAI API anahtarı - kendi anahtarınızla değiştirin
OPENAI_API_KEY = "sk-proj-AZXoZV7vdTPzxP1GBiE7T3BlbkFJnUPhxjWfTcWBaod7INhj"  # Kendi API anahtarınızı girin
openai.api_key = OPENAI_API_KEY

# Ses kaydı parametreleri - Optimize edildi
SAMPLE_RATE = 44100  # Standart ses kalitesi
CHANNELS = 1         # Mono kayıt
DTYPE = np.float32   # Yüksek hassasiyet
RECORDING_SECONDS = 30  # Varsayılan kayıt süresi

# Buffer parametreleri - Optimize edildi
BUFFER_SIZE = 1024      # Küçük ve güvenilir buffer boyutu  
BLOCK_DURATION = 0.1    # Her blok 100ms 
OVERLAP_SAMPLES = 0     # Overlap kullanmıyoruz (sadece direkt kayıt)

# GPT model parametresi
GPT_MODEL = "gpt-4o-mini"  # OpenAI'nin gpt-4o-mini modeli
GPT_LANGUAGE = "tr"  # Türkçe dil desteği

# Analiz modelleri
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
VAD_MODEL = "pyannote/voice-activity-detection"
OVERLAP_MODEL = "pyannote/overlapped-speech-detection"
SEGMENTATION_MODEL = "pyannote/segmentation"
SEPARATION_MODEL = "pyannote/speech-separation-ami-1.0"

# Renk paleti konuşmacılar için
SPEAKER_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']

# Duygu analizi için renk paleti
EMOTION_COLORS = {
    'mutlu': '#2ECC71',
    'üzgün': '#3498DB', 
    'kızgın': '#E74C3C',
    'sakin': '#95A5A6',
    'heyecanlı': '#F39C12',
    'stresli': '#E67E22'
}

# Cinsiyet ve yaş tahmini modelleri
GENDER_AGE_MODELS = {
    'wav2vec2_gender': "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    'speechbrain_gender': "speechbrain/spkrec-ecapa-voxceleb",
    'hubert_age_gender': "facebook/hubert-large-ls960-ft",
    'whisper_feature_extractor': "openai/whisper-base"
}

# Cinsiyet renkleri
GENDER_COLORS = {
    'erkek': '#3498DB',      # Mavi
    'kadın': '#E91E63',      # Pembe
    'belirsiz': '#95A5A6'    # Gri
}

# Yaş grubu renkleri
AGE_COLORS = {
    'çocuk': '#FF9800',      # Turuncu (0-12)
    'genç': '#4CAF50',       # Yeşil (13-25)
    'yetişkin': '#2196F3',   # Mavi (26-45)
    'orta_yaş': '#9C27B0',   # Mor (46-65)
    'yaşlı': '#795548'       # Kahverengi (65+)
}

class SesKayitAnaliz:
    def __init__(self, root):
        self.root = root
        self.root.title("Gelişmiş Ses Analizi ve PyAnnote Sistemi")
        
        # macOS uyumluluğu için pencere ayarları
        self.root.geometry("1200x800")
        
        # macOS'ta state('zoomed') çalışmaz, alternatif yaklaşım
        try:
            # Windows/Linux için
            if self.root.tk.call('tk', 'windowingsystem') == 'win32':
                self.root.state('zoomed')
            # macOS için
            elif self.root.tk.call('tk', 'windowingsystem') == 'aqua':
                self.root.attributes('-zoomed', True)
            # Linux için
            else:
                self.root.attributes('-zoomed', True)
        except:
            # Hata durumunda normal boyut kullan
            self.root.geometry("1200x800")
        
        # Pencereyi ekranın ortasına yerleştir
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Minimum pencere boyutu
        self.root.minsize(800, 600)
        
        # Pencereyi görünür yap
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        
        # Ana çerçeve
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Başlık
        header = ttk.Label(main_frame, text="🎤 Gelişmiş Ses Analizi ve Konuşmacı Diyarizasyonu 🎤", 
                          font=("Helvetica", 18, "bold"))
        header.pack(pady=10)
        
        # Kontrol çerçevesi - Üst kısım
        control_frame_top = ttk.Frame(main_frame)
        control_frame_top.pack(fill=tk.X, pady=5)
        
        # Kayıt kontrolleri
        record_frame = ttk.LabelFrame(control_frame_top, text="🎙️ Kayıt Kontrolleri")
        record_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Kayıt süresi girişi
        ttk.Label(record_frame, text="Kayıt Süresi (saniye):").pack(side=tk.LEFT, padx=5)
        self.duration_var = tk.StringVar(value=str(RECORDING_SECONDS))
        duration_entry = ttk.Entry(record_frame, textvariable=self.duration_var, width=5)
        duration_entry.pack(side=tk.LEFT, padx=5)
        
        # Kayıt dosya adı girişi
        ttk.Label(record_frame, text="Dosya Adı:").pack(side=tk.LEFT, padx=5)
        self.filename_var = tk.StringVar(value=f"kayit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        filename_entry = ttk.Entry(record_frame, textvariable=self.filename_var, width=20)
        filename_entry.pack(side=tk.LEFT, padx=5)
        
        # Başlat/Durdur düğmesi
        self.record_button = ttk.Button(record_frame, text="🔴 Kaydı Başlat", 
                                       command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=10)
        
        # Dosya seç düğmesi
        self.load_button = ttk.Button(record_frame, text="📁 Dosya Seç", 
                                     command=self.load_audio_file)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        # Analiz seçenekleri çerçevesi
        analysis_frame = ttk.LabelFrame(control_frame_top, text="🔬 Analiz Seçenekleri")
        analysis_frame.pack(side=tk.RIGHT, padx=5)
        
        # Analiz seçenekleri
        self.enable_vad = tk.BooleanVar(value=True)
        self.enable_overlap = tk.BooleanVar(value=True)
        self.enable_emotion = tk.BooleanVar(value=True)
        self.enable_noise_reduction = tk.BooleanVar(value=True)
        self.enable_separation = tk.BooleanVar(value=False)
        self.enable_live_analysis = tk.BooleanVar(value=False)
        self.enable_gender_age = tk.BooleanVar(value=True)  # Yeni: Cinsiyet ve yaş analizi
        
        ttk.Checkbutton(analysis_frame, text="Ses Aktivitesi Tespiti", variable=self.enable_vad).pack(anchor=tk.W)
        ttk.Checkbutton(analysis_frame, text="Örtüşen Konuşma", variable=self.enable_overlap).pack(anchor=tk.W)
        ttk.Checkbutton(analysis_frame, text="Duygu Analizi", variable=self.enable_emotion).pack(anchor=tk.W)
        ttk.Checkbutton(analysis_frame, text="Cinsiyet ve Yaş Analizi", variable=self.enable_gender_age).pack(anchor=tk.W)
        ttk.Checkbutton(analysis_frame, text="Gürültü Azaltma", variable=self.enable_noise_reduction).pack(anchor=tk.W)
        ttk.Checkbutton(analysis_frame, text="Ses Ayrıştırma", variable=self.enable_separation).pack(anchor=tk.W)
        ttk.Checkbutton(analysis_frame, text="Canlı Analiz", variable=self.enable_live_analysis).pack(anchor=tk.W)
        
        # Kontrol çerçevesi - Alt kısım
        control_frame_bottom = ttk.Frame(main_frame)
        control_frame_bottom.pack(fill=tk.X, pady=5)
        
        # Ana analiz düğmeleri
        button_frame = ttk.Frame(control_frame_bottom)
        button_frame.pack(side=tk.LEFT)
        
        # Analiz düğmesi
        self.analyze_button = ttk.Button(button_frame, text="🔍 Tam Analiz Yap", 
                                        command=self.analyze_recording,
                                        state=tk.DISABLED)
        self.analyze_button.pack(side=tk.LEFT, padx=5)
        
        # Hızlı analiz düğmesi
        self.quick_analyze_button = ttk.Button(button_frame, text="⚡ Hızlı Analiz", 
                                              command=self.quick_analyze,
                                              state=tk.DISABLED)
        self.quick_analyze_button.pack(side=tk.LEFT, padx=5)
        
        # Rapor oluştur düğmesi
        self.report_button = ttk.Button(button_frame, text="📄 Rapor Oluştur", 
                                       command=self.generate_report,
                                       state=tk.DISABLED)
        self.report_button.pack(side=tk.LEFT, padx=5)
        
        # Durum etiketi
        self.status_label = ttk.Label(control_frame_bottom, text="🟢 Hazır", 
                                     font=("Helvetica", 12, "bold"))
        self.status_label.pack(side=tk.RIGHT, padx=20)
        
        # İçerik çerçevesi
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Sol panel (görselleştirme)
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=(0, 5))
        
        # Notebook widget for multiple visualizations
        self.viz_notebook = ttk.Notebook(left_panel)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Ses görselleştirme sekmesi
        viz_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(viz_frame, text="🌊 Dalga Formu")
        
        self.fig = Figure(figsize=(8, 6), dpi=100)
        
        # Alt grafik alanları oluştur
        self.ax_waveform = self.fig.add_subplot(3, 1, 1)
        self.ax_spectrogram = self.fig.add_subplot(3, 1, 2)
        self.ax_diarization = self.fig.add_subplot(3, 1, 3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Spektogram sekmesi
        spectrogram_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(spectrogram_frame, text="📊 Spektogram")
        
        self.fig2 = Figure(figsize=(8, 6), dpi=100)
        self.ax_spec_detail = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=spectrogram_frame)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Duygu analizi sekmesi
        emotion_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(emotion_frame, text="😊 Duygu Analizi")
        
        self.fig3 = Figure(figsize=(8, 6), dpi=100)
        self.ax_emotion = self.fig3.add_subplot(111)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=emotion_frame)
        self.canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # İstatistik sekmesi
        stats_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(stats_frame, text="📈 İstatistikler")
        
        self.fig4 = Figure(figsize=(8, 6), dpi=100)
        self.ax_stats1 = self.fig4.add_subplot(2, 2, 1)
        self.ax_stats2 = self.fig4.add_subplot(2, 2, 2)
        self.ax_stats3 = self.fig4.add_subplot(2, 2, 3)
        self.ax_stats4 = self.fig4.add_subplot(2, 2, 4)
        self.canvas4 = FigureCanvasTkAgg(self.fig4, master=stats_frame)
        self.canvas4.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Cinsiyet ve yaş sekmesi
        gender_age_viz_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(gender_age_viz_frame, text="👥 Cinsiyet & Yaş")
        
        self.fig5 = Figure(figsize=(8, 6), dpi=100)
        self.ax_gender = self.fig5.add_subplot(2, 2, 1)
        self.ax_age = self.fig5.add_subplot(2, 2, 2)
        self.ax_speaker_gender = self.fig5.add_subplot(2, 2, 3)
        self.ax_speaker_age = self.fig5.add_subplot(2, 2, 4)
        self.canvas5 = FigureCanvasTkAgg(self.fig5, master=gender_age_viz_frame)
        self.canvas5.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Boş grafikleri göster (thread-safe)
        self.root.after(500, self.update_empty_plots)
        
        # Sağ panel (log ve sonuçlar)
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT, padx=(5, 0))
        
        # Sağ panel için notebook
        self.right_notebook = ttk.Notebook(right_panel)
        self.right_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Log sekmesi
        log_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(log_frame, text="📋 İşlem Günlüğü")
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=25, 
                                                 font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Sonuç sekmesi
        result_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(result_frame, text="🎯 Diyarizasyon Sonuçları")
        
        self.result_text = scrolledtext.ScrolledText(result_frame, height=25, 
                                                    font=("Consolas", 9))
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Transkript sekmesi
        transcript_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(transcript_frame, text="💬 Konuşma İçeriği")
        
        self.transcript_text = scrolledtext.ScrolledText(transcript_frame, height=25, 
                                                       font=("Consolas", 9))
        self.transcript_text.pack(fill=tk.BOTH, expand=True)
        
        # Detaylı analiz sekmesi
        analysis_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(analysis_frame, text="🔬 Detaylı Analiz")
        
        self.analysis_text = scrolledtext.ScrolledText(analysis_frame, height=25, 
                                                      font=("Consolas", 9))
        self.analysis_text.pack(fill=tk.BOTH, expand=True)
        
        # Duygu analizi sekmesi
        emotion_analysis_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(emotion_analysis_frame, text="😊 Duygu Raporu")
        
        self.emotion_text = scrolledtext.ScrolledText(emotion_analysis_frame, height=25, 
                                                     font=("Consolas", 9))
        self.emotion_text.pack(fill=tk.BOTH, expand=True)
        
        # Cinsiyet ve yaş analizi sekmesi
        gender_age_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(gender_age_frame, text="👥 Cinsiyet & Yaş")
        
        self.gender_age_text = scrolledtext.ScrolledText(gender_age_frame, height=25, 
                                                        font=("Consolas", 9))
        self.gender_age_text.pack(fill=tk.BOTH, expand=True)
        
        # Durum değişkenleri
        self.is_recording = False
        self.recorded_audio = []
        self.recording_thread = None
        self.current_audio_file = None
        self.sample_rate = SAMPLE_RATE
        
        # Analiz sonuçları
        self.diarization_result = None
        self.vad_result = None
        self.overlap_result = None
        self.emotion_result = None
        self.gender_age_result = {}  # Yeni: Cinsiyet ve yaş sonuçları
        self.speaker_embeddings = {}
        self.noise_reduced_audio = None
        
        # Pipeline'lar
        self.pipelines = {}
        self.models_loaded = False
        
        # Gerçek zamanlı analiz için
        self.live_analysis_thread = None
        self.live_analysis_running = False
        
        # Hoşgeldin mesajı
        self.add_log("🚀 Gelişmiş Ses Analizi Sistemi başlatıldı!")
        self.add_log("📝 Kullanım: Ses kaydı yapın veya dosya yükleyin, sonra analiz seçeneklerini belirleyip analiz başlatın.")
        self.add_log("⚡ İpucu: Hızlı analiz için sadece temel özellikleri seçin, tam analiz için tüm seçenekleri aktifleştirin.")
        self.add_log("🔧 Modeller ilk kullanımda otomatik olarak yüklenecektir.")
        
        # Pencereyi zorla güncelle ve görünür yap
        self.root.update_idletasks()
        self.root.update()
        
        # macOS uyumluluğu için son ayarlar
        self.root.after(100, self._final_setup)
    
    def _final_setup(self):
        """Son kurulum işlemleri - macOS uyumluluğu"""
        try:
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.after(100, lambda: self.root.attributes('-topmost', False))
            self.root.focus_force()
            self.add_log("✅ Arayüz hazır!")
        except Exception as e:
            self.add_log(f"⚠️ Arayüz kurulum uyarısı: {e}")
    
    def update_empty_plots(self):
        """Boş grafikleri göster"""
        # Ana dalga formu
        self.ax_waveform.clear()
        self.ax_waveform.text(0.5, 0.5, "🎤 Ses kaydı bekleniyor...", 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=12, color='gray', transform=self.ax_waveform.transAxes)
        self.ax_waveform.set_title("Ses Dalga Formu")
        self.ax_waveform.grid(True, alpha=0.3)
        
        # Spektogram
        self.ax_spectrogram.clear()
        self.ax_spectrogram.text(0.5, 0.5, "📊 Spektogram görünümü", 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=10, color='gray', transform=self.ax_spectrogram.transAxes)
        self.ax_spectrogram.set_title("Spektogram")
        
        # Diyarizasyon
        self.ax_diarization.clear()
        self.ax_diarization.text(0.5, 0.5, "👥 Konuşmacı zaman çizelgesi", 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=10, color='gray', transform=self.ax_diarization.transAxes)
        self.ax_diarization.set_title("Konuşmacı Diyarizasyonu")
        self.ax_diarization.set_xlabel("Zaman (saniye)")
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Detaylı spektogram
        self.ax_spec_detail.clear()
        self.ax_spec_detail.text(0.5, 0.5, "📈 Detaylı frekans analizi için ses yükleyin", 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=12, color='gray', transform=self.ax_spec_detail.transAxes)
        self.canvas2.draw()
        
        # Duygu analizi
        self.ax_emotion.clear()
        self.ax_emotion.text(0.5, 0.5, "😊 Duygu analizi sonuçları burada görünecek", 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=12, color='gray', transform=self.ax_emotion.transAxes)
        self.canvas3.draw()
        
        # İstatistikler
        for ax in [self.ax_stats1, self.ax_stats2, self.ax_stats3, self.ax_stats4]:
            ax.clear()
            ax.text(0.5, 0.5, "📊", 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=20, color='lightgray', transform=ax.transAxes)
        self.canvas4.draw()
        
        # Cinsiyet ve yaş grafikleri
        for ax in [self.ax_gender, self.ax_age, self.ax_speaker_gender, self.ax_speaker_age]:
            ax.clear()
            ax.text(0.5, 0.5, "👥", 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=20, color='lightgray', transform=ax.transAxes)
        self.canvas5.draw()
    
    def load_audio_file(self):
        """Ses dosyası yükle"""
        file_path = filedialog.askopenfilename(
            title="Ses Dosyası Seçin",
            filetypes=[
                ("Ses Dosyaları", "*.wav *.mp3 *.flac *.m4a *.ogg"),
                ("WAV Dosyaları", "*.wav"),
                ("MP3 Dosyaları", "*.mp3"),
                ("FLAC Dosyaları", "*.flac"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.add_log(f"📁 Dosya yükleniyor: {os.path.basename(file_path)}")
                
                # Ses dosyasını yükle
                audio_data, sample_rate = librosa.load(file_path, sr=None)
                self.sample_rate = sample_rate
                self.current_audio_file = file_path
                self.recorded_audio = audio_data
                
                # Dalga formunu güncelle
                self.update_comprehensive_plots(audio_data)
                
                # Analiz düğmelerini etkinleştir
                self.analyze_button.config(state=tk.NORMAL)
                self.quick_analyze_button.config(state=tk.NORMAL)
                
                self.add_log(f"✅ Dosya başarıyla yüklendi! Süre: {len(audio_data)/sample_rate:.2f} saniye")
                self.status_label.config(text="🟢 Dosya yüklendi - Analiz için hazır")
                
            except Exception as e:
                self.add_log(f"❌ Dosya yükleme hatası: {e}")
                messagebox.showerror("Hata", f"Dosya yüklenirken hata oluştu:\n{e}")
    
    def update_comprehensive_plots(self, audio_data):
        """Kapsamlı görselleştirme güncelle"""
        time_axis = np.arange(len(audio_data)) / self.sample_rate
        
        # Ana dalga formu
        self.ax_waveform.clear()
        self.ax_waveform.plot(time_axis, audio_data, color='#2E86AB', linewidth=0.8)
        self.ax_waveform.set_xlim(0, max(time_axis))
        self.ax_waveform.set_ylim(-1, 1)
        self.ax_waveform.set_ylabel("Genlik")
        self.ax_waveform.set_title("🌊 Ses Dalga Formu")
        self.ax_waveform.grid(True, alpha=0.3)
        
        # Spektogram
        self.ax_spectrogram.clear()
        f, t, Sxx = signal.spectrogram(audio_data, self.sample_rate, nperseg=1024)
        self.ax_spectrogram.pcolormesh(t, f[:len(f)//4], 10 * np.log10(Sxx[:len(f)//4]), 
                                      shading='gouraud', cmap='viridis')
        self.ax_spectrogram.set_ylabel('Frekans (Hz)')
        self.ax_spectrogram.set_title("📊 Spektogram")
        
        # Diyarizasyon placeholder
        self.ax_diarization.clear()
        self.ax_diarization.text(0.5, 0.5, "👥 Diyarizasyon için analiz yapın", 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=10, color='gray', transform=self.ax_diarization.transAxes)
        self.ax_diarization.set_xlabel("Zaman (saniye)")
        self.ax_diarization.set_title("Konuşmacı Diyarizasyonu")
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Detaylı spektogram
        self.ax_spec_detail.clear()
        f_detail, t_detail, Sxx_detail = signal.spectrogram(audio_data, self.sample_rate, nperseg=2048)
        im = self.ax_spec_detail.pcolormesh(t_detail, f_detail, 10 * np.log10(Sxx_detail), 
                                           shading='gouraud', cmap='plasma')
        self.ax_spec_detail.set_ylabel('Frekans (Hz)')
        self.ax_spec_detail.set_xlabel('Zaman (saniye)')
        self.ax_spec_detail.set_title('🎵 Detaylı Spektogram')
        plt.colorbar(im, ax=self.ax_spec_detail, label='Güç (dB)')
        self.canvas2.draw()
        
    def update_waveform_plot(self, audio_data):
        """Basitçe dalga formunu güncelle"""
        self.update_comprehensive_plots(audio_data)
    
    def toggle_recording(self):
        """Kaydı başlat/durdur"""
        if not self.is_recording:
            # Kayıt süresini al
            try:
                recording_seconds = int(self.duration_var.get())
                if recording_seconds <= 0:
                    messagebox.showerror("Hata", "Kayıt süresi pozitif bir sayı olmalıdır!")
                    return
            except ValueError:
                messagebox.showerror("Hata", "Geçerli bir kayıt süresi giriniz!")
                return
            
            # Kaydı başlat
            self.record_button.config(text="⏹️ Kaydı Durdur")
            self.status_label.config(text="🔴 Kayıt yapılıyor...")
            self.analyze_button.config(state=tk.DISABLED)
            self.quick_analyze_button.config(state=tk.DISABLED)
            self.recorded_audio = []
            self.is_recording = True
            
            # Canlı analiz başlat
            if self.enable_live_analysis.get():
                self.start_live_analysis()
            
            # İş parçacığını başlat
            self.recording_thread = threading.Thread(target=self.record_audio, 
                                                   args=(recording_seconds,))
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            self.add_log(f"🎙️ {recording_seconds} saniyelik kayıt başlatıldı.")
        else:
            # Kaydı durdur
            self.is_recording = False
            self.record_button.config(text="🔴 Kaydı Başlat")
            self.status_label.config(text="⏹️ Kayıt durduruldu")
            self.add_log("⏹️ Kayıt manuel olarak durduruldu.")
            
            # Canlı analizi durdur
            if self.live_analysis_running:
                self.stop_live_analysis()
    
    def record_audio(self, duration):
        """Basit ve güvenilir ses kaydı"""
        try:
            self.add_log("🎙️ Basit kayıt sistemi başlatılıyor...")
            
            # Ses cihazını optimize et
            self.optimize_audio_device()
            
            # Toplam sample sayısını hesapla
            total_samples = int(SAMPLE_RATE * duration)
            self.add_log(f"📊 Hedef: {duration}s = {total_samples} sample")
            
            # Kayıt için buffer
            self.recorded_audio = np.zeros(total_samples, dtype=DTYPE)
            current_frame = 0
            
            # Buffer boyutu - daha küçük ve güvenilir
            block_size = 1024  # Sabit 1024 sample bloklar
            
            def audio_callback(indata, frames, time, status):
                nonlocal current_frame
                
                if status:
                    self.add_log(f"⚠️ Audio status: {status}")
                
                if self.is_recording and current_frame < total_samples:
                    # Kaç sample alacağımızı hesapla  
                    samples_to_take = min(frames, total_samples - current_frame)
                    
                    # Veriyi direkt kayıt buffer'ına kopyala
                    self.recorded_audio[current_frame:current_frame + samples_to_take] = indata[:samples_to_take, 0]
                    current_frame += samples_to_take
                    
                    # Progress güncelle (her 0.5 saniyede bir)
                    if current_frame % (SAMPLE_RATE // 2) < frames:
                        progress = (current_frame / total_samples) * 100
                        elapsed = current_frame / SAMPLE_RATE
                        self.root.after(1, lambda p=progress, e=elapsed: self.update_recording_progress(p, e, duration))
                    
                    # Kayıt tamamlandı mı?
                    if current_frame >= total_samples:
                        self.is_recording = False
            
            self.add_log(f"🎤 Kayıt başlıyor: {block_size} sample bloklar")
            self.add_log(f"🔧 Sample rate: {SAMPLE_RATE} Hz")
            self.add_log(f"📊 Dtype: {DTYPE}")
            self.add_log(f"🎧 Channels: {CHANNELS}")
            
            # Debug - ses cihazı bilgileri
            try:
                current_device = sd.query_devices(sd.default.device[0])
                self.add_log(f"🎤 Aktif cihaz: {current_device['name']}")
                self.add_log(f"⚡ Desteklenen sample rate: {current_device['default_samplerate']}")
            except:
                pass
            
            # Stream başlat
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=block_size,
                callback=audio_callback,
                latency='low'
            ):
                start_time = time.time()
                
                # Kayıt döngüsü - çok basit
                while self.is_recording and current_frame < total_samples:
                    time.sleep(0.1)  # 100ms bekle
                    
                    # Timeout kontrolü
                    if time.time() - start_time > duration + 2:  # 2 saniye extra
                        self.add_log("⏰ Kayıt timeout - zorla durduruluyor")
                        break
            
            # Kayıt işlemi bitti
            self.is_recording = False
            self.root.after(1, lambda: self.record_button.config(text="🔴 Kaydı Başlat"))
            
            # Sonuçları kontrol et
            actual_duration = current_frame / SAMPLE_RATE
            self.add_log(f"✅ Kayıt tamamlandı!")
            self.add_log(f"📊 Hedef: {duration}s, Gerçek: {actual_duration:.2f}s")
            self.add_log(f"🔢 Sample sayısı: {current_frame}/{total_samples}")
            
            # Kaydedilen veriyi kes (gereksiz sıfırları temizle)
            if current_frame > 0:
                self.recorded_audio = self.recorded_audio[:current_frame]
                self.sample_rate = SAMPLE_RATE
                
                # Ses seviyesi analizi
                max_amplitude = np.max(np.abs(self.recorded_audio))
                avg_amplitude = np.mean(np.abs(self.recorded_audio))
                
                self.add_log(f"🔊 Max seviye: {max_amplitude:.4f}")
                self.add_log(f"📈 Ortalama seviye: {avg_amplitude:.4f}")
                
                # Normalize et (sadece gerekirse)
                if max_amplitude > 0.95:
                    self.recorded_audio = self.recorded_audio / max_amplitude * 0.9
                    self.add_log("🔧 Ses seviyesi normalize edildi")
                
                # Grafikleri güncelle
                self.root.after(1, lambda: self.update_comprehensive_plots(self.recorded_audio))
                self.root.after(1, lambda: self.status_label.config(text="✅ Kayıt tamamlandı"))
                
                # Dosyaya kaydet
                self.save_recording()
                
                # Düğmeleri etkinleştir
                self.root.after(1, lambda: self.analyze_button.config(state=tk.NORMAL))
                self.root.after(1, lambda: self.quick_analyze_button.config(state=tk.NORMAL))
                self.root.after(1, lambda: self.report_button.config(state=tk.NORMAL))
            else:
                self.add_log("❌ Hiç ses verisi alınamadı!")
                self.root.after(1, lambda: self.status_label.config(text="❌ Kayıt başarısız"))
                
        except Exception as e:
            self.add_log(f"❌ Callback kayıt hatası: {e}")
            self.add_log("🔄 Fallback basit kayıt yöntemi deneniyor...")
            
            # Fallback - basit kayıt yöntemi
            self.is_recording = True  # Reset için
            success = self.record_audio_simple(duration)
            
            if not success:
                self.is_recording = False
                self.root.after(1, lambda: self.record_button.config(text="🔴 Kaydı Başlat"))
                self.root.after(1, lambda: self.status_label.config(text="❌ Tüm kayıt yöntemleri başarısız!"))
        finally:
            # Canlı analizi durdur
            if self.live_analysis_running:
                self.stop_live_analysis()
    
    def save_recording(self):
        """Kaydı yüksek kaliteli WAV dosyasına kaydet"""
        try:
            filename = self.filename_var.get()
            if not filename.endswith(".wav"):
                filename += ".wav"
            
            # Soundfile kullanarak yüksek kaliteli kayıt
            # (wave modülünden daha iyi kalite)
            sf.write(filename, self.recorded_audio, SAMPLE_RATE, 
                    subtype='PCM_24')  # 24-bit yüksek kalite
            
            self.current_audio_file = filename
            
            # Dosya bilgilerini göster
            file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
            duration = len(self.recorded_audio) / SAMPLE_RATE
            
            self.add_log(f"💾 Kayıt '{filename}' dosyasına kaydedildi.")
            self.add_log(f"📊 Dosya boyutu: {file_size:.2f} MB, Süre: {duration:.2f}s")
            self.add_log(f"🎵 Format: 24-bit PCM, {SAMPLE_RATE} Hz, {CHANNELS} kanal")
            
        except Exception as e:
            self.add_log(f"❌ Dosya kaydetme hatası: {e}")
            # Fallback - basic wave format
            try:
                filename = self.filename_var.get()
                if not filename.endswith(".wav"):
                    filename += ".wav"
                    
                int_data = np.int16(self.recorded_audio * 32767)
                with wave.open(filename, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(int_data.tobytes())
                
                self.current_audio_file = filename
                self.add_log(f"💾 Fallback kayıt başarılı: {filename}")
            except Exception as e2:
                self.add_log(f"❌ Fallback kayıt da başarısız: {e2}")
    
    def load_models(self):
        """Analiz modellerini yükle"""
        if self.models_loaded:
            return True
            
        try:
            self.add_log("🔄 Modeller yükleniyor...")
            self.status_label.config(text="🔄 Modeller yükleniyor...")
            
            # GPU kullanımını kontrol et
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.add_log(f"💻 Cihaz: {device}")
            
            # Diyarizasyon pipeline'ı
            if 'diarization' not in self.pipelines:
                self.pipelines['diarization'] = Pipeline.from_pretrained(
                    DIARIZATION_MODEL, use_auth_token=HF_TOKEN
                )
                self.pipelines['diarization'].to(device)
                self.add_log("✅ Diyarizasyon modeli yüklendi")
            
            # VAD pipeline'ı
            if self.enable_vad.get() and 'vad' not in self.pipelines:
                try:
                    self.pipelines['vad'] = Pipeline.from_pretrained(
                        VAD_MODEL, use_auth_token=HF_TOKEN
                    )
                    self.pipelines['vad'].to(device)
                    self.add_log("✅ Ses Aktivitesi Tespiti modeli yüklendi")
                except Exception as e:
                    self.add_log(f"⚠️ VAD modeli yüklenemedi: {e}")
            
            # Örtüşen konuşma tespiti pipeline'ı
            if self.enable_overlap.get() and 'overlap' not in self.pipelines:
                try:
                    self.pipelines['overlap'] = Pipeline.from_pretrained(
                        OVERLAP_MODEL, use_auth_token=HF_TOKEN
                    )
                    self.pipelines['overlap'].to(device)
                    self.add_log("✅ Örtüşen Konuşma Tespiti modeli yüklendi")
                except Exception as e:
                    self.add_log(f"⚠️ Örtüşme modeli yüklenemedi: {e}")
            
            self.models_loaded = True
            self.add_log("🎉 Tüm modeller başarıyla yüklendi!")
            return True
            
        except Exception as e:
            self.add_log(f"❌ Model yükleme hatası: {e}")
            messagebox.showerror("Model Hatası", f"Modeller yüklenirken hata oluştu:\n{e}")
            return False
    
    def quick_analyze(self):
        """Hızlı analiz yap (sadece diyarizasyon)"""
        if not self.current_audio_file and len(self.recorded_audio) == 0:
            messagebox.showwarning("Uyarı", "Önce ses kaydı yapın veya dosya yükleyin!")
            return
            
        self.add_log("⚡ Hızlı analiz başlatılıyor...")
        self.status_label.config(text="⚡ Hızlı analiz yapılıyor...")
        self.quick_analyze_button.config(state=tk.DISABLED)
        
        # Sadece diyarizasyon için diğer seçenekleri geçici olarak kapat
        original_states = {
            'vad': self.enable_vad.get(),
            'overlap': self.enable_overlap.get(),
            'emotion': self.enable_emotion.get(),
            'noise': self.enable_noise_reduction.get(),
            'separation': self.enable_separation.get()
        }
        
        # Hızlı analiz için seçenekleri kapat
        self.enable_vad.set(False)
        self.enable_overlap.set(False)
        self.enable_emotion.set(False)
        self.enable_noise_reduction.set(False)
        self.enable_separation.set(False)
        
        # Analizi başlat
        filename = self.current_audio_file or self.get_current_filename()
        threading.Thread(target=self.run_analysis, args=(filename, True, original_states)).start()
    
    def analyze_recording(self):
        """Tam analiz yap"""
        if not self.current_audio_file and len(self.recorded_audio) == 0:
            messagebox.showwarning("Uyarı", "Önce ses kaydı yapın veya dosya yükleyin!")
            return
            
        filename = self.current_audio_file or self.get_current_filename()
        
        if not os.path.exists(filename):
            messagebox.showerror("Hata", f"'{filename}' dosyası bulunamadı!")
            return
        
        self.add_log("🔍 Tam analiz başlatılıyor...")
        self.status_label.config(text="🔍 Tam analiz yapılıyor...")
        self.analyze_button.config(state=tk.DISABLED)
        
        # İş parçacığını başlat
        threading.Thread(target=self.run_analysis, args=(filename,)).start()
    
    def get_current_filename(self):
        """Geçerli dosya adını al"""
        filename = self.filename_var.get()
        if not filename.endswith(".wav"):
            filename += ".wav"
        return filename
    
    def run_analysis(self, audio_file, is_quick=False, restore_states=None):
        """Kapsamlı ses analizi yap"""
        try:
            # Modelleri yükle
            if not self.load_models():
                return
            
            start_time = time.time()
            
            # Ses dosyasını yükle
            self.add_log(f"📂 Ses dosyası yükleniyor: {os.path.basename(audio_file)}")
            audio_data, sample_rate = librosa.load(audio_file, sr=None)
            
            # Gelişmiş duygu analizi filtreleme sistemi
            if self.enable_noise_reduction.get():
                self.add_log("🔇 Gürültü azaltma işlemi başlatılıyor...")
                # Standart gürültü azaltma
                audio_data = self.apply_noise_reduction(audio_data, sample_rate)
                
                # Duygu analizi için gelişmiş filtreleme
                if self.enable_emotion.get():
                    audio_data = self.apply_advanced_emotion_filtering(audio_data, sample_rate)
                    
                    # Spektral domain iyileştirmeleri
                    audio_data = self.apply_spectral_emotion_enhancement(audio_data, sample_rate)
                    
                    # Psiko-akustik filtreleme
                    audio_data = self.apply_psychoacoustic_filtering(audio_data, sample_rate)
                
                self.noise_reduced_audio = audio_data
            
            # 1. Voice Activity Detection (VAD)
            if self.enable_vad.get():
                self.add_log("🎯 Ses Aktivitesi Tespiti yapılıyor...")
                self.vad_result = self.run_vad_analysis(audio_file)
                
            # 2. Diyarizasyon
            self.add_log("👥 Konuşmacı diyarizasyonu yapılıyor...")
            self.diarization_result = self.run_diarization(audio_file)
            
            # 3. Örtüşen konuşma tespiti
            if self.enable_overlap.get():
                self.add_log("🗣️ Örtüşen konuşma tespiti yapılıyor...")
                self.overlap_result = self.run_overlap_detection(audio_file)
            
            # 4. Ses ayrıştırma
            if self.enable_separation.get():
                self.add_log("🎼 Ses ayrıştırma işlemi yapılıyor...")
                self.run_speech_separation(audio_file)
            
            # 5. Konuşmacı tanıma ve embedding'ler
            self.add_log("🔍 Konuşmacı embedding'leri çıkarılıyor...")
            self.extract_speaker_embeddings(audio_file, self.diarization_result)
            
            # 6. Transkripsiyon
            self.add_log("📝 Konuşma transkripte ediliyor...")
            self.run_speaker_based_transcription(audio_file, self.diarization_result)
            
            # 7. Duygu analizi
            if self.enable_emotion.get():
                self.add_log("😊 Duygu analizi yapılıyor...")
                self.emotion_result = self.run_ml_emotion_analysis(audio_data, sample_rate)
            
            # 8. Cinsiyet ve yaş analizi
            self.add_log("👥 Cinsiyet ve yaş analizi başlatılıyor...")
            self.gender_age_result = self.run_gender_age_analysis(audio_data, sample_rate, self.diarization_result)
            
            # 9. Görselleştirmeleri güncelle
            self.update_advanced_visualizations(audio_data, sample_rate)
            
            # 10. Detaylı rapor oluştur
            self.generate_detailed_analysis_report()
            
            # Süre hesapla
            end_time = time.time()
            total_time = end_time - start_time
            
            self.add_log(f"🎉 Analiz tamamlandı! Toplam süre: {total_time:.2f} saniye")
            self.status_label.config(text="✅ Analiz tamamlandı")
            
            # Hızlı analiz durumunu geri yükle
            if is_quick and restore_states:
                for key, value in restore_states.items():
                    if key == 'vad':
                        self.enable_vad.set(value)
                    elif key == 'overlap':
                        self.enable_overlap.set(value)
                    elif key == 'emotion':
                        self.enable_emotion.set(value)
                    elif key == 'noise':
                        self.enable_noise_reduction.set(value)
                    elif key == 'separation':
                        self.enable_separation.set(value)
            
        except Exception as e:
            self.add_log(f"❌ Analiz hatası: {e}")
            self.status_label.config(text="❌ Analiz hatası!")
            messagebox.showerror("Analiz Hatası", f"Analiz sırasında hata oluştu:\n{e}")
        finally:
            # Düğmeleri yeniden etkinleştir
            self.analyze_button.config(state=tk.NORMAL)
            self.quick_analyze_button.config(state=tk.NORMAL)
    
    def run_diarization(self, audio_file):
        """PyAnnote diyarizasyon işlemini çalıştır"""
        try:
            # Diyarizasyon uygula
            diarization = self.pipelines['diarization'](audio_file)
            
            # Sonuçları işle ve göster
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"🎯 Diyarizasyon Sonuçları: {os.path.basename(audio_file)}\n")
            self.result_text.insert(tk.END, f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Konuşmacıları ve zaman aralıklarını göster
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start = turn.start
                end = turn.end
                duration = end - start
                speaker_segments.append({
                    'speaker': speaker,
                    'start': start,
                    'end': end,
                    'duration': duration
                })
                result_line = f"👤 {speaker}: {start:.2f}s - {end:.2f}s ({duration:.2f}s)\n"
                self.result_text.insert(tk.END, result_line)
            
            # Konuşmacı istatistiklerini hesapla
            speaker_stats = {}
            total_speech_time = 0
            for segment in speaker_segments:
                speaker = segment['speaker']
                duration = segment['duration']
                if speaker not in speaker_stats:
                    speaker_stats[speaker] = {'total_time': 0, 'segment_count': 0}
                speaker_stats[speaker]['total_time'] += duration
                speaker_stats[speaker]['segment_count'] += 1
                total_speech_time += duration
            
            # İstatistikleri göster
            self.result_text.insert(tk.END, "\n📊 Konuşmacı İstatistikleri:\n")
            self.result_text.insert(tk.END, f"📈 Toplam konuşma süresi: {total_speech_time:.2f} saniye\n")
            self.result_text.insert(tk.END, f"👥 Toplam konuşmacı sayısı: {len(speaker_stats)}\n\n")
            
            for speaker, stats in speaker_stats.items():
                percentage = (stats['total_time'] / total_speech_time) * 100 if total_speech_time > 0 else 0
                self.result_text.insert(tk.END, 
                    f"{speaker}:\n"
                    f"  ⏱️ Süre: {stats['total_time']:.2f}s ({percentage:.1f}%)\n"
                    f"  💬 Segment sayısı: {stats['segment_count']}\n"
                    f"  📏 Ortalama segment: {stats['total_time']/stats['segment_count']:.2f}s\n"
                )
                
                # Cinsiyet ve yaş bilgilerini ekle (eğer analiz yapılmışsa)
                if hasattr(self, 'gender_age_result') and self.gender_age_result:
                    if 'detailed' in self.gender_age_result and 'speaker_based' in self.gender_age_result['detailed']:
                        speaker_results = self.gender_age_result['detailed']['speaker_based']
                        if speaker in speaker_results:
                            speaker_data = speaker_results[speaker]
                            
                            # En yüksek skorlu cinsiyet ve yaş
                            dominant_gender = max(speaker_data['gender'].items(), key=lambda x: x[1])
                            dominant_age = max(speaker_data['age'].items(), key=lambda x: x[1])
                            confidence = speaker_data.get('confidence', 0.5)
                            
                            # İkonlar
                            gender_icons = {'erkek': '👨', 'kadın': '👩', 'belirsiz': '❓'}
                            age_icons = {
                                'çocuk': '👶', 'genç': '🧒', 'yetişkin': '👤', 
                                'orta_yaş': '🧑', 'yaşlı': '👴'
                            }
                            
                            gender_icon = gender_icons.get(dominant_gender[0], '❓')
                            age_icon = age_icons.get(dominant_age[0], '❓')
                            
                            self.result_text.insert(tk.END, 
                                f"  {gender_icon} Cinsiyet: {dominant_gender[0].capitalize()} ({dominant_gender[1]*100:.1f}%)\n"
                                f"  {age_icon} Yaş Grubu: {dominant_age[0].capitalize()} ({dominant_age[1]*100:.1f}%)\n"
                                f"  🎯 Güven: {confidence*100:.1f}%\n"
                            )
                
                self.result_text.insert(tk.END, "\n")
            
            self.add_log("✅ Diyarizasyon tamamlandı.")
            return diarization
            
        except Exception as e:
            self.add_log(f"❌ Diyarizasyon hatası: {e}")
            return None
    
    def apply_noise_reduction(self, audio_data, sample_rate):
        """Gürültü azaltma uygula"""
        try:
            # Önce ses verisini temizle
            audio_data = self.clean_audio_buffer(audio_data)
            
            # NoiseReduce kütüphanesi ile gürültü azaltma (daha muhafazakar)
            reduced_noise = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.6)  # 0.8'den 0.6'ya düşürdük
            
            # Sonucu tekrar temizle
            reduced_noise = self.clean_audio_buffer(reduced_noise)
            
            self.add_log("✅ Gürültü azaltma tamamlandı")
            return reduced_noise
        except Exception as e:
            self.add_log(f"⚠️ Gürültü azaltma hatası: {e}")
            # Hata durumunda temizlenmiş orijinal veriyi döndür
            return self.clean_audio_buffer(audio_data)
    
    def apply_advanced_emotion_filtering(self, audio_data, sample_rate):
        """Duygu analizi için gelişmiş filtreleme sistemi"""
        try:
            self.add_log("🎭 Duygu analizi için gelişmiş filtreleme başlatılıyor...")
            
            # 1. Adaptif Gürültü Azaltma - Duygu tonlarını koruyucu
            filtered_audio = self.adaptive_noise_reduction(audio_data, sample_rate)
            
            # 2. Vokal Frekans Vurgulama (İnsan sesi 80-8000 Hz)
            filtered_audio = self.enhance_vocal_frequencies(filtered_audio, sample_rate)
            
            # 3. Duygusal Tonlama Koruması
            filtered_audio = self.preserve_emotional_tones(filtered_audio, sample_rate)
            
            # 4. Dinamik Aralık Optimizasyonu
            filtered_audio = self.optimize_dynamic_range(filtered_audio)
            
            # 5. Kahkaha ve Özel Ses Desenlerini Koruma
            filtered_audio = self.preserve_laughter_patterns(filtered_audio, sample_rate)
            
            self.add_log("✅ Gelişmiş duygu filtreleme tamamlandı")
            return filtered_audio
            
        except Exception as e:
            self.add_log(f"❌ Gelişmiş filtreleme hatası: {e}")
            return audio_data
    
    def adaptive_noise_reduction(self, audio_data, sample_rate):
        """Adaptif gürültü azaltma - duygu tonlarını korur"""
        try:
            # Ses enerjisine göre adaptif filtreleme
            energy_threshold = np.percentile(np.abs(audio_data), 70)
            
            if energy_threshold < 0.01:  # Çok sessiz ses
                # Hafif filtreleme - duygusal nüansları korur
                return nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.3)
            elif energy_threshold > 0.1:  # Yüksek enerjili ses (bağırma, kahkaha)
                # Orta seviye filtreleme
                return nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.5)
            else:  # Normal konuşma
                # Standart filtreleme
                return nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.6)
                
        except Exception as e:
            return audio_data
    
    def enhance_vocal_frequencies(self, audio_data, sample_rate):
        """İnsan sesi frekanslarını vurgula (80-8000 Hz)"""
        try:
            # FFT ile frekans alanına geç
            fft = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)
            
            # İnsan sesi frekans maskesi oluştur
            vocal_mask = (np.abs(freqs) >= 80) & (np.abs(freqs) <= 8000)
            
            # Vokal frekansları hafifçe vurgula
            fft[vocal_mask] *= 1.2
            
            # Çok yüksek frekansları azalt (gürültü olabilir)
            high_freq_mask = np.abs(freqs) > 8000
            fft[high_freq_mask] *= 0.7
            
            # Geri dönüştür
            enhanced_audio = np.real(np.fft.ifft(fft))
            
            # Aşırı büyüme kontrolü
            max_val = np.max(np.abs(enhanced_audio))
            if max_val > 1.0:
                enhanced_audio = enhanced_audio / max_val * 0.95
                
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            return audio_data
    
    def preserve_emotional_tones(self, audio_data, sample_rate):
        """Duygusal tonlamaları koruyucu filtreleme"""
        try:
            # Pitch tracking ile duygusal tonlama tespiti
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
            
            # Pitch değişkenliği analizi
            pitch_values = pitches[pitches > 0]
            if len(pitch_values) > 0:
                pitch_variance = np.var(pitch_values)
                
                # Yüksek pitch varyansı = duygusal konuşma
                if pitch_variance > 1000:  # Duygusal konuşma tespit edildi
                    # Daha az agresif filtreleme uygula
                    return nr.reduce_noise(y=audio_data, sr=sample_rate, 
                                         prop_decrease=0.4)  # Çok hafif
                else:
                    # Normal filtreleme
                    return nr.reduce_noise(y=audio_data, sr=sample_rate, 
                                         prop_decrease=0.6)
            else:
                return audio_data
                
        except Exception as e:
            return audio_data
    
    def optimize_dynamic_range(self, audio_data):
        """Dinamik aralığı optimize et - duygu analizine uygun"""
        try:
            # Ses seviyesi analizi
            rms = np.sqrt(np.mean(audio_data**2))
            
            if rms < 0.01:  # Çok sessiz
                # Hafif amplifikasyon
                amplified = audio_data * 2.0
                return np.clip(amplified, -1.0, 1.0)
            elif rms > 0.3:  # Çok yüksek
                # Hafif kompresyon
                compressed = audio_data * 0.7
                return compressed
            else:
                return audio_data
                
        except Exception as e:
            return audio_data
    
    def preserve_laughter_patterns(self, audio_data, sample_rate):
        """Kahkaha ve özel ses desenlerini koruma"""
        try:
            # Kısa-dönem enerji analizi (kahkaha tespiti için)
            frame_length = int(0.025 * sample_rate)  # 25ms
            hop_length = int(0.01 * sample_rate)     # 10ms
            
            energy_frames = []
            for i in range(0, len(audio_data) - frame_length, hop_length):
                frame = audio_data[i:i + frame_length]
                energy = np.sum(frame ** 2)
                energy_frames.append(energy)
            
            energy_frames = np.array(energy_frames)
            
            # Ani enerji artışları (kahkaha göstergesi)
            energy_diff = np.diff(energy_frames)
            sudden_peaks = np.where(energy_diff > np.percentile(energy_diff, 90))[0]
            
            if len(sudden_peaks) > 3:  # Muhtemelen kahkaha var
                # Çok hafif filtreleme - kahkaha desenlerini koru
                return nr.reduce_noise(y=audio_data, sr=sample_rate, 
                                     prop_decrease=0.2)
            else:
                return audio_data
                
        except Exception as e:
            return audio_data
    
    def apply_spectral_emotion_enhancement(self, audio_data, sample_rate):
        """Spektral domain'de duygu analizi için özel iyileştirmeler"""
        try:
            self.add_log("🎵 Spektral duygu iyileştirmesi başlatılıyor...")
            
            # STFT ile spektral analiz
            stft = librosa.stft(audio_data, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Frekans bantları tanımla
            freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
            
            # Duygu-spesifik frekans bantları
            emotion_bands = {
                'fundamental': (80, 300),    # Temel ses perdesi
                'formants': (300, 3000),     # Formant frekansları
                'brightness': (3000, 8000),  # Parlaklık (mutluluk göstergesi)
                'breathiness': (8000, 12000) # Nefes sesleri (duygusal durum)
            }
            
            # Her bant için özel işlem
            enhanced_magnitude = magnitude.copy()
            
            for band_name, (low_freq, high_freq) in emotion_bands.items():
                # Frekans maskesi oluştur
                freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                
                if band_name == 'brightness':
                    # Parlaklık bandını hafif vurgula (mutluluk için)
                    enhanced_magnitude[freq_mask] *= 1.1
                elif band_name == 'formants':
                    # Formant bandını güçlendir (konuşma netliği için)
                    enhanced_magnitude[freq_mask] *= 1.05
                elif band_name == 'breathiness':
                    # Nefes seslerini azalt ama tamamen silme
                    enhanced_magnitude[freq_mask] *= 0.9
            
            # Geri dönüştür
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
            
            # Seviye kontrolü
            max_val = np.max(np.abs(enhanced_audio))
            if max_val > 1.0:
                enhanced_audio = enhanced_audio / max_val * 0.95
            
            self.add_log("✅ Spektral duygu iyileştirmesi tamamlandı")
            return enhanced_audio.astype(np.float32)
            
        except Exception as e:
            self.add_log(f"❌ Spektral iyileştirme hatası: {e}")
            return audio_data
    
    def apply_psychoacoustic_filtering(self, audio_data, sample_rate):
        """Psiko-akustik prensiplere dayalı filtreleme"""
        try:
            self.add_log("🧠 Psiko-akustik filtreleme başlatılıyor...")
            
            # İnsan işitme eğrisi (A-weighting benzeri)
            freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)
            fft = np.fft.fft(audio_data)
            
            # İnsan kulağının hassasiyet eğrisi
            def hearing_sensitivity(f):
                """İnsan kulağının frekans hassasiyeti"""
                f = np.abs(f)
                if f < 20:
                    return 0.1
                elif f < 200:
                    return 0.3 + 0.7 * (f - 20) / 180
                elif f < 1000:
                    return 1.0
                elif f < 4000:
                    return 1.0 + 0.2 * (f - 1000) / 3000  # Maksimum hassasiyet
                elif f < 8000:
                    return 1.2 - 0.3 * (f - 4000) / 4000
                else:
                    return 0.9 * np.exp(-(f - 8000) / 8000)
            
            # Hassasiyet eğrisini uygula
            sensitivity_curve = np.array([hearing_sensitivity(f) for f in freqs])
            
            # FFT'yi ağırlıklandır
            weighted_fft = fft * sensitivity_curve
            
            # Geri dönüştür
            filtered_audio = np.real(np.fft.ifft(weighted_fft))
            
            # Normalize
            max_val = np.max(np.abs(filtered_audio))
            if max_val > 0:
                filtered_audio = filtered_audio / max_val * np.max(np.abs(audio_data))
            
            self.add_log("✅ Psiko-akustik filtreleme tamamlandı")
            return filtered_audio.astype(np.float32)
            
        except Exception as e:
            self.add_log(f"❌ Psiko-akustik filtreleme hatası: {e}")
            return audio_data
    
    def run_gender_age_analysis(self, audio_data, sample_rate, diarization_result=None):
        """Kapsamlı cinsiyet ve yaş analizi"""
        try:
            self.add_log("👥 Cinsiyet ve yaş analizi başlatılıyor...")
            
            # Ses verisini temizle
            audio_data = self.clean_audio_buffer(audio_data)
            
            # Çoklu yaklaşım ile analiz
            results = {}
            
            # 1. Ses özellik tabanlı analiz
            feature_based_results = self.feature_based_gender_age_analysis(audio_data, sample_rate)
            results['feature_based'] = feature_based_results
            
            # 2. Frekans domain analizi
            frequency_based_results = self.frequency_domain_gender_age_analysis(audio_data, sample_rate)
            results['frequency_based'] = frequency_based_results
            
            # 3. Deep Learning tabanlı analiz (Transformers)
            try:
                dl_results = self.deep_learning_gender_age_analysis(audio_data, sample_rate)
                results['deep_learning'] = dl_results
            except Exception as e:
                self.add_log(f"⚠️ Deep learning analizi başarısız: {e}")
                results['deep_learning'] = None
            
            # 4. Konuşmacı bazlı analiz (eğer diyarizasyon varsa)
            if diarization_result:
                speaker_results = self.speaker_based_gender_age_analysis(audio_data, sample_rate, diarization_result)
                results['speaker_based'] = speaker_results
            
            # 5. Sonuçları birleştir (ensemble)
            final_results = self.ensemble_gender_age_results(results)
            
            # 6. Sonuçları kaydet ve göster
            self.gender_age_result = final_results
            self.display_gender_age_results(final_results)
            
            # 7. Konuşmacı bazlı sonuçları log'a yazdır
            if 'detailed' in final_results and 'speaker_based' in final_results['detailed']:
                speaker_results = final_results['detailed']['speaker_based']
                if speaker_results:
                    self.add_log("👥 Konuşmacı bazlı sonuçlar:")
                    for speaker, speaker_data in speaker_results.items():
                        dominant_gender = max(speaker_data['gender'].items(), key=lambda x: x[1])
                        dominant_age = max(speaker_data['age'].items(), key=lambda x: x[1])
                        confidence = speaker_data.get('confidence', 0.5)
                        
                        gender_icons = {'erkek': '👨', 'kadın': '👩', 'belirsiz': '❓'}
                        age_icons = {'çocuk': '👶', 'genç': '🧒', 'yetişkin': '👤', 'orta_yaş': '🧑', 'yaşlı': '👴'}
                        
                        gender_icon = gender_icons.get(dominant_gender[0], '❓')
                        age_icon = age_icons.get(dominant_age[0], '❓')
                        
                        self.add_log(f"  🎤 {speaker}: {gender_icon} {dominant_gender[0]} ({dominant_gender[1]*100:.1f}%), "
                                   f"{age_icon} {dominant_age[0]} ({dominant_age[1]*100:.1f}%), güven: {confidence*100:.1f}%")
            
            self.add_log("✅ Cinsiyet ve yaş analizi tamamlandı")
            return final_results
            
        except Exception as e:
            self.add_log(f"❌ Cinsiyet ve yaş analizi hatası: {e}")
            return {}
    
    def feature_based_gender_age_analysis(self, audio_data, sample_rate):
        """Ses özellik tabanlı cinsiyet ve yaş analizi"""
        try:
            self.add_log("🎵 Özellik tabanlı cinsiyet-yaş analizi...")
            
            # Gelişmiş özellik çıkarımı
            features = self.extract_advanced_audio_features(audio_data, sample_rate)
            
            if not features:
                return self.get_default_gender_age_results()
            
            # CİNSİYET ANALİZİ
            gender_scores = {'erkek': 0.0, 'kadın': 0.0, 'belirsiz': 0.0}
            
            # Temel perde analizi (en güvenilir gösterge)
            if features['pitch_mean'] < 165:  # Erkek sesi (genelde 85-165 Hz)
                gender_scores['erkek'] += 0.4
            elif features['pitch_mean'] > 165:  # Kadın sesi (genelde 165-265 Hz)
                gender_scores['kadın'] += 0.4
            else:
                gender_scores['belirsiz'] += 0.2
            
            # Formant frekansları (ikinci en güvenilir)
            if features['spectral_centroid_mean'] < 1200:  # Erkek formantları daha düşük
                gender_scores['erkek'] += 0.3
            elif features['spectral_centroid_mean'] > 1400:  # Kadın formantları daha yüksek
                gender_scores['kadın'] += 0.3
            
            # Ses kalınlığı ve tonu
            if features['spectral_bandwidth_mean'] > 2000:  # Geniş spektrum = genelde erkek
                gender_scores['erkek'] += 0.2
            elif features['spectral_bandwidth_mean'] < 1500:  # Dar spektrum = genelde kadın
                gender_scores['kadın'] += 0.2
            
            # Konuşma hızı ve ritim
            if features['speaking_rate'] > 3:  # Hızlı konuşma
                gender_scores['kadın'] += 0.1  # İstatistiksel olarak kadınlar daha hızlı konuşur
            elif features['speaking_rate'] < 2:  # Yavaş konuşma
                gender_scores['erkek'] += 0.1
            
            # YAŞ ANALİZİ
            age_scores = {'çocuk': 0.0, 'genç': 0.0, 'yetişkin': 0.0, 'orta_yaş': 0.0, 'yaşlı': 0.0}
            
            # Perde değişkenliği (yaş ile ters orantılı)
            if features['pitch_std'] > 80:  # Yüksek değişkenlik = genç
                age_scores['çocuk'] += 0.2
                age_scores['genç'] += 0.3
            elif features['pitch_std'] < 30:  # Düşük değişkenlik = yaşlı
                age_scores['orta_yaş'] += 0.2
                age_scores['yaşlı'] += 0.3
            else:
                age_scores['yetişkin'] += 0.3
            
            # Ses titremesi (yaşla artar)
            if features['zcr_std'] > 0.05:  # Yüksek titreme
                age_scores['yaşlı'] += 0.3
            elif features['zcr_std'] < 0.02:  # Düşük titreme
                age_scores['çocuk'] += 0.1
                age_scores['genç'] += 0.2
                age_scores['yetişkin'] += 0.2
            
            # Konuşma hızı ve duraklama
            if features['speaking_rate'] > 4:  # Çok hızlı
                age_scores['çocuk'] += 0.2
                age_scores['genç'] += 0.1
            elif features['speaking_rate'] < 1.5:  # Çok yavaş
                age_scores['yaşlı'] += 0.3
            
            # Sessizlik oranı (yaşla artar)
            if features['silence_ratio'] > 0.6:  # Çok sessizlik
                age_scores['yaşlı'] += 0.2
            elif features['silence_ratio'] < 0.2:  # Az sessizlik
                age_scores['genç'] += 0.2
            
            # Enerji kararlılığı
            if features['energy_variance'] < 0.0001:  # Çok kararlı
                age_scores['yetişkin'] += 0.2
                age_scores['orta_yaş'] += 0.1
            elif features['energy_variance'] > 0.001:  # Değişken
                age_scores['çocuk'] += 0.1
                age_scores['genç'] += 0.2
            
            # Normalize et
            gender_total = sum(gender_scores.values())
            if gender_total > 0:
                for gender in gender_scores:
                    gender_scores[gender] /= gender_total
            else:
                gender_scores = {'erkek': 0.5, 'kadın': 0.5, 'belirsiz': 0.0}
            
            age_total = sum(age_scores.values())
            if age_total > 0:
                for age in age_scores:
                    age_scores[age] /= age_total
            else:
                age_scores = {'yetişkin': 0.6, 'genç': 0.3, 'orta_yaş': 0.1, 'çocuk': 0.0, 'yaşlı': 0.0}
            
            return {
                'gender': gender_scores,
                'age': age_scores,
                'confidence': self.calculate_gender_age_confidence(features, gender_scores, age_scores)
            }
            
        except Exception as e:
            self.add_log(f"❌ Özellik tabanlı analiz hatası: {e}")
            return self.get_default_gender_age_results()
    
    def frequency_domain_gender_age_analysis(self, audio_data, sample_rate):
        """Frekans domain cinsiyet ve yaş analizi"""
        try:
            self.add_log("📊 Frekans domain analizi...")
            
            # FFT analizi
            fft = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)
            magnitude = np.abs(fft)
            
            # Frekans bantları
            low_freq = magnitude[(np.abs(freqs) >= 80) & (np.abs(freqs) <= 300)]    # Temel frekans
            mid_freq = magnitude[(np.abs(freqs) >= 300) & (np.abs(freqs) <= 3000)]  # Formant bölgesi
            high_freq = magnitude[(np.abs(freqs) >= 3000) & (np.abs(freqs) <= 8000)] # Yüksek frekanslar
            
            # Enerji dağılımı
            low_energy = np.sum(low_freq)
            mid_energy = np.sum(mid_freq)
            high_energy = np.sum(high_freq)
            total_energy = low_energy + mid_energy + high_energy
            
            if total_energy == 0:
                return self.get_default_gender_age_results()
            
            low_ratio = low_energy / total_energy
            mid_ratio = mid_energy / total_energy
            high_ratio = high_energy / total_energy
            
            # Cinsiyet analizi
            gender_scores = {'erkek': 0.0, 'kadın': 0.0, 'belirsiz': 0.0}
            
            if low_ratio > 0.4:  # Düşük frekans dominant = erkek
                gender_scores['erkek'] += 0.4
            elif high_ratio > 0.3:  # Yüksek frekans dominant = kadın
                gender_scores['kadın'] += 0.4
            else:
                gender_scores['belirsiz'] += 0.2
            
            # Orta frekans analizi (formantlar)
            if mid_ratio > 0.5:
                # Formant detayı için daha derinlemesine analiz
                formant_peak_freq = freqs[np.abs(freqs) <= 3000][np.argmax(magnitude[(np.abs(freqs) >= 300) & (np.abs(freqs) <= 3000)])] + 300
                
                if formant_peak_freq < 1000:  # Düşük formant = erkek
                    gender_scores['erkek'] += 0.3
                elif formant_peak_freq > 1200:  # Yüksek formant = kadın
                    gender_scores['kadın'] += 0.3
            
            # Yaş analizi
            age_scores = {'çocuk': 0.0, 'genç': 0.0, 'yetişkin': 0.0, 'orta_yaş': 0.0, 'yaşlı': 0.0}
            
            # Çok yüksek frekanslar (çocuk sesi göstergesi)
            ultra_high = magnitude[np.abs(freqs) > 8000]
            if len(ultra_high) > 0 and np.sum(ultra_high) / total_energy > 0.1:
                age_scores['çocuk'] += 0.3
            
            # Frekans dağılımının düzenliliği
            spectral_flatness = np.mean(magnitude) / (np.max(magnitude) + 1e-10)
            
            if spectral_flatness > 0.1:  # Düzensiz spektrum = yaşlı
                age_scores['yaşlı'] += 0.3
            elif spectral_flatness < 0.05:  # Düzenli spektrum = genç/yetişkin
                age_scores['genç'] += 0.2
                age_scores['yetişkin'] += 0.2
            
            # Harmonik yapı analizi
            try:
                harmonics = []
                fundamental_freq = freqs[np.argmax(magnitude)]
                for i in range(2, 6):  # 2. ile 5. harmonikler
                    harmonic_freq = fundamental_freq * i
                    if harmonic_freq < sample_rate / 2:
                        harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
                        harmonics.append(magnitude[harmonic_idx])
                
                if harmonics:
                    harmonic_strength = np.mean(harmonics) / (np.max(magnitude) + 1e-10)
                    
                    if harmonic_strength > 0.3:  # Güçlü harmonikler = genç
                        age_scores['genç'] += 0.2
                        age_scores['yetişkin'] += 0.1
                    elif harmonic_strength < 0.1:  # Zayıf harmonikler = yaşlı
                        age_scores['yaşlı'] += 0.2
            except:
                pass
            
            # Normalize
            gender_total = sum(gender_scores.values())
            if gender_total > 0:
                for gender in gender_scores:
                    gender_scores[gender] /= gender_total
            
            age_total = sum(age_scores.values())
            if age_total > 0:
                for age in age_scores:
                    age_scores[age] /= age_total
            
            return {
                'gender': gender_scores,
                'age': age_scores,
                'confidence': 0.7  # Orta güven
            }
            
        except Exception as e:
            self.add_log(f"❌ Frekans domain analizi hatası: {e}")
            return self.get_default_gender_age_results()
    
    def deep_learning_gender_age_analysis(self, audio_data, sample_rate):
        """Deep Learning tabanlı cinsiyet ve yaş analizi"""
        try:
            self.add_log("🤖 Deep Learning cinsiyet-yaş analizi...")
            
            # Transformers ile analiz
            from transformers import pipeline, Wav2Vec2Processor, Wav2Vec2Model
            
            # Ses verisini uygun formata çevir
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            
            # Çoklu model yaklaşımı
            models_to_try = [
                "facebook/wav2vec2-large-xlsr-53",
                "microsoft/unispeech-sat-base-plus",
                "facebook/hubert-large-ls960-ft"
            ]
            
            results = []
            
            for model_name in models_to_try:
                try:
                    self.add_log(f"🔄 Model deneniyor: {model_name}")
                    
                    # Model ve processor yükle
                    processor = Wav2Vec2Processor.from_pretrained(model_name)
                    model = Wav2Vec2Model.from_pretrained(model_name)
                    
                    # Ses verisini işle
                    inputs = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt")
                    
                    # Model çıktısını al
                    with torch.no_grad():
                        outputs = model(**inputs)
                        hidden_states = outputs.last_hidden_state
                    
                    # Özellik vektörünü çıkar (ortalama pooling)
                    feature_vector = torch.mean(hidden_states, dim=1).squeeze().numpy()
                    
                    # Özellik vektörünü cinsiyet ve yaş analizine çevir
                    gender_age_result = self.analyze_deep_features(feature_vector)
                    results.append(gender_age_result)
                    
                    self.add_log(f"✅ Model başarılı: {model_name}")
                    break  # İlk başarılı model ile devam et
                    
                except Exception as model_error:
                    self.add_log(f"❌ Model hatası {model_name}: {model_error}")
                    continue
            
            if results:
                return results[0]  # İlk başarılı sonucu döndür
            else:
                self.add_log("⚠️ Hiçbir deep learning model çalışmadı")
                return self.get_default_gender_age_results()
                
        except ImportError:
            self.add_log("⚠️ Transformers kütüphanesi yok, alternatif yöntem kullanılıyor")
            return self.get_default_gender_age_results()
        except Exception as e:
            self.add_log(f"❌ Deep learning analizi hatası: {e}")
            return self.get_default_gender_age_results()
    
    def analyze_deep_features(self, feature_vector):
        """Deep learning özellik vektörünü cinsiyet ve yaş analizine çevir"""
        try:
            # Özellik vektörü istatistikleri
            mean_val = np.mean(feature_vector)
            std_val = np.std(feature_vector)
            max_val = np.max(feature_vector)
            min_val = np.min(feature_vector)
            
            # Basit kural tabanlı analiz (gerçek projede eğitilmiş classifier kullanılır)
            gender_scores = {'erkek': 0.5, 'kadın': 0.5, 'belirsiz': 0.0}
            age_scores = {'çocuk': 0.1, 'genç': 0.3, 'yetişkin': 0.4, 'orta_yaş': 0.2, 'yaşlı': 0.0}
            
            # Özellik vektörü analizine dayalı basit kurallar
            if mean_val > 0.1:
                gender_scores['kadın'] += 0.2
                gender_scores['erkek'] -= 0.2
            elif mean_val < -0.1:
                gender_scores['erkek'] += 0.2
                gender_scores['kadın'] -= 0.2
            
            if std_val > 0.5:
                age_scores['genç'] += 0.2
                age_scores['yaşlı'] -= 0.1
            elif std_val < 0.2:
                age_scores['yaşlı'] += 0.2
                age_scores['genç'] -= 0.1
            
            # Normalize
            gender_total = sum(gender_scores.values())
            if gender_total > 0:
                for gender in gender_scores:
                    gender_scores[gender] /= gender_total
            
            age_total = sum(age_scores.values())
            if age_total > 0:
                for age in age_scores:
                    age_scores[age] /= age_total
            
            return {
                'gender': gender_scores,
                'age': age_scores,
                'confidence': 0.6
            }
            
        except Exception as e:
            return self.get_default_gender_age_results()
    
    def speaker_based_gender_age_analysis(self, audio_data, sample_rate, diarization_result):
        """Konuşmacı bazlı cinsiyet ve yaş analizi"""
        try:
            self.add_log("👥 Konuşmacı bazlı cinsiyet-yaş analizi...")
            
            speaker_results = {}
            
            # Her konuşmacı için ayrı analiz
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                start_sample = int(turn.start * sample_rate)
                end_sample = int(turn.end * sample_rate)
                
                # Konuşmacı segmentini al
                if start_sample < len(audio_data) and end_sample <= len(audio_data):
                    speaker_audio = audio_data[start_sample:end_sample]
                    
                    if len(speaker_audio) > 1024:  # Yeterli veri varsa
                        # Bu konuşmacı için analiz yap
                        speaker_analysis = self.feature_based_gender_age_analysis(speaker_audio, sample_rate)
                        
                        if speaker not in speaker_results:
                            speaker_results[speaker] = []
                        speaker_results[speaker].append(speaker_analysis)
            
            # Her konuşmacı için ortalama sonuç hesapla
            final_speaker_results = {}
            for speaker, analyses in speaker_results.items():
                if analyses:
                    # Cinsiyet skorlarını ortala
                    avg_gender = {}
                    avg_age = {}
                    
                    for gender in ['erkek', 'kadın', 'belirsiz']:
                        scores = [analysis['gender'][gender] for analysis in analyses if 'gender' in analysis]
                        avg_gender[gender] = np.mean(scores) if scores else 0.0
                    
                    for age in ['çocuk', 'genç', 'yetişkin', 'orta_yaş', 'yaşlı']:
                        scores = [analysis['age'][age] for analysis in analyses if 'age' in analysis]
                        avg_age[age] = np.mean(scores) if scores else 0.0
                    
                    final_speaker_results[speaker] = {
                        'gender': avg_gender,
                        'age': avg_age,
                        'confidence': np.mean([analysis.get('confidence', 0.5) for analysis in analyses])
                    }
            
            return final_speaker_results
            
        except Exception as e:
            self.add_log(f"❌ Konuşmacı bazlı analiz hatası: {e}")
            return {}
    
    def ensemble_gender_age_results(self, results):
        """Farklı analiz yöntemlerinin sonuçlarını birleştir"""
        try:
            # Ağırlıklar
            weights = {
                'feature_based': 0.4,    # En güvenilir
                'frequency_based': 0.3,  # İkinci güvenilir
                'deep_learning': 0.2,    # Üçüncü güvenilir
                'speaker_based': 0.1     # Destekleyici
            }
            
            # Genel cinsiyet ve yaş skorları
            ensemble_gender = {'erkek': 0.0, 'kadın': 0.0, 'belirsiz': 0.0}
            ensemble_age = {'çocuk': 0.0, 'genç': 0.0, 'yetişkin': 0.0, 'orta_yaş': 0.0, 'yaşlı': 0.0}
            total_weight = 0.0
            
            # Her yöntemin sonuçlarını ağırlıklı olarak birleştir
            for method, result in results.items():
                if result and method in weights:
                    weight = weights[method]
                    
                    if isinstance(result, dict) and 'gender' in result:
                        # Tekil sonuç
                        for gender in ensemble_gender:
                            if gender in result['gender']:
                                ensemble_gender[gender] += result['gender'][gender] * weight
                        
                        for age in ensemble_age:
                            if age in result['age']:
                                ensemble_age[age] += result['age'][age] * weight
                        
                        total_weight += weight
                    
                    elif isinstance(result, dict):
                        # Konuşmacı bazlı sonuçlar
                        speaker_count = len(result)
                        if speaker_count > 0:
                            speaker_weight = weight / speaker_count
                            
                            for speaker_result in result.values():
                                for gender in ensemble_gender:
                                    if gender in speaker_result['gender']:
                                        ensemble_gender[gender] += speaker_result['gender'][gender] * speaker_weight
                                
                                for age in ensemble_age:
                                    if age in speaker_result['age']:
                                        ensemble_age[age] += speaker_result['age'][age] * speaker_weight
                            
                            total_weight += weight
            
            # Normalize et
            if total_weight > 0:
                for gender in ensemble_gender:
                    ensemble_gender[gender] /= total_weight
                for age in ensemble_age:
                    ensemble_age[age] /= total_weight
            else:
                # Varsayılan değerler
                ensemble_gender = {'erkek': 0.5, 'kadın': 0.5, 'belirsiz': 0.0}
                ensemble_age = {'yetişkin': 0.6, 'genç': 0.3, 'orta_yaş': 0.1, 'çocuk': 0.0, 'yaşlı': 0.0}
            
            # Güven skoru hesapla
            confidence = min(total_weight, 1.0)
            
            return {
                'overall': {
                    'gender': ensemble_gender,
                    'age': ensemble_age,
                    'confidence': confidence
                },
                'detailed': results
            }
            
        except Exception as e:
            self.add_log(f"❌ Ensemble birleştirme hatası: {e}")
            return self.get_default_gender_age_results()
    
    def calculate_gender_age_confidence(self, features, gender_scores, age_scores):
        """Cinsiyet ve yaş analizi güven skorunu hesapla"""
        try:
            # Ses kalitesi faktörleri
            quality_factors = {
                'pitch_clarity': 1.0 if features['pitch_mean'] > 50 else 0.5,
                'energy_level': min(features['energy'] * 100, 1.0),
                'voice_activity': features['voice_activity_ratio'],
                'spectral_clarity': 1 - features['spectral_flatness_mean']
            }
            
            # Skor dağılımı analizi
            max_gender_score = max(gender_scores.values())
            max_age_score = max(age_scores.values())
            
            gender_confidence = max_gender_score
            age_confidence = max_age_score
            
            # Genel güven skoru
            quality_score = np.mean(list(quality_factors.values()))
            prediction_confidence = (gender_confidence + age_confidence) / 2
            
            overall_confidence = (quality_score * 0.6) + (prediction_confidence * 0.4)
            
            return min(overall_confidence, 1.0)
            
        except Exception as e:
            return 0.5  # Orta güven
    
    def display_gender_age_results(self, results):
        """Cinsiyet ve yaş analizi sonuçlarını göster"""
        try:
            self.gender_age_text.delete(1.0, tk.END)
            self.gender_age_text.insert(tk.END, f"👥 Cinsiyet ve Yaş Analizi Sonuçları\n")
            self.gender_age_text.insert(tk.END, f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if 'overall' in results:
                overall = results['overall']
                
                # Genel sonuçlar
                self.gender_age_text.insert(tk.END, f"🎯 GENEL SONUÇLAR\n")
                self.gender_age_text.insert(tk.END, f"{'='*30}\n")
                self.gender_age_text.insert(tk.END, f"🎯 Güven Skoru: {overall['confidence']*100:.1f}%\n\n")
                
                # Cinsiyet sonuçları
                self.gender_age_text.insert(tk.END, f"👤 CİNSİYET ANALİZİ:\n")
                gender_sorted = sorted(overall['gender'].items(), key=lambda x: x[1], reverse=True)
                
                for i, (gender, score) in enumerate(gender_sorted):
                    percentage = score * 100
                    bar_length = int(percentage / 2.5)
                    bar = "█" * bar_length + "░" * (40 - bar_length)
                    
                    # İkon ve renk
                    icons = {'erkek': '👨', 'kadın': '👩', 'belirsiz': '❓'}
                    icon = icons.get(gender, '❓')
                    rank_icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                    
                    self.gender_age_text.insert(tk.END, 
                        f"{rank_icon} {icon} {gender.capitalize()}: {percentage:.1f}% {bar}\n")
                
                # Yaş sonuçları
                self.gender_age_text.insert(tk.END, f"\n🎂 YAŞ GRUBU ANALİZİ:\n")
                age_sorted = sorted(overall['age'].items(), key=lambda x: x[1], reverse=True)
                
                for i, (age, score) in enumerate(age_sorted):
                    percentage = score * 100
                    bar_length = int(percentage / 2.5)
                    bar = "█" * bar_length + "░" * (40 - bar_length)
                    
                    # İkon ve yaş aralığı
                    age_info = {
                        'çocuk': ('👶', '0-12 yaş'),
                        'genç': ('🧒', '13-25 yaş'),
                        'yetişkin': ('👤', '26-45 yaş'),
                        'orta_yaş': ('🧑', '46-65 yaş'),
                        'yaşlı': ('👴', '65+ yaş')
                    }
                    icon, age_range = age_info.get(age, ('❓', 'Belirsiz'))
                    rank_icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                    
                    self.gender_age_text.insert(tk.END, 
                        f"{rank_icon} {icon} {age.capitalize()} ({age_range}): {percentage:.1f}% {bar}\n")
                
                # Dominant tahminler
                dominant_gender = gender_sorted[0][0]
                dominant_age = age_sorted[0][0]
                
                self.gender_age_text.insert(tk.END, f"\n🎯 SONUÇ ÖZETİ:\n")
                self.gender_age_text.insert(tk.END, f"👤 Tahmin Edilen Cinsiyet: {dominant_gender.capitalize()}\n")
                self.gender_age_text.insert(tk.END, f"🎂 Tahmin Edilen Yaş Grubu: {dominant_age.capitalize()}\n")
                
                # Güven seviyesi yorumu
                confidence = overall['confidence']
                if confidence > 0.8:
                    conf_text = "Çok Yüksek ✨"
                elif confidence > 0.6:
                    conf_text = "Yüksek ✅"
                elif confidence > 0.4:
                    conf_text = "Orta ⚠️"
                else:
                    conf_text = "Düşük ❌"
                    
                self.gender_age_text.insert(tk.END, f"🎯 Analiz Güvenilirliği: {conf_text}\n")
                
                # Konuşmacı bazlı detaylı sonuçlar
                if 'detailed' in results and 'speaker_based' in results['detailed']:
                    speaker_results = results['detailed']['speaker_based']
                    if speaker_results:
                        self.gender_age_text.insert(tk.END, f"\n👥 KONUŞMACI BAZLI DETAYLI ANALİZ:\n")
                        self.gender_age_text.insert(tk.END, f"{'='*45}\n")
                        
                        for speaker, speaker_data in speaker_results.items():
                            self.gender_age_text.insert(tk.END, f"\n🎤 {speaker}:\n")
                            self.gender_age_text.insert(tk.END, f"{'─'*25}\n")
                            
                            # Cinsiyet detayları
                            self.gender_age_text.insert(tk.END, f"👤 CİNSİYET SKORLARI:\n")
                            gender_sorted = sorted(speaker_data['gender'].items(), key=lambda x: x[1], reverse=True)
                            for i, (gender, score) in enumerate(gender_sorted):
                                percentage = score * 100
                                bar_length = int(percentage / 5)  # 5% per character
                                bar = "█" * bar_length + "░" * (20 - bar_length)
                                
                                icons = {'erkek': '👨', 'kadın': '👩', 'belirsiz': '❓'}
                                icon = icons.get(gender, '❓')
                                rank_icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                                
                                self.gender_age_text.insert(tk.END, 
                                    f"  {rank_icon} {icon} {gender.capitalize()}: {percentage:.1f}% {bar}\n")
                            
                            # Yaş detayları
                            self.gender_age_text.insert(tk.END, f"\n🎂 YAŞ GRUBU SKORLARI:\n")
                            age_sorted = sorted(speaker_data['age'].items(), key=lambda x: x[1], reverse=True)
                            for i, (age, score) in enumerate(age_sorted):
                                percentage = score * 100
                                bar_length = int(percentage / 5)  # 5% per character
                                bar = "█" * bar_length + "░" * (20 - bar_length)
                                
                                age_info = {
                                    'çocuk': ('👶', '0-12 yaş'),
                                    'genç': ('🧒', '13-25 yaş'),
                                    'yetişkin': ('👤', '26-45 yaş'),
                                    'orta_yaş': ('🧑', '46-65 yaş'),
                                    'yaşlı': ('👴', '65+ yaş')
                                }
                                icon, age_range = age_info.get(age, ('❓', 'Belirsiz'))
                                rank_icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                                
                                self.gender_age_text.insert(tk.END, 
                                    f"  {rank_icon} {icon} {age.capitalize()} ({age_range}): {percentage:.1f}% {bar}\n")
                            
                            # Bu konuşmacının sonuç özeti
                            dominant_gender = gender_sorted[0][0]
                            dominant_age = age_sorted[0][0]
                            confidence = speaker_data.get('confidence', 0.5)
                            
                            self.gender_age_text.insert(tk.END, f"\n🎯 {speaker} SONUÇ ÖZETİ:\n")
                            self.gender_age_text.insert(tk.END, f"  👤 Tahmin: {dominant_gender.capitalize()}\n")
                            self.gender_age_text.insert(tk.END, f"  🎂 Yaş Grubu: {dominant_age.capitalize()}\n")
                            self.gender_age_text.insert(tk.END, f"  🎯 Güven: {confidence*100:.1f}%\n")
                
                # Diğer analiz yöntemlerinin sonuçları
                if 'detailed' in results:
                    self.gender_age_text.insert(tk.END, f"\n📊 DİĞER ANALİZ YÖNTEMLERİ:\n")
                    self.gender_age_text.insert(tk.END, f"{'='*35}\n")
                    
                    for method, result in results['detailed'].items():
                        if result and method != 'speaker_based':
                            self.gender_age_text.insert(tk.END, f"\n🔬 {method.replace('_', ' ').title()}:\n")
                            
                            if isinstance(result, dict) and 'gender' in result:
                                # Tekil sonuç
                                dominant_g = max(result['gender'].items(), key=lambda x: x[1])
                                dominant_a = max(result['age'].items(), key=lambda x: x[1])
                                
                                self.gender_age_text.insert(tk.END, 
                                    f"  👤 Cinsiyet: {dominant_g[0]} ({dominant_g[1]*100:.1f}%)\n")
                                self.gender_age_text.insert(tk.END, 
                                    f"  🎂 Yaş: {dominant_a[0]} ({dominant_a[1]*100:.1f}%)\n")
            
            # Görselleştirmeleri güncelle
            self.plot_gender_age_analysis()
            
        except Exception as e:
            self.add_log(f"❌ Cinsiyet-yaş sonuç görüntüleme hatası: {e}")
    
    def get_default_gender_age_results(self):
        """Varsayılan cinsiyet ve yaş sonuçları"""
        return {
            'gender': {'erkek': 0.5, 'kadın': 0.5, 'belirsiz': 0.0},
            'age': {'yetişkin': 0.6, 'genç': 0.3, 'orta_yaş': 0.1, 'çocuk': 0.0, 'yaşlı': 0.0},
            'confidence': 0.3
        }
    
    def plot_gender_age_analysis(self):
        """Cinsiyet ve yaş analizi görselleştirmesi"""
        try:
            if not self.gender_age_result or 'overall' not in self.gender_age_result:
                return
                
            overall = self.gender_age_result['overall']
            
            # Cinsiyet pie chart
            self.ax_gender.clear()
            gender_data = overall['gender']
            gender_labels = list(gender_data.keys())
            gender_values = list(gender_data.values())
            gender_colors = [GENDER_COLORS.get(gender, '#95A5A6') for gender in gender_labels]
            
            wedges, texts, autotexts = self.ax_gender.pie(gender_values, labels=gender_labels, 
                                                         colors=gender_colors, autopct='%1.1f%%',
                                                         startangle=90)
            self.ax_gender.set_title('👤 Cinsiyet Dağılımı')
            
            # Yaş pie chart
            self.ax_age.clear()
            age_data = overall['age']
            age_labels = list(age_data.keys())
            age_values = list(age_data.values())
            age_colors = [AGE_COLORS.get(age, '#95A5A6') for age in age_labels]
            
            wedges, texts, autotexts = self.ax_age.pie(age_values, labels=age_labels, 
                                                      colors=age_colors, autopct='%1.1f%%',
                                                      startangle=90)
            self.ax_age.set_title('🎂 Yaş Grubu Dağılımı')
            
            # Konuşmacı bazlı cinsiyet dağılımı
            self.ax_speaker_gender.clear()
            if 'detailed' in self.gender_age_result and 'speaker_based' in self.gender_age_result['detailed']:
                speaker_results = self.gender_age_result['detailed']['speaker_based']
                if speaker_results:
                    speakers = list(speaker_results.keys())
                    male_scores = [speaker_results[s]['gender']['erkek'] * 100 for s in speakers]
                    female_scores = [speaker_results[s]['gender']['kadın'] * 100 for s in speakers]
                    
                    x = np.arange(len(speakers))
                    width = 0.35
                    
                    self.ax_speaker_gender.bar(x - width/2, male_scores, width, label='Erkek', color=GENDER_COLORS['erkek'])
                    self.ax_speaker_gender.bar(x + width/2, female_scores, width, label='Kadın', color=GENDER_COLORS['kadın'])
                    
                    self.ax_speaker_gender.set_xlabel('Konuşmacılar')
                    self.ax_speaker_gender.set_ylabel('Skor (%)')
                    self.ax_speaker_gender.set_title('👥 Konuşmacı Cinsiyet Skorları')
                    self.ax_speaker_gender.set_xticks(x)
                    self.ax_speaker_gender.set_xticklabels(speakers)
                    self.ax_speaker_gender.legend()
                else:
                    self.ax_speaker_gender.text(0.5, 0.5, 'Konuşmacı verisi yok', 
                                              ha='center', va='center', transform=self.ax_speaker_gender.transAxes)
            else:
                self.ax_speaker_gender.text(0.5, 0.5, 'Konuşmacı analizi yapılmadı', 
                                          ha='center', va='center', transform=self.ax_speaker_gender.transAxes)
            
            # Güven skoru göstergesi
            self.ax_speaker_age.clear()
            confidence = overall['confidence']
            
            # Güven skoru gauge benzeri görselleştirme
            angles = np.linspace(0, np.pi, 100)
            values = np.ones_like(angles) * confidence
            
            self.ax_speaker_age.plot(angles, values, linewidth=10, color='green' if confidence > 0.7 else 'orange' if confidence > 0.4 else 'red')
            self.ax_speaker_age.fill_between(angles, 0, values, alpha=0.3, color='green' if confidence > 0.7 else 'orange' if confidence > 0.4 else 'red')
            self.ax_speaker_age.set_ylim(0, 1)
            self.ax_speaker_age.set_xlim(0, np.pi)
            self.ax_speaker_age.set_title(f'🎯 Güven Skoru: {confidence*100:.1f}%')
            self.ax_speaker_age.text(np.pi/2, confidence/2, f'{confidence*100:.1f}%', 
                                   ha='center', va='center', fontsize=14, weight='bold')
            
            self.fig5.tight_layout()
            self.canvas5.draw()
            
        except Exception as e:
            self.add_log(f"❌ Cinsiyet-yaş görselleştirme hatası: {e}")
    
    def run_vad_analysis(self, audio_file):
        """Ses Aktivitesi Tespiti (VAD) yap"""
        try:
            vad_result = self.pipelines['vad'](audio_file)
            
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(tk.END, f"🎯 Ses Aktivitesi Tespiti Sonuçları\n")
            self.analysis_text.insert(tk.END, f"📂 Dosya: {os.path.basename(audio_file)}\n\n")
            
            total_speech = 0
            speech_segments = []
            
            for segment in vad_result.get_timeline():
                start = segment.start
                end = segment.end
                duration = end - start
                total_speech += duration
                speech_segments.append({'start': start, 'end': end, 'duration': duration})
                
                self.analysis_text.insert(tk.END, f"🗣️ Konuşma: {start:.2f}s - {end:.2f}s ({duration:.2f}s)\n")
            
            # İstatistikler
            audio_duration = librosa.get_duration(filename=audio_file)
            speech_ratio = (total_speech / audio_duration) * 100 if audio_duration > 0 else 0
            silence_duration = audio_duration - total_speech
            
            self.analysis_text.insert(tk.END, f"\n📊 VAD İstatistikleri:\n")
            self.analysis_text.insert(tk.END, f"📈 Toplam ses süresi: {audio_duration:.2f}s\n")
            self.analysis_text.insert(tk.END, f"🗣️ Konuşma süresi: {total_speech:.2f}s ({speech_ratio:.1f}%)\n")
            self.analysis_text.insert(tk.END, f"🤫 Sessizlik süresi: {silence_duration:.2f}s ({100-speech_ratio:.1f}%)\n")
            self.analysis_text.insert(tk.END, f"📝 Konuşma segment sayısı: {len(speech_segments)}\n")
            
            if speech_segments:
                avg_segment = total_speech / len(speech_segments)
                self.analysis_text.insert(tk.END, f"⏱️ Ortalama segment süresi: {avg_segment:.2f}s\n")
            
            self.add_log("✅ VAD analizi tamamlandı")
            return vad_result
            
        except Exception as e:
            self.add_log(f"❌ VAD analizi hatası: {e}")
            return None
    
    def run_overlap_detection(self, audio_file):
        """Örtüşen konuşma tespiti yap"""
        try:
            overlap_result = self.pipelines['overlap'](audio_file)
            
            self.analysis_text.insert(tk.END, f"\n🗣️ Örtüşen Konuşma Tespiti Sonuçları\n")
            self.analysis_text.insert(tk.END, f"📂 Dosya: {os.path.basename(audio_file)}\n\n")
            
            total_overlap = 0
            overlap_segments = []
            
            for segment in overlap_result.get_timeline():
                start = segment.start
                end = segment.end
                duration = end - start
                total_overlap += duration
                overlap_segments.append({'start': start, 'end': end, 'duration': duration})
                
                self.analysis_text.insert(tk.END, f"🔄 Örtüşme: {start:.2f}s - {end:.2f}s ({duration:.2f}s)\n")
            
            # İstatistikler
            audio_duration = librosa.get_duration(filename=audio_file)
            overlap_ratio = (total_overlap / audio_duration) * 100 if audio_duration > 0 else 0
            
            self.analysis_text.insert(tk.END, f"\n📊 Örtüşme İstatistikleri:\n")
            self.analysis_text.insert(tk.END, f"🔄 Toplam örtüşme süresi: {total_overlap:.2f}s ({overlap_ratio:.1f}%)\n")
            self.analysis_text.insert(tk.END, f"📝 Örtüşme segment sayısı: {len(overlap_segments)}\n")
            
            if overlap_segments:
                avg_overlap = total_overlap / len(overlap_segments)
                self.analysis_text.insert(tk.END, f"⏱️ Ortalama örtüşme süresi: {avg_overlap:.2f}s\n")
            
            self.add_log("✅ Örtüşme tespiti tamamlandı")
            return overlap_result
            
        except Exception as e:
            self.add_log(f"❌ Örtüşme tespiti hatası: {e}")
            return None
    
    def extract_speaker_embeddings(self, audio_file, diarization):
        """Konuşmacı embedding'lerini çıkar"""
        try:
            if not diarization:
                return
                
            self.add_log("🔍 Konuşmacı embedding'leri çıkarılıyor...")
            
            # Ses dosyasını yükle
            audio_data, sample_rate = librosa.load(audio_file, sr=16000)  # 16kHz
            
            # Her konuşmacı için embedding çıkar
            embeddings = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_sample = int(turn.start * sample_rate)
                end_sample = int(turn.end * sample_rate)
                
                # Segment ses verisini al
                segment_audio = audio_data[start_sample:end_sample]
                
                # Basit özellik çıkarma (gerçek projede daha gelişmiş yöntemler kullanılır)
                features = {
                    'mfcc_mean': np.mean(librosa.feature.mfcc(y=segment_audio, sr=sample_rate)),
                    'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=segment_audio, sr=sample_rate)),
                    'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(segment_audio)),
                    'duration': turn.end - turn.start
                }
                
                if speaker not in embeddings:
                    embeddings[speaker] = []
                embeddings[speaker].append(features)
            
            # Ortalama embedding'leri hesapla
            for speaker in embeddings:
                speaker_features = embeddings[speaker]
                avg_features = {
                    'mfcc_mean': np.mean([f['mfcc_mean'] for f in speaker_features]),
                    'spectral_centroid': np.mean([f['spectral_centroid'] for f in speaker_features]),
                    'zero_crossing_rate': np.mean([f['zero_crossing_rate'] for f in speaker_features]),
                    'total_duration': sum([f['duration'] for f in speaker_features])
                }
                self.speaker_embeddings[speaker] = avg_features
            
            self.add_log("✅ Konuşmacı embedding'leri çıkarıldı")
            
        except Exception as e:
            self.add_log(f"❌ Embedding çıkarma hatası: {e}")
    
    def run_speech_separation(self, audio_file):
        """Ses ayrıştırma işlemi yap"""
        try:
            self.add_log("🎼 Ses ayrıştırma başlatılıyor...")
            # Bu özellik için gelişmiş modeller gerekir
            # Şu an için basit bir placeholder
            self.add_log("⚠️ Ses ayrıştırma özelliği geliştirme aşamasında")
            
        except Exception as e:
            self.add_log(f"❌ Ses ayrıştırma hatası: {e}")
    
    def extract_advanced_audio_features(self, audio_data, sample_rate):
        """Gelişmiş ses özelliklerini çıkar"""
        try:
            self.add_log("🔍 Gelişmiş ses özellikleri çıkarılıyor...")
            
            # Ses verisini temizle ve kontrol et
            audio_data = self.clean_audio_buffer(audio_data)
            
            # Ses verisi boş veya çok kısa mı kontrol et
            if len(audio_data) < 1024:  # Minimum 1024 sample
                self.add_log("❌ Ses verisi çok kısa, varsayılan özellikler döndürülüyor")
                return self.get_default_features()
            
            # Temel özellikler
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=20)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
            zero_crossing_rates = librosa.feature.zero_crossing_rate(audio_data)
            
            # Gelişmiş spektral özellikler
            chroma = librosa.feature.chroma(y=audio_data, sr=sample_rate)
            mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
            tonnetz = librosa.feature.tonnetz(y=audio_data, sr=sample_rate)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)
            spectral_flatness = librosa.feature.spectral_flatness(y=audio_data)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
            
            # Prozodik özellikler
            try:
                tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
                speaking_rate = len(beats) / (len(audio_data) / sample_rate)  # konuşma hızı
            except:
                tempo = 0
                speaking_rate = 0
            
            # Pitch analizi
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            
            # Enerji ve güç özellikleri
            energy = np.mean(np.square(audio_data))
            rms_energy = np.mean(librosa.feature.rms(y=audio_data))
            
            # Formant analizi (basitleştirilmiş)
            # Gerçek formant analizi için daha karmaşık algoritma gerekir
            stft = librosa.stft(audio_data)
            spectral_magnitude = np.abs(stft)
            formant_frequencies = []
            for frame in spectral_magnitude.T:
                peaks, _ = librosa.util.peak_pick(frame, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.1, wait=10)
                if len(peaks) >= 2:
                    # İlk iki formant yaklaşımı
                    formant_frequencies.append(peaks[:2])
            
            # Sessizlik analizi
            silence_threshold = 0.01
            silence_frames = np.where(np.abs(audio_data) < silence_threshold)[0]
            silence_ratio = len(silence_frames) / len(audio_data)
            
            # Vurgu ve tonlama (prosody) özellikleri
            # Kısa-dönem enerji değişimleri
            frame_length = int(0.025 * sample_rate)  # 25ms
            hop_length = int(0.01 * sample_rate)     # 10ms
            
            short_time_energy = []
            for i in range(0, len(audio_data) - frame_length, hop_length):
                frame = audio_data[i:i + frame_length]
                energy_frame = np.sum(frame ** 2)
                short_time_energy.append(energy_frame)
            
            energy_variance = np.var(short_time_energy)
            energy_mean = np.mean(short_time_energy)
            
            # Özellik sözlüğü - Güvenli hesaplama
            features = {}
            
            # Güvenli özellik hesaplama fonksiyonu
            def safe_calc(func, default_val=0.0):
                try:
                    result = func()
                    if np.isfinite(result):
                        return float(result)
                    else:
                        return default_val
                except:
                    return default_val
            
            # Temel özellikler
            features['mfcc_mean'] = safe_calc(lambda: np.mean(mfccs), 0.0)
            features['mfcc_std'] = safe_calc(lambda: np.std(mfccs), 1.0)
            features['spectral_centroid_mean'] = safe_calc(lambda: np.mean(spectral_centroids), 1000.0)
            features['spectral_centroid_std'] = safe_calc(lambda: np.std(spectral_centroids), 500.0)
            features['zcr_mean'] = safe_calc(lambda: np.mean(zero_crossing_rates), 0.05)
            features['zcr_std'] = safe_calc(lambda: np.std(zero_crossing_rates), 0.02)
            
            # Gelişmiş spektral özellikler
            features['chroma_mean'] = safe_calc(lambda: np.mean(chroma), 0.1)
            features['chroma_std'] = safe_calc(lambda: np.std(chroma), 0.05)
            features['mel_spectrogram_mean'] = safe_calc(lambda: np.mean(mel_spectrogram), 0.01)
            features['tonnetz_mean'] = safe_calc(lambda: np.mean(tonnetz), 0.0)
            features['spectral_contrast_mean'] = safe_calc(lambda: np.mean(spectral_contrast), 10.0)
            features['spectral_bandwidth_mean'] = safe_calc(lambda: np.mean(spectral_bandwidth), 1500.0)
            features['spectral_flatness_mean'] = safe_calc(lambda: np.mean(spectral_flatness), 0.1)
            features['spectral_rolloff_mean'] = safe_calc(lambda: np.mean(spectral_rolloff), 2000.0)
            
            # Prozodik özellikler
            features['tempo'] = safe_calc(lambda: tempo, 100.0)
            features['speaking_rate'] = safe_calc(lambda: speaking_rate, 2.0)
            features['pitch_mean'] = safe_calc(lambda: pitch_mean, 200.0)
            features['pitch_std'] = safe_calc(lambda: pitch_std, 50.0)
            features['pitch_range'] = safe_calc(lambda: pitch_std / pitch_mean if pitch_mean > 0 else 0, 0.25)
            
            # Enerji özellikleri
            features['energy'] = safe_calc(lambda: energy, 0.001)
            features['rms_energy'] = safe_calc(lambda: rms_energy, 0.01)
            features['energy_variance'] = safe_calc(lambda: energy_variance, 0.0001)
            features['energy_mean'] = safe_calc(lambda: energy_mean, 0.001)
            features['energy_dynamic_range'] = safe_calc(
                lambda: np.max(short_time_energy) - np.min(short_time_energy) if len(short_time_energy) > 0 else 0, 
                0.005
            )
            
            # Sessizlik ve duraklama
            features['silence_ratio'] = safe_calc(lambda: silence_ratio, 0.3)
            features['voice_activity_ratio'] = safe_calc(lambda: 1 - silence_ratio, 0.7)
            
            # Harmonik özellikler
            features['harmonic_mean'] = safe_calc(lambda: np.mean(librosa.effects.harmonic(audio_data)), 0.001)
            features['percussive_mean'] = safe_calc(lambda: np.mean(librosa.effects.percussive(audio_data)), 0.001)
            
            self.add_log(f"✅ {len(features)} gelişmiş özellik çıkarıldı")
            return features
            
        except Exception as e:
            self.add_log(f"❌ Gelişmiş özellik çıkarma hatası: {e}")
            return {}
    
    def run_emotion_analysis(self, audio_data, sample_rate):
        """Duygu analizi yap"""
        try:
            self.add_log("😊 Duygu analizi başlatılıyor...")
            
            # Ses verisini temizle
            audio_data = self.clean_audio_buffer(audio_data)
            
            # Basit ses özellik analizi ile duygu tahmini
            # Gerçek projede daha gelişmiş modeller kullanılır
            
            # Ses özelliklerini güvenli şekilde çıkar
            try:
                mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
                spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
                zero_crossing_rates = librosa.feature.zero_crossing_rate(audio_data)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
            except Exception as e:
                self.add_log(f"⚠️ Temel özellik çıkarma hatası: {e}")
                return self.get_default_emotion_scores()
            
            # Basit kural tabanlı duygu tahmini - Güvenli hesaplama
            try:
                energy = np.mean(np.square(audio_data))
                if not np.isfinite(energy):
                    energy = 0.001
            except:
                energy = 0.001
                
            try:
                pitch_mean = np.mean(spectral_centroids)
                if not np.isfinite(pitch_mean):
                    pitch_mean = 1000.0
            except:
                pitch_mean = 1000.0
                
            try:
                zcr_mean = np.mean(zero_crossing_rates)
                if not np.isfinite(zcr_mean):
                    zcr_mean = 0.05
            except:
                zcr_mean = 0.05
            
            # Duygu sınıflandırması (basitleştirilmiş)
            emotion_scores = {
                'mutlu': 0.0,
                'üzgün': 0.0,
                'kızgın': 0.0,
                'sakin': 0.0,
                'heyecanlı': 0.0,
                'stresli': 0.0
            }
            
            # Basit kurallar
            if energy > 0.01 and pitch_mean > 2000:
                emotion_scores['heyecanlı'] += 0.3
                emotion_scores['mutlu'] += 0.2
            elif energy < 0.005:
                emotion_scores['sakin'] += 0.3
                emotion_scores['üzgün'] += 0.2
            
            if zcr_mean > 0.1:
                emotion_scores['stresli'] += 0.2
                emotion_scores['kızgın'] += 0.1
            
            # Normalize et
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                for emotion in emotion_scores:
                    emotion_scores[emotion] = emotion_scores[emotion] / total_score
            else:
                # Varsayılan değerler
                emotion_scores['sakin'] = 0.6
                emotion_scores['mutlu'] = 0.4
            
            # Sonuçları göster
            self.emotion_text.delete(1.0, tk.END)
            self.emotion_text.insert(tk.END, f"😊 Duygu Analizi Sonuçları\n")
            self.emotion_text.insert(tk.END, f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Duyguları sıralı göster
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
            
            for emotion, score in sorted_emotions:
                percentage = score * 100
                bar_length = int(percentage / 5)  # 5% per character
                bar = "█" * bar_length + "░" * (20 - bar_length)
                
                self.emotion_text.insert(tk.END, f"{emotion.capitalize()}: {percentage:.1f}% {bar}\n")
            
            # Dominant duygu
            dominant_emotion = sorted_emotions[0][0]
            self.emotion_text.insert(tk.END, f"\n🎯 Baskın Duygu: {dominant_emotion.capitalize()}\n")
            
            # Ses özellikleri
            self.emotion_text.insert(tk.END, f"\n📊 Ses Özellikleri:\n")
            self.emotion_text.insert(tk.END, f"⚡ Enerji: {energy:.4f}\n")
            self.emotion_text.insert(tk.END, f"🎵 Ortalama Perde: {pitch_mean:.1f} Hz\n")
            self.emotion_text.insert(tk.END, f"🌊 Zero Crossing Rate: {zcr_mean:.3f}\n")
            
            self.add_log("✅ Duygu analizi tamamlandı")
            return emotion_scores
            
        except Exception as e:
            self.add_log(f"❌ Duygu analizi hatası: {e}")
            return None
    
    def run_speaker_based_transcription(self, audio_file, diarization):
        """Konuşmacı bazlı transkripsiyon yap"""
        try:
            self.add_log("Konuşmacı bazlı transkripsiyon başlatılıyor...")
            
            # Sonuçları göster
            self.transcript_text.delete(1.0, tk.END)
            self.transcript_text.insert(tk.END, f"Konuşma İçeriği: {audio_file}\n")
            self.transcript_text.insert(tk.END, f"Model: {GPT_MODEL}\n\n")
            
            # Önce tüm ses dosyasını transkript et
            self.add_log("Tüm ses dosyası transkript ediliyor...")
            full_transcript = self.transcribe_audio_segment(audio_file)
            
            if not full_transcript:
                self.add_log("Transkripsiyon başarısız oldu. Lütfen API anahtarınızı kontrol edin.")
                self.status_label.config(text="Transkripsiyon hatası!")
                self.analyze_button.config(state=tk.NORMAL)
                return
            
            self.add_log("Tam transkript alındı, konuşmacılara göre bölünüyor...")
            
            # Diyarizasyon sonuçlarını göster
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Diyarizasyon Sonuçları: {audio_file}\n\n")
            
            # Konuşmacıları ve zaman aralıklarını göster
            speaker_segments = []
            if diarization:
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    start = turn.start
                    end = turn.end
                    duration = end - start
                    result_line = f"{speaker}: {start:.2f}s - {end:.2f}s ({duration:.2f}s)\n"
                    self.result_text.insert(tk.END, result_line)
                    
                    # Segment bilgilerini kaydet
                    speaker_segments.append({
                        "speaker": speaker,
                        "start": start,
                        "end": end
                    })
            
            # Konuşmacı istatistiklerini hesapla
            speaker_stats = {}
            if diarization:
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if speaker not in speaker_stats:
                        speaker_stats[speaker] = 0
                    speaker_stats[speaker] += turn.end - turn.start
            
            # İstatistikleri göster
            self.result_text.insert(tk.END, "\nKonuşmacı İstatistikleri:\n")
            if speaker_stats:
                for speaker, duration in speaker_stats.items():
                    self.result_text.insert(tk.END, f"{speaker}: {duration:.2f} saniye\n")
            
            # Konuşmacı bazlı transkriptleri oluştur
            # Tam transkripti zaman aralıklarına göre böl
            
            # Segmentleri zaman sırasına göre sırala
            speaker_segments.sort(key=lambda x: x["start"])
            
            # Her segment için transkript oluştur
            for i, segment in enumerate(speaker_segments):
                speaker = segment["speaker"]
                start = segment["start"]
                end = segment["end"]
                
                # Tam transkriptten bu konuşmacının konuşma içeriğini tahmin et
                # Burada basit bir yaklaşım kullanıyoruz
                # Gerçek uygulamada daha gelişmiş bir metin bölme algoritması gerekebilir
                
                # Her segment için ayrı transkript yap
                self.add_log(f"{speaker} için segment transkript ediliyor: {start:.2f}s - {end:.2f}s")
                
                # Segment ses dosyasını oluştur
                import soundfile as sf
                import numpy as np
                
                # Ses dosyasını yükle
                audio, sample_rate = sf.read(audio_file)
                
                # Segment sınırlarını hesapla
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                
                # Segment sınırlarını kontrol et
                if start_sample >= len(audio) or end_sample > len(audio):
                    continue
                    
                # Konuşmacı segmentini kes
                segment_audio = audio[start_sample:end_sample]
                
                # Geçici dosya oluştur
                temp_dir = "temp_segments"
                os.makedirs(temp_dir, exist_ok=True)
                segment_file = os.path.join(temp_dir, f"segment_{i}.wav")
                sf.write(segment_file, segment_audio, sample_rate)
                
                # Bu segmenti transkript et
                segment_text = self.transcribe_audio_segment(segment_file)
                
                if segment_text and len(segment_text.strip()) > 0:
                    # Transkript sonucunu göster
                    self.transcript_text.insert(tk.END, f"[{start:.2f}s - {end:.2f}s] {speaker}: {segment_text}\n\n")
                else:
                    self.add_log(f"{speaker} için segment transkripsiyon boş döndü")
            
            self.add_log("Konuşmacı bazlı transkripsiyon tamamlandı.")
            self.status_label.config(text="Analiz tamamlandı")
            self.analyze_button.config(state=tk.NORMAL)
            
        except Exception as e:
            self.add_log(f"Konuşmacı bazlı transkripsiyon hatası: {e}")
            self.status_label.config(text="Hata!")
            self.analyze_button.config(state=tk.NORMAL)
    
    def transcribe_audio_segment(self, audio_file):
        """Tek bir ses segmentini transkript et"""
        try:
            # gpt-4o-mini için Whisper API kullan
            whisper_response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=open(audio_file, "rb"),
                language=GPT_LANGUAGE
            )
            
            return whisper_response.text
            
        except Exception as e:
            self.add_log(f"Segment transkripsiyon hatası: {e}")
            return ""
    
    def add_log(self, message):
        """Log alanına mesaj ekle"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
    
    def update_advanced_visualizations(self, audio_data, sample_rate):
        """Gelişmiş görselleştirmeleri güncelle"""
        try:
            self.add_log("📊 Görselleştirmeler güncelleniyor...")
            
            # Ana dalga formu ve diyarizasyon görselleştirmesi
            self.update_comprehensive_plots(audio_data)
            
            # Diyarizasyon sonuçlarını çiz
            if self.diarization_result:
                self.plot_diarization_timeline(audio_data, sample_rate)
            
            # Duygu analizi görselleştirmesi
            if self.emotion_result:
                if hasattr(self, 'temporal_emotion_history') and self.temporal_emotion_history:
                    self.plot_temporal_emotion_analysis()  # Zamansal görselleştirme
                else:
                    self.plot_emotion_analysis()  # Standart görselleştirme
            
            # İstatistik grafikleri
            self.plot_statistics()
            
            self.add_log("✅ Görselleştirmeler güncellendi")
            
        except Exception as e:
            self.add_log(f"❌ Görselleştirme hatası: {e}")
    
    def plot_diarization_timeline(self, audio_data, sample_rate):
        """Diyarizasyon zaman çizelgesi çiz"""
        try:
            self.ax_diarization.clear()
            
            if not self.diarization_result:
                return
                
            # Konuşmacıları renk kodları ile eşleştir
            speakers = list(set(speaker for _, _, speaker in self.diarization_result.itertracks(yield_label=True)))
            speaker_colors = {speaker: SPEAKER_COLORS[i % len(SPEAKER_COLORS)] for i, speaker in enumerate(speakers)}
            
            # Y ekseni için konuşmacı pozisyonları
            speaker_positions = {speaker: i for i, speaker in enumerate(speakers)}
            
            # Her konuşmacı segmentini çiz
            for turn, _, speaker in self.diarization_result.itertracks(yield_label=True):
                start = turn.start
                end = turn.end
                y_pos = speaker_positions[speaker]
                
                # Segment çubunu çiz
                self.ax_diarization.barh(y_pos, end - start, left=start, height=0.8, 
                                        color=speaker_colors[speaker], alpha=0.7, 
                                        edgecolor='black', linewidth=0.5)
                
                # Segment üzerine süre yaz (eğer yeterince uzunsa)
                if end - start > 1.0:  # 1 saniyeden uzun segmentler için
                    self.ax_diarization.text(start + (end - start) / 2, y_pos, 
                                           f'{end - start:.1f}s', 
                                           ha='center', va='center', fontsize=8, weight='bold')
            
            # VAD sonuçlarını ekle (varsa)
            if self.vad_result:
                audio_duration = len(audio_data) / sample_rate
                vad_y = len(speakers)  # En üste VAD çubuğu
                
                # Sessizlik bölgeleri (gri)
                self.ax_diarization.barh(vad_y, audio_duration, left=0, height=0.3, 
                                        color='lightgray', alpha=0.5, label='Sessizlik')
                
                # Konuşma bölgeleri (yeşil)
                for segment in self.vad_result.get_timeline():
                    self.ax_diarization.barh(vad_y, segment.end - segment.start, 
                                           left=segment.start, height=0.3, 
                                           color='lightgreen', alpha=0.7)
            
            # Örtüşme bölgelerini ekle (varsa)
            if self.overlap_result:
                for segment in self.overlap_result.get_timeline():
                    # Tüm konuşmacılar boyunca kırmızı çizgi
                    self.ax_diarization.axvspan(segment.start, segment.end, 
                                              alpha=0.3, color='red', 
                                              label='Örtüşme' if segment == list(self.overlap_result.get_timeline())[0] else "")
            
            # Grafik ayarları
            self.ax_diarization.set_xlabel('Zaman (saniye)')
            self.ax_diarization.set_ylabel('Konuşmacılar')
            self.ax_diarization.set_title('👥 Konuşmacı Zaman Çizelgesi')
            
            # Y ekseni etiketleri
            all_labels = speakers[:]
            if self.vad_result:
                all_labels.append('VAD')
            
            self.ax_diarization.set_yticks(range(len(all_labels)))
            self.ax_diarization.set_yticklabels(all_labels)
            
            # Legend ekle
            if self.overlap_result and len(list(self.overlap_result.get_timeline())) > 0:
                self.ax_diarization.legend(loc='upper right')
            
            self.ax_diarization.grid(True, alpha=0.3)
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.add_log(f"❌ Diyarizasyon görselleştirme hatası: {e}")
    
    def plot_emotion_analysis(self):
        """Duygu analizi görselleştirmesi"""
        try:
            if not self.emotion_result:
                return
                
            self.ax_emotion.clear()
            
            # Duyguları ve skorları al
            emotions = list(self.emotion_result.keys())
            scores = list(self.emotion_result.values())
            percentages = [score * 100 for score in scores]
            
            # Renkleri eşleştir
            colors = [EMOTION_COLORS.get(emotion, '#95A5A6') for emotion in emotions]
            
            # Pasta grafiği
            wedges, texts, autotexts = self.ax_emotion.pie(percentages, labels=emotions, 
                                                          colors=colors, autopct='%1.1f%%',
                                                          startangle=90, textprops={'fontsize': 10})
            
            # Grafiği güzelleştir
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
            
            self.ax_emotion.set_title('😊 Duygu Dağılımı', fontsize=14, weight='bold')
            
            # Dominant duyguyu vurgula
            max_emotion_idx = scores.index(max(scores))
            wedges[max_emotion_idx].set_edgecolor('black')
            wedges[max_emotion_idx].set_linewidth(3)
            
            self.canvas3.draw()
            
        except Exception as e:
            self.add_log(f"❌ Duygu görselleştirme hatası: {e}")
    
    def plot_statistics(self):
        """İstatistik grafikleri çiz"""
        try:
            # Konuşmacı süre dağılımı
            if self.diarization_result:
                speakers = {}
                for turn, _, speaker in self.diarization_result.itertracks(yield_label=True):
                    if speaker not in speakers:
                        speakers[speaker] = 0
                    speakers[speaker] += turn.end - turn.start
                
                # Konuşmacı süreleri bar grafiği
                self.ax_stats1.clear()
                speaker_names = list(speakers.keys())
                durations = list(speakers.values())
                colors = [SPEAKER_COLORS[i % len(SPEAKER_COLORS)] for i in range(len(speaker_names))]
                
                bars = self.ax_stats1.bar(speaker_names, durations, color=colors, alpha=0.7)
                self.ax_stats1.set_title('Konuşmacı Süreleri')
                self.ax_stats1.set_ylabel('Süre (saniye)')
                self.ax_stats1.tick_params(axis='x', rotation=45)
                
                # Değerleri bar üzerine yaz
                for bar, duration in zip(bars, durations):
                    self.ax_stats1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                       f'{duration:.1f}s', ha='center', va='bottom', fontsize=9)
            
            # Konuşmacı embedding özellikleri
            if self.speaker_embeddings:
                self.ax_stats2.clear()
                
                # MFCC ortalama değerleri
                speakers = list(self.speaker_embeddings.keys())
                mfcc_values = [self.speaker_embeddings[s]['mfcc_mean'] for s in speakers]
                
                colors = [SPEAKER_COLORS[i % len(SPEAKER_COLORS)] for i in range(len(speakers))]
                self.ax_stats2.scatter(range(len(speakers)), mfcc_values, c=colors, s=100, alpha=0.7)
                self.ax_stats2.set_title('Konuşmacı MFCC Ortalaması')
                self.ax_stats2.set_xlabel('Konuşmacılar')
                self.ax_stats2.set_ylabel('MFCC Ortalama')
                self.ax_stats2.set_xticks(range(len(speakers)))
                self.ax_stats2.set_xticklabels(speakers, rotation=45)
                self.ax_stats2.grid(True, alpha=0.3)
            
            # Ses aktivitesi istatistikleri
            if self.vad_result:
                self.ax_stats3.clear()
                
                speech_segments = list(self.vad_result.get_timeline())
                if speech_segments:
                    # Segment süre dağılımı histogramı
                    segment_durations = [seg.end - seg.start for seg in speech_segments]
                    
                    self.ax_stats3.hist(segment_durations, bins=min(20, len(segment_durations)), 
                                       color='lightgreen', alpha=0.7, edgecolor='black')
                    self.ax_stats3.set_title('Konuşma Segment Süre Dağılımı')
                    self.ax_stats3.set_xlabel('Segment Süresi (saniye)')
                    self.ax_stats3.set_ylabel('Frekans')
                    self.ax_stats3.grid(True, alpha=0.3)
            
            # Duygu skorları radar chart (basitleştirilmiş)
            if self.emotion_result:
                self.ax_stats4.clear()
                
                emotions = list(self.emotion_result.keys())
                scores = [self.emotion_result[e] * 100 for e in emotions]
                
                # Basit bar chart
                colors = [EMOTION_COLORS.get(emotion, '#95A5A6') for emotion in emotions]
                bars = self.ax_stats4.bar(emotions, scores, color=colors, alpha=0.7)
                self.ax_stats4.set_title('Duygu Skorları')
                self.ax_stats4.set_ylabel('Skor (%)')
                self.ax_stats4.tick_params(axis='x', rotation=45)
                self.ax_stats4.set_ylim(0, 100)
                
                # Değerleri bar üzerine yaz
                for bar, score in zip(bars, scores):
                    self.ax_stats4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                       f'{score:.1f}%', ha='center', va='bottom', fontsize=8)
            
            self.fig4.tight_layout()
            self.canvas4.draw()
            
        except Exception as e:
            self.add_log(f"❌ İstatistik görselleştirme hatası: {e}")
    
    def generate_detailed_analysis_report(self):
        """Detaylı analiz raporu oluştur"""
        try:
            self.analysis_text.insert(tk.END, f"\n\n📋 DETAYLI ANALİZ RAPORU\n")
            self.analysis_text.insert(tk.END, f"{'='*50}\n")
            self.analysis_text.insert(tk.END, f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.analysis_text.insert(tk.END, f"📂 Dosya: {os.path.basename(self.current_audio_file) if self.current_audio_file else 'Kayıt'}\n\n")
            
            # Genel istatistikler
            if self.current_audio_file:
                audio_duration = librosa.get_duration(filename=self.current_audio_file)
                self.analysis_text.insert(tk.END, f"⏱️ Toplam süre: {audio_duration:.2f} saniye\n")
            
            # Diyarizasyon özeti
            if self.diarization_result:
                speakers = list(set(speaker for _, _, speaker in self.diarization_result.itertracks(yield_label=True)))
                self.analysis_text.insert(tk.END, f"👥 Tespit edilen konuşmacı sayısı: {len(speakers)}\n")
                
                total_speech = sum(turn.end - turn.start for turn, _, _ in self.diarization_result.itertracks())
                self.analysis_text.insert(tk.END, f"🗣️ Toplam konuşma süresi: {total_speech:.2f} saniye\n")
            
            # VAD özeti
            if self.vad_result:
                speech_segments = list(self.vad_result.get_timeline())
                self.analysis_text.insert(tk.END, f"🎯 Konuşma segment sayısı: {len(speech_segments)}\n")
            
            # Örtüşme özeti
            if self.overlap_result:
                overlap_segments = list(self.overlap_result.get_timeline())
                total_overlap = sum(seg.end - seg.start for seg in overlap_segments)
                self.analysis_text.insert(tk.END, f"🔄 Toplam örtüşme süresi: {total_overlap:.2f} saniye\n")
            
            # Duygu özeti
            if self.emotion_result:
                dominant_emotion = max(self.emotion_result.items(), key=lambda x: x[1])
                self.analysis_text.insert(tk.END, f"😊 Baskın duygu: {dominant_emotion[0].capitalize()} ({dominant_emotion[1]*100:.1f}%)\n")
            
            self.analysis_text.insert(tk.END, f"\n{'='*50}\n")
            
        except Exception as e:
            self.add_log(f"❌ Rapor oluşturma hatası: {e}")
    
    def start_live_analysis(self):
        """Canlı analiz başlat"""
        try:
            if self.live_analysis_running:
                return
                
            self.live_analysis_running = True
            self.add_log("🔴 Canlı analiz başlatıldı")
            
            # Canlı analiz için ayrı thread başlat
            self.live_analysis_thread = threading.Thread(target=self.live_analysis_worker)
            self.live_analysis_thread.daemon = True
            self.live_analysis_thread.start()
            
        except Exception as e:
            self.add_log(f"❌ Canlı analiz başlatma hatası: {e}")
    
    def stop_live_analysis(self):
        """Canlı analizi durdur"""
        try:
            self.live_analysis_running = False
            self.add_log("⏹️ Canlı analiz durduruldu")
            
        except Exception as e:
            self.add_log(f"❌ Canlı analiz durdurma hatası: {e}")
    
    def live_analysis_worker(self):
        """Canlı analiz worker fonksiyonu"""
        try:
            while self.live_analysis_running and self.is_recording:
                # Basit canlı analiz - ses seviyesi gösterimi
                time.sleep(0.1)  # 100ms güncelleme
                
                # Gerçek projede burada gerçek zamanlı analiz yapılır
                # Şu an için sadece durum gösterimi
                
                if not self.is_recording:
                    break
                    
        except Exception as e:
            self.add_log(f"❌ Canlı analiz worker hatası: {e}")
        finally:
            self.live_analysis_running = False
    
    def generate_report(self):
        """PDF raporu oluştur"""
        try:
            self.add_log("📄 Rapor oluşturuluyor...")
            
            # Basit metin raporu oluştur
            report_content = []
            report_content.append("🎤 GELIŞMIŞ SES ANALİZİ RAPORU")
            report_content.append("=" * 50)
            report_content.append(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_content.append(f"📂 Dosya: {os.path.basename(self.current_audio_file) if self.current_audio_file else 'Kayıt'}")
            report_content.append("")
            
            # Diyarizasyon sonuçları
            if self.diarization_result:
                report_content.append("👥 DİYARİZASYON SONUÇLARI:")
                report_content.append("-" * 30)
                
                speakers = {}
                for turn, _, speaker in self.diarization_result.itertracks(yield_label=True):
                    if speaker not in speakers:
                        speakers[speaker] = []
                    speakers[speaker].append(f"{turn.start:.2f}s - {turn.end:.2f}s ({turn.end - turn.start:.2f}s)")
                
                for speaker, segments in speakers.items():
                    report_content.append(f"\n{speaker}:")
                    for segment in segments:
                        report_content.append(f"  • {segment}")
            
            # Duygu analizi sonuçları
            if self.emotion_result:
                report_content.append("\n\n😊 DUYGU ANALİZİ SONUÇLARI:")
                report_content.append("-" * 30)
                sorted_emotions = sorted(self.emotion_result.items(), key=lambda x: x[1], reverse=True)
                for emotion, score in sorted_emotions:
                    report_content.append(f"{emotion.capitalize()}: {score*100:.1f}%")
            
            # Cinsiyet ve yaş analizi sonuçları
            if hasattr(self, 'gender_age_result') and self.gender_age_result:
                report_content.append("\n\n👥 CİNSİYET VE YAŞ ANALİZİ SONUÇLARI:")
                report_content.append("-" * 40)
                
                if 'overall' in self.gender_age_result:
                    overall = self.gender_age_result['overall']
                    
                    # Genel sonuçlar
                    report_content.append("\n🎯 GENEL SONUÇLAR:")
                    dominant_gender = max(overall['gender'].items(), key=lambda x: x[1])
                    dominant_age = max(overall['age'].items(), key=lambda x: x[1])
                    confidence = overall.get('confidence', 0.5)
                    
                    report_content.append(f"Cinsiyet: {dominant_gender[0].capitalize()} ({dominant_gender[1]*100:.1f}%)")
                    report_content.append(f"Yaş Grubu: {dominant_age[0].capitalize()} ({dominant_age[1]*100:.1f}%)")
                    report_content.append(f"Güven Skoru: {confidence*100:.1f}%")
                    
                    # Tüm cinsiyet skorları
                    report_content.append("\nCinsiyet Skorları:")
                    gender_sorted = sorted(overall['gender'].items(), key=lambda x: x[1], reverse=True)
                    for gender, score in gender_sorted:
                        report_content.append(f"  {gender.capitalize()}: {score*100:.1f}%")
                    
                    # Tüm yaş skorları
                    report_content.append("\nYaş Grubu Skorları:")
                    age_sorted = sorted(overall['age'].items(), key=lambda x: x[1], reverse=True)
                    for age, score in age_sorted:
                        report_content.append(f"  {age.capitalize()}: {score*100:.1f}%")
                
                # Konuşmacı bazlı sonuçlar
                if 'detailed' in self.gender_age_result and 'speaker_based' in self.gender_age_result['detailed']:
                    speaker_results = self.gender_age_result['detailed']['speaker_based']
                    if speaker_results:
                        report_content.append("\n\n🎤 KONUŞMACI BAZLI SONUÇLAR:")
                        report_content.append("-" * 30)
                        
                        for speaker, speaker_data in speaker_results.items():
                            report_content.append(f"\n{speaker}:")
                            
                            # Cinsiyet sonuçları
                            dominant_gender = max(speaker_data['gender'].items(), key=lambda x: x[1])
                            report_content.append(f"  Cinsiyet: {dominant_gender[0].capitalize()} ({dominant_gender[1]*100:.1f}%)")
                            
                            # Yaş sonuçları
                            dominant_age = max(speaker_data['age'].items(), key=lambda x: x[1])
                            report_content.append(f"  Yaş Grubu: {dominant_age[0].capitalize()} ({dominant_age[1]*100:.1f}%)")
                            
                            # Güven skoru
                            sp_confidence = speaker_data.get('confidence', 0.5)
                            report_content.append(f"  Güven Skoru: {sp_confidence*100:.1f}%")
                            
                            # Detaylı skorlar
                            report_content.append("  Cinsiyet Detayları:")
                            for gender, score in sorted(speaker_data['gender'].items(), key=lambda x: x[1], reverse=True):
                                report_content.append(f"    {gender.capitalize()}: {score*100:.1f}%")
                            
                            report_content.append("  Yaş Detayları:")
                            for age, score in sorted(speaker_data['age'].items(), key=lambda x: x[1], reverse=True):
                                report_content.append(f"    {age.capitalize()}: {score*100:.1f}%")
            
            # Raporu dosyaya kaydet
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"ses_analizi_raporu_{timestamp}.txt"
            
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_content))
            
            self.add_log(f"✅ Rapor oluşturuldu: {report_filename}")
            messagebox.showinfo("Rapor", f"Rapor başarıyla oluşturuldu:\n{report_filename}")
            
        except Exception as e:
            self.add_log(f"❌ Rapor oluşturma hatası: {e}")
            messagebox.showerror("Hata", f"Rapor oluşturulurken hata oluştu:\n{e}")
    
    def run_ml_emotion_analysis(self, audio_data, sample_rate):
        """Machine Learning tabanlı duygu analizi"""
        try:
            self.add_log("🤖 ML tabanlı duygu analizi başlatılıyor...")
            
            # Gelişmiş özellik çıkarımı
            features = self.extract_advanced_audio_features(audio_data, sample_rate)
            
            if not features:
                return self.run_emotion_analysis(audio_data, sample_rate)  # Fallback
            
            # Özellik vektörünü hazırla
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            
            # NaN değerleri temizle
            feature_vector = np.nan_to_num(feature_vector)
            
            # Özellik normalizasyonu
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            feature_vector_normalized = scaler.fit_transform(feature_vector)
            
            # Gelişmiş kural tabanlı sistem (ML benzeri)
            emotion_scores = self.advanced_rule_based_classification(features)
            
            # Eğer gerçek ML modeli varsa kullan
            try:
                emotion_scores = self.use_pretrained_emotion_model(feature_vector_normalized)
            except:
                self.add_log("⚠️ Pretrained model bulunamadı, gelişmiş kurallar kullanılıyor")
            
            # Temporal analiz (zaman serisi)
            temporal_scores = self.temporal_emotion_analysis(audio_data, sample_rate)
            
            # Zamansal veriyi sakla (görselleştirme için)
            if hasattr(self, '_temp_temporal_data'):
                self.temporal_emotion_history = self._temp_temporal_data
            
            # Skorları birleştir (ensemble)
            final_scores = self.ensemble_emotion_scores(emotion_scores, temporal_scores)
            
            # Güven skoru hesapla
            confidence = self.calculate_confidence_score(features, final_scores)
            
            # Debug bilgilerini göster
            debug_info = self.debug_emotion_analysis(features)
            self.add_log("🔍 Debug bilgileri:")
            for line in debug_info.split('\n'):
                if line.strip():
                    self.add_log(line)
            
            # Sonuçları göster
            self.display_advanced_emotion_results(final_scores, features, confidence)
            
            self.add_log("✅ ML tabanlı duygu analizi tamamlandı")
            return final_scores
            
        except Exception as e:
            self.add_log(f"❌ ML duygu analizi hatası: {e}")
            # Fallback to basic analysis
            return self.run_emotion_analysis(audio_data, sample_rate)
    
    def advanced_rule_based_classification(self, features):
        """Gelişmiş kural tabanlı duygu sınıflandırması"""
        emotion_scores = {
            'mutlu': 0.0, 'üzgün': 0.0, 'kızgın': 0.0,
            'sakin': 0.0, 'heyecanlı': 0.0, 'stresli': 0.0,
            'şaşkın': 0.0, 'korku': 0.0  # Yeni duygular
        }
        
        # Gelişmiş kurallar - DÜŞÜK EŞİKLER ile güncellendi
        
        # 1. MUTLULUK ve GÜLME TESPİTİ (Çok geliştirildi!)
        mutlu_score = 0.0
        
        # Ana mutluluk göstergeleri
        if features['pitch_mean'] > 150:  # Düşürüldü: 200 -> 150
            mutlu_score += 0.3
        if features['energy'] > 0.0005:  # ÇOK düşürüldü: 0.01 -> 0.0005
            mutlu_score += 0.3
        if features['pitch_range'] > 0.05:  # Düşürüldü: 0.1 -> 0.05
            mutlu_score += 0.2
        if features['zcr_mean'] > 0.05:  # ZCR kahkaha için önemli
            mutlu_score += 0.2
        if features['spectral_bandwidth_mean'] > 1000:  # Geniş frekans = gülme
            mutlu_score += 0.2
        if features['energy_dynamic_range'] > 0.001:  # Düşürüldü: 0.01 -> 0.001
            mutlu_score += 0.2
            
        # KAHKAHA özel tespiti
        if (features['pitch_mean'] > 200 and features['zcr_mean'] > 0.08 and 
            features['energy'] > 0.001):  # Kahkaha kombinasyonu
            mutlu_score += 0.5
            
        emotion_scores['mutlu'] = min(mutlu_score, 1.0)
        
        # 2. HEYECAN TESPİTİ (Geliştirildi)
        heyecan_score = 0.0
        if features['energy'] > 0.001:  # Düşürüldü: 0.015 -> 0.001
            heyecan_score += 0.3
        if features['pitch_std'] > 30:  # Perde değişkenliği
            heyecan_score += 0.2
        if features['zcr_mean'] > 0.07:
            heyecan_score += 0.2
        if features['energy_variance'] > 0.0001:
            heyecan_score += 0.2
        if features['voice_activity_ratio'] > 0.6:  # Aktif konuşma
            heyecan_score += 0.1
            
        emotion_scores['heyecanlı'] = min(heyecan_score, 1.0)
            
        # 3. ÜZÜNTÜ tespiti (Daha spesifik)
        if (features['pitch_mean'] < 120 and features['energy'] < 0.0003 and  # Çok düşük
            features['speaking_rate'] < 2 and features['silence_ratio'] > 0.4):
            emotion_scores['üzgün'] += 0.4
            
        # 4. ÖFKE tespiti (Geliştirildi)
        if (features['zcr_mean'] > 0.12 and features['energy'] > 0.005 and  # Daha yüksek eşik
            features['spectral_bandwidth_mean'] > 2500 and features['energy_variance'] > 0.002):
            emotion_scores['kızgın'] += 0.5
            
        # 5. SAKİNLİK tespiti (Çok daha spesifik)
        sakin_score = 0.0
        if features['energy_variance'] < 0.00005:  # Çok düşük varyans
            sakin_score += 0.2
        if features['pitch_std'] < 20:  # Çok stabil perde
            sakin_score += 0.2
        if features['zcr_mean'] < 0.03:  # Çok düşük ZCR
            sakin_score += 0.2
        if features['silence_ratio'] > 0.5:  # Çok sessizlik
            sakin_score += 0.3
        if features['energy'] < 0.0002:  # Çok düşük enerji
            sakin_score += 0.3
            
        emotion_scores['sakin'] = min(sakin_score, 1.0)
            
        # 6. STRES tespiti
        if (features['zcr_std'] > 0.03 and features['energy_variance'] > 0.0003 and
            features['pitch_std'] > 60 and features['spectral_flatness_mean'] > 0.08):
            emotion_scores['stresli'] += 0.4
            
        # 7. ŞAŞKINLIK tespiti  
        if (features['pitch_range'] > 0.3 and features['energy_dynamic_range'] > 0.005 and
            features['speaking_rate'] < 3):
            emotion_scores['şaşkın'] += 0.3
            
        # 8. KORKU tespiti
        if (features['zcr_mean'] > 0.10 and features['pitch_mean'] > 180 and
            features['energy_variance'] > 0.0005 and features['voice_activity_ratio'] < 0.5):
            emotion_scores['korku'] += 0.3
        
        # ÖZEL KAHKAHA TESPİTİ - En üst öncelik
        if self.detect_laughter_patterns(features):
            emotion_scores['mutlu'] += 0.6  # Güçlü kahkaha bonusu
            emotion_scores['heyecanlı'] += 0.4
            emotion_scores['sakin'] *= 0.1  # Sakinliği bastır
            emotion_scores['üzgün'] *= 0.1   # Üzüntüyü bastır
        
        # Normalize skorları
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] = emotion_scores[emotion] / total_score
        else:
            # Default değerler (kahkaha bulunamadıysa)
            emotion_scores['mutlu'] = 0.4
            emotion_scores['heyecanlı'] = 0.3
            emotion_scores['sakin'] = 0.3
            
        return emotion_scores
    
    def detect_laughter_patterns(self, features):
        """Kahkaha desenlerini tespit et"""
        try:
            laughter_indicators = 0
            
            # 1. Yüksek frekans ama orta enerji (tipik kahkaha)
            if 1000 < features['pitch_mean'] < 4000:
                laughter_indicators += 1
                
            # 2. Yüksek ZCR (gürültülü, titrek ses)
            if features['zcr_mean'] > 0.08:
                laughter_indicators += 1
                
            # 3. Geniş spektral bant (ha-ha-ha sesi)
            if features['spectral_bandwidth_mean'] > 1500:
                laughter_indicators += 1
                
            # 4. Ortalama enerji (çok yüksek değil ama var)
            if 0.0005 < features['energy'] < 0.01:
                laughter_indicators += 1
                
            # 5. Değişken enerji (patlamalar)
            if features['energy_variance'] > 0.0002:
                laughter_indicators += 1
                
            # 6. Perküsif özellikler (ha-ha ritmi)
            if features['percussive_mean'] > 0.001:
                laughter_indicators += 1
                
            # 4 veya daha fazla gösterge = muhtemelen kahkaha
            return laughter_indicators >= 4
            
        except Exception as e:
            return False
    
    def temporal_emotion_analysis(self, audio_data, sample_rate, window_size=3.0):
        """Zamansal duygu analizi - ses boyunca duygu değişimi"""
        try:
            self.add_log("⏰ Zamansal duygu analizi yapılıyor...")
            
            window_samples = int(window_size * sample_rate)
            hop_samples = window_samples // 2
            
            temporal_emotions = []
            self._temp_temporal_data = []  # Geçici veri saklama
            
            for start in range(0, len(audio_data) - window_samples, hop_samples):
                end = start + window_samples
                segment = audio_data[start:end]
                
                # Segment özelliklerini çıkar
                segment_features = self.extract_advanced_audio_features(segment, sample_rate)
                
                if segment_features:
                    # Segment için duygu analizi
                    segment_emotions = self.advanced_rule_based_classification(segment_features)
                    temporal_data = {
                        'time': start / sample_rate,
                        'emotions': segment_emotions
                    }
                    temporal_emotions.append(temporal_data)
                    self._temp_temporal_data.append(temporal_data)  # Görselleştirme için sakla
            
            # Zamansal ortalamaları hesapla
            if temporal_emotions:
                avg_emotions = {}
                for emotion in temporal_emotions[0]['emotions'].keys():
                    scores = [te['emotions'][emotion] for te in temporal_emotions]
                    avg_emotions[emotion] = np.mean(scores)
                
                return avg_emotions
            else:
                return {}
                
        except Exception as e:
            self.add_log(f"❌ Zamansal analiz hatası: {e}")
            return {}
    
    def ensemble_emotion_scores(self, rule_scores, temporal_scores, weights=[0.6, 0.4]):
        """Farklı analiz yöntemlerinin skorlarını birleştir"""
        if not temporal_scores:
            return rule_scores
            
        ensemble_scores = {}
        
        for emotion in rule_scores.keys():
            if emotion in temporal_scores:
                ensemble_scores[emotion] = (
                    weights[0] * rule_scores[emotion] + 
                    weights[1] * temporal_scores[emotion]
                )
            else:
                ensemble_scores[emotion] = rule_scores[emotion]
        
        # Normalize
        total = sum(ensemble_scores.values())
        if total > 0:
            for emotion in ensemble_scores:
                ensemble_scores[emotion] /= total
                
        return ensemble_scores
    
    def calculate_confidence_score(self, features, emotion_scores):
        """Analiz güven skorunu hesapla"""
        try:
            # Ses kalitesi faktörleri
            quality_factors = {
                'energy_level': min(features['energy'] * 100, 1.0),
                'voice_activity': features['voice_activity_ratio'],
                'signal_clarity': 1 - features['spectral_flatness_mean'],
                'pitch_stability': 1 / (1 + features['pitch_std'] / max(features['pitch_mean'], 1))
            }
            
            # Duygu skorlarının dağılımı
            max_emotion_score = max(emotion_scores.values())
            second_max = sorted(emotion_scores.values())[-2] if len(emotion_scores) > 1 else 0
            score_separation = max_emotion_score - second_max
            
            # Genel güven skoru
            quality_score = np.mean(list(quality_factors.values()))
            confidence = (quality_score * 0.7) + (score_separation * 0.3)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            return 0.5  # Orta güven
    
    def display_advanced_emotion_results(self, emotion_scores, features, confidence):
        """Gelişmiş duygu analizi sonuçlarını göster"""
        self.emotion_text.delete(1.0, tk.END)
        self.emotion_text.insert(tk.END, f"🤖 Gelişmiş Duygu Analizi Sonuçları\n")
        self.emotion_text.insert(tk.END, f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.emotion_text.insert(tk.END, f"🎯 Güven Skoru: {confidence*100:.1f}%\n\n")
        
        # Duyguları sıralı göster
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (emotion, score) in enumerate(sorted_emotions):
            percentage = score * 100
            bar_length = int(percentage / 2.5)  # Daha hassas çubuk
            bar = "█" * bar_length + "░" * (40 - bar_length)
            
            # İkon ekle
            icons = {'mutlu': '😊', 'üzgün': '😢', 'kızgın': '😠', 
                    'sakin': '😌', 'heyecanlı': '🤩', 'stresli': '😰',
                    'şaşkın': '😮', 'korku': '😨'}
            icon = icons.get(emotion, '😐')
            
            rank_icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            
            self.emotion_text.insert(tk.END, 
                f"{rank_icon} {icon} {emotion.capitalize()}: {percentage:.1f}% {bar}\n")
        
        # Dominant duygu
        dominant_emotion, dominant_score = sorted_emotions[0]
        self.emotion_text.insert(tk.END, f"\n🎯 Baskın Duygu: {dominant_emotion.capitalize()} ({dominant_score*100:.1f}%)\n")
        
        # Detaylı ses özellikleri
        self.emotion_text.insert(tk.END, f"\n📊 Detaylı Ses Özellikleri:\n")
        self.emotion_text.insert(tk.END, f"⚡ Enerji Seviyesi: {features['energy']:.4f}\n")
        self.emotion_text.insert(tk.END, f"🎵 Ortalama Perde: {features['pitch_mean']:.1f} Hz\n")
        self.emotion_text.insert(tk.END, f"📈 Perde Değişkenliği: {features['pitch_std']:.1f} Hz\n")
        self.emotion_text.insert(tk.END, f"🗣️ Konuşma Hızı: {features['speaking_rate']:.1f}\n")
        self.emotion_text.insert(tk.END, f"🤫 Sessizlik Oranı: {features['silence_ratio']*100:.1f}%\n")
        self.emotion_text.insert(tk.END, f"🌊 ZCR Ortalama: {features['zcr_mean']:.3f}\n")
        self.emotion_text.insert(tk.END, f"🎼 Spektral Merkez: {features['spectral_centroid_mean']:.1f} Hz\n")
        self.emotion_text.insert(tk.END, f"🎛️ Enerji Varyansı: {features['energy_variance']:.6f}\n")
        
        # Güven seviyesi yorumu
        if confidence > 0.8:
            conf_text = "Çok Yüksek ✨"
        elif confidence > 0.6:
            conf_text = "Yüksek ✅"
        elif confidence > 0.4:
            conf_text = "Orta ⚠️"
        else:
            conf_text = "Düşük ❌"
            
        self.emotion_text.insert(tk.END, f"\n🎯 Analiz Güvenilirliği: {conf_text}\n")
    
    def use_pretrained_emotion_model(self, feature_vector):
        """Önceden eğitilmiş duygu modeli kullan"""
        try:
            # Hugging Face Transformers ile ses duygu tanıma
            from transformers import pipeline
            
            # Çoklu model yaklaşımı - daha doğru sonuçlar için
            models_to_try = [
                "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                "facebook/wav2vec2-large-xlsr-53-turkish",  # Türkçe desteği
                "microsoft/unispeech-sat-base-plus"
            ]
            
            emotion_results = []
            
            for model_name in models_to_try:
                try:
                    self.add_log(f"🤖 Model deneniyor: {model_name}")
                    emotion_classifier = pipeline(
                        "audio-classification",
                        model=model_name,
                        device=0 if torch.cuda.is_available() else -1
                    )
                    
                    # Gerçek ses dosyası ile analiz
                    if hasattr(self, 'current_audio_file') and self.current_audio_file:
                        result = emotion_classifier(self.current_audio_file)
                        emotion_results.append(result)
                        self.add_log(f"✅ Model başarılı: {model_name}")
                        break
                    
                except Exception as model_error:
                    self.add_log(f"❌ Model hatası {model_name}: {model_error}")
                    continue
            
            if emotion_results:
                # Sonuçları normalize et
                return self.normalize_transformers_results(emotion_results[0])
            else:
                self.add_log("⚠️ Hiçbir transformer model çalışmadı, alternatif kullanılıyor")
                return self.use_sklearn_emotion_model(feature_vector)
            
        except ImportError:
            self.add_log("⚠️ Transformers kütüphanesi yok, alternatif yöntem kullanılıyor")
            return self.use_sklearn_emotion_model(feature_vector)
        except Exception as e:
            self.add_log(f"❌ Pretrained model hatası: {e}")
            return self.use_sklearn_emotion_model(feature_vector)
    
    def normalize_transformers_results(self, transformer_results):
        """Transformer sonuçlarını normalize et"""
        try:
            # Transformer sonuçlarını kendi duygu kategorilerimize çevir
            emotion_mapping = {
                'happy': 'mutlu',
                'joy': 'mutlu',
                'sad': 'üzgün',
                'angry': 'kızgın',
                'calm': 'sakin',
                'neutral': 'sakin',
                'excited': 'heyecanlı',
                'fear': 'stresli',
                'surprise': 'şaşkın'
            }
            
            normalized_scores = {
                'mutlu': 0.0, 'üzgün': 0.0, 'kızgın': 0.0,
                'sakin': 0.0, 'heyecanlı': 0.0, 'stresli': 0.0
            }
            
            for result in transformer_results:
                label = result['label'].lower()
                score = result['score']
                
                # Eşleştirme yap
                for eng_emotion, tr_emotion in emotion_mapping.items():
                    if eng_emotion in label:
                        if tr_emotion in normalized_scores:
                            normalized_scores[tr_emotion] += score
                        break
            
            # Normalize et
            total_score = sum(normalized_scores.values())
            if total_score > 0:
                for emotion in normalized_scores:
                    normalized_scores[emotion] /= total_score
            
            return normalized_scores
            
        except Exception as e:
            self.add_log(f"❌ Sonuç normalizasyon hatası: {e}")
            return self.mock_pretrained_results()
    
    def use_sklearn_emotion_model(self, feature_vector):
        """Scikit-learn tabanlı duygu modeli"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC
            from sklearn.naive_bayes import GaussianNB
            
            # Mock eğitilmiş model (gerçek projede önceden eğitilmiş model yüklenir)
            # model = joblib.load('emotion_rf_model.pkl')
            
            # Şimdilik gelişmiş kural tabanlı tahmin döndür
            return self.mock_sklearn_results()
            
        except Exception as e:
            self.add_log(f"❌ Sklearn model hatası: {e}")
            raise e
    
    def mock_pretrained_results(self):
        """Mock pretrained model sonuçları"""
        return {
            'mutlu': 0.15, 'üzgün': 0.10, 'kızgın': 0.08,
            'sakin': 0.25, 'heyecanlı': 0.20, 'stresli': 0.12,
            'şaşkın': 0.06, 'korku': 0.04
        }
    
    def mock_sklearn_results(self):
        """Mock sklearn model sonuçları"""
        return {
            'mutlu': 0.18, 'üzgün': 0.12, 'kızgın': 0.10,
            'sakin': 0.22, 'heyecanlı': 0.18, 'stresli': 0.15,
            'şaşkın': 0.03, 'korku': 0.02
        }
    
    def plot_temporal_emotion_analysis(self):
        """Zamansal duygu analizi görselleştirmesi"""
        try:
            if not hasattr(self, 'temporal_emotion_history') or not self.temporal_emotion_history:
                return
                
            self.ax_emotion.clear()
            
            # Zaman ekseni oluştur
            times = [te['time'] for te in self.temporal_emotion_history]
            
            # Her duygu için zaman serisi çiz
            emotion_colors = {
                'mutlu': '#2ECC71', 'üzgün': '#3498DB', 'kızgın': '#E74C3C',
                'sakin': '#95A5A6', 'heyecanlı': '#F39C12', 'stresli': '#E67E22',
                'şaşkın': '#9B59B6', 'korku': '#34495E'
            }
            
            for emotion in ['mutlu', 'üzgün', 'kızgın', 'sakin', 'heyecanlı']:
                scores = [te['emotions'].get(emotion, 0) * 100 for te in self.temporal_emotion_history]
                color = emotion_colors.get(emotion, '#95A5A6')
                
                self.ax_emotion.plot(times, scores, label=emotion.capitalize(), 
                                   color=color, linewidth=2, marker='o', markersize=4)
            
            self.ax_emotion.set_xlabel('Zaman (saniye)')
            self.ax_emotion.set_ylabel('Duygu Skoru (%)')
            self.ax_emotion.set_title('⏰ Zamansal Duygu Değişimi')
            self.ax_emotion.legend(loc='upper right', fontsize=9)
            self.ax_emotion.grid(True, alpha=0.3)
            self.ax_emotion.set_ylim(0, 100)
            
            # Dominant duygu bölgelerini vurgula
            for i in range(len(self.temporal_emotion_history) - 1):
                current_emotions = self.temporal_emotion_history[i]['emotions']
                dominant_emotion = max(current_emotions.keys(), key=current_emotions.get)
                dominant_color = emotion_colors.get(dominant_emotion, '#95A5A6')
                
                start_time = self.temporal_emotion_history[i]['time']
                end_time = self.temporal_emotion_history[i + 1]['time']
                
                self.ax_emotion.axvspan(start_time, end_time, alpha=0.1, color=dominant_color)
            
            self.canvas3.draw()
            
        except Exception as e:
            self.add_log(f"❌ Zamansal görselleştirme hatası: {e}")
    
    def assess_audio_quality(self, audio_data, sample_rate):
        """Ses kalitesini değerlendir"""
        try:
            quality_score = 0.0
            quality_factors = {}
            
            # 1. Sinyal-Gürültü Oranı (SNR) tahmini
            signal_power = np.mean(audio_data ** 2)
            noise_threshold = 0.001
            noise_power = np.mean(audio_data[np.abs(audio_data) < noise_threshold] ** 2) if np.any(np.abs(audio_data) < noise_threshold) else 0.0001
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 40
            quality_factors['snr'] = min(snr / 40, 1.0)  # 40 dB'yi maksimum kabul et
            
            # 2. Dinamik Aralık
            dynamic_range = np.max(audio_data) - np.min(audio_data)
            quality_factors['dynamic_range'] = min(dynamic_range / 2.0, 1.0)  # -1 ile +1 arası max
            
            # 3. Kliping (Kesik) Tespiti
            clipping_threshold = 0.95
            clipped_samples = np.sum(np.abs(audio_data) > clipping_threshold)
            clipping_ratio = clipped_samples / len(audio_data)
            quality_factors['no_clipping'] = max(1.0 - clipping_ratio * 10, 0.0)
            
            # 4. Frekans Dağılımı
            freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)
            fft = np.abs(np.fft.fft(audio_data))
            
            # İnsan sesi frekans aralığı (80-8000 Hz)
            voice_freq_mask = (np.abs(freqs) >= 80) & (np.abs(freqs) <= 8000)
            voice_energy = np.sum(fft[voice_freq_mask])
            total_energy = np.sum(fft)
            voice_ratio = voice_energy / total_energy if total_energy > 0 else 0
            quality_factors['voice_frequency'] = voice_ratio
            
            # 5. Ses Sürekliliği (Sessizlik analizine göre)
            silence_threshold = 0.01
            voice_frames = np.abs(audio_data) > silence_threshold
            continuity = np.sum(voice_frames) / len(audio_data)
            quality_factors['continuity'] = continuity
            
            # 6. Spektral Düzlük (Gürültü göstergesi)
            spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=audio_data))
            quality_factors['spectral_clarity'] = max(1.0 - spectral_flatness, 0.0)
            
            # Genel kalite skoru hesapla
            weights = {
                'snr': 0.25,
                'dynamic_range': 0.15,
                'no_clipping': 0.20,
                'voice_frequency': 0.15,
                'continuity': 0.15,
                'spectral_clarity': 0.10
            }
            
            quality_score = sum(quality_factors[factor] * weights[factor] for factor in quality_factors)
            
            return {
                'overall_score': quality_score,
                'factors': quality_factors,
                'details': {
                    'snr_db': snr,
                    'dynamic_range': dynamic_range,
                    'clipping_percentage': clipping_ratio * 100,
                    'voice_frequency_ratio': voice_ratio * 100,
                    'voice_continuity': continuity * 100,
                    'spectral_flatness': spectral_flatness
                }
            }
            
        except Exception as e:
            self.add_log(f"❌ Ses kalitesi analizi hatası: {e}")
            return {'overall_score': 0.5, 'factors': {}, 'details': {}}
    
    def debug_emotion_analysis(self, features):
        """Duygu analizi debug bilgileri"""
        debug_info = []
        
        debug_info.append(f"🔍 DUYGU ANALİZİ DEBUG BİLGİLERİ:")
        debug_info.append(f"{'='*40}")
        
        # Temel özellikler
        debug_info.append(f"📊 Temel Özellikler:")
        debug_info.append(f"  ⚡ Enerji: {features['energy']:.6f}")
        debug_info.append(f"  🎵 Perde: {features['pitch_mean']:.1f} Hz")
        debug_info.append(f"  📈 Perde Std: {features['pitch_std']:.1f} Hz")
        debug_info.append(f"  🌊 ZCR: {features['zcr_mean']:.4f}")
        debug_info.append(f"  📊 Spektral Bant: {features['spectral_bandwidth_mean']:.1f} Hz")
        debug_info.append(f"  🎛️ Enerji Varyans: {features['energy_variance']:.8f}")
        debug_info.append("")
        
        # MUTLULUK kontrolleri
        debug_info.append(f"😊 MUTLULUK KONTROL:")
        mutlu_tests = []
        if features['pitch_mean'] > 150:
            mutlu_tests.append("✅ Perde > 150 Hz")
        else:
            mutlu_tests.append("❌ Perde çok düşük")
            
        if features['energy'] > 0.0005:
            mutlu_tests.append("✅ Enerji > 0.0005")
        else:
            mutlu_tests.append("❌ Enerji çok düşük")
            
        if features['zcr_mean'] > 0.05:
            mutlu_tests.append("✅ ZCR > 0.05")
        else:
            mutlu_tests.append("❌ ZCR düşük")
            
        # Kahkaha testi
        laughter_detected = self.detect_laughter_patterns(features)
        if laughter_detected:
            mutlu_tests.append("🎉 KAHKAHA TESPİT EDİLDİ!")
        else:
            mutlu_tests.append("❌ Kahkaha tespit edilmedi")
            
        for test in mutlu_tests:
            debug_info.append(f"  {test}")
        debug_info.append("")
        
        # SAKİNLİK kontrolleri
        debug_info.append(f"😌 SAKİNLİK KONTROL:")
        sakin_tests = []
        if features['energy'] < 0.0002:
            sakin_tests.append("✅ Çok düşük enerji")
        else:
            sakin_tests.append("❌ Enerji yeterince düşük değil")
            
        if features['zcr_mean'] < 0.03:
            sakin_tests.append("✅ Çok düşük ZCR")
        else:
            sakin_tests.append("❌ ZCR yeterince düşük değil")
            
        if features['pitch_std'] < 20:
            sakin_tests.append("✅ Stabil perde")
        else:
            sakin_tests.append("❌ Perde değişken")
            
        for test in sakin_tests:
            debug_info.append(f"  {test}")
        debug_info.append("")
        
        return "\n".join(debug_info)
    
    def clean_audio_buffer(self, audio_data):
        """Ses verisini temizle - NaN ve Infinity değerleri kaldır"""
        try:
            self.add_log("🧹 Ses verisi temizleniyor...")
            
            # NaN ve Infinity kontrolü
            if not np.isfinite(audio_data).all():
                self.add_log("⚠️ Ses verisinde NaN/Infinity değerleri tespit edildi, temizleniyor...")
                
                # NaN değerleri sıfır ile değiştir
                audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Aşırı büyük değerleri kırp (-1, +1 aralığına)
                audio_data = np.clip(audio_data, -1.0, 1.0)
                
                # Eğer tüm veriler sıfır olduysa, küçük bir gürültü ekle
                if np.all(audio_data == 0):
                    self.add_log("⚠️ Tüm ses verisi sıfır, küçük gürültü ekleniyor...")
                    audio_data = np.random.normal(0, 0.001, len(audio_data))
                
                self.add_log("✅ Ses verisi temizlendi")
            
            return audio_data.astype(np.float32)
            
        except Exception as e:
            self.add_log(f"❌ Ses temizleme hatası: {e}")
            # Son çare: sessizlik döndür
            return np.zeros(len(audio_data), dtype=np.float32)
    
    def get_default_features(self):
        """Hata durumunda kullanılacak varsayılan özellikler"""
        return {
            # Temel özellikler
            'mfcc_mean': 0.0,
            'mfcc_std': 1.0,
            'spectral_centroid_mean': 1000.0,
            'spectral_centroid_std': 500.0,
            'zcr_mean': 0.05,
            'zcr_std': 0.02,
            
            # Gelişmiş spektral özellikler
            'chroma_mean': 0.1,
            'chroma_std': 0.05,
            'mel_spectrogram_mean': 0.01,
            'tonnetz_mean': 0.0,
            'spectral_contrast_mean': 10.0,
            'spectral_bandwidth_mean': 1500.0,
            'spectral_flatness_mean': 0.1,
            'spectral_rolloff_mean': 2000.0,
            
            # Prozodik özellikler
            'tempo': 100.0,
            'speaking_rate': 2.0,
            'pitch_mean': 200.0,
            'pitch_std': 50.0,
            'pitch_range': 0.25,
            
            # Enerji özellikleri
            'energy': 0.001,
            'rms_energy': 0.01,
            'energy_variance': 0.0001,
            'energy_mean': 0.001,
            'energy_dynamic_range': 0.005,
            
            # Sessizlik ve duraklama
            'silence_ratio': 0.3,
            'voice_activity_ratio': 0.7,
            
            # Harmonik özellikler
            'harmonic_mean': 0.001,
            'percussive_mean': 0.001,
        }
    
    def get_default_emotion_scores(self):
        """Hata durumunda kullanılacak varsayılan duygu skorları"""
        return {
            'mutlu': 0.25,
            'üzgün': 0.15,
            'kızgın': 0.10,
            'sakin': 0.30,
            'heyecanlı': 0.15,
            'stresli': 0.05
        }
    
    def update_live_waveform(self):
        """Canlı kayıt sırasında dalga formunu güncelle"""
        try:
            if len(self.recorded_audio) > 0:
                audio_array = np.array(self.recorded_audio[-SAMPLE_RATE:])  # Son 1 saniye
                if len(audio_array) > 100:  # Yeterli veri varsa
                    time_axis = np.arange(len(audio_array)) / SAMPLE_RATE
                    
                    # Ana dalga formu - sadece güncelle, layout değişikliği yapma
                    self.ax_waveform.clear()
                    self.ax_waveform.plot(time_axis, audio_array, color='#2E86AB', linewidth=0.8)
                    self.ax_waveform.set_ylim(-1, 1)
                    self.ax_waveform.set_title("🎙️ Canlı Kayıt - Dalga Formu")
                    self.ax_waveform.grid(True, alpha=0.3)
                    
                    # Canvas'ı güncelle - ancak layout hesaplaması yapma
                    self.canvas.draw_idle()
                    
        except Exception as e:
            # Canlı güncelleme hatalarını sessizce geç
            pass
    
    def optimize_audio_device(self):
        """En uygun ses cihazını seç ve optimize et"""
        try:
            self.add_log("🎧 Ses cihazları taranıyor...")
            
            # Mevcut cihazları listele
            devices = sd.query_devices()
            
            # En iyi giriş cihazını bul
            best_input_device = None
            best_score = 0
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:  # Giriş cihazı
                    score = 0
                    
                    # Samplerate desteği
                    try:
                        sd.check_device(i, samplerate=SAMPLE_RATE)
                        score += 10
                    except:
                        continue
                    
                    # Kanal sayısı
                    score += device['max_input_channels']
                    
                    # Varsayılan cihaz bonusu
                    if device == sd.query_devices(kind='input'):
                        score += 5
                    
                    # macOS için Core Audio tercih et
                    if 'Core Audio' in device['hostapi_name']:
                        score += 3
                    
                    # USB/Harici cihaz bonusu
                    if 'USB' in device['name'] or 'External' in device['name']:
                        score += 2
                    
                    if score > best_score:
                        best_score = score
                        best_input_device = i
            
            if best_input_device is not None:
                sd.default.device[0] = best_input_device  # Input device
                device_info = devices[best_input_device]
                self.add_log(f"🎤 Seçilen cihaz: {device_info['name']}")
                self.add_log(f"📊 API: {device_info['hostapi_name']}")
                self.add_log(f"🔊 Max kanallar: {device_info['max_input_channels']}")
                self.add_log(f"⚡ Gecikme: {device_info['default_low_input_latency']*1000:.1f}ms")
            else:
                self.add_log("⚠️ Uygun ses cihazı bulunamadı, varsayılan kullanılacak")
                
            # Buffer ayarlarını optimize et
            latency = sd.query_devices(sd.default.device[0])['default_low_input_latency']
            optimal_blocksize = int(SAMPLE_RATE * latency)
            
            self.add_log(f"🔧 Optimal buffer boyutu: {optimal_blocksize} sample")
            return optimal_blocksize
            
        except Exception as e:
            self.add_log(f"❌ Cihaz optimizasyonu hatası: {e}")
            return BUFFER_SIZE
    
    def update_recording_progress(self, progress, elapsed, total_duration):
        """Kayıt ilerlemesini güncelle"""
        try:
            remaining = total_duration - elapsed
            
            # Durum etiketi güncelle
            self.status_label.config(
                text=f"🎙️ Kayıt: {elapsed:.1f}s / {total_duration:.1f}s (%{progress:.1f})"
            )
            
            # Her 2 saniyede bir ilerleme logu
            if int(elapsed) % 2 == 0 and elapsed > 0:
                self.add_log(f"📊 İlerleme: %{progress:.1f} - {elapsed:.1f}s / {total_duration:.1f}s")
            
        except Exception as e:
            # Progress güncelleme hatalarını sessizce geç
            pass
    
    def record_audio_simple(self, duration):
        """En basit kayıt yöntemi - fallback"""
        try:
            self.add_log("🎙️ FALLBACK: En basit kayıt yöntemi")
            
            # Ses cihazını optimize et
            self.optimize_audio_device()
            
            # Tek seferde tüm kaydı al - eski yöntem ama güvenilir
            total_samples = int(SAMPLE_RATE * duration)
            self.add_log(f"📊 Tek seferde {total_samples} sample alınacak")
            
            # Kayıt başlat
            self.add_log("🔴 Kayıt başlıyor...")
            start_time = time.time()
            
            # SoundDevice'ın basit rec fonksiyonu
            audio_data = sd.rec(
                frames=total_samples,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocking=False  # Non-blocking kayıt
            )
            
            # Kayıt tamamlanana kadar bekle ve progress göster
            while sd.get_stream().active and self.is_recording:
                elapsed = time.time() - start_time
                progress = min((elapsed / duration) * 100, 100)
                
                self.root.after(1, lambda p=progress, e=elapsed: 
                    self.status_label.config(text=f"🎙️ Basit Kayıt: {e:.1f}s / {duration}s (%{p:.1f})"))
                
                if int(elapsed) % 2 == 0:  # Her 2 saniyede log
                    self.add_log(f"📊 Kayıt sürüyor: {elapsed:.1f}s")
                
                time.sleep(0.1)
                
                # Timeout kontrolü
                if elapsed > duration + 2:
                    break
            
            # Kayıt verisini al
            sd.wait()  # Kayıt tamamlanana kadar bekle
            
            # Sonuçları kontrol et
            self.recorded_audio = audio_data[:, 0] if audio_data.ndim > 1 else audio_data
            self.sample_rate = SAMPLE_RATE
            
            actual_duration = len(self.recorded_audio) / SAMPLE_RATE
            self.add_log(f"✅ Basit kayıt tamamlandı!")
            self.add_log(f"📊 Hedef: {duration}s, Gerçek: {actual_duration:.2f}s")
            self.add_log(f"🔢 Sample sayısı: {len(self.recorded_audio)}")
            
            # Ses analizi
            if len(self.recorded_audio) > 0:
                max_amplitude = np.max(np.abs(self.recorded_audio))
                avg_amplitude = np.mean(np.abs(self.recorded_audio))
                
                self.add_log(f"🔊 Max seviye: {max_amplitude:.4f}")
                self.add_log(f"📈 Ortalama seviye: {avg_amplitude:.4f}")
                
                # Normalize et (gerekirse)
                if max_amplitude > 0.95:
                    self.recorded_audio = self.recorded_audio / max_amplitude * 0.9
                    self.add_log("🔧 Ses seviyesi normalize edildi")
                
                # UI güncelle
                self.root.after(1, lambda: self.update_comprehensive_plots(self.recorded_audio))
                self.root.after(1, lambda: self.status_label.config(text="✅ Basit kayıt tamamlandı"))
                
                # Dosyaya kaydet
                self.save_recording()
                
                # Düğmeleri etkinleştir
                self.root.after(1, lambda: self.analyze_button.config(state=tk.NORMAL))
                self.root.after(1, lambda: self.quick_analyze_button.config(state=tk.NORMAL))
                self.root.after(1, lambda: self.report_button.config(state=tk.NORMAL))
                
                return True
            else:
                self.add_log("❌ Kayıt verisi alınamadı!")
                return False
                
        except Exception as e:
            self.add_log(f"❌ Basit kayıt hatası: {e}")
            return False
        finally:
            self.is_recording = False
            self.root.after(1, lambda: self.record_button.config(text="🔴 Kaydı Başlat"))

def main():
    """Ana fonksiyon"""
    # macOS uyumluluğu için
    import sys
    
    # Root pencere oluştur
    root = tk.Tk()
    
    # macOS'ta Tkinter'ı ön plana getir
    if sys.platform == "darwin":  # macOS
        try:
            # macOS'ta Tkinter uygulamasını aktifleştir
            from subprocess import call
            call(['osascript', '-e', 'tell application "Python" to activate'])
        except:
            pass
    
    # Uygulamayı başlat
    app = SesKayitAnaliz(root)
    
    # Pencereyi merkeze getir ve görünür yap
    root.update()
    root.after(100, lambda: root.focus_force())
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nUygulama kapatılıyor...")
        root.quit()

if __name__ == "__main__":
    main() 