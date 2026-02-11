# 🏀 CoachTrack Oracle v19.0 - Bio-Metric Intelligence

**CoachTrack Oracle** è la piattaforma definitiva per il coaching NBA moderno. Questa versione introduce il monitoraggio avanzato della composizione corporea e un'integrazione profonda con l'intelligenza artificiale per l'ottimizzazione delle performance e della salute degli atleti.

---

## 🚀 Novità Versione 18.0

* **📊 Visual Tracking Progress:** Barra di avanzamento dinamica durante l'analisi video YOLO per monitorare lo stato del processamento in tempo reale.
* **⚖️ Bio-Metric Architect:** Monitoraggio completo della composizione corporea (Peso, BMI, % Grasso, Massa Muscolare, Massa Ossea, % Acqua).
* **🥗 AI-Driven Nutrition:** The Oracle genera ora piani alimentari e di recupero ultra-personalizzati basati sui dati impedenziometrici reali.
* **⚔️ War Room 2.0:** Confronto radar potenziato che incrocia efficienza tecnica (Tiro) e potenza fisica (Massa Muscolare/HRV).
* **📘 Manuale Integrato:** Documentazione scaricabile in PDF direttamente dalla dashboard per un onboarding rapido.

---

## 🛠️ Architettura e Pulizia Repository

Per garantire la massima stabilità, il repository è stato ottimizzato eliminando moduli ridondanti. L'app gira esclusivamente sui seguenti file core:

* **`app.py`**: Il motore centrale v18.0.
* **`requirements.txt`**: Dipendenze (ultralytics, groq, opencv, ecc.).
* **`packages.txt`**: Librerie di sistema Linux (libgl1, libglib2.0-0).
* **`.gitignore`**: Esclusione file temporanei e DB locali.

---

## 🤖 The Oracle: Come Interagire

L'AI "The Oracle" ha ora accesso a parametri fisici granulari. Puoi interrogarlo per:
1. **Analisi del Rischio:** "Controlla se il calo di HRV di [Nome] è legato alla disidratazione."
2. **Ottimizzazione Peso:** "Suggerisci una dieta per aumentare la massa muscolare senza alterare la velocità di spostamento laterale."
3. **Pianificazione Tattica:** "Qual è il quintetto più 'fresco' fisicamente per l'ultimo quarto?"

---

## 📦 Deployment Rapido

1. Assicurati che le cartelle `.devcontainer`, `data` e `modules` siano state rimosse.
2. Carica i file core nel tuo repository.
3. Configura la tua `GROQ_API_KEY` nei segreti di Streamlit.
4. Avvia l'app e scarica il manuale dalla sidebar per iniziare.

---
© 2026 CoachTrack Elite - *Developing the future of NBA coaching.*
