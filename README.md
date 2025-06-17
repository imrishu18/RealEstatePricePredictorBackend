# 🏠 Real Estate Price Predictor

A modern, responsive, and intelligent web application that allows users to predict property prices in Bengaluru using machine learning. Built using **Next.js**, **Tailwind CSS**, **FastAPI**, and a trained **XGBoost model**, the app provides an intuitive UI, clean UX, and smart results.

---

## 🚀 Live Demo

🔗 [View Live Application](https://real-estate-price-predictor-fronten.vercel.app)  
🔗 [Backend API](https://real-estate-api-z0un.onrender.com)

---

## 🌟 Features

- 📊 Real-time property price predictions
- 📐 Location-sensitive pricing with smart feature engineering
- 🧠 ML-powered (XGBoost + SHAP for explainability)
- ✨ Beautiful UI, responsive layout, animated visuals
- 💳 EMI estimator & price breakdown
- ✅ Validations & fallback handling
- 🔧 Backend-frontend integration with REST API

---

## 🖼️ Pages Overview

### 1️⃣ Home Page
- Gradient text with “Real Estate Price Predictor”
- ⚡ Animated particle background
- CTA: “🚀 Get Started”
- Icons showing tech used

### 2️⃣ Predict Page
- Input fields:
  - Total Sqft
  - Bathrooms (select dropdown)
  - Balconies (select dropdown)
  - BHK
  - Location (searchable list of 60+ Bengaluru areas)
- 🔎 Smart location filtering & validation
- 🎨 Floating particle animation

### 3️⃣ Results Page
- ✅ Predicted Price (in Lakhs)
- 📏 Price Per Sqft
- 📊 Price Breakdown: Base, Amenities, Location Premium
- 💳 EMI Calculator (customizable loan term + interest)
- 🔁 Predict Again button

---

## 🧠 Machine Learning Model

- **Dataset**: [Bengaluru House Price Data (Kaggle)](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data)
- **Preprocessing**:
  - Handled mixed `total_sqft` formats and units
  - Removed outliers using IQR and domain rules
  - Encoded categorical location field
  - Applied `log1p(price)` to reduce skew
  - Scaled features using `StandardScaler`
- **Model**: `XGBoostRegressor`
  - Hyperparameter-tuned using `RandomizedSearchCV`
  - Trained with 5-fold CV

### ✅ Final Model Metrics

| Metric     | Value (Test Set) |
|------------|------------------|
| MAE        | `16.77 Lakhs`    |
| RMSE       | `47.45 Lakhs`    |
| R² Score   | `0.627`          |

> Predictions are de-log transformed using `expm1()` to obtain real-world values.

---

## 🔧 Tech Stack

| Layer      | Tools Used                                 |
|------------|---------------------------------------------|
| Frontend   | Next.js, Tailwind CSS, Framer Motion        |
| Backend    | FastAPI, Uvicorn                            |
| ML Model   | XGBoost, Scikit-learn, SHAP, joblib         |
| Deployment | Vercel (frontend), Render (backend)         |

---

## 📈 Explainability

- ✅ SHAP plots (summary & force) are generated separately for analysis  
- Stored in: `app/artifacts/shap_summary.png`, `shap_force_plot.html`

---

## 🚀 Deployment

### 1. Backend (FastAPI on Render)

1. Push `backend/` folder to GitHub
2. Go to [Render](https://render.com)
3. Create new Web Service:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
4. Deploy and get the public URL: https://real-estate-api-z0un.onrender.com


### 2. Frontend (Next.js on Vercel)

1. Push `frontend/` folder to GitHub
2. Go to [Vercel](https://vercel.com)
3. Import GitHub repo → Vercel detects Next.js automatically
4. Add environment variable: NEXT_PUBLIC_API_URL=https://real-estate-api-z0un.onrender.com
5. Click **Deploy**

---

## 🛠️ Run Locally

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev

```
---

## 🔗 Project Links

- 🖥️ **Frontend Repository**: [RealEstatePricePredictorFrontend (GitHub)](https://github.com/imrishu18/RealEstatePricePredictorFrontend)
- 🛠️ **Backend Repository**: [RealEstatePricePredictorBackend (GitHub)](https://github.com/imrishu18/RealEstatePricePredictorBackend)
- 📓 **Google Colab Notebook**: [Run the Model on Google Colab](https://colab.research.google.com/drive/1_vV1FyKQ_lqpPKhKBZtVUTR_dDterorm?usp=sharing)

---

## 👨‍💻 Author

**Rishu Raj**  
🔗 [GitHub](https://github.com/imrishu18)  
💼 [LinkedIn](https://www.linkedin.com/in/your-link) *(Coming soon...)*

---

## 📜 License

This project is open-source and free to use under the MIT License.
