# 🏠 Real Estate Price Predictor

A modern, responsive, and intelligent web application that allows users to predict property prices in Bengaluru using machine learning. Built using **Next.js**, **Tailwind CSS**, **FastAPI**, and a trained **XGBoost model**, the app provides an intuitive UI, clean UX, and smart results.

---

## 🚀 Live Demo

🔗 [View Live Application](https://real-estate-price-predictor-fronten.vercel.app)  
🔗 [Backend API](https://real-estate-api-z0un.onrender.com)

---

## 🌟 Features

- 📊 Real-time property price predictions
- ✨ Stunning UI with animated backgrounds
- 📍 Location-specific predictions with premium adjustments
- 🧠 Machine Learning powered (XGBoost)
- 🖥️ Fully responsive and mobile-friendly
- 🔒 Input validation and error handling

---

## 🖼️ Pages Overview

### 1️⃣ Landing Page (Home)

- Gradient text heading: `Real Estate Price Predictor`
- Emojis for property type hints
- CTA: “🚀 Get Started”
- Animated background using `particles.js` canvas effect
- Built with icons: Python, FastAPI, ML Model, Next.js, Tailwind CSS

### 2️⃣ Predict Page

- Input fields:
  - Total Sqft
  - Price per Sqft
  - Bathrooms (select dropdown)
  - Balconies (select dropdown)
  - Location (input with datalist from 50+ popular Bengaluru areas)
- Smart validation: user must select valid location
- Background animation with floating particles

### 3️⃣ Result Page

- Displays Predicted Price 💰
- 💹 Price Breakdown:
  - Base Cost
  - Amenities Cost
  - Location Premium
- 💳 Estimated EMI calculator based on prediction
- Button to re-predict or go back

---

## 🧠 ML Model

- Dataset: [Bengaluru House Price Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data)
- Libraries used:
  - `pandas`, `numpy`, `scikit-learn`, `xgboost`
- Preprocessing:
  - Categorical encoding, outlier removal, null handling
- Model:
  - Trained using XGBoost Regressor
  - Stored via `joblib` and served via FastAPI

---

## 🔧 Tech Stack

| Frontend | Backend | Machine Learning |
|----------|---------|------------------|
| Next.js  | FastAPI | XGBoost, Pandas  |
| Tailwind CSS | Uvicorn | Scikit-learn |
| Framer Motion | REST API | joblib     |

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

## 📊 Model Performance

The model predicts the **log-transformed house prices** to ensure stability and reduce the impact of outliers. The following models were evaluated:

| Model                 | Metric                          | Value     |
|----------------------|----------------------------------|-----------|
| **Linear Regression** | RMSE (Validation Set)            | `0.5276`  |
| **XGBoost Regressor** | Cross-Validated RMSE (CV=5)      | `0.0514`  |

> ✅ **RMSE (Root Mean Squared Error)** is used as the evaluation metric. A lower RMSE indicates better performance in predicting house prices (log scale).

### 📌 Notes
- The `price` variable was log-transformed using `log1p(price)` for model training to improve prediction stability.
- **XGBoost Regressor** outperformed the baseline **Linear Regression**, showing its ability to capture non-linear patterns in real estate data.
- The performance metric (RMSE) is calculated on the **log-scale**. You can apply `expm1()` to convert predictions back to actual price values.
- Cross-validation (CV=5) was used to ensure robust evaluation of the XGBoost model.


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

---

## 🔗 Project Links

- 🖥️ **Frontend Repository**: [RealEstatePricePredictorFrontend (GitHub)](https://github.com/imrishu18/RealEstatePricePredictorFrontend)
- 🛠️ **Backend Repository**: [RealEstatePricePredictorBackend (GitHub)](https://github.com/imrishu18/RealEstatePricePredictorBackend)
- 📓 **Google Colab Notebook**: [Run the Model on Google Colab](https://colab.research.google.com/drive/1Zhdsix8XywGV4iEeTodskOFno7SQsYLc?usp=sharing)

---

## 👨‍💻 Author

**Rishu Raj**  
🔗 [GitHub](https://github.com/imrishu18)  
💼 [LinkedIn][Coming soon...](https://www.linkedin.com/in/your-link)

---

## 📜 License

This project is open-source and free to use under the MIT License.
