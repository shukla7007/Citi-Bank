from flask import Flask, render_template, request, send_file,redirect, url_for
from flask import flash, get_flashed_messages
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from fpdf import FPDF
from datetime import datetime
import matplotlib
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

# Use non-interactive backend
matplotlib.use('Agg')

app = Flask(__name__)

app.secret_key = 'your_secret_key_here'  # Needed for flashing messages

# Load models for credit risk
model = joblib.load("random_forest_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Credit risk columns
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
num_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']


@app.route('/')
def home():
    return render_template('index.html')

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    phone = db.Column(db.String(20))
    address = db.Column(db.String(200))
    city = db.Column(db.String(100))
    zip = db.Column(db.String(20))
    id_type = db.Column(db.String(50))
    id_filename = db.Column(db.String(200))
    selfie_filename = db.Column(db.String(200))

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'uploads'

db.init_app(app)

# Run this once to create tables
with app.app_context():
    db.create_all()

@app.route('/onboarding-ekyc', methods=['POST'])
def submit_onboarding():
    # Get form data
    full_name = request.form['full_name']
    email = request.form['email']
    phone = request.form['phone']
    address = request.form['address']
    city = request.form['city']
    zip_code = request.form['zip']
    id_type = request.form['id_type']

    # Check if email already exists before touching files
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        flash("A user with this email already exists. Please use a different one.")
        return render_template('onboarding_ekyc.html',
                               full_name=full_name,
                               email=email,
                               phone=phone,
                               address=address,
                               city=city,
                               zip=zip_code)

    # Handle files
    id_file = request.files['id_upload']
    selfie = request.files['selfie_upload']

    id_filename = secure_filename(id_file.filename)
    selfie_filename = secure_filename(selfie.filename)

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    id_file.save(os.path.join(app.config['UPLOAD_FOLDER'], id_filename))
    selfie.save(os.path.join(app.config['UPLOAD_FOLDER'], selfie_filename))

    # Save to DB
    user = User(
        full_name=full_name,
        email=email,
        phone=phone,
        address=address,
        city=city,
        zip=zip_code,
        id_type=id_type,
        id_filename=id_filename,
        selfie_filename=selfie_filename
    )

    db.session.add(user)
    db.session.commit()

    return redirect(url_for('onboarding_complete'))

@app.route('/onboarding_complete')
def onboarding_complete():
    return render_template('onboarding_complete.html')

@app.route('/onboarding-ekyc')
def onboarding_ekyc():
    return render_template('onboarding_ekyc.html')


@app.route('/credit-risk')
def credit_risk():
    return render_template('credit_risk.html')


@app.route('/credit-risk-charts')
def credit_risk_charts():
    return render_template('charts.html')


@app.route('/tax-calculator', methods=['GET', 'POST'])
def tax_calculator():
    result = None

    if request.method == 'POST':
        try:
            income = float(request.form['income'])
            extra_income = float(request.form['extra_income'])
            age = request.form['age']
            deductions = float(request.form['deductions'])

            total_income = income + extra_income - deductions

            if total_income <= 0:
                result = {'error': 'Total taxable income must be greater than 0.'}
            else:
                if age == '<40':
                    tax_rate = 0.3 if total_income > 800000 else 0
                elif age == '>=40&<60':
                    tax_rate = 0.4 if total_income > 800000 else 0
                else:
                    tax_rate = 0.1 if total_income > 800000 else 0

                tax = (total_income - 800000) * tax_rate if tax_rate > 0 else 0
                tax = round(tax, 2)
                net_income = round(total_income - tax, 2)

                result = {
                    'tax': tax,
                    'net_income': net_income
                }

                record = {
                    "Gross Income": income,
                    "Extra Income": extra_income,
                    "Age Group": age,
                    "Deductions": deductions,
                    "Tax": tax,
                    "Net Income": net_income,
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                if os.path.exists("tax_history.csv"):
                    df = pd.read_csv("tax_history.csv")
                    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
                else:
                    df = pd.DataFrame([record])
                df.to_csv("tax_history.csv", index=False)

                generate_tax_chart(df)

        except Exception as e:
            result = {'error': f"An error occurred: {e}"}

    return render_template('tax.html', result=result)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_data = {
            'person_age': float(request.form['person_age']),
            'person_income': float(request.form['person_income']),
            'person_emp_length': float(request.form['person_emp_length']),
            'loan_amnt': float(request.form['loan_amnt']),
            'loan_int_rate': float(request.form['loan_int_rate']),
            'loan_percent_income': float(request.form['loan_percent_income']),
            'cb_person_cred_hist_length': float(request.form['cb_person_cred_hist_length']),
            'person_home_ownership': request.form['person_home_ownership'],
            'loan_intent': request.form['loan_intent'],
            'loan_grade': request.form['loan_grade'],
            'cb_person_default_on_file': request.form['cb_person_default_on_file']
        }

        input_df = pd.DataFrame([user_data])
        encoded_cats = encoder.transform(input_df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))
        scaled_nums = scaler.transform(input_df[num_cols])
        scaled_df = pd.DataFrame(scaled_nums, columns=num_cols)

        final_input = pd.concat([scaled_df, encoded_df], axis=1)

        prediction = model.predict(final_input)[0]
        proba = model.predict_proba(final_input)[0][prediction]
        confidence = round(proba * 100, 2)
        risk_label = "High Risk" if prediction == 1 else "Low Risk"

        record = {
            **user_data,
            "Prediction": prediction,
            "Risk Label": risk_label,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        if os.path.exists("y_pred.csv"):
            y_df = pd.read_csv("y_pred.csv")
            y_df = pd.concat([y_df, pd.DataFrame([record])], ignore_index=True)
        else:
            y_df = pd.DataFrame([record])
        y_df.to_csv("y_pred.csv", index=False)

        generate_visualization()

        return render_template(
            'result.html',
            prediction=risk_label,
            input_data=user_data,
            confidence=confidence
        )

    except Exception as e:
        return f"An error occurred: {e}"


@app.route('/download_csv')
def download_csv():
    try:
        return send_file("y_pred.csv", as_attachment=True)
    except Exception as e:
        return f"CSV download failed: {e}"


@app.route('/download_pdf')
def download_pdf():
    try:
        y_pred = pd.read_csv("y_pred.csv")
        last_row = y_pred.iloc[-1]

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="Credit Risk Prediction Report", ln=True, align='C')
        pdf.ln(5)
        pdf.cell(200, 10, txt=f"Generated on: {last_row['Timestamp']}", ln=True)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Latest Prediction Summary:", ln=True)
        pdf.set_font("Arial", size=12)

        for col in y_pred.columns:
            if col not in ["Prediction", "Timestamp"]:
                pdf.cell(200, 10, txt=f"{col.replace('_', ' ').capitalize()}: {last_row[col]}", ln=True)
        pdf.cell(200, 10, txt=f"Prediction: {last_row['Risk Label']}", ln=True)

        if os.path.exists("static/prediction_plot.png"):
            pdf.image("static/prediction_plot.png", x=30, w=150)

        pdf.output("report.pdf")
        return send_file("report.pdf", as_attachment=True)
    except Exception as e:
        return f"PDF generation failed: {e}"


@app.route('/download_tax_pdf')
def download_tax_pdf():
    try:
        df = pd.read_csv("tax_history.csv")
        last_row = df.iloc[-1]

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="Tax Calculation Report", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Generated on: {last_row['Timestamp']}", ln=True)
        pdf.ln(5)

        for col in df.columns:
            pdf.cell(200, 10, txt=f"{col}: {last_row[col]}", ln=True)

        if os.path.exists("static/tax_plot.png"):
            pdf.image("static/tax_plot.png", x=30, w=150)

        pdf.output("tax_report.pdf")
        return send_file("tax_report.pdf", as_attachment=True)
    except Exception as e:
        return f"Tax PDF generation failed: {e}"


@app.route('/download_tax_csv')
def download_tax_csv():
    try:
        return send_file("tax_history.csv", as_attachment=True)
    except Exception as e:
        return f"CSV download failed: {e}"


def generate_visualization():
    try:
        y_pred = pd.read_csv("y_pred.csv")
        predictions = y_pred["Prediction"].astype(int)
        low_risk = (predictions == 0).sum()
        high_risk = (predictions == 1).sum()

        plt.figure(figsize=(6, 6))
        plt.pie([low_risk, high_risk], explode=(0, 0.1), labels=['Low Risk', 'High Risk'],
                colors=['#28a745', '#dc3545'], autopct='%1.1f%%', shadow=True, startangle=140)
        plt.title('Credit Risk Prediction Distribution')
        plt.axis('equal')
        plt.tight_layout()

        if not os.path.exists("static"):
            os.makedirs("static")
        plt.savefig("static/prediction_plot.png")
        plt.close()
    except Exception as e:
        print("Visualization Error:", e)


def generate_tax_chart(df):
    try:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Timestamp', y='Tax', marker='o')
        plt.xticks(rotation=45)
        plt.title("Tax Over Time")
        plt.tight_layout()
        plt.savefig("static/tax_plot.png")
        plt.close()
    except Exception as e:
        print("Tax Chart Error:", e)


if __name__ == '__main__':
    app.run(debug=True)
