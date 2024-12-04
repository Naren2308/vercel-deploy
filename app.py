from flask import Flask, render_template, request, send_from_directory
import os
import datetime
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

app = Flask(__name__)
app.config['PREDICTIONS_FOLDER'] = 'static/predictions'

# Load model and scaler
model = load_model('lstm_model.h5')
scaler = joblib.load('scaler.pkl')

# Load and preprocess the dataset
data = pd.read_csv('NFLX_dataset.csv')
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data.set_index('Date', inplace=True)
data = data[['Close']]

# Feature Engineering (same as during training)
data['5_MA'] = data['Close'].rolling(window=5).mean()
data['30_MA'] = data['Close'].rolling(window=30).mean()
data['Volatility'] = data['Close'].rolling(window=5).std()
data['Returns'] = data['Close'].pct_change()
data.dropna(inplace=True)

# Scale the data with all features
data_scaled = scaler.transform(data[['Close', '5_MA', '30_MA', 'Volatility', 'Returns']])

# Define lookback period for the prediction
lookback = 120

# Custom Date Prediction Function
def predict_custom_dates(custom_dates):
    # Convert custom dates to datetime objects
    custom_dates_datetime = []
    for date in custom_dates:
        try:
            custom_dates_datetime.append(datetime.datetime.strptime(date, '%d-%m-%Y'))
        except ValueError:
            print(f"Invalid date format: {date}. Skipping this date.")

    # If no valid dates are provided, return early
    if not custom_dates_datetime:
        return [], []

    # Prepare input for prediction (last 120 data points)
    last_lookback_data = data_scaled[-lookback:]
    last_input = np.array([last_lookback_data])

    custom_predictions = []
    for _ in custom_dates_datetime:
        prediction = model.predict(last_input)
        custom_predictions.append(prediction[0, 0])

        # Update the input for the next prediction step
        prediction_padded = np.concatenate((prediction, np.zeros((1, 4))), axis=1)
        prediction_reshaped = prediction_padded.reshape(1, 1, 5)
        last_input = np.concatenate([last_input[:, 1:, :], prediction_reshaped], axis=1)

    custom_predictions = np.array(custom_predictions).reshape(-1, 1)
    custom_predictions = scaler.inverse_transform(np.concatenate((custom_predictions, np.zeros((custom_predictions.shape[0], 4))), axis=1))[:, 0]

    return custom_dates_datetime, custom_predictions

# Save prediction image
def save_prediction_image(predicted_dates, predicted_prices):
    if not os.path.exists(app.config['PREDICTIONS_FOLDER']):
        os.makedirs(app.config['PREDICTIONS_FOLDER'])

    timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    prediction_filename = f"prediction_{timestamp}.png"
    filepath = os.path.join(app.config['PREDICTIONS_FOLDER'], prediction_filename)

    # Plotting actual data and predictions
    plt.figure(figsize=(14, 8))
    last_month_data = data[-120:]
    last_month_actual = scaler.inverse_transform(data_scaled[-120:, :])[:, 0]
    plt.plot(last_month_data.index, last_month_actual, color='blue', linewidth=1.5, label='Actual Price (Last 120 Days)')
    plt.plot(predicted_dates, predicted_prices, color='purple', linestyle='-.', marker='o', markersize=6, linewidth=2, label='Forecasted Custom Dates')
    plt.title('Netflix Stock Price Prediction', fontsize=16, weight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Stock Price (USD)', fontsize=14)
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', fontsize=12)
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return prediction_filename

@app.route('/', methods=['GET', 'POST'])
def index():
    # Analyze the last 120 days
    last_120_days_data = data[-120:]
    first_price = last_120_days_data['Close'].iloc[0]
    last_price = last_120_days_data['Close'].iloc[-1]
    price_change = last_price - first_price
    percentage_change = (price_change / first_price) * 100

    insights = {
        'first_price': round(first_price, 2),
        'last_price': round(last_price, 2),
        'price_change': round(price_change, 2),
        'percentage_change': round(percentage_change, 2),
        'trend': "rose" if price_change > 0 else "fell"
    }

    if request.method == 'POST':
        date_input = request.form.get('dates')
        custom_dates = [date.strip() for date in date_input.split(",")]

        # Generate predictions
        predicted_dates, predicted_prices = predict_custom_dates(custom_dates)

        # Calculate price changes
        price_changes = [predicted_prices[i] - predicted_prices[i - 1] if i > 0 else 0 for i in range(len(predicted_prices))]
        percentage_changes = [(change / predicted_prices[i - 1] * 100 if i > 0 else 0) for i, change in enumerate(price_changes)]

        # Format dates and combine them with prices and changes
        formatted_dates = [date.strftime('%d-%m-%Y') for date in predicted_dates]
        predictions = list(zip(formatted_dates, predicted_prices, price_changes, percentage_changes))

        # Save the prediction image and get the filename
        image_url = save_prediction_image(predicted_dates, predicted_prices)

        # Pass the filename and predictions to be displayed on the front end
        return render_template('index.html', image_url=image_url, predictions=predictions, insights=insights)

    return render_template('index.html', image_url=None, insights=insights)

# Serve prediction images
@app.route('/static/predictions/<filename>')
def display_image(filename):
    return send_from_directory(app.config['PREDICTIONS_FOLDER'], filename)

if __name__ == '__main__':
    app.run()
