# Develop time series modelling AI algorithms capable of processing vast amounts of financial data and generating actionable investment insights.
#  Implement intuitive and user-friendly chat interfaces, leveraging NLP techniques for natural and fluid interaction.
#  Conduct rigorous testing and validation to verify the accuracy, reliability, and effectiveness of the AI investment advisor across diverse market conditions.

import streamlit as st
import spacy
import functools 
import numpy as np
# from transformers import DistilBertTokenizer,DistilBer
from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import pipeline
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
import matplotlib.pyplot as plt
import yfinance
import warnings
warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_md")
from spacy.matcher import Matcher,PhraseMatcher
matcher = Matcher(nlp.vocab)
phrase_matcher = PhraseMatcher(nlp.vocab,attr="LOWER")


sector_keywords = ["technology","finance", "healthcare", "consumer goods", "energy", "utilities", "materials", "industrials", "telecommunications", "real estate", "consumer services", "transportation", "agriculture", "media and entertainment", "government"]

def find_most_similar_sector(text):
    doc = nlp(text)
    max_similarity = 0
    most_similar_sector = ""
    for token in doc:
        for sector in sector_keywords:
            similarity = nlp(sector).similarity(token)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_sector = sector
    if max_similarity > 0.6:
        return most_similar_sector
    else:
        return "others"

def get_top_companies(sector, top_n=10):
    tickers = {
        "technology": ["AAPL", "MSFT", "GOOGL", "META", "INTC", "CSCO", "ORCL", "IBM", "NVDA", "ADBE"],
        "finance": ["JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "BK", "USB", "SCHW"],
        "healthcare": ["JNJ", "PFE", "ABBV", "MRK", "TMO", "GILD", "AMGN", "CVS", "ANTM", "BMY"],
        "consumer goods": ["PG", "KO", "PEP", "PM", "MO", "UL", "CL", "KMB", "GIS", "KHC"],
        "energy": ["XOM", "CVX", "COP", "EOG", "OXY", "PSX", "VLO", "MPC", "SLB", "HAL"],
        "utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PPL", "ED", "PEG"],
        "materials": ["LIN", "DD", "SHW", "APD", "NEM", "ECL", "PPG", "LYB", "MLM", "FCX"],
        "industrials": ["BA", "MMM", "GE", "HON", "UNP", "UPS", "RTX", "CAT", "LMT", "DE"],
        "telecommunications": ["VZ", "T", "TMUS", "CHTR", "CMCSA", "VOD", "AMX", "TEF", "ORAN", "TU"],
        "real estate": ["AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "SBAC", "WY", "AVB", "EQR"],
        "consumer services": ["AMZN", "BABA", "HD", "WMT", "LOW", "COST", "CVS", "TGT", "JD", "BKNG"],
        "transportation": ["FDX", "UPS", "DAL", "LUV", "UAL", "CSX", "NSC", "KSU", "UNP", "EXPD"],
        "agriculture": ["ADM", "BG", "CF", "CTVA", "DE", "MOS", "TSN", "FMC", "AGRO", "CRESY"],
        "media and entertainment": ["DIS", "NFLX", "CMCSA", "CHTR", "T", "VZ", "ATVI", "EA", "TTWO", "LYV"],
        "government": ["GD", "LMT", "NOC", "BAH", "SAIC", "LDOS", "BWXT", "TXT", "MANT", "PRSP"],
        "other":["AAPL", "MSFT", "GOOGL", "FB", "AMZN", "NVDA", "IBM", "TSLA", "CRM", "ADBE"]
    }
    
    if sector in tickers:
        selected_tickers = tickers[sector]
    else:
        return []
    return selected_tickers[:top_n]
    
    
def plot_stock_data(tickers):
    plt.figure(figsize=(10, 6))
    for ticker in tickers:
        data = fetch_data(ticker)
        if not data.empty:
            plt.plot(data.index, data['Close'], label=ticker)
    
    plt.title('Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return plt
@functools.lru_cache(maxsize=128)
def fetch_data(ticker):
    data = yfinance.download(ticker,start='2000-01-01')
    return data

risk_pattern = [
    # {"POS": {"IN": ["ADJ", "NOUN"]}, "OP": "?"},  
    {"POS": {"NOT_IN": ["PUNCT"]}, "OP": "?"},
    {"LOWER": {"IN": ["low", "medium", "high"]}, "OP": "?"},
    {"LOWER": {"IN": ["risk", "risky", "volatility", "volatile"]}}
]

def calculate_volatility(data):
    data['daily_return'] = data['Close'].pct_change()
    volatility = data['daily_return'].std()* (252 ** 0.5)
    return volatility

def calculate_net_profit(data):
    data['Year'] = data.index.year
    yearly_prices = data['Close'].resample('Y').last()
    yearly_profit = yearly_prices.pct_change().dropna()*100
    return yearly_profit

def plot_volatility(tickers,risk='medium'):
    risk_threshold = {
        'low': 0.2,
        'medium': 0.35,
        'high': 0.5
    }
    
    ticker_data = []
    plt.figure(figsize=(10, 6))
    for ticker in tickers:
        data = fetch_data(ticker)
        try:
            company_name = yfinance.Ticker(ticker).info['longName']
        except KeyError:
            ticker_info = yfinance.Ticker(ticker).info
            company_name = ticker_info.get('longName',ticker)
        if not data.empty:
            volatility = calculate_volatility(data)
            plt.bar(ticker, volatility,label=company_name)
            if volatility < risk_threshold[risk]:
                ticker_data.append(ticker)
            
    if risk in risk_threshold:
        plt.axhline(y=risk_threshold[risk], color='r', linestyle='--', label=f'{risk.capitalize()} Risk Threshold')
        
    plt.title('Volatility of Stocks')
    plt.xlabel('Stock Ticker')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt, ticker_data
    


from sentence_transformers import SentenceTransformer, util

# Load the pre-trained sentence transformer model
model = SentenceTransformer('all-mpnet-base-v2')

investment_goals_mapping = {
    "retirement": ["retirement", "retirement fund", "retirement savings"],
    "education": ["education", "college", "higher education", "education funding", "saving for education"],
    "home purchase": ["buying a home", "down payment", "house purchase", "real estate investment"],
    "wealth accumulation": ["building wealth", "asset accumulation", "wealth growth", "capital appreciation"],
    "emergency savings": ["emergency fund", "emergency savings", "financial safety net", "unexpected expenses"],
    "major purchases": ["major purchases", "large purchases", "buying a car", "home renovations"],
    "vacation": ["vacation", "travel", "holiday planning", "saving for vacation"],
    "debt repayment": ["debt repayment", "paying off debt", "debt reduction", "loan repayment"],
    "healthcare": ["healthcare", "medical expenses", "health savings", "wellness"],
    "charitable giving": ["charitable giving", "donations", "charity", "philanthropy"],
    "estate planning": ["estate planning", "inheritance planning", "financial security for heirs"],
    "business investment": ["business investment", "starting a business", "entrepreneurship", "business growth"],
    "tax planning": ["tax planning", "tax minimization", "tax liability management"],
    "legacy building": ["legacy building", "impact investing", "supporting causes", "creating a lasting impact"],
    "growth": ["investment growth", "capital appreciation", "value increase", "wealth growth"]
}

def find_most_similar_investment_goal(text):
    # Encode the input text
    text_embedding = model.encode(text, convert_to_tensor=True)
    
    max_similarity = 0
    most_similar_goal = "others"
    
    for goal, keywords in investment_goals_mapping.items():
        for keyword in keywords:
            # Encode the keyword
            keyword_embedding = model.encode(keyword, convert_to_tensor=True)
            
            # Compute the cosine similarity
            similarity = util.pytorch_cos_sim(text_embedding, keyword_embedding).item()
            
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_goal = goal
    
    if max_similarity > 0.5:
        return most_similar_goal
    else:
        return "others"

def map_word_to_category(word):
    for category, words in investment_goals_mapping.items():
        if word.lower() in [w.lower() for w in words]:
            return category
    return "others"


matcher.add("RISK", [risk_pattern])

def extract_entities(text):
    #named entities recognition and pattern matching
    doc = nlp(text)
    matches = matcher(doc)
    
    entities = {
        "ORG": [],
        "RISK": [],
        "AMOUNT": [],
        "DATE": []
    }
    
    # Extract named entities for organizations and amounts
    for ent in doc.ents:
        if ent.label_ == "ORG":
            entities["ORG"].append(ent.text)
        elif ent.label_ == "MONEY":
            entities["AMOUNT"].append(ent.text)
        elif ent.label_ == 'DATE':
            entities["DATE"].append(ent.text)
        
    
    # Extract custom matched entities
    for match_id, start, end in matches:
        span = doc[start:end]
        entity_label = nlp.vocab.strings[match_id]
        entities[entity_label].append(span.text)
    
    if entities['RISK']:
        risk_terms = set(entities['RISK'])
        risk_level = 'medium'
        for term in risk_terms:
            if 'high' in term:
                risk_level = 'high'
                break
            elif 'medium' in term:
                risk_level = 'medium'
            elif 'low' in term:
                risk_level = 'low'
        entities['RISK'] = risk_level
    else:
        entities['RISK'] = 'medium'
    # print(entities)
    entities['DATE'] = ' '.join(entities['DATE'])

    entities['SECTOR'] = find_most_similar_sector(text)
    entities['INVESTMENT_GOAL'] = find_most_similar_investment_goal(text)
    
    if not entities['AMOUNT']:
        entities['AMOUNT'] = ['100000']
        
    print(entities['AMOUNT'])
    if not entities['DATE']:
        entities['DATE'] = ['5 years']

    return entities

def fundamental_analysis(ticker):
    stock = yfinance.Ticker(ticker)
    info = stock.info
    return {
        'Ticker': ticker,
        'P/E Ratio': info.get('trailingPE', 'N/A'),
        'Earnings Growth': info.get('earningsGrowth', 'N/A'),
        'Dividend Yield': info.get('dividendYield', 'N/A'),
        'Debt-to-Equity': info.get('debtToEquity', 'N/A'),
        'ROE': info.get('returnOnEquity', 'N/A')
    }

def technical_analysis(ticker):
    data = fetch_data(ticker)
    if data.empty:
        return {}

    ma50 = data['Close'].rolling(window=50).mean().iloc[-1]
    ma200 = data['Close'].rolling(window=200).mean().iloc[-1]
    rsi = calculate_rsi(data['Close'])
    macd = calculate_macd(data['Close'])
    
    return {
        'Ticker': ticker,
        'MA50': ma50,
        'MA200': ma200,
        'RSI': rsi,
        'MACD': macd
    }

def calculate_rsi(close_prices, period=14):
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).iloc[-1]

def calculate_macd(close_prices, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = close_prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close_prices.ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd.iloc[-1] - signal.iloc[-1]

def evaluate_stocks(tickers):
    fundamental_results = [fundamental_analysis(ticker) for ticker in tickers]
    technical_results = [technical_analysis(ticker) for ticker in tickers]
    combined_results = []
    summaries = []
    for fundamental, technical in zip(fundamental_results, technical_results):
        score = 0
        if fundamental['P/E Ratio'] != 'N/A' and fundamental['P/E Ratio'] < 20:
            score += 1
        if fundamental['Earnings Growth'] != 'N/A' and fundamental['Earnings Growth'] > 0.1:
            score += 1
        if fundamental['Debt-to-Equity'] != 'N/A' and fundamental['Debt-to-Equity'] < 1:
            score += 1
        if fundamental['ROE'] != 'N/A' and fundamental['ROE'] > 0.15:
            score += 1
        if technical.get('MACD', 0) > 0:
            score += 1

        combined_results.append({'Ticker': fundamental['Ticker'], 'Score': score, 'Fundamental': fundamental, 'Technical': technical})
        summaries.append(generate_summary(fundamental,technical))
    combined_results = sorted(combined_results, key=lambda x: x['Score'], reverse=True)
    
    best_companines = combined_results[0]['Ticker']
    
    return combined_results,best_companines,summaries

def predict_stock_prices(ticker,time = 5,amount=100000):
    data = fetch_data(ticker)
    data = data['Close']
    np.random.seed(42) #creating random noise to account for unexpected situations
    
    model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=12)
    fit = model.fit()
    forecast = fit.forecast(int(365*time))
    noise = np.random.normal(0,forecast.std()*0.2,len(forecast))
    forecast_with_noise = forecast+noise
    
    last_date = data.index[-1]
    forecast_dates = pd.date_range(start=last_date, periods=len(forecast_with_noise))
    # print(len(forecast_dates),len(forecast_with_noise))
    # print(forecast_dates)
    plt.figure(figsize=(10, 6))
    # plt.plot(data.index, data, label='Historical')
    plt.plot(forecast_dates, forecast_with_noise, label='Forecast',color='r')
    plt.title(f'Stock Price Prediction for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if not forecast_with_noise.empty:
        initial_price = data.iloc[-1].item()
        final_forecasted_price = forecast_with_noise.iloc[-1].item()
        print(amount)
        
        net_profit = (final_forecasted_price - initial_price) * (float(amount) / initial_price)
        print(net_profit)
        print(initial_price,final_forecasted_price)
        expected_returns = net_profit / amount
    else:
        net_profit = 0
        expected_returns = 0
    
    return plt.gcf(),net_profit,expected_returns
    

def generate_summary(fundamental, technical):
    model_name = 't5-base'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    summarizer_model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Combine fundamental and technical data into a single input string
    input_text = (
        f"The stock {fundamental['Ticker']} has the following fundamental metrics: "
        f"a P/E Ratio of {fundamental['P/E Ratio']}, "
        f"an Earnings Growth of {fundamental['Earnings Growth']}, "
        f"a Dividend Yield of {fundamental['Dividend Yield']}, "
        f"a Debt-to-Equity ratio of {fundamental['Debt-to-Equity']}, "
        f"and a Return on Equity (ROE) of {fundamental['ROE']}. "
        f"In terms of technical indicators, it has a 50-Day Moving Average of {technical['MA50']}, "
        f"a 200-Day Moving Average of {technical['MA200']}, "
        f"an RSI of {technical['RSI']}, "
        f"and a MACD of {technical['MACD']}."
    )

    # Tokenize the input text
    inputs = tokenizer.encode(f"explain the financial analysis of Company: {fundamental['Ticker']}" + input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the summary
    summary_ids = summarizer_model.generate(inputs, max_length=250, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the output tokens into a string
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

import re
def convert_date(date_str):
    years_pattern = re.compile(r'(\d+)\s*years?')
    months_pattern = re.compile(r'(\d+)\s*months?')
    days_pattern = re.compile(r'(\d+)\s*days?')
    
    years_match = years_pattern.match(date_str)
    if years_match:
        return float(years_match.group(1))
    months_match = months_pattern.match(date_str)
    if months_match:
        return float(months_match.group(1))/12
    days_match = days_pattern.match(date_str)
    if days_match:
        return float(days_match.group(1))/365
    return 5.0    

st.title('AI Investment Chatbot 🤖')

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['text'])

prompt =st.chat_input("")
if prompt:
    with st.chat_message('user'):
        st.markdown(prompt)
    
    st.session_state.messages.append({'role':'user','text':prompt})
    
    response = extract_entities(prompt)
    with st.chat_message('assistant'):
        # st.markdown(response)
        sector = response.get('SECTOR','other')
        risk = response.get('RISK','medium')
        time = response.get('DATE','5 years')
        time = convert_date(time)
        amount = response.get('AMOUNT','100000')[0]
        
        def extract_amount(amount_str):
            # Regular expression to extract the numeric value
            match = re.search(r'\d+', amount_str)
            if match:
                return int(match.group(0))
            else:
                # Default to 100000 if no numeric value is found
                return 100000
        amount = extract_amount(amount)
        tickers = get_top_companies(sector)
        # print(time)
        # prompt_for_summarizer = 
        st.write('Here is the plot of closing prices of the stock')
        st.pyplot(plot_stock_data(tickers))
        
        st.write('Here is the volatility plot based on your risk preference:')
        st.pyplot(plot_volatility(tickers,risk)[0])
        results,best_comp,summaries = evaluate_stocks(plot_volatility(tickers,risk)[1])
        # for result in results:
        #     st.write(f"**Ticker**: {result['Ticker']}, **Score**: {result['Score']}")
        #     st.write("Fundamental Analysis:", result['Fundamental'])
        #     st.write("Technical Analysis:", result['Technical'])
        #     st.write("---")
        
        for summary in summaries:
            st.markdown(summary)
        st.write(f"Best Company suited for your needs from my analysis is : {best_comp}")
        
        st.write('Here is the stock price prediction for the best company:')
        # st.pyplot(predict_stock_prices(best_comp,time))
        
        plot,net_profit, expected_returns = predict_stock_prices(best_comp,time,amount)
        st.pyplot(plot)

        st.write(f"Net Profit: ${net_profit:.2f}")
        st.write(f"Expected Returns: {expected_returns:.2%}")
        
        assistant_content = f"""
        Sector: {sector}
        Risk: {risk}
        Time: {time:.2f} years
        Amount: USD {amount}
        Best Company: {best_comp}
        Net Profit: ${net_profit:.2f}
        Expected Returns: {expected_returns:.2%}
        """
        st.session_state.messages.append({'role': 'assistant', 'text': assistant_content})
        
    
    # st.session_state.messages.append({'role':'assistant','text':response})