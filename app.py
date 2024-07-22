import streamlit as st
import spacy 
import torch
import time
import yfinance
nlp = spacy.load("en_core_web_sm")

from spacy.matcher import Matcher,PhraseMatcher
matcher = Matcher(nlp.vocab)
phrase_matcher = PhraseMatcher(nlp.vocab,attr="LOWER")

sector_patterns = [
    [{"LOWER": {"IN": ["technology",'tech', "finance", "healthcare", "consumer", "goods", "energy", "utilities", "materials", "industrials", "telecommunications", "real", "estate", "consumer", "services", "transportation", "agriculture", "media", "entertainment", "education", "government", "environmental", "investments"]}}]
]
# Define pattern for risk
risk_pattern = [
    {"POS": {"IN": ["ADJ", "NOUN"]}, "OP": "?"},  
    {"LOWER": {"IN": ["risk", "risky", "volatility", "volatile"]}}
]



investment_goal_patterns = [
    [{"LOWER": {"IN": ["retirement", "retirement", "retirement fund", "retirement savings"]}}],
    [{"LOWER": {"IN": ["education", "college", "higher education", "education funding", "saving for education"]}}],
    [{"LOWER": {"IN": ["buying a home", "down payment", "house", "home purchase", "real estate"]}}],
    [{"LOWER": {"IN": ["building wealth", "accumulating assets", "growing wealth", "wealth accumulation"]}}],
    [{"LOWER": {"IN": ["emergency fund", "emergency savings", "financial safety net", "unexpected expenses"]}}],
    [{"LOWER": {"IN": ["major purchases", "large purchases", "buying a car", "home renovations"]}}],
    [{"LOWER": {"IN": ["vacation", "travel", "holiday", "vacation planning", "saving for vacation"]}}],
    [{"LOWER": {"IN": ["debt repayment", "paying off debt", "reducing debt", "credit card balances", "loan repayment"]}}],
    [{"LOWER": {"IN": ["health", "medical expenses", "wellness", "preventive care"]}}],
    [{"LOWER": {"IN": ["charitable giving", "donations", "charity", "philanthropy", "donor-advised fund"]}}],
    [{"LOWER": {"IN": ["estate planning", "legacy", "inheritance", "financial security for heirs"]}}],
    [{"LOWER": {"IN": ["business investment", "starting a business", "entrepreneurial ventures", "business growth"]}}],
    [{"LOWER": {"IN": ["tax planning", "minimizing taxes", "tax liabilities", "tax-advantaged accounts"]}}],
    [{"LOWER": {"IN": ["legacy building", "impact investing", "supporting causes", "lasting impact"]}}],
    [{"LOWER": {"IN": ["growth",'grow', "capital appreciation", "value increase", "growing money", "investment growth"]}}]
]

word_to_category_mapping = {
    "retirement": ["retirement", "retirement fund", "retirement savings"],
    "education": ["education", "college", "higher education", "education funding", "saving for education"],
    "home purchase": ["buying a home", "down payment", "house", "home purchase", "real estate"],
    "wealth accumulation": ["building wealth", "accumulating assets", "growing wealth", "wealth accumulation"],
    "emergency savings": ["emergency fund", "emergency savings", "financial safety net", "unexpected expenses"],
    "major purchases": ["major purchases", "large purchases", "buying a car", "home renovations"],
    "vacation": ["vacation", "travel", "holiday", "vacation planning", "saving for vacation"],
    "debt repayment": ["debt repayment", "paying off debt", "reducing debt", "credit card balances", "loan repayment"],
    "healthcare": ["health", "medical expenses", "wellness", "preventive care"],
    "charitable giving": ["charitable giving", "donations", "charity", "philanthropy", "donor-advised fund"],
    "estate planning": ["estate planning", "legacy", "inheritance", "financial security for heirs"],
    "business investment": ["business investment", "starting a business", "entrepreneurial ventures", "business growth"],
    "tax planning": ["tax planning", "minimizing taxes", "tax liabilities", "tax-advantaged accounts"],
    "legacy building": ["legacy building", "impact investing", "supporting causes", "lasting impact"],
    "growth": ["growth", "grow", "capital appreciation", "value increase", "growing money", "investment growth"]
}

def map_word_to_category(word):
    for category, words in word_to_category_mapping.items():
        if word.lower() in [w.lower() for w in words]:
            return category
    return "others"
   


for pattern in investment_goal_patterns:
    matcher.add("INVESTMENT_GOAL", [pattern])



matcher.add("SECTOR", sector_patterns)
matcher.add("RISK", [risk_pattern])

def extract_entities(text):
    #named entities recognition and pattern matching
    doc = nlp(text)
    matches = matcher(doc)
    
    entities = {
        "ORG": [],
        "SECTOR": [],
        "RISK": [],
        "INVESTMENT_GOAL": [],
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
    
    for risk in entities['RISK']:
        if risk.lower() == 'risky' or risk.lower() == 'volatility' or risk.lower() == 'volatile':
            entities['RISK'] = 'high'
        elif 'low' in risk.lower():
            entities['RISK'] = 'low'
        elif 'high' in risk.lower():
            entities['RISK'] = 'high'
        else:
            entities['RISK'] = 'medium'
    # print(entities)
    entities['DATE'] = ' '.join(entities['DATE'])
    
    if entities['INVESTMENT_GOAL']:
        entities['INVESTMENT_GOAL'] = list(set(map_word_to_category(goal) for goal in entities['INVESTMENT_GOAL']))
    else:
        entities['INVESTMENT_GOAL'] = ['growth']
    
    if not entities['AMOUNT']:
        entities['AMOUNT'] = ['100000']
        
    if not entities['DATE']:
        entities['DATE'] = ['5 years']
    
    return entities

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
        st.markdown(response)
    
    st.session_state.messages.append({'role':'assistant','text':response})