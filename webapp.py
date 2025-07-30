import streamlit as st
import pandas as pd
import pickle
import os
import joblib

# --- Feature definitions ---
categorical_features = ['Coin', 'Side', 'Direction', 'classification']
numerical_features = ['Execution Price', 'Size Tokens', 'Size USD', 'Start Position', 'Fee']

categorical_values = {
    'Coin': [
        '@107', 'AAVE', 'DYDX', 'AIXBT', 'GMX', 'EIGEN', 'HYPE', 'SOL', 'SUI', 'DOGE',
        'ETH', 'kPEPE', 'TRUMP', 'ONDO', 'ENA', 'LINK', 'XRP', 'S', 'BNB', 'BERA', 'WIF',
        'LAYER', 'MKR', 'KAITO', 'IP', 'JUP', 'USUAL', 'ADA', 'BTC', 'PURR/USDC', 'ZRO',
        '@7', '@19', '@21', '@44', '@48', '@11', '@15', '@46', '@61', '@28', '@45', '@9',
        '@41', '@38', 'kSHIB', 'GRASS', 'TAO', 'AVAX', '@2', '@6', '@8', '@10', '@12', '@16',
        '@17', '@35', '@26', '@24', '@32', '@29', '@31', '@33', '@34', '@36', '@37', '@47',
        '@53', '@74', 'RUNE', 'CANTO', 'NTRN', 'BLUR', 'ZETA', 'MINA', 'MANTA', 'RNDR',
        'WLD', 'kBONK', 'ALT', 'INJ', 'STG', 'ZEN', 'MAVIA', 'PIXEL', 'ILV', 'FET', 'STRK',
        'CAKE', 'STX', 'ACE', 'PENDLE', 'AR', 'XAI', 'APE', 'MEME', 'NEAR', 'SEI', 'FTM',
        'MYRO', 'BIGTIME', 'IMX', 'BADGER', 'POLYX', 'OP', 'TNSR', 'MAV', 'TIA', 'MERL',
        'TON', 'PURR', 'ME', 'CRV', 'BRETT', 'CHILLGUY', 'MOODENG', 'VIRTUAL', 'COMP',
        'FARTCOIN', 'AI16Z', 'GRIFFAIN', 'ZEREBRO', 'SPX', 'MELANIA', 'PENGU', 'JELLY',
        'VVV', 'VINE', 'TST', 'ARK', 'YGG', 'POPCAT', 'NIL', 'MOVE', 'BABY', 'RENDER',
        'PROMPT', 'WCT', 'OGN', 'HYPER', 'ZORA', 'BIO', 'INIT', 'TURBO', 'ARB', '@142',
        'JTO', 'PYTH', 'MATIC', 'HPOS', 'FXS', 'FIL', 'SHIA', 'PEOPLE', 'UNI', 'SUSHI',
        'LOOM', 'USTC', 'RLB', 'ETC', 'GAS', 'BANANA', 'UNIBOT', 'CYBER', 'GMT', 'ENS',
        'DYM', 'ETHFI', '@4', '@3', '@20', '@14', '@51', '@125', '@109', '@85', '@59', '@23',
        '@78', 'GOAT', 'LDO', 'PNUT', '@138', '@151', '@18', '@25', '@42', '@13', '@40',
        '@39', '@30', '@103', '@100', '@135', '@116', '@147', '@152', '@153', 'ANIME', 'W',
        'ORDI', 'IO', 'GALA', 'AI', 'NEIROETH', 'SAND', 'BOME', 'MEW', 'SUPER', 'ALGO',
        'HBAR', 'APT', 'BLAST', 'kNEIRO', 'ATOM', 'DOT', 'kFLOKI', 'TRX', 'FTT', '@1',
        '@83', '@124', '@112', '@95', 'LISTA', 'LTC', 'RSR', 'MNT', '@113', 'ZK', '@117',
        '@86', '@68', '@63', 'MORPHO', 'OM', 'REZ', 'REQ', '@49', 'BNT', 'SCR', 'IOTA',
        '@93', '@114', '@123', 'PAXG'
    ],
    'Side': ['BUY', 'SELL'],
    'Direction': ['Buy', 'Sell', 'Open Long', 'Close Long', 'Spot Dust Conversion', 'Open Short',
                  'Close Short', 'Long > Short', 'Short > Long', 'Auto-Deleveraging',
                  'Liquidated Isolated Short', 'Settlement'],
    'classification': ['Extreme Greed', 'Extreme Fear', 'Fear', 'Greed', 'Neutral']
}

# --- Custom CSS for background, card, and responsive design ---
st.markdown("""
    <style>
        html, body, .stApp {
            height: 100%;
            min-height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        }
        # .main-card {
        #     background: white !important;
        #     border-radius: 18px;
        #     box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        #     padding: 2.5rem 2rem 2rem 2rem;
        #     margin-top: 2rem;
        #     margin-bottom: 2rem;
        #     max-width: 600px;
        #     margin-left: auto;
        #     margin-right: auto;
        # }
        .section-divider {
            border: none;
            border-top: 2px solid #764ba2;
            margin: 2rem 0 1.5rem 0;
        }
        .footer {
            color: #fff;
            text-align: center;
            margin-top: 2rem;
            margin-bottom: 0.5rem;
            font-size: 1.1em;
            opacity: 0.8;
        }
        @media (max-width: 700px) {
            .main-card {
                padding: 1.2rem 0.5rem 1.2rem 0.5rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# --- App title and description ---
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 1rem; text-align: center; max-width: 600px; margin: 2rem auto 0 auto;">
        <h1 style="color: white; margin-bottom: 0.5em;">🔮 Trade Prediction </h1>
    </div>
    """, unsafe_allow_html=True
)

# --- Main card container (only one, not empty) ---
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    with st.form("trade_form"):
        st.markdown('<span style="font-size:1.5em;">🎯 <b style="color:#000000;">Categorical Features</b></span>', unsafe_allow_html=True)
        cat_inputs = {}
        for feature in categorical_features:
            options = ["--Select--"] + categorical_values[feature]
            cat_inputs[feature] = st.selectbox(
                f"Select {feature}", options, key=feature
            )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown('<span style="font-size:1.5em;">📈 <b style="color:#000000;">Numerical Features</b></span>', unsafe_allow_html=True)
        num_inputs = {}
        for feature in numerical_features:
            num_inputs[feature] = st.number_input(
                f"Enter {feature}", value=None, format="%.2f", key=feature, placeholder="None"
            )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮 Predict Trade Outcome", use_container_width=True)

    # Validation: check if any dropdown is still "--Select--"
    if submitted:
        if any(v == "--Select--" for v in cat_inputs.values()):
            st.error("Please select a value for all categorical features.")
        else:
            # --- DataFrame creation ---
            data = {**cat_inputs, **num_inputs}
            df = pd.DataFrame([data])[categorical_features + numerical_features]

            # --- Model loading ---
            preprocessor_path = os.path.join('outputs', 'preprocessor.pkl')
            model_path = os.path.join('outputs', 'xgb_model.pkl')

            if not os.path.exists(preprocessor_path):
                st.error("Preprocessor model not found. Please ensure preprocessor.pkl exists in the outputs folder.")
                st.stop()
            if not os.path.exists(model_path):
                st.error("Prediction model not found. Please ensure xgb_model.pkl exists in the outputs folder.")
                st.stop()

            with open(preprocessor_path, 'rb') as f:
                preprocessor = joblib.load(f)
            with open(model_path, 'rb') as f:
                model = joblib.load(f)

            # --- Prediction ---
            features_array = preprocessor.transform(df[categorical_features + numerical_features])
            cat_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
            all_cols = list(cat_columns) + numerical_features
            features = pd.DataFrame(features_array, columns=all_cols)

            prediction = model.predict(features)[0]

            # --- Result display ---
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            if prediction == 1:
                st.markdown(
                    '<div style="background: linear-gradient(135deg, #28a745, #20c997); color: white; padding: 1.5rem; border-radius: 1rem; text-align: center; font-size: 1.3em; margin-top: 1.5rem;">'
                    '🎉 <b>Profit (Success)</b> - This trade is predicted to be <b>profitable</b>!</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div style="background: linear-gradient(135deg, #dc3545, #fd7e14); color: white; padding: 1.5rem; border-radius: 1rem; text-align: center; font-size: 1.3em; margin-top: 1.5rem;">'
                    '📉 <b>Loss (Failure)</b> - This trade is predicted to result in a <b>loss</b>.</div>',
                    unsafe_allow_html=True
                )

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("#### Prediction Details")
            st.write(f"**Prediction Value:** {int(prediction)}")
            st.write(f"**DataFrame Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.markdown("#### Submitted Data")
            st.json(data)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown(
    "<div class='footer'>Made by Ayush Dhoble</div>",
    unsafe_allow_html=True
)