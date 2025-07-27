import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .stSelectbox > div[data-baseweb="select"] > div {
            background-color: #1c1c1c;
            color: white;
        }
        .stSelectbox div[data-baseweb="select"] span {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #ff4b4b !important;
    }
    
    .stProgress > div > div > div {
        background-color: rgba(255, 75, 75, 0.1) !important;
    }
    
    .circular-progress {
        position: relative;
        width: 120px;
        height: 120px;
        margin: 20px auto;
        border-radius: 50%;
        background: #2a2a2a;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .circular-progress svg {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        transform: rotate(-90deg);
    }
    
    .circular-progress svg circle {
        fill: none;
        stroke-width: 8;
        stroke-linecap: round;
    }
    
    .bg-circle {
        stroke: #444;
    }
    
    .progress-circle {
        stroke: #ff4b4b;
        stroke-dasharray: 283;
        stroke-dashoffset: 283;
        transition: stroke-dashoffset 1s ease-in-out;
    }
    
    .progress-content {
        position: relative;
        z-index: 1;
        text-align: center;
        color: white;
    }
    
    .progress-title {
        font-size: 12px;
        margin-bottom: 2px;
        color: #ff4b4b;
        font-weight: bold;
    }
    
    .progress-value {
        font-size: 18px;
        font-weight: bold;
        color: white;
    }
    
    .progress-icon {
        font-size: 14px;
        margin-bottom: 2px;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class DiabetesPredictor:
    def __init__(self):
        self.theta = None
        self.x1_mean = None
        self.x1_std = None
        self.cost_history = []
        self.is_trained = False
        
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def train_model(self, df, lr=0.01, steps=1000):
        # Features and target
        x1 = df.drop('Outcome', axis=1).values
        y = df['Outcome'].values.reshape(-1, 1)
        
        # Normalize features
        self.x1_mean = x1.mean(axis=0)
        self.x1_std = x1.std(axis=0)
        x1_norm = (x1 - self.x1_mean) / self.x1_std
        
        # Add bias term
        x = np.hstack((np.ones((x1_norm.shape[0], 1)), x1_norm))
        m, n = x.shape
        
        # Initialize parameters
        self.theta = np.zeros((n, 1))
        self.cost_history = []
        
        # Gradient Descent with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(steps):
            z = x @ self.theta
            h = self.sigmoid(z)
            gradient = (1 / m) * (x.T @ (h - y))
            self.theta = self.theta - (lr * gradient)
            
            # Cost function
            eps = 1e-8
            cost = (-1 / m) * np.sum((y * np.log(h + eps)) + ((1 - y) * np.log(1 - h + eps)))
            self.cost_history.append(cost)
            
            # Update progress
            if i % 100 == 0:
                progress_bar.progress((i + 1) / steps)
                status_text.text(f'Training Progress: Step {i+1}/{steps} | Cost: {cost:.4f}')
        
        progress_bar.progress(1.0)
        status_text.text('Training Complete! ✅')
        
        self.is_trained = True
        return x, y
    
    def predict(self, user_input):
        if not self.is_trained:
            return None
        
        # Normalize user input
        user_input_norm = (user_input - self.x1_mean) / self.x1_std
        user_input_with_bias = np.hstack(([1], user_input_norm))
        
        # Predict
        z = np.dot(user_input_with_bias, self.theta)
        probability = self.sigmoid(z)[0]
        
        return probability, z[0]

# Initialize the app
st.markdown('<h1 class="main-header">🧬 Diabetes Prediction</h1>', unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = DiabetesPredictor()

# Initialize with default dataset
try:
    df = pd.read_csv('diabetes.csv')
except FileNotFoundError:
    # Create a sample dataset if diabetes.csv is not found
    np.random.seed(42)
    n_samples = 768
    df = pd.DataFrame({
        'Pregnancies': np.random.randint(0, 17, n_samples),
        'Glucose': np.random.normal(120, 30, n_samples).clip(0, 300),
        'BloodPressure': np.random.normal(70, 20, n_samples).clip(0, 200),
        'SkinThickness': np.random.normal(20, 15, n_samples).clip(0, 100),
        'Insulin': np.random.normal(80, 100, n_samples).clip(0, 900),
        'BMI': np.random.normal(32, 7, n_samples).clip(15, 70),
        'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, n_samples),
        'Age': np.random.randint(21, 81, n_samples),
        'Outcome': np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
    })

# Train model automatically
if not st.session_state.predictor.is_trained:
    with st.spinner("Loading and training model..."):
        x, y = st.session_state.predictor.train_model(df, lr=0.01, steps=1000)
        
        # Calculate training metrics
        train_prob = st.session_state.predictor.sigmoid(x @ st.session_state.predictor.theta)
        y_pred = (train_prob >= 0.5).astype(int)
        
        # Calculate metrics
        TP = np.sum((y_pred == 1) & (y == 1))
        TN = np.sum((y_pred == 0) & (y == 0))
        FP = np.sum((y_pred == 1) & (y == 0))
        FN = np.sum((y_pred == 0) & (y == 1))
        
        accuracy = (TP + TN) / len(y)
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Store metrics in session state
        st.session_state.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
        }

# Load data
@st.cache_data
def load_data():
    return df

df = load_data()

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📈 Dataset Overview")
    
    # Dataset info
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("📋 Total Records", len(df))
    with col_info2:
        st.metric("📊 Features", len(df.columns) - 1)
    with col_info3:
        diabetic_count = df['Outcome'].sum()
        st.metric("🔴 Diabetic Cases", f"{diabetic_count} ({diabetic_count/len(df)*100:.1f}%)")
    
    # Display dataset
    with st.expander("👀 View Dataset"):
        st.dataframe(df.head(10), use_container_width=True)
        
    # Dataset statistics
    with st.expander("📊 Dataset Statistics"):
        st.write(df.describe())

with col2:
    st.subheader("🎯 Model Status")
    st.success("✅ Model Ready")
    st.info("Model has been trained and is ready for predictions.")

# Model Performance Section
st.markdown("---")
st.subheader("📊 Model Performance")

# Function to create circular progress chart
def create_circular_progress(value, title, icon):
    fig = go.Figure(go.Pie(
        values=[value, 1-value],
        labels=['Progress', 'Remaining'],
        hole=0.7,
        marker_colors=['#ff4b4b', '#2a2a2a'],
        textinfo='none',
        hoverinfo='none',
        showlegend=False
    ))
    
    fig.update_layout(
        annotations=[
            dict(text=f'{icon}<br>{title}<br><b>{value:.3f}</b>', 
                 x=0.5, y=0.5, font_size=28, showarrow=False,
                 font=dict(color='white'))
        ],
        width=250, height=250,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Metrics display with circular progress bars
col1, col2, col3, col4 = st.columns(4)
metrics = st.session_state.metrics

with col1:
    fig_acc = create_circular_progress(metrics['accuracy'], 'Accuracy', '🎯')
    st.plotly_chart(fig_acc, use_container_width=True, config={'displayModeBar': False})

with col2:
    fig_prec = create_circular_progress(metrics['precision'], 'Precision', '🎪')
    st.plotly_chart(fig_prec, use_container_width=True, config={'displayModeBar': False})

with col3:
    fig_rec = create_circular_progress(metrics['recall'], 'Recall', '🔍')
    st.plotly_chart(fig_rec, use_container_width=True, config={'displayModeBar': False})

with col4:
    fig_f1 = create_circular_progress(metrics['f1'], 'F1-Score', '⚖️')
    st.plotly_chart(fig_f1, use_container_width=True, config={'displayModeBar': False})

# Visualizations
col1, col2 = st.columns(2)

with col1:
    # Cost function plot
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(
        y=st.session_state.predictor.cost_history,
        mode='lines',
        name='Training Cost',
        line=dict(color='#74b9ff', width=2)
    ))
    fig_cost.update_layout(
        title="📉 Training Cost Over Time",
        xaxis_title="Iteration",
        yaxis_title="Cost",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig_cost, use_container_width=True)

with col2:
    # Confusion Matrix
    fig_cm = px.imshow(
        [[metrics['TN'], metrics['FP']], [metrics['FN'], metrics['TP']]],
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Non-Diabetic', 'Diabetic'],
        y=['Non-Diabetic', 'Diabetic'],
        color_continuous_scale='Blues',
        title="🎯 Confusion Matrix"
    )
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            value = [[metrics['TN'], metrics['FP']], [metrics['FN'], metrics['TP']]][i][j]
            fig_cm.add_annotation(
                x=j, y=i,
                text=str(int(value)),
                showarrow=False,
                font=dict(color="white" if value > max(metrics['TN'], metrics['TP'], metrics['FP'], metrics['FN'])/2 else "black", size=16)
            )
    
    fig_cm.update_layout(height=500)
    st.plotly_chart(fig_cm, use_container_width=True)


# Insert logistic regression sigmoid image here
st.markdown("### 📉 Logistic Regression Sigmoid Visualization")
st.image("images/train.png", caption="Actual vs Predicted Class with Sigmoid Curve", use_container_width=True)


# Prediction Section
st.markdown("---")
st.subheader("🔮 Make Predictions")

# User input form
with st.form("prediction_form"):
    st.markdown("### Enter Your Health Information:")
    
    col1, col2 = st.columns(2)
    
    feature_names = df.columns[:-1].tolist()
    user_inputs = {}
    
    for i, feature in enumerate(feature_names):
        if feature == 'DiabetesPedigreeFunction':
            with col1 if i % 2 == 0 else col2:
                st.markdown("**Family History of Diabetes:**")
                family_history = st.selectbox(
                    "Select your family history:",
                    [
                        "No one in my family has diabetes",
                        "One parent or sibling has diabetes", 
                        "Multiple parents or siblings have diabetes",
                        "Only extended family has diabetes"
                    ],
                    key="family_history"
                )
                
                health_conscious = st.selectbox(
                    "Are you and your family health-conscious?",
                    ["Yes", "No"],
                    key="health_conscious"
                )
                
                # Calculate diabetes pedigree function
                if family_history == "No one in my family has diabetes":
                    pedigree_value = 0.2
                elif family_history == "One parent or sibling has diabetes":
                    pedigree_value = 0.5
                elif family_history == "Multiple parents or siblings have diabetes":
                    pedigree_value = 1.0
                else:  # Extended family
                    pedigree_value = 0.4
                
                if health_conscious == "Yes":
                    pedigree_value *= 0.7
                    
                user_inputs[feature] = pedigree_value
                
        else:
            with col1 if i % 2 == 0 else col2:
                # Set appropriate ranges and defaults for each feature
                if feature == 'Pregnancies':
                    value = st.number_input(f"{feature}:", min_value=0, max_value=20, value=0, key=feature)
                elif feature == 'Glucose':
                    value = st.number_input(f"{feature} (mg/dL):", min_value=0, max_value=300, value=120, key=feature)
                elif feature == 'BloodPressure':
                    value = st.number_input(f"{feature} (mmHg):", min_value=0, max_value=200, value=80, key=feature)
                elif feature == 'SkinThickness':
                    value = st.number_input(f"{feature} (mm):", min_value=0, max_value=100, value=20, key=feature)
                elif feature == 'Insulin':
                    value = st.number_input(f"{feature} (μU/mL):", min_value=0, max_value=900, value=100, key=feature)
                elif feature == 'BMI':
                    value = st.number_input(f"{feature}:", min_value=0.0, max_value=70.0, value=25.0, step=0.1, key=feature)
                elif feature == 'Age':
                    value = st.number_input(f"{feature}:", min_value=1, max_value=120, value=30, key=feature)
                else:
                    value = st.number_input(f"{feature}:", value=0.0, key=feature)
                
                user_inputs[feature] = value
    
    submitted = st.form_submit_button("🔮 Predict Diabetes Risk", type="primary", use_container_width=True)
    
    if submitted:
        # Make prediction
        user_values = np.array([user_inputs[feature] for feature in feature_names])
        probability, z_value = st.session_state.predictor.predict(user_values)
        
        # Display prediction
        if probability >= 0.5:
            st.markdown(f"""
            <div class="prediction-box" style="background: linear-gradient(135deg, #e17055 0%, #d63031 100%);">
                <h2>🔴 High Risk of Diabetes</h2>
                <h3>Probability: {probability:.1%}</h3>
                <p>The model indicates a higher likelihood of diabetes risk.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box" style="background: linear-gradient(135deg, #00b894 0%, #00a085 100%);">
                <h2>🟢 Low Risk of Diabetes</h2>
                <h3>Probability: {probability:.1%}</h3>
                <p>The model indicates a lower likelihood of diabetes risk.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Sigmoid visualization with user prediction
        z_vals = np.linspace(-10, 10, 200)
        prob_vals = st.session_state.predictor.sigmoid(z_vals)
        
        fig_sigmoid = go.Figure()
        
        # Sigmoid curve
        fig_sigmoid.add_trace(go.Scatter(
            x=z_vals,
            y=prob_vals,
            mode='lines',
            name='Sigmoid Function',
            line=dict(color='#74b9ff', width=3)
        ))
        
        # Decision boundary
        fig_sigmoid.add_hline(
            y=0.5, 
            line_dash="dash", 
            line_color="gray",
            annotation_text="Decision Boundary (50%)"
        )
        
        # User prediction point
        fig_sigmoid.add_trace(go.Scatter(
            x=[z_value],
            y=[probability],
            mode='markers',
            name='Your Prediction',
            marker=dict(
                size=15,
                color='#ff4b4b',
                symbol='circle',
                line=dict(width=2, color='white')
            )
        ))
        
        fig_sigmoid.update_layout(
            title="📈 Sigmoid Function with Your Prediction",
            xaxis_title="z = θᵀx (Decision Score)",
            yaxis_title="Probability of Diabetes",
            template="plotly_white",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig_sigmoid, use_container_width=True)
        
        # Disclaimer
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Important Disclaimer</h4>
            <p>This prediction is based on statistical modeling and should not be considered as medical advice. 
            The model considers family history and lifestyle factors, but actual diabetes risk depends on many 
            complex factors. Please consult with a healthcare professional for proper medical assessment and advice.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p style='text-align: center; font-size:18px;'>Made By: <a href="https://github.com/cursed-0men" target="_blank" style="color:#ff4b4b; text-decoration:none; font-weight:bold;">Dhyey Savaliya</a></p>
    <p>🧬 Diabetes Prediction | Built with Streamlit❤️ & Logistic Regression</p>
    <p><em>For educational purposes only. Not a substitute for professional medical advice.</em></p>
</div>
""", unsafe_allow_html=True)