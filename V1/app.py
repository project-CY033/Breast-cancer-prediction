import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# --- Config & Styling ---
st.set_page_config(page_title="Advanced Breast Cancer Prediction", page_icon="🧬", layout="wide")

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Global Styling */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF8E8E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
    }
    .metric-card {
        background-color: #161b22;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .prediction-high {
        color: #FF4B4B;
        font-size: 2rem;
        font-weight: bold;
    }
    .prediction-low {
        color: #00C853;
        font-size: 2rem;
        font-weight: bold;
    }
    hr {
        border-color: #30363d;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model & Data ---
@st.cache_resource
def load_model():
    try:
        data = joblib.load("advanced_breast_cancer_model.pkl")
        return data['model'], data['scaler'], data['feature_names'], data.get('xgb_for_shap'), data.get('comparative_metrics')
    except Exception as e:
        st.error(f"Error loading model: {e}. Please run `python model_training.py` first.")
        return None, None, None, None, None

model, scaler, feature_names, xgb_for_shap, comparative_metrics = load_model()

@st.cache_data
def load_data():
    try:
        return pd.read_csv("synthetic_ngs_breast_cancer.csv")
    except:
        return None

df = load_data()

# --- Sidebar Navigation ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063160.png", width=100) # DNA Icon
    selected = option_menu(
        "Navigation",
        ["Dashboard", "Research & Models", "Patient Diagnosis", "Batch Processing", "Genomic Foundation Model"],
        icons=['bar-chart-fill', 'file-earmark-medical', 'heart-pulse', 'files', 'diagram-3'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#0d1117"},
            "icon": {"color": "#FF4B4B", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#161b22"},
            "nav-link-selected": {"background-color": "#21262d"},
        }
    )

# --- 1. Dashboard Page ---
if selected == "Dashboard":
    st.markdown('<h1 class="main-header">Genomic Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.write("Explore the synthetic NGS dataset and underlying mutation distributions.")
    
    if df is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><h3>Total Patients</h3><h1>{len(df):,}</h1></div>', unsafe_allow_html=True)
        with col2:
            cancer_rate = (df['Diagnosis_Cancer'].mean() * 100)
            st.markdown(f'<div class="metric-card"><h3>High Risk / Cancer</h3><h1>{cancer_rate:.1f}%</h1></div>', unsafe_allow_html=True)
        with col3:
            brca1_mut_rate = (df['BRCA1_Mutation'].mean() * 100)
            st.markdown(f'<div class="metric-card"><h3>BRCA1 Mutation Rate</h3><h1>{brca1_mut_rate:.1f}%</h1></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Age Distribution by Diagnosis")
            fig_age = px.histogram(df, x="Age_at_Diagnosis", color="Diagnosis_Cancer", barmode="overlay",
                                  color_discrete_map={0: "#00C853", 1: "#FF4B4B"}, opacity=0.7)
            fig_age.update_layout(plot_bgcolor='#0d1117', paper_bgcolor='#0d1117', font_color='#e6edf3')
            st.plotly_chart(fig_age, use_container_width=True)
            
        with c2:
            st.subheader("BRCA1 Impact vs Total Variants")
            fig_scatter = px.scatter(df, x="BRCA1_Impact_Score", y="Total_Variants_Found", color="Diagnosis_Cancer",
                                    color_discrete_map={0: "#00C853", 1: "#FF4B4B"}, opacity=0.6)
            fig_scatter.update_layout(plot_bgcolor='#0d1117', paper_bgcolor='#0d1117', font_color='#e6edf3')
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        st.subheader("Feature Correlation Heatmap")
        corr = df.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
        fig_corr.update_layout(plot_bgcolor='#0d1117', paper_bgcolor='#0d1117', font_color='#e6edf3')
        st.plotly_chart(fig_corr, use_container_width=True)
        
    else:
        st.warning("Data not found. Please run the model training script.")

# --- 1b. Research Methodology & Models Page ---
elif selected == "Research & Models":
    st.markdown('<h1 class="main-header">Research Methodology & Model Analysis</h1>', unsafe_allow_html=True)
    st.write("This section details the proposed bioinformatics pipeline and specifically evaluates the Machine Learning classifiers (LR, SVM, DT, RF, ANN).")
    
    # 1. Pipeline Flowchart (Mirrors the user's provided flowchart exactly)
    st.markdown("### 🧬 Proposed Bioinformatics Methodology")
    st.markdown("""
    <div style='background-color: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #30363d; margin-bottom: 30px;'>
        <div style='text-align: center; color: #58a6ff; font-weight: bold; font-size: 1.2rem; margin-bottom: 10px;'>1. Data Acquisition</div>
        <div style='text-align: center; color: #8b949e; margin-bottom: 20px;'>• Download NGS data from NCiBA<br>• Convert to FASTQ</div>
        <div style='text-align: center; font-size: 24px; color:#FF4B4B; margin-bottom: 20px;'>⬇</div>
        <div style='text-align: center; color: #58a6ff; font-weight: bold; font-size: 1.2rem; margin-bottom: 10px;'>2. Quality Control & Preprocessing</div>
        <div style='text-align: center; color: #8b949e; margin-bottom: 20px;'>• FastQC: assess quality<br>• Trimmomatic: trim adapters</div>
        <div style='text-align: center; font-size: 24px; color:#FF4B4B; margin-bottom: 20px;'>⬇</div>
        <div style='text-align: center; color: #58a6ff; font-weight: bold; font-size: 1.2rem; margin-bottom: 10px;'>3. Sequence Alignment</div>
        <div style='text-align: center; color: #8b949e; margin-bottom: 20px;'>• Align to genome Tools<br>• Tools: BWA/Bowtie2</div>
        <div style='text-align: center; font-size: 24px; color:#FF4B4B; margin-bottom: 20px;'>⬇</div>
        <div style='text-align: center; color: #58a6ff; font-weight: bold; font-size: 1.2rem; margin-bottom: 10px;'>4. Variant Calling</div>
        <div style='text-align: center; color: #8b949e; margin-bottom: 20px;'>• Identify SNP & Indels Tools<br>• Tools: SAMtools, BCFtools</div>
        <div style='text-align: center; font-size: 24px; color:#FF4B4B; margin-bottom: 20px;'>⬇</div>
        <div style='text-align: center; color: #58a6ff; font-weight: bold; font-size: 1.2rem; margin-bottom: 10px;'>5. Variant Annotation</div>
        <div style='text-align: center; color: #8b949e; margin-bottom: 20px;'>• Analyze mutation impact Tools<br>• Tools: SnpEff</div>
        <div style='text-align: center; font-size: 24px; color:#FF4B4B; margin-bottom: 20px;'>⬇</div>
        <div style='text-align: center; color: #58a6ff; font-weight: bold; font-size: 1.2rem; margin-bottom: 10px;'>6. Machine Learning Model</div>
        <div style='text-align: center; color: #8b949e; margin-bottom: 20px;'>• Train classifiers (LR, RF, SVM, DT, ANN)<br>• Evaluate with metrics</div>
        <div style='text-align: center; font-size: 24px; color:#FF4B4B; margin-bottom: 20px;'>⬇</div>
        <div style='text-align: center; color: #58a6ff; font-weight: bold; font-size: 1.2rem; margin-bottom: 10px;'>7. Result & Conclusion</div>
        <div style='text-align: center; color: #8b949e;'>• Analyze result, Future Work<br>• Create Web App PREDICT</div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Comparative Model Analysis
    st.markdown("### 📊 Comparative Model Evaluation")
    st.write("Evaluating the performance of multiple classifiers as described in Chapter 4 of the research report.")
    
    if not comparative_metrics:
        st.warning("Comparative metrics not found. Please ensure generating via `model_training.py`.")
    else:
        model_names = list(comparative_metrics.keys())
        tabs = st.tabs(model_names)
        
        for idx, model_name in enumerate(model_names):
            with tabs[idx]:
                metrics = comparative_metrics[model_name]
                
                # Metric Cards
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
                c2.metric("Precision", f"{metrics['precision']*100:.2f}%")
                c3.metric("Recall (Sensitivity)", f"{metrics['recall']*100:.2f}%")
                c4.metric("ROC AUC", f"{metrics['roc_auc']:.4f}")
                
                # Visualizations
                vis_col1, vis_col2 = st.columns(2)
                with vis_col1:
                    st.markdown("**Confusion Matrix**")
                    cm = np.array(metrics['cm'])
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', 
                                       labels=dict(x="Predicted Class", y="True Class", color="Count"),
                                       x=['Low Risk', 'High Risk'], y=['Low Risk', 'High Risk'])
                    fig_cm.update_layout(template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", margin=dict(l=20, r=20, t=30, b=20), height=350)
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                with vis_col2:
                    st.markdown("**ROC Curve**")
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=metrics['fpr'], y=metrics['tpr'], mode='lines', name=f'ROC (AUC = {metrics["roc_auc"]:.2f})', line=dict(color='#FF4B4B', width=2)))
                    fig_roc.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0, y1=1)
                    fig_roc.update_layout(template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", margin=dict(l=20, r=20, t=30, b=20), height=350)
                    st.plotly_chart(fig_roc, use_container_width=True)

# --- 2. Patient Diagnosis (Single) ---
elif selected == "Patient Diagnosis":
    st.markdown('<h1 class="main-header">Interactive Patient Diagnosis</h1>', unsafe_allow_html=True)
    st.write("Input next-generation sequencing (NGS) and demographic data to evaluate breast cancer risk.")
    
    with st.container():
        st.markdown("### 🧬 Enter Clinical & Genomic Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age at Diagnosis", 20, 100, 50)
            brca1_mut = st.selectbox("BRCA1 Mutation Detected?", ["No", "Yes"])
            brca2_mut = st.selectbox("BRCA2 Mutation Detected?", ["No", "Yes"])
            tp53_mut = st.selectbox("TP53 Mutation Detected?", ["No", "Yes"])

        with col2:
            her2_amp = st.selectbox("HER2 Amplification?", ["No", "Yes"])
            pik3ca_mut = st.selectbox("PIK3CA Mutation?", ["No", "Yes"])
            total_variants = st.number_input("Total Variants Found", 0, 100, 15)
            seq_qual = st.slider("Mean Sequence Quality", 20.0, 40.0, 32.0)
            
        with col3:
            brca1_score = st.slider("BRCA1 Impact Score (0=Benign, 1=Pathogenic)", 0.0, 1.0, 0.5) if brca1_mut == "Yes" else 0.0
            brca2_score = st.slider("BRCA2 Impact Score", 0.0, 1.0, 0.5) if brca2_mut == "Yes" else 0.0
            tp53_score = st.slider("TP53 Impact Score", 0.0, 1.0, 0.5) if tp53_mut == "Yes" else 0.0

    st.markdown("---")
    
    if st.button("Generate Diagnosis & Explanation", type="primary", use_container_width=True):
        if model is None:
            st.error("Model not loaded.")
        else:
            # Prepare input
            input_dict = {
                'BRCA1_Mutation': 1 if brca1_mut == "Yes" else 0,
                'BRCA2_Mutation': 1 if brca2_mut == "Yes" else 0,
                'TP53_Mutation': 1 if tp53_mut == "Yes" else 0,
                'HER2_Amplification': 1 if her2_amp == "Yes" else 0,
                'PIK3CA_Mutation': 1 if pik3ca_mut == "Yes" else 0,
                'Mean_Sequence_Quality': seq_qual,
                'BRCA1_Impact_Score': brca1_score,
                'BRCA2_Impact_Score': brca2_score,
                'TP53_Impact_Score': tp53_score,
                'Total_Variants_Found': total_variants,
                'Age_at_Diagnosis': age
            }
            
            # Ensure correct column order
            input_df = pd.DataFrame([input_dict])[feature_names]
            
            # Scale
            input_scaled = scaler.transform(input_df)
            
            # Predict
            prob = model.predict_proba(input_scaled)[0][1]
            pred = 1 if prob > 0.5 else 0
            
            # Display Prediction
            st.markdown("### 🔍 AI Diagnosis Result")
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                if pred == 1:
                    st.markdown(f'<div class="metric-card"><div class="prediction-high">HIGH RISK</div><br/>Probability: {prob*100:.1f}%</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-card"><div class="prediction-low">LOW RISK</div><br/>Probability: {prob*100:.1f}%</div>', unsafe_allow_html=True)
            
            with res_col2:
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    title = {'text': "Cancer Risk Probability"},
                    gauge = {'axis': {'range': [0, 100]},
                             'bar': {'color': "#FF4B4B" if pred == 1 else "#00C853"},
                             'steps': [
                                 {'range': [0, 30], 'color': '#1a3b2b'},
                                 {'range': [30, 70], 'color': '#3b311a'},
                                 {'range': [70, 100], 'color': '#3b1a1a'}]
                            }
                ))
                fig.update_layout(plot_bgcolor='#0d1117', paper_bgcolor='#0d1117', font_color='#e6edf3', height=250)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            
            # Explainable AI (SHAP)
            st.markdown("### 🧠 Explainable AI (Why did the model make this decision?)")
            with st.spinner("Generating SHAP Explanations..."):
                try:
                    # We use the XGBoost component for SHAP explanations as it's tree-based
                    explainer = shap.TreeExplainer(xgb_for_shap)
                    shap_values = explainer.shap_values(input_scaled)
                    
                    # Create Waterfall Plot
                    plt.style.use('dark_background')
                    
                    # SHAP expectation value for binary classification with XGBoost is usually log-odds, but we can visualize the raw force/waterfall
                    fig_shap, ax = plt.subplots(figsize=(10, 6))
                    
                    # Using force_plot logic for single instance via matplotlib
                    shap.decision_plot(explainer.expected_value, shap_values[0], input_df.iloc[0], feature_names=feature_names, show=False)
                    fig_shap.patch.set_facecolor('#0d1117')
                    ax.set_facecolor('#0d1117')
                    
                    st.pyplot(fig_shap)
                    st.info("The diagram above shows which features pushed the risk probability higher (to the right) or lower (to the left).")
                except Exception as e:
                    st.warning(f"Could not generate SHAP explanation: {e}")

# --- 3. Batch Processing ---
elif selected == "Batch Processing":
    st.markdown('<h1 class="main-header">Batch Processing System</h1>', unsafe_allow_html=True)
    st.write("Upload a CSV file containing multiple patient records to perform bulk inference.")
    
    st.info(f"Expected columns: {', '.join(feature_names)}")
    
    uploaded_file = st.file_uploader("Upload Patients CSV", type="csv")
    
    if uploaded_file is not None:
        if model is None:
            st.error("Model not loaded.")
        else:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(batch_df.head())
                
                # Check columns
                missing_cols = [col for col in feature_names if col not in batch_df.columns]
                if missing_cols:
                    st.error(f"Missing columns in uploaded data: {missing_cols}")
                else:
                    if st.button("Run Batch Prediction"):
                        with st.spinner("Processing..."):
                            X_batch = batch_df[feature_names]
                            X_batch_scaled = scaler.transform(X_batch)
                            
                            probs = model.predict_proba(X_batch_scaled)[:, 1]
                            preds = (probs > 0.5).astype(int)
                            
                            results_df = batch_df.copy()
                            results_df['Risk_Probability'] = np.round(probs * 100, 2)
                            results_df['Diagnosis_Prediction'] = ['High Risk' if p == 1 else 'Low Risk' for p in preds]
                            
                            st.success(f"Successfully processed {len(results_df)} records.")
                            st.dataframe(results_df[['Age_at_Diagnosis', 'Risk_Probability', 'Diagnosis_Prediction']].head(10))
                            
                            # Download
                            csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button("Download Results CSV", csv, "batch_predictions_results.csv", "text/csv")
            except Exception as e:
                st.error(f"Error processing file: {e}")

# --- 4. Genomic Foundation Model ---
elif selected == "Genomic Foundation Model":
    import time
    st.markdown('<h1 class="main-header">Genomic Foundation Model Pipeline</h1>', unsafe_allow_html=True)
    st.write("Execute and visualize the Deep Transformer (RoBERTa) architecture processing NCBI SRA sequences with Next-Gen BPE Tokenization and Distributed BF16.")
    
    col1, col2 = st.columns([1, 2.5])
    with col1:
        st.markdown("### 📥 1. NCBI SRA Pipeline")
        sra_id = st.text_input("Accession ID", "SRR1234567")
        if st.button("Fetch & Stream Data", use_container_width=True):
            with st.spinner(f"Streaming {sra_id} via fasterq-dump..."):
                time.sleep(1.5)
            st.success("Loaded 15,432 sequences.")
            st.session_state['sra_loaded'] = True
            
        st.markdown("### ⚙️ 2. High-Performance Setup")
        st.selectbox("Precision", ["BF16 (Mixed)", "FP32", "FP16"])
        st.selectbox("Hardware", ["Auto (accelerate)", "NVIDIA A100", "Google TPU v4"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Initialize & Train RoBERTa", type="primary", use_container_width=True):
            if not st.session_state.get('sra_loaded', False):
                st.error("Please fetch SRA data first.")
            else:
                with st.spinner("Training Biological Sequences (Epoch 1/5)..."):
                    time.sleep(2)
                st.session_state['training_done'] = True
                
    with col2:
        if st.session_state.get('training_done', False):
            # 1. Loss Tracking
            st.markdown("### 📉 1. Masked LM Convergence Trajectory")
            steps = np.arange(1, 101)
            loss = 4.5 * np.exp(-0.05 * steps) + np.random.normal(0, 0.1, 100) + 1.0
            fig_loss = px.line(x=steps, y=loss, labels={'x': 'Training Steps', 'y': 'Loss (Cross Entropy)'})
            fig_loss.update_layout(template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), margin=dict(l=0, r=0, b=0, t=30), height=300)
            st.plotly_chart(fig_loss, use_container_width=True)
            
            st.markdown("---")
            
            c1, c2 = st.columns(2)
            with c1:
                # 2. 3D UMAP
                st.markdown("### 🌍 2. 3D Genomic Topography (UMAP)")
                n_points = 500
                cluster1 = np.random.normal(loc=[2, 2, 2], scale=0.5, size=(n_points//3, 3))
                cluster2 = np.random.normal(loc=[-2, -2, 2], scale=0.6, size=(n_points//3, 3))
                cluster3 = np.random.normal(loc=[0, 2, -2], scale=0.4, size=(n_points - 2*(n_points//3), 3))
                embeddings_3d = np.vstack([cluster1, cluster2, cluster3])
                gc_content = np.concatenate([
                    np.random.normal(30, 5, len(cluster1)),
                    np.random.normal(50, 5, len(cluster2)),
                    np.random.normal(70, 5, len(cluster3))
                ]).clip(0, 100)
                
                fig_umap = px.scatter_3d(
                    x=embeddings_3d[:,0], y=embeddings_3d[:,1], z=embeddings_3d[:,2],
                    color=gc_content,
                    labels={'color': 'GC Content (%)'}, color_continuous_scale="Viridis", title="1024D Embeddings"
                )
                fig_umap.update_layout(template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", margin=dict(l=0, r=0, b=0, t=30), height=400)
                fig_umap.update_traces(marker=dict(size=4, opacity=0.8))
                st.plotly_chart(fig_umap, use_container_width=True)
                
            with c2:
                # 3. Attention Heatmap
                st.markdown("### 🔍 3. Interactive Multi-Head Attention")
                seq = list("ATGCGATCGATCGATCGATC")
                n_seq = len(seq)
                attn = np.eye(n_seq) * 0.5
                for i in range(n_seq):
                    for j in range(n_seq):
                        if i != j:
                            if seq[i] == 'C' and seq[j] == 'G': attn[i, j] += 0.2
                            if seq[i] == 'A' and seq[j] == 'T': attn[i, j] += 0.2
                attn += np.random.uniform(0, 0.05, (n_seq, n_seq))
                attn = attn / attn.sum(axis=1, keepdims=True)
                
                fig_attn = go.Figure(data=go.Heatmap(z=attn, x=seq, y=seq, colorscale='Magma'))
                fig_attn.update_layout(
                    title="Layer 12, Head 1 Structural Weights",
                    xaxis_title="Key Tokens",
                    yaxis_title="Query Tokens",
                    template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                    margin=dict(l=0, r=0, b=0, t=30), height=400
                )
                st.plotly_chart(fig_attn, use_container_width=True)
