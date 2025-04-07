import streamlit as st
from enum import Enum
from typing import List, Dict, Any

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

def render_risk_level(risk_level: str):
    """Render the risk level with appropriate color coding"""
    color_map = {
        "low": "green",
        "medium": "orange",
        "high": "red"
    }
    
    st.markdown(f"### Risk Level: <span style='color:{color_map[risk_level]};'>{risk_level.upper()}</span>", unsafe_allow_html=True)
    
    # Create a progress bar to visualize risk level
    risk_values = {"low": 0.33, "medium": 0.66, "high": 1.0}
    st.progress(risk_values[risk_level])

def render_security_concerns(concerns: List[Dict[str, Any]]):
    """Render the list of security concerns"""
    st.markdown("### Security Concerns")
    
    if not concerns:
        st.info("No security concerns detected.")
        return
    
    for concern in concerns:
        with st.expander(f"{concern['category']} ({concern['riskLevel'].upper()})"):
            st.markdown(f"**Description:** {concern['description']}")
            st.markdown(f"**Confidence:** {concern['confidence']:.2f}")
            
            if concern.get('timestamp'):
                st.markdown(f"**Timestamp:** {concern['timestamp']}")
            
            if concern.get('screenshotUrl'):
                st.image(concern['screenshotUrl'], caption=f"Evidence for {concern['category']}")

def render_recommendations(recommendations: List[str]):
    """Render the list of recommendations"""
    st.markdown("### Recommendations")
    
    if not recommendations:
        st.info("No specific recommendations.")
        return
    
    for i, recommendation in enumerate(recommendations):
        st.markdown(f"{i+1}. {recommendation}")

def render_security_analysis(analysis_data: Dict[str, Any]):
    """Main function to render the security analysis"""
    st.title("Security Analysis")
    
    # Render risk level
    render_risk_level(analysis_data['overallRiskLevel'])
    
    # Render security concerns
    render_security_concerns(analysis_data['securityConcerns'])
    
    # Render recommendations
    render_recommendations(analysis_data['recommendations']) 