import streamlit as st
from datetime import datetime
from header import set_header

# Call the global configuration
set_header()

st.title("streamlit sample")
    
st.markdown(
    """
# Hello!
            
Welcome to my FullstackGPT Portfolio!
            
Here are the apps I made:
            
- [x] [DocumentGPT](/DocumentGPT)
- [x] [PrivateGPT](/PrivateGPT)
- [x] [QuizGPT](/QuizGPT)
- [x] [SiteGPT](/SiteGPT)
- [x] [MeetingGPT](/MeetingGPT)
- [x] [InvestorGPT](/InvestorGPT)
"""
)

st.markdown(f"""
    <div style="position: fixed;
                bottom: 10px;
                right: 10px;
                padding: 10px;
                border-radius: 5px;
                ">
        {datetime.today().strftime("%H:%M:%S")}
    </div>
    """, unsafe_allow_html=True)