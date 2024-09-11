import streamlit as st

def set_header():
    st.set_page_config(
        page_title="Streamlit Sample",
        page_icon="🌈",
        layout="wide",  # This makes the layout wide, giving more space
        # initial_sidebar_state="collapsed",  # This hides the auto-sidebar by default
        menu_items={  # You can also remove unnecessary menu items if needed
            'Get help': None,
            'Report a bug': None,
            'About': None
        }
    )    
    # create_sidebar()

    
# def create_sidebar():
    
#     # Divider and "samples" section
#     st.sidebar.title("samples")    
    
#     # Add additional items or sample apps under the new section
#     st.sidebar.write("[sample_tab_selectbox](sample_tab_selectbox)")
#     # st.sidebar.write("[Sample App 2](sample2)")
#     # st.sidebar.markdown("---")  # Horizontal rule (divider)

# Function to load pages manually
